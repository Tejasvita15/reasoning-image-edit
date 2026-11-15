"""CPU refinement contracts with a trainable mock model; no upstream downloads."""

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.rq2.refine_unireason import refine_with_cot


class MockCache:
    def __init__(self, layers):
        self.layers = layers
        self.context = []
        self.tensors = []


class MockModel:
    def __init__(self):
        self.config = SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=2))
        # Trainable weights expose accidental autograd graphs in cached tensors.
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.modes = []
        self.updates = []
        self.prepared_images = []
        self.cfg_inputs = []
        self.generation = None

    def _update(self, cache, kind, content):
        self.modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        cache.tensors.append(self.weight * (len(cache.context) + 1))
        cache.context.append((kind, content))
        self.updates.append((kind, content))
        return cache

    def _prepare_images(self, kind, *, curr_kvlens, curr_rope, images,
                        transforms, new_token_ids):
        self.prepared_images.append((kind, images, transforms, new_token_ids))
        content = tuple(image.info["label"] for image in images)
        return {"content": content}, [curr_kvlens[0] + 1], [curr_rope[0] + 1]

    def prepare_vae_images(self, **kwargs):
        return self._prepare_images("vae", **kwargs)

    def forward_cache_update_vae(self, vae, cache, *, content):
        self.vae = vae
        return self._update(cache, "vae", content)

    def prepare_vit_images(self, **kwargs):
        return self._prepare_images("vit", **kwargs)

    def forward_cache_update_vit(self, cache, *, content):
        return self._update(cache, "vit", content)

    def prepare_prompts(self, *, curr_kvlens, curr_rope, prompts, tokenizer,
                        new_token_ids):
        return {"content": tuple(prompts)}, [curr_kvlens[0] + 1], [curr_rope[0] + 1]

    def forward_cache_update_text(self, cache, *, content):
        return self._update(cache, "text", content)

    def prepare_vae_latent_cfg(self, *, curr_kvlens, curr_rope, image_sizes):
        index = len(self.cfg_inputs)
        result = {f"cfg_{name}": f"branch{index}:{name}" for name in
                  ("packed_position_ids", "packed_query_indexes",
                   "key_values_lens", "packed_key_value_indexes")}
        self.cfg_inputs.append((curr_kvlens[:], curr_rope[:], image_sizes, result))
        return result

    def prepare_vae_latent(self, *, curr_kvlens, curr_rope, image_sizes,
                           new_token_ids):
        height, width = image_sizes[0]
        self.noise = torch.rand(height // 16 * (width // 16) * 64)
        return {"packed_noise": self.noise, "latent_marker": "main"}

    def generate_image(self, **kwargs):
        self.modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        self.generation = kwargs
        return [kwargs["packed_noise"]]


class MockVAE:
    def decode(self, latent):
        self.mode = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
        self.latent = latent.clone()
        return torch.nn.functional.interpolate(latent[:, :3] * 2 - 1, scale_factor=8,
                                                mode="nearest")


@pytest.fixture
def refinement(monkeypatch):
    for name in ("modeling", "modeling.bagel", "modeling.bagel.qwen2_navit"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["modeling.bagel.qwen2_navit"].NaiveCache = MockCache
    autocast = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    cleanup = Mock()
    monkeypatch.setattr(torch.amp, "autocast", autocast)
    monkeypatch.setattr(torch.cuda, "empty_cache", cleanup)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    source = Image.new("RGB", (32, 16), "red")
    source.info["label"] = "source"
    first = Image.new("RGB", (32, 16), "blue")
    first.info["label"] = "first"
    model, vae = MockModel(), MockVAE()
    output_size = Mock(side_effect=lambda image: image.size)
    runner = SimpleNamespace(model=model, vae_model=vae, tokenizer=object(),
                             new_token_ids={"image": 7}, vae_transform=object(),
                             vit_transform=object(), _move=lambda value: value,
                             _output_size=output_size)

    def call(seed=42):
        return refine_with_cot(runner, src_image=source, first_image=first,
                               enhance_prompt="refine the original instruction",
                               think_text="the cached plan", seed=seed,
                               num_timesteps=17, cfg_text_scale=4.25, cfg_img_scale=1.75)

    with torch.random.fork_rng(devices=[]):
        yield SimpleNamespace(call=call, model=model, vae=vae, runner=runner,
                              source=source, first=first, output_size=output_size,
                              autocast=autocast, cleanup=cleanup)


def test_trainable_cache_construction_and_deepcopy_run_in_inference_mode(refinement):
    with torch.enable_grad():
        output = refinement.call()

    assert isinstance(output, Image.Image)
    assert refinement.model.weight.requires_grad
    assert refinement.model.modes == [(False, True)] * 5
    assert refinement.vae.mode == (False, True)
    main = refinement.model.generation["past_key_values"]
    snapshot = refinement.model.generation["cfg_text_past_key_values"]
    assert snapshot is not main
    for original, copied in zip(main.tensors, snapshot.tensors):
        assert not original.requires_grad and original.grad_fn is None
        assert not copied.requires_grad and copied.grad_fn is None
        assert original.data_ptr() != copied.data_ptr()
        assert torch.equal(original, copied)


def test_refinement_keeps_images_and_text_in_their_guidance_branches(refinement):
    refinement.call()
    model = refinement.model
    image_context = [("vae", ("source", "first")), ("vit", ("source", "first"))]
    assert model.updates == image_context + [
        ("text", ("refine the original instruction",)),
        ("text", ("the cached plan",)),
    ]
    assert model.generation["past_key_values"].context == image_context + [
        ("text", ("the cached plan",))]
    assert model.generation["cfg_text_past_key_values"].context == image_context
    assert model.generation["cfg_img_past_key_values"].context == [
        ("text", ("refine the original instruction",))]
    for kind, images, transforms, token_ids in model.prepared_images:
        assert images == [refinement.source, refinement.first]
        assert transforms is getattr(refinement.runner, f"{kind}_transform")
        assert token_ids is refinement.runner.new_token_ids
    assert model.vae is refinement.vae
    assert [entry[:3] for entry in model.cfg_inputs] == [
        ([2], [2], [(16, 32)]), ([1], [1], [(16, 32)])]


def test_generation_parameters_and_decode_layout_are_preserved(refinement):
    image = refinement.call()
    args = refinement.model.generation
    assert {name: args[name] for name in ("num_timesteps", "cfg_text_scale", "cfg_img_scale",
                                        "cfg_type", "cfg_interval", "cfg_renorm_min",
                                        "cfg_renorm_type", "timestep_shift", "latent_marker")} == {
        "num_timesteps": 17, "cfg_text_scale": 4.25, "cfg_img_scale": 1.75,
        "cfg_type": "serial_text_img", "cfg_interval": [0.0, 1.0],
        "cfg_renorm_min": 0.0, "cfg_renorm_type": "text_channel",
        "timestep_shift": 3.0, "latent_marker": "main",
    }
    for prefix, cfg in zip(("cfg_text", "cfg_img"), refinement.model.cfg_inputs):
        for name, value in cfg[3].items():
            assert args[f"{prefix}_{name.removeprefix('cfg_')}"] == value
    assert args["packed_noise"] is refinement.model.noise
    refinement.output_size.assert_called_once_with(refinement.first)
    assert refinement.vae.latent.shape == (1, 16, 2, 4)
    assert image.mode == "RGB" and image.size == (32, 16)
    refinement.cleanup.assert_called_once_with()
    assert refinement.autocast.call_count == 5
    for call in refinement.autocast.call_args_list:
        assert call.args == ("cuda",) and call.kwargs == {"dtype": torch.bfloat16}


def test_seed_replays_identical_pixels_and_distinguishes_another_seed(refinement):
    first = refinement.call(seed=1234)
    torch.rand(13)
    repeated = refinement.call(seed=1234)
    different = refinement.call(seed=7)
    assert first.tobytes() == repeated.tobytes()
    assert first.tobytes() != different.tobytes()
