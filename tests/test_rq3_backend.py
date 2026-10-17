"""CPU tests of the scientific intervention and backend instrumentation.

These do not claim a Step1X GPU deployment test: that requires the documented
development pilot and determinism gate on the actual experiment hardware.
"""

from contextlib import nullcontext
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Union
import unittest

import numpy as np
from PIL import Image

from src.rq3.backend import REGIONE_DEFAULTS, Step1XBackend, _replace
from src.rq3.regione import AttentionRecorder, NoRIKVProcessor, RegionTrace


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def detach(self):
        return self

    def contiguous(self):
        return self

    def view(self, dtype):
        return Array(self.values.view(dtype))

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def flatten(self):
        return Array(self.values.flatten())

    def tolist(self):
        return self.values.tolist()


class BackendTests(unittest.TestCase):
    def test_no_rikv_passes_only_current_tokens_and_reduced_rope(self):
        observed = {}

        def native(attn, hidden, encoder, mask, rope):
            observed.update(hidden=hidden, encoder=encoder, rope=rope)
            # A simple attention-like aggregation makes omitted context
            # observable: only supplied current-stream values contribute.
            return float(hidden.values.sum() + encoder.values.sum())

        reduced = Array(np.array([[[1.], [2.]]]))
        encoder = Array(np.array([[[4.]]]))
        reduced_rope = object()
        processor = NoRIKVProcessor(native)
        output = processor(None, reduced, encoder, image_rotary_emb=reduced_rope, tag="cond")
        self.assertEqual(output, 7.)
        self.assertIs(observed["hidden"], reduced)
        self.assertIs(observed["rope"], reduced_rope)
        self.assertFalse(any("cache" in key for key in vars(processor)))

    def test_trace_distinguishes_refresh_reduced_calls_and_velocity_reuse(self):
        trace = RegionTrace(True, True, True)
        trace.total_latent_tokens = trace.condition_tokens = 4
        manager = SimpleNamespace(current_step=5, warmup_step=6, post_step=2,
                                  inference_step=28, prev_refresh_step=15,
                                  edited_ids=Array([[1, 2]]), height=32, width=32,
                                  patch_size=2, vae_scale_factor=8)
        recorder = AttentionRecorder(lambda *args, **kwargs: None, trace, manager, True)
        recorder(None, Array(np.zeros((1, 8, 1))), tag="cond")
        trace.transformer_call(Array(np.zeros((1, 8, 1))), "cond")
        trace.active_tokens = 4
        trace.finish_step(5, manager)
        self.assertEqual(trace.steps[0]["rikv_cache_writes"], 1)
        self.assertTrue(trace.steps[0]["refresh_event"])
        self.assertEqual(trace.steps[0]["rho_t"], 1.)
        self.assertEqual(trace.steps[0]["selected_fraction"], .5)
        manager.current_step = 6
        recorder(None, Array(np.zeros((1, 2, 1))), tag="cond")
        trace.transformer_call(Array(np.zeros((1, 2, 1))), "cond")
        trace.active_tokens = 2
        trace.finish_step(6, manager)
        manager.current_step = 7
        trace.finish_step(7, manager)
        output = trace.as_dict()
        self.assertEqual(output["generator_calls"], 2)
        self.assertEqual(output["full_image_generator_calls"], 1)
        self.assertEqual(output["reduced_token_generator_calls"], 1)
        self.assertEqual(output["avd_cache_reuse_steps"], 1)
        self.assertEqual(output["steps"][1]["rikv_context_reuses"], 1)
        self.assertEqual(output["selected_regions"][0]["selected_indices"], [1, 2])

    def test_no_rikv_preserves_scheduler_refresh_logging(self):
        trace = RegionTrace(True, False, True)
        trace.total_latent_tokens = trace.condition_tokens = trace.active_tokens = 4
        manager = SimpleNamespace(edited_ids=Array([[1, 2]]), warmup_step=6, prev_refresh_step=15)
        trace.transformer_call(Array(np.zeros((1, 8, 1))), "cond")
        trace.finish_step(15, manager)
        self.assertTrue(trace.steps[0]["refresh_event"])
        self.assertEqual(trace.steps[0]["rikv_cache_writes"], 0)

    def test_method_override_is_restored_after_failure(self):
        class Subject:
            def method(self):
                return "original"
        subject = Subject()
        with self.assertRaises(RuntimeError):
            with _replace(subject, "method", lambda: "wrapped"):
                self.assertEqual(subject.method(), "wrapped")
                raise RuntimeError("failure")
        self.assertNotIn("method", subject.__dict__)
        self.assertEqual(subject.method(), "original")

    def test_replay_instrumentation_counts_real_calls_and_preserves_noise(self):
        backend = Step1XBackend.__new__(Step1XBackend)
        image = Image.new("RGB", (32, 32), "red")
        calls = []

        class Generator:
            def __init__(self, **kwargs):
                pass
            def manual_seed(self, seed):
                self.seed = seed
                return self

        class FakePipe:
            def __init__(self):
                self.transformer = SimpleNamespace(forward=lambda **kwargs: None,
                                                   transformer_blocks=[], single_transformer_blocks=[])
                self.scheduler = SimpleNamespace(step=lambda noise, time, sample, **kwargs: [sample])
                self.vae = SimpleNamespace(decode=lambda value, **kwargs: image)

            def encode_image(self, image):
                return None, image, image.size, 32, 32

            def encode_prompt(self, prompt):
                return prompt

            def prepare_latents(self, generator):
                noise = np.random.default_rng(generator.seed).normal(size=(1, 4, 3)).astype("float32")
                return Array(noise), Array(np.zeros((1, 4, 3))), None, None

            def __call__(self, **kwargs):
                calls.append(kwargs)
                self.encode_image(kwargs["image"])
                self.encode_prompt(kwargs["prompt"])
                latents, _, _, _ = self.prepare_latents(kwargs["generator"])
                for step in range(28):
                    for _ in range(2):
                        self.transformer.forward(hidden_states=Array(np.zeros((1, 8, 3))))
                    self.scheduler.step(None, step, latents)
                    kwargs["callback_on_step_end"](self, step, step, {})
                self.vae.decode(latents)
                return SimpleNamespace(images=[image])

        backend.pipe = FakePipe()
        backend.config = {"preprocessing": {"size_level": 1024}}
        backend.regione_params = dict(REGIONE_DEFAULTS)
        backend.inplace = SimpleNamespace(MANAGER=None, Step1XEditV1P2Manager=lambda: object())
        backend.torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None,
                                       max_memory_allocated=lambda: 1234), Generator=Generator,
                                       inference_mode=nullcontext, uint8=np.uint8)
        backend._sync = lambda: None
        backend._seed = lambda seed: None
        backend._resident = lambda name: None
        backend._configure_condition = lambda condition: (RegionTrace(False, False, False), None)
        first = backend.generate(image, "original prompt", 42, "A")
        second = backend.generate(image, "rewritten prompt", 42, "B")
        third = backend.generate(image, "original prompt", 7, "A")
        self.assertEqual(first["initial_noise_sha256"], second["initial_noise_sha256"])
        self.assertNotEqual(first["initial_noise_sha256"], third["initial_noise_sha256"])
        self.assertEqual(first["regione"]["generator_calls"], 56)
        self.assertEqual(first["regione"]["full_image_generator_calls"], 56)
        self.assertEqual(first["regione"]["avd_cache_reuse_steps"], 0)
        self.assertEqual(first["preprocessing"]["inference_size"], [32, 32])
        self.assertGreaterEqual(first["timings"]["measured_replay_seconds"], first["timings"]["diffusion_seconds"])
        for call in calls:
            self.assertFalse(call["enable_thinking_mode"])
            self.assertFalse(call["enable_reflection_mode"])
            self.assertEqual(call["max_try_cnt"], 1)
            self.assertEqual(call["num_inference_steps"], 28)
            self.assertEqual(call["true_cfg_scale"], 6)


class UpstreamAttentionTests(unittest.TestCase):
    """Run real pinned processor code on tiny CPU tensors, without model loads.

    The imports surrounding the processors require CUDA-only Triton. Loading
    their AST definitions isolates the actual tensor code from those imports;
    only Triton's scatter-linear primitive is substituted with a CPU operation.
    Supply RQ3_DIFFUSERS_SOURCE/RQ3_REGIONE_SOURCE or use external/ checkouts.
    """

    def test_pinned_processors_remove_cached_context_in_both_block_types(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Install torch for the CPU tensor test")
        diffusers = Path(os.environ.get("RQ3_DIFFUSERS_SOURCE", "external/diffusers"))
        regione = Path(os.environ.get("RQ3_REGIONE_SOURCE", "external/RegionE"))
        transformer_file = diffusers / "src/diffusers/models/transformers/transformer_step1x_edit.py"
        regione_file = regione / "RegionE/Step1XEditV1P2/inplace.py"
        if not transformer_file.exists() or not regione_file.exists():
            self.skipTest("Pinned upstream source checkouts required for numerical integration test")
        namespace = {"torch": torch, "F": torch.nn.functional, "Optional": Optional,
                     "Union": Union, "Tuple": Tuple, "flash_attn": None}

        def load_definitions(path, names):
            tree = ast.parse(path.read_text())
            nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
            self.assertEqual({node.name for node in nodes}, set(names))
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

        def sdpa(query, key, value, attn_mask=None, backend=None):
            out = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attn_mask)
            return out.transpose(1, 2)

        def cpu_partially_linear(hidden, weight, bias, selection, cache):
            cache[:, selection] = torch.nn.functional.linear(hidden, weight, bias)

        namespace.update(dispatch_attention_fn=sdpa, _partially_linear=cpu_partially_linear)
        load_definitions(diffusers / "src/diffusers/models/embeddings.py", ["apply_rotary_emb"])
        load_definitions(transformer_file, ["_get_projections", "_get_fused_projections",
                                           "_get_qkv_projections", "Step1XEditAttnProcessor"])
        load_definitions(regione_file, ["RegionEStep1XEditAttnProcessor"])
        torch.manual_seed(6)
        text = torch.randn(1, 2, 4)
        images = torch.randn(1, 6, 4)  # 3 generated + 3 source-reference tokens
        edit_ids = torch.tensor([[0, 2]])
        active_images = images[:, [0, 2]] + .25
        full_rope = (torch.ones(8, 4), torch.zeros(8, 4))
        reduced_rope = (torch.ones(4, 4), torch.zeros(4, 4))

        def linear_identity():
            layer = torch.nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                layer.weight.copy_(torch.eye(4))
                layer.bias.zero_()
            return layer

        with torch.inference_mode():
            for single in (False, True):
                with self.subTest(single_block=single):
                    manager = SimpleNamespace(current_step=5, warmup_step=6, post_step=2,
                                              inference_step=28, prev_refresh_step=15,
                                              edited_ids=edit_ids, txt_length=2, neg_txt_length=2,
                                              image_rotary_emb=full_rope)
                    namespace["MANAGER"] = manager
                    attn = SimpleNamespace(heads=1, fused_projections=False,
                                           to_q=linear_identity(), to_k=linear_identity(), to_v=linear_identity(),
                                           norm_q=torch.nn.Identity(), norm_k=torch.nn.Identity(),
                                           added_kv_proj_dim=None if single else 4,
                                           add_q_proj=linear_identity(), add_k_proj=linear_identity(), add_v_proj=linear_identity(),
                                           norm_added_q=torch.nn.Identity(), norm_added_k=torch.nn.Identity(),
                                           to_out=[torch.nn.Identity(), torch.nn.Identity()], to_add_out=torch.nn.Identity())
                    native = namespace["Step1XEditAttnProcessor"]()
                    nocache = NoRIKVProcessor(native)
                    cached = namespace["RegionEStep1XEditAttnProcessor"](single)
                    full_hidden = torch.cat([text, images], dim=1) if single else images
                    reduced_hidden = torch.cat([text, active_images], dim=1) if single else active_images
                    encoder = None if single else text

                    def flatten(result):
                        return result if single else torch.cat([result[1], result[0]], dim=1)

                    warmup_cached = cached(attn, full_hidden, encoder, image_rotary_emb=full_rope, tag="cond")
                    warmup_native = nocache(attn, full_hidden, encoder, image_rotary_emb=full_rope, tag="cond")
                    torch.testing.assert_close(flatten(warmup_cached), flatten(warmup_native), rtol=1e-6, atol=1e-6)
                    manager.current_step = 6
                    reduced_cached = flatten(cached(attn, reduced_hidden, encoder, image_rotary_emb=reduced_rope, tag="cond"))
                    reduced_no_context = flatten(nocache(attn, reduced_hidden, encoder, image_rotary_emb=reduced_rope, tag="cond"))
                    reduced_all = torch.cat([text, active_images], dim=1)
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        reduced_all[:, None], reduced_all[:, None], reduced_all[:, None])[:, 0]
                    torch.testing.assert_close(reduced_no_context, expected, rtol=1e-6, atol=1e-6)
                    self.assertGreater(torch.max(torch.abs(reduced_cached - reduced_no_context)).item(), .01)
                    # The cache retains full context, whereas the no-RIKV
                    # processor has no cache and sees four text/edit tokens.
                    self.assertEqual(cached.k_cache_even.shape[1], 8 if single else 6)
                    self.assertEqual(reduced_no_context.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
