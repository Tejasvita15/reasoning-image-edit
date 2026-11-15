# src/unireason_runner.py

"""
UniReason (BAGEL-based) wrapper for RQ experiments.

Key differences from Step1XRunner
----------------------------------
- Custom BAGEL model stack; NOT a diffusers pipeline.
  Requires the UniReason repository to be on sys.path
  (set model.unireason_repo in the config).

- Two CFG guidance scales:
    cfg_text_scale  (config: true_cfg_scale  -- shared key)
    cfg_img_scale   (config: cfg_img_scale)

- Thinking produces a full <think>...</think> chain that is
  APPENDED to the instruction in the KV context, unlike Step1X
  where the rewrite replaces the prompt.

- No CPU offload equivalent; model is always fully on one GPU.

- Single-process / single-GPU: the upstream eval scripts use
  torchrun with 8 GPUs; we run one process on one device.

Config layout
-------------
model:
  backend        : unireason
  model_id       : Alex11556666/UniReason   (for logging)
  model_path     : /path/to/downloaded/model/dir
  unireason_repo : /path/to/UniReason/UniReason
  max_latent_size: 64          # optional, default 64
  dtype          : bfloat16    # validator compat; always autocast bfloat16
  cpu_offload    : false        # validator compat; not used

generation:
  num_inference_steps  : 50    # maps to num_timesteps
  true_cfg_scale       : 4.0   # maps to cfg_text_scale
  cfg_img_scale        : 1.5   # image guidance (BAGEL-specific)
  enable_thinking_mode : false  # True = live thinking at inference time
  enable_reflection_mode: false # unused; must be false
  max_try_cnt          : 1      # unused; must be 1
"""

import copy
import os
import sys
import time
from typing import Any, Dict, Optional

import torch

from src.seeding import (
    get_rq1_stage_seeds,
    set_global_seed,
)


SYSTEM_PROMPT = (
    "You should first think about the planning process "
    "in the mind and then generate the image. \n"
    "The planning process is enclosed within <think> </think> "
    "tags, i.e. <think> planning process here </think> image here"
)


class UniReasonRunner:
    """
    UniReason-Edit single-GPU wrapper.

    Interface contract with Step1XRunner
    -------------------------------------
    run() returns a dict with the same keys so that build_rq2_record
    and ExperimentLogger work without changes.

    The one extra keyword argument is cached_thought (str | None):
    - None       → no pre-computed thought; behaviour follows
                   generation["enable_thinking_mode"]
    - str        → use this think_output directly; skip live thinking
                   (thinking-cached mode, equivalent to Step1X's
                   rewrite-replay)

    THINKING_IS_APPENDED = True tells run_rq2.py that the thought
    cache entry is appended alongside the original instruction, not
    used as a prompt replacement.
    """

    THINKING_IS_APPENDED = True

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_cfg = config["model"]

        repo_path = model_cfg.get("unireason_repo")
        if repo_path:
            repo_path = os.path.expanduser(repo_path)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        from safetensors.torch import load_file

        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from modeling.bagel import (
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.autoencoder import load_ae

        model_path = os.path.expanduser(model_cfg["model_path"])
        print(f"Loading UniReason from: {model_path}")

        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        self.vae_model, vae_config = load_ae(
            local_path=os.path.join(model_path, "ae.safetensors")
        )

        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 378, 14)

        max_latent_size = int(model_cfg.get("max_latent_size", 64))

        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=max_latent_size,
        )

        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config
        )

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(
            self.tokenizer
        )

        ema_path = os.path.join(model_path, "ema.safetensors")
        state_dict = load_file(ema_path, device="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"State dict loaded: {msg}")
        if msg.missing_keys:
            raise RuntimeError(
                f"UniReason weights incomplete: {len(msg.missing_keys)} missing keys, "
                f"e.g. {msg.missing_keys[:5]}"
            )
        del state_dict

        if not torch.cuda.is_available():
            raise RuntimeError("UniReason requires CUDA.")

        self.device = "cuda:0"
        self.model = model.to(self.device).eval()
        self.vae_model = self.vae_model.to(self.device).eval()

        print("UniReason loaded.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _move(self, gen_input: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in gen_input.items():
            if isinstance(v, torch.Tensor):
                gen_input[k] = v.to(self.device)
        return gen_input

    @staticmethod
    def _output_size(image) -> tuple:
        """
        Compute (w, h) for the generated image from the source image
        size, clamped to [512, 1024] and aligned to stride 16.
        Mirrors the logic in gen_images_mp_kris.py apply_scale().
        """
        max_sz, min_sz = 1024, 512

        def div16(v):
            return max(16, int(round(v / 16) * 16))

        w, h = image.size
        scale = min(max_sz / max(w, h), 1.0)
        scale = max(scale, min_sz / min(w, h))
        w, h = div16(round(w * scale)), div16(round(h * scale))

        if max(w, h) > max_sz:
            s = max_sz / max(w, h)
            w, h = div16(round(w * s)), div16(round(h * s))

        return w, h

    def _count_tokens(self, text: Optional[str]) -> Optional[int]:
        if not text:
            return 0
        try:
            return len(
                self.tokenizer(text, add_special_tokens=False)["input_ids"]
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Thinking phase
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _think(
        self,
        images,
        instruction: str,
        thinking_seed: int,
        max_length: int = 2048,
    ) -> str:
        """
        Generate the UniReason reasoning chain for one instruction.

        Seeds the global RNG with thinking_seed before text generation
        so that re-runs with the same seed reproduce the same output.

        Returns the decoded think_output string (may contain
        <think>...</think> tags followed by the reformulated instruction).
        """
        from modeling.bagel.qwen2_navit import NaiveCache

        model = self.model
        tokenizer = self.tokenizer
        new_token_ids = self.new_token_ids

        set_global_seed(thinking_seed)

        past_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
        lens, rope = [0], [0]

        inp, lens, rope = model.prepare_prompts(
            curr_kvlens=lens, curr_rope=rope,
            prompts=[SYSTEM_PROMPT],
            tokenizer=tokenizer, new_token_ids=new_token_ids,
        )
        inp = self._move(inp)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            past_kv = model.forward_cache_update_text(past_kv, **inp)

        for image in images:
            inp, lens, rope = model.prepare_vae_images(
                curr_kvlens=lens, curr_rope=rope,
                images=[image], transforms=self.vae_transform,
                new_token_ids=new_token_ids,
            )
            inp = self._move(inp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                past_kv = model.forward_cache_update_vae(
                    self.vae_model, past_kv, **inp
                )

            inp, lens, rope = model.prepare_vit_images(
                curr_kvlens=lens, curr_rope=rope,
                images=[image], transforms=self.vit_transform,
                new_token_ids=new_token_ids,
            )
            inp = self._move(inp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                past_kv = model.forward_cache_update_vit(past_kv, **inp)

        # Thinking uses a deepcopy so it does not pollute the main context.
        tmp_kv = copy.deepcopy(past_kv)
        tmp_lens = copy.deepcopy(lens)
        tmp_rope = copy.deepcopy(rope)

        inp, tmp_lens, tmp_rope = model.prepare_prompts(
            curr_kvlens=tmp_lens, curr_rope=tmp_rope,
            prompts=[instruction],
            tokenizer=tokenizer, new_token_ids=new_token_ids,
        )
        inp = self._move(inp)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tmp_kv = model.forward_cache_update_text(tmp_kv, **inp)

        inp = model.prepare_start_tokens(tmp_lens, tmp_rope, new_token_ids)
        inp = self._move(inp)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.generate_text(
                past_key_values=tmp_kv,
                max_length=max_length,
                do_sample=True,
                temperature=0.3,
                end_token_id=new_token_ids["eos_token_id"],
                **inp,
            )
            raw = tokenizer.decode(tokens[:, 0])
            parts = raw.split("<|im_end|>")[0].split("<|im_start|>")
            if len(parts) < 2:
                raise RuntimeError(
                    f"Thinking output has no <|im_start|> block: {raw[:200]!r}"
                )
            think_output = parts[1]

        return think_output

    # ------------------------------------------------------------------
    # Generation phase
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate(
        self,
        images,
        instruction: str,
        think_output: Optional[str],
        diffusion_seed: int,
        num_timesteps: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
    ):
        """
        Generate the edited image, with or without a pre-computed thought.

        think_output != None  (thinking-on path)
          KV context:  [SYSTEM_PROMPT] [VAE+ViT images] [instruction] [think_output]
          cfg_img KV:  [SYSTEM_PROMPT] [instruction] [think_output]
          cfg_text KV: snapshot after [SYSTEM_PROMPT] [VAE+ViT images]

        think_output is None  (thinking-off path)
          KV context:  [VAE+ViT images] [instruction]
          cfg_img KV:  [instruction]
          cfg_text KV: snapshot after [VAE+ViT images]
        """
        from modeling.bagel.qwen2_navit import NaiveCache
        from PIL import Image as PILImage

        model = self.model
        tokenizer = self.tokenizer
        new_token_ids = self.new_token_ids

        w, h = self._output_size(images[0])
        set_global_seed(diffusion_seed)

        past_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
        lens, rope = [0], [0]

        # ---- system prompt (thinking-on only) ----
        if think_output is not None:
            inp, lens, rope = model.prepare_prompts(
                curr_kvlens=lens, curr_rope=rope,
                prompts=[SYSTEM_PROMPT],
                tokenizer=tokenizer, new_token_ids=new_token_ids,
            )
            inp = self._move(inp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                past_kv = model.forward_cache_update_text(past_kv, **inp)

        # ---- source image (VAE + ViT) ----
        for image in images:
            inp, lens, rope = model.prepare_vae_images(
                curr_kvlens=lens, curr_rope=rope,
                images=[image], transforms=self.vae_transform,
                new_token_ids=new_token_ids,
            )
            inp = self._move(inp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                past_kv = model.forward_cache_update_vae(
                    self.vae_model, past_kv, **inp
                )

            inp, lens, rope = model.prepare_vit_images(
                curr_kvlens=lens, curr_rope=rope,
                images=[image], transforms=self.vit_transform,
                new_token_ids=new_token_ids,
            )
            inp = self._move(inp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                past_kv = model.forward_cache_update_vit(past_kv, **inp)

        # ---- cfg_text: snapshot after images (unconditional) ----
        cfg_text_kv = copy.deepcopy(past_kv)
        inp_cfg_text = model.prepare_vae_latent_cfg(
            curr_kvlens=lens, curr_rope=rope, image_sizes=[(h, w)],
        )
        inp_cfg_text = self._move(inp_cfg_text)

        # ---- cfg_img: text only, no source image tokens ----
        cfg_img_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
        ci_lens, ci_rope = [0], [0]

        if think_output is not None:
            for text in [SYSTEM_PROMPT, instruction, think_output]:
                inp, ci_lens, ci_rope = model.prepare_prompts(
                    curr_kvlens=ci_lens, curr_rope=ci_rope,
                    prompts=[text],
                    tokenizer=tokenizer, new_token_ids=new_token_ids,
                )
                inp = self._move(inp)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    cfg_img_kv = model.forward_cache_update_text(
                        cfg_img_kv, **inp
                    )
        else:
            inp, ci_lens, ci_rope = model.prepare_prompts(
                curr_kvlens=ci_lens, curr_rope=ci_rope,
                prompts=[instruction],
                tokenizer=tokenizer, new_token_ids=new_token_ids,
            )
            inp = self._move(inp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cfg_img_kv = model.forward_cache_update_text(
                    cfg_img_kv, **inp
                )

        inp_cfg_img = model.prepare_vae_latent_cfg(
            curr_kvlens=ci_lens, curr_rope=ci_rope, image_sizes=[(h, w)],
        )
        inp_cfg_img = self._move(inp_cfg_img)

        # ---- main branch: instruction [+ think_output] ----
        texts = [instruction]
        if think_output is not None:
            texts.append(think_output)

        for text in texts:
            inp, lens, rope = model.prepare_prompts(
                curr_kvlens=lens, curr_rope=rope,
                prompts=[text],
                tokenizer=tokenizer, new_token_ids=new_token_ids,
            )
            inp = self._move(inp)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                past_kv = model.forward_cache_update_text(past_kv, **inp)

        inp_gen = model.prepare_vae_latent(
            curr_kvlens=lens, curr_rope=rope,
            image_sizes=[(h, w)], new_token_ids=new_token_ids,
        )
        inp_gen = self._move(inp_gen)

        # ---- diffusion ----
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            unpacked = model.generate_image(
                past_key_values=past_kv,
                cfg_text_past_key_values=cfg_text_kv,
                cfg_img_past_key_values=cfg_img_kv,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_type="serial_text_img",
                cfg_interval=[0.0, 1.0],
                cfg_renorm_min=0.0,
                cfg_renorm_type="text_channel",
                timestep_shift=3.0,
                **inp_gen,
                cfg_text_packed_position_ids=inp_cfg_text["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=inp_cfg_text["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=inp_cfg_text["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=inp_cfg_text["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=inp_cfg_img["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=inp_cfg_img["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=inp_cfg_img["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=inp_cfg_img["cfg_packed_key_value_indexes"],
            )

        latent = unpacked[0]
        latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, h // 8, w // 8)
        decoded = self.vae_model.decode(latent)
        arr = (
            (decoded * 0.5 + 0.5)
            .clamp(0, 1)[0]
            .permute(1, 2, 0)
            .mul(255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return PILImage.fromarray(arr)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        images,
        instruction: str,
        example_uid: str,
        base_seed: int,
        cached_thought: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run one UniReason generation.

        cached_thought  : pre-computed think_output string from the
                          thought cache.  When provided the thinking
                          phase is skipped (thinking-cached mode).
                          Pass None to follow enable_thinking_mode in
                          the config (True = live, False = off).

        Returns the same dict keys as Step1XRunner.run() so that
        build_rq2_record and ExperimentLogger work unchanged.
        """
        generation_cfg = self.config["generation"]

        if len(images) != 1:
            raise ValueError(
                f"{example_uid}: requires exactly one source image, "
                f"got {len(images)}."
            )

        image_input = images[0]

        num_timesteps = int(generation_cfg["num_inference_steps"])
        cfg_text_scale = float(generation_cfg["true_cfg_scale"])
        cfg_img_scale = float(generation_cfg.get("cfg_img_scale", 2.0))
        thinking_live = bool(generation_cfg["enable_thinking_mode"])

        stage_seeds = get_rq1_stage_seeds(
            base_seed=base_seed,
            example_uid=example_uid,
            max_try_cnt=1,
        )

        torch.cuda.reset_peak_memory_stats()
        self._sync_cuda()
        pipeline_start = time.perf_counter()

        # ---- thinking phase ----
        think_output = None
        latency_thinking_sec = None

        if cached_thought is not None:
            think_output = cached_thought
        elif thinking_live:
            self._sync_cuda()
            think_start = time.perf_counter()
            think_output = self._think(
                images=[image_input],
                instruction=instruction,
                thinking_seed=stage_seeds["thinking"],
            )
            self._sync_cuda()
            latency_thinking_sec = time.perf_counter() - think_start

        # ---- generation phase ----
        final_image = self._generate(
            images=[image_input],
            instruction=instruction,
            think_output=think_output,
            diffusion_seed=stage_seeds["diffusion_0"],
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
        )

        self._sync_cuda()
        latency_pipeline_sec = time.perf_counter() - pipeline_start

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        if thinking_live:
            stop_reason = "thinking_live"
        elif cached_thought is not None:
            stop_reason = "thinking_cached"
        else:
            stop_reason = "single_pass"

        return {
            "final_image": final_image,
            "candidate_images": [final_image],
            "reformat_prompt": think_output,
            "think_info": think_output,
            "thinking_token_count": self._count_tokens(think_output),
            "best_info": None,
            "reflection_info": None,
            "reflection_token_count": None,
            "attempt_count": 1,
            "selected_candidate_index": 0,
            "retry_triggered": False,
            "stop_reason": stop_reason,
            "selected_is_last_candidate": True,
            "base_seed": base_seed,
            "stage_seeds": stage_seeds,
            "latency_pipeline_sec": latency_pipeline_sec,
            "latency_thinking_sec": latency_thinking_sec,
            "latency_diffusion_attempts_sec": [],
            "latency_reflection_rounds_sec": [],
            "latency_offload_sec": None,
            "peak_vram_gb": peak_vram_gb,
        }