"""CUDA Step1X-v1p2 backend for the standalone RQ3 experiment.

Heavy dependencies are lazy: planning, analysis and unit tests do not download
weights. One backend per process is required because upstream RegionE has a
module-global manager. The existing RQ1 runner is not changed.
"""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from .regione import AttentionRecorder, NoRIKVProcessor, RegionTrace


DIFFUSERS_REVISION = "f5f1c98fa00cb4d0479af1b1b1c17d724345963a"
REGIONE_REVISION = "df66d2e60710f66ce801e8b7c151d22c3239d114"
THINKING_TOP_P_BACKEND = "transformers_cpu"
REGIONE_DEFAULTS = {"num_inference_steps": 28, "warmup_step": 6, "post_step": 2,
                    "refresh_step": "16", "threshold": 0.88,
                    "cache_threshold": 0.02, "erosion_dilation": True}


def _verify_import(module, expected_path, revision):
    """Fail closed on an import from a different checkout or a dirty source."""
    root = Path(expected_path).resolve()
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(root):
        raise RuntimeError(f"{module.__name__} loaded from {source}; install editable source at {root}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != revision or dirty:
        raise RuntimeError(f"{root}: expected clean revision {revision}; found {head}, dirty={bool(dirty)}")


@contextmanager
def _replace(obj, name, replacement):
    """Restore an instance override without leaving a bound method behind."""
    previous = obj.__dict__.get(name)
    had_override = name in obj.__dict__
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if had_override:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


class _CPUTopPLogitsWarper:
    """Run the original Transformers nucleus filter on CPU under strict mode.

    PyTorch 2.5.1 rejects floating-point CUDA cumsum in deterministic mode.
    Reusing the native processor on CPU preserves its threshold/tie/min-token
    rules without changing global determinism or copying upstream tensor code.
    The result returns to the sampling device; the seeded multinomial stays
    native. CPU transfers and filtering are included in the thinking timer.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, input_ids, scores):
        return self.processor(input_ids.to("cpu"), scores.to("cpu")).to(scores.device)


@contextmanager
def _deterministic_top_p(text_encoder):
    """Replace only the generated top-p processor, at its original position.

    Transformers 4.55.4 appends sampling warpers *after* custom processors.
    Passing logits_processor= would change their order and leave native CUDA
    top-p active. Scope this override to the current encoder invocation and
    restore it on both success and error; other models are never modified.
    """
    from transformers.generation.logits_process import TopPLogitsWarper

    original = text_encoder._get_logits_processor

    def build_processors(*args, **kwargs):
        processors = original(*args, **kwargs)
        indices = [index for index, processor in enumerate(processors)
                   if type(processor) is TopPLogitsWarper]
        if len(indices) != 1 or processors[indices[0]].top_p != .9:
            raise RuntimeError("Expected exactly one native top_p=.9 processor for the RQ3 thinker")
        index = indices[0]
        processors[index] = _CPUTopPLogitsWarper(processors[index])
        return processors

    with _replace(text_encoder, "_get_logits_processor", build_processors):
        yield


class Step1XBackend:
    def __init__(self, config):
        self.config = config
        # Must be set before the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Real RQ3 generation requires an NVIDIA CUDA GPU; use --dry-run on this machine.")
        import diffusers
        import RegionE
        from diffusers import Step1XEditPipelineV1P2
        from diffusers.models.transformers.transformer_step1x_edit import Step1XEditAttnProcessor
        from diffusers.pipelines.step1x_edit.pipeline_step1x_edit_thinker import Step1XEditThinker
        from .prepare import download_model_snapshot
        from RegionE.Step1XEditV1P2 import inplace

        source = config["source"]
        for module, key, supported_revision in ((diffusers, "diffusers", DIFFUSERS_REVISION),
                                                (RegionE, "regione", REGIONE_REVISION)):
            if source[f"{key}_revision"] != supported_revision:
                raise ValueError(f"The backend has been audited for {key}@{supported_revision}; revalidate before changing it")
            _verify_import(module, config.get("_paths", {}).get(f"{key}_path", source[f"{key}_path"]), supported_revision)
        self.torch, self.np = torch, np
        self.inplace, self.native_processor = inplace, Step1XEditAttnProcessor
        self.offload_mode = config["model"].get("offload_mode", "model")
        if self.offload_mode not in ("model", "none"):
            raise ValueError("RQ3 supports model-level swapping or no offloading, never sequential offloading")
        if config["generation"]["num_inference_steps"] != 28 or config["generation"]["true_cfg_scale"] != 6:
            raise ValueError("RQ3 is fixed at 28 steps and true_cfg_scale=6")
        params = config["regione"].get("params", REGIONE_DEFAULTS)
        if params != REGIONE_DEFAULTS:
            raise ValueError("RQ3 requires the pinned v1p2 RegionE defaults")
        self.regione_params = dict(params)
        if config.get("thinking", {"temperature": .7, "top_p": .9, "max_new_tokens": 128}) != {
            "temperature": .7, "top_p": .9, "max_new_tokens": 128
        }:
            raise ValueError("The pinned native thinker samples at temperature=.7, top_p=.9, max_new_tokens=128")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dtype = getattr(torch, config["model"]["dtype"])
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Frozen bfloat16 precision requires a GPU with bfloat16 support")

        model = config["model"]
        snapshot = download_model_snapshot(model)
        # Loading from the immutable local snapshot also ensures the native
        # AutoProcessor re-loads resolve to the same checkpoint's local files.
        self.pipe = Step1XEditPipelineV1P2.from_pretrained(snapshot, torch_dtype=dtype, local_files_only=True)
        processor_path = Path(self.pipe.processor.tokenizer.name_or_path).resolve()
        if not processor_path.is_relative_to(Path(snapshot).resolve()):
            raise RuntimeError("Native processor did not resolve inside the pinned model snapshot")
        self.thinker = Step1XEditThinker(self.pipe.text_encoder, self.pipe.processor)
        for name in ("text_encoder", "transformer", "vae"):
            getattr(self.pipe, name).eval()
        self._active = None
        self._offload_seconds = 0.0
        self._resident_classes = {}
        self._set_execution_device()
        if self.offload_mode == "none":
            self.pipe.to("cuda")
        # Upstream uses `False` on ImportError but checks `is not None`. Force
        # its working SDPA branch, with the same PyTorch attention family as A/B.
        self.inplace.flash_attn = None
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        try:
            driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            driver = None
        self.runtime_metadata = {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "driver": driver,
            "diffusers": diffusers.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "regione": importlib.metadata.version("RegionE"),
            "accelerate": importlib.metadata.version("accelerate"),
            "triton": importlib.metadata.version("triton"),
            "device_name": properties.name, "total_vram_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "attention_backend": "pytorch_sdpa", "offload_mode": self.offload_mode,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "thinking_top_p_backend": THINKING_TOP_P_BACKEND,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
        }

    def _set_execution_device(self):
        # RegionE changes only this instance's __class__. An instance-local
        # subclass provides an explicit device while all models rest on CPU.
        # We never replace the base class's property or methods globally.
        base = self.pipe.__class__
        if base not in self._resident_classes:
            device = self.torch.device("cuda")
            self._resident_classes[base] = type("RQ3" + base.__name__, (base,),
                                               {"_execution_device": property(lambda instance: device)})
        self.pipe.__class__ = self._resident_classes[base]

    def _sync(self):
        self.torch.cuda.synchronize()

    def _resident(self, name):
        if self.offload_mode == "none" or name == self._active:
            return
        self._sync()
        start = time.perf_counter()
        if self._active is not None:
            getattr(self.pipe, self._active).to("cpu")
        if name is not None:
            getattr(self.pipe, name).to("cuda")
        self._sync()
        self._offload_seconds += time.perf_counter() - start
        self._active = name

    def _seed(self, seed):
        random.seed(seed)
        self.np.random.seed(seed % (2**32))
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def think(self, image, instruction, seed):
        """Sample one native rewrite with the global RNG, never run diffusion."""
        self._seed(seed)
        self._offload_seconds = 0.0
        self.torch.cuda.reset_peak_memory_stats()
        self._sync()
        start = time.perf_counter()
        captured = {}
        original = self.pipe.text_encoder.generate

        def capture_generate(*args, **kwargs):
            for key, expected in (("do_sample", True), ("temperature", .7), ("top_p", .9), ("max_new_tokens", 128)):
                if kwargs.get(key) != expected:
                    raise RuntimeError(f"Unexpected native thinker sampling setting: {key}")
            generated = original(*args, **kwargs)
            ids = generated[:, kwargs["input_ids"].shape[1]:]
            captured["token_count"] = int(ids.shape[1])
            captured["raw_text"] = self.thinker.reflector.client.processor.batch_decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return generated

        try:
            self._resident("text_encoder")
            with (self.torch.inference_mode(),
                  _deterministic_top_p(self.pipe.text_encoder),
                  _replace(self.pipe.text_encoder, "generate", capture_generate)):
                rewrite = self.thinker.think(image, instruction)
            if not rewrite or "token_count" not in captured:
                raise RuntimeError("The native thinker returned no rewrite or token trace")
        finally:
            self._resident(None)
            self._sync()
        return {"rewrite": rewrite, **captured,
                "top_p_backend": THINKING_TOP_P_BACKEND,
                "token_count_includes_eos": True,
                "seconds": time.perf_counter() - start - self._offload_seconds,
                "offload_seconds": self._offload_seconds,
                "peak_vram_bytes": self.torch.cuda.max_memory_allocated()}

    def _configure_condition(self, condition):
        if condition not in "ABCDEF" or len(condition) != 1:
            raise ValueError(f"Unknown RQ3 condition {condition}")
        # Always reconstruct processors/scheduler; no KV or AVD cache survives
        # between samples, seeds, timing repetitions, or experimental arms.
        self.inplace.unwarp_modules(self.pipe)
        self.inplace.MANAGER = self.inplace.Step1XEditV1P2Manager()
        enabled = condition in "CDEF"
        rikv = enabled and condition != "E"
        avd = enabled and condition != "F"
        if enabled:
            params = dict(self.regione_params)
            if not avd:
                params["cache_threshold"] = -1.0
            self.inplace.warp_modules(self.pipe, **params)
        self._set_execution_device()
        trace = RegionTrace(enabled, rikv, avd)
        manager = self.inplace.MANAGER if enabled else None
        for block in list(self.pipe.transformer.transformer_blocks) + list(self.pipe.transformer.single_transformer_blocks):
            processor = block.attn.processor
            if enabled and not rikv:
                native = self.native_processor()
                native._attention_backend = "native"
                processor = NoRIKVProcessor(native)
            elif not enabled:
                processor._attention_backend = "native"
            block.attn.set_processor(AttentionRecorder(processor, trace, manager, rikv))
        return trace, manager

    def _clear_kv_tensors(self):
        # RegionE processors are plain Python objects, so Module.to('cpu')
        # does not move their KV tensors. Explicitly release them before
        # loading the VAE/thinker, including after a failed invocation.
        for block in list(self.pipe.transformer.transformer_blocks) + list(self.pipe.transformer.single_transformer_blocks):
            processor = block.attn.processor
            if isinstance(processor, AttentionRecorder):
                processor.manager = None
                processor = processor.processor
            for name in ("k_cache_odd", "k_cache_even", "v_cache_odd", "v_cache_even"):
                if hasattr(processor, name):
                    setattr(processor, name, None)

    def generate(self, image, prompt, seed, condition):
        """Replay one instruction with thinking/reflection explicitly disabled.

        `diffusion_seconds` measures the complete denoising loop (both CFG
        branches, scheduler, RegionE decisions), excluding model transfers.
        `measured_replay_seconds` is directly timed; adding a cached thinking
        charge produces a reconstructed full-pipeline cost, not a direct timer.
        """
        from contextlib import ExitStack
        trace, manager = self._configure_condition(condition)
        self._seed(seed)
        self._offload_seconds = 0.0
        self.torch.cuda.reset_peak_memory_stats()
        timings = {key: 0.0 for key in ("preprocess_seconds", "prompt_encode_seconds",
                   "latent_prepare_seconds", "diffusion_seconds", "postprocess_seconds")}
        metadata = {"input_size": list(image.size), "mode": "step1x_native",
                    "size_level": self.config["preprocessing"]["size_level"]}
        state = {"diffusion_start": None, "postprocess_start": None}
        initial_noise = {}

        def timed(function, key, resident=None, after=None):
            def wrapper(*args, **kwargs):
                if resident:
                    self._resident(resident)
                self._sync()
                start = time.perf_counter()
                result = function(*args, **kwargs)
                self._sync()
                timings[key] += time.perf_counter() - start
                if after:
                    after(result)
                return result
            return wrapper

        def record_image(result):
            metadata["inference_size"] = [int(result[3]), int(result[4])]

        def record_latents(result):
            trace.total_latent_tokens = int(result[0].shape[1])
            trace.condition_tokens = int(result[1].shape[1])
            # Retain a small detached tensor and hash after the timed invocation.
            # Subsequent scheduler updates allocate new tensors in pinned code.
            initial_noise["tensor"] = result[0].detach()
            self._resident("transformer")
            self._sync()
            state["diffusion_start"] = time.perf_counter()

        original_forward = self.pipe.transformer.forward

        def transformer_forward(*args, **kwargs):
            trace.transformer_call(kwargs["hidden_states"],
                                   (kwargs.get("joint_attention_kwargs") or {}).get("tag"))
            return original_forward(*args, **kwargs)

        original_step = self.pipe.scheduler.step

        def scheduler_step(model_output, timestep, sample, *args, **kwargs):
            trace.active_tokens = int(sample.shape[1])
            return original_step(model_output, timestep, sample, *args, **kwargs)

        original_decode = self.pipe.vae.decode

        def decode(*args, **kwargs):
            self._sync()
            timings["diffusion_seconds"] = time.perf_counter() - state["diffusion_start"]
            self._clear_kv_tensors()
            self._resident("vae")
            self._sync()
            state["postprocess_start"] = time.perf_counter()
            return original_decode(*args, **kwargs)

        def callback(pipe, step, timestep, callback_kwargs):
            trace.finish_step(step, manager)
            return callback_kwargs

        self._sync()
        start = time.perf_counter()
        try:
            with ExitStack() as stack, self.torch.inference_mode():
                for obj, name, wrapper in (
                    (self.pipe, "encode_image", timed(self.pipe.encode_image, "preprocess_seconds", after=record_image)),
                    (self.pipe, "encode_prompt", timed(self.pipe.encode_prompt, "prompt_encode_seconds", resident="text_encoder")),
                    (self.pipe, "prepare_latents", timed(self.pipe.prepare_latents, "latent_prepare_seconds", resident="vae", after=record_latents)),
                    (self.pipe.transformer, "forward", transformer_forward),
                    (self.pipe.scheduler, "step", scheduler_step),
                    (self.pipe.vae, "decode", decode),
                ):
                    stack.enter_context(_replace(obj, name, wrapper))
                result = self.pipe(
                    image=image, prompt=prompt, size_level=self.config["preprocessing"]["size_level"],
                    num_inference_steps=28, true_cfg_scale=6.0, guidance_scale=6.0,
                    enable_thinking_mode=False, enable_reflection_mode=False, max_try_cnt=1,
                    num_images_per_prompt=1, output_type="pil", return_dict=True,
                    negative_prompt="", generator=self.torch.Generator(device="cuda").manual_seed(seed),
                    callback_on_step_end=callback, callback_on_step_end_tensor_inputs=[],
                )
                self._sync()
                if state["postprocess_start"] is not None:
                    timings["postprocess_seconds"] = time.perf_counter() - state["postprocess_start"]
        finally:
            self._clear_kv_tensors()
            self._resident(None)
            self._sync()
            self.inplace.MANAGER = self.inplace.Step1XEditV1P2Manager()
        timings["measured_replay_seconds"] = time.perf_counter() - start
        timings["pipeline_seconds"] = timings["measured_replay_seconds"]
        timings["offload_seconds"] = self._offload_seconds
        if len(result.images) != 1 or len(trace.steps) != 28:
            raise RuntimeError("RQ3 requires one candidate and exactly 28 observed scheduler steps")
        if condition == "F" and any(step["avd_cache_reused"] for step in trace.steps):
            raise RuntimeError("AVD-off arm unexpectedly reused a velocity cache")
        output = result.images[0]
        metadata["output_size"] = list(output.size)
        noise = initial_noise["tensor"].contiguous().view(self.torch.uint8).cpu().numpy().tobytes()
        regione_log = trace.as_dict()
        return {"image": output, "timings": timings, "preprocessing": metadata,
                "peak_vram_bytes": self.torch.cuda.max_memory_allocated(),
                "initial_noise_sha256": hashlib.sha256(noise).hexdigest(),
                "regione": {**regione_log, "params": dict(self.regione_params),
                            "attention_backend": "pytorch_sdpa",
                            "avd_cache_threshold": -1.0 if condition == "F" else self.regione_params["cache_threshold"]}}
