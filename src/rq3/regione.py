"""Small, auditable adapters around the pinned RegionE implementation.

No denoising or cache-reuse decision is reimplemented here. RegionE owns those
decisions. The no-RIKV intervention replaces its attention processor with native
attention on the *current reduced stream*: excluded image/reference keys never
enter attention. Recomputing the full stream would be a different intervention.
"""

from collections import Counter


class NoRIKVProcessor:
    """Native attention, accepting RegionE's conditional/unconditional tag."""

    def __init__(self, native_processor):
        self.native = native_processor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, tag=None):
        # Native attention applies the reduced RoPE to both Q and K. In
        # particular, do not use MANAGER.image_rotary_emb (the full KV grid).
        return self.native(attn, hidden_states, encoder_hidden_states,
                           attention_mask, image_rotary_emb)


class AttentionRecorder:
    """Count actual attention invocations without tensor copies or CUDA syncs."""

    def __init__(self, processor, trace, manager=None, rikv_enabled=False):
        self.processor = processor
        self.trace = trace
        self.manager = manager
        self.rikv_enabled = rikv_enabled

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, tag=None):
        if self.manager is not None:
            result = self.processor(attn, hidden_states, encoder_hidden_states,
                                    attention_mask, image_rotary_emb, tag=tag)
        else:
            result = self.processor(attn, hidden_states, encoder_hidden_states,
                                    attention_mask, image_rotary_emb)
        counts = self.trace.attention_counts
        counts["attention_calls"] += 1
        if self.rikv_enabled:
            m = self.manager
            if m.current_step == m.warmup_step - 1 or m.current_step == m.prev_refresh_step:
                counts["rikv_cache_writes"] += 1
            elif m.warmup_step - 1 < m.current_step <= m.inference_step - m.post_step - 1:
                counts["rikv_context_reuses"] += 1
        return result


class RegionTrace:
    """Step records derived from observed transformer and scheduler calls."""

    def __init__(self, enabled, rikv_enabled, avd_enabled):
        self.enabled = enabled
        self.rikv_enabled = rikv_enabled
        self.avd_enabled = avd_enabled
        self.steps = []
        self.selected_regions = []
        self.attention_counts = Counter()
        self.calls = []
        self.total_latent_tokens = None
        self.condition_tokens = None
        self.active_tokens = None

    def transformer_call(self, hidden_states, tag=None):
        tokens = int(hidden_states.shape[1])
        full = tokens == self.total_latent_tokens + self.condition_tokens
        self.calls.append({"branch": tag or ("cond" if not self.calls else "uncond"),
                           "input_tokens": tokens, "full_image": full})

    def finish_step(self, step, manager=None):
        total = self.total_latent_tokens
        selected = int(manager.edited_ids.shape[1]) if manager is not None and manager.edited_ids is not None else total
        full_calls = sum(call["full_image"] for call in self.calls)
        record = {
            "step": step, "rho_t": self.active_tokens / total,
            "selected_fraction": selected / total,
            "active_tokens": self.active_tokens, "total_latent_tokens": total,
            "full_image_generator_calls": full_calls,
            "reduced_token_generator_calls": len(self.calls) - full_calls,
            "avd_cache_reused": self.enabled and not self.calls,
            "generator_calls": self.calls,
            **{key: self.attention_counts[key] for key in
               ("attention_calls", "rikv_cache_writes", "rikv_context_reuses")},
        }
        record["refresh_event"] = bool(manager is not None and
                                       (step == manager.warmup_step - 1 or step == manager.prev_refresh_step))
        self.steps.append(record)
        # RegionE selects the mask once after the warmup. Copy it only after
        # inference (the backend materializes this tensor outside the timer).
        if manager is not None and step == manager.warmup_step - 1:
            self.selected_regions.append({
                "step": step, "grid_height": manager.height // (manager.patch_size * manager.vae_scale_factor),
                "grid_width": manager.width // (manager.patch_size * manager.vae_scale_factor),
                "selected_indices": manager.edited_ids.detach(),
            })
        self.calls = []
        self.attention_counts.clear()

    def as_dict(self):
        regions = [{**region, "selected_indices": region["selected_indices"].cpu().flatten().tolist()}
                   for region in self.selected_regions]
        full = sum(step["full_image_generator_calls"] for step in self.steps)
        reduced = sum(step["reduced_token_generator_calls"] for step in self.steps)
        return {"enabled": self.enabled, "rikv_enabled": self.rikv_enabled,
                "avd_enabled": self.avd_enabled, "steps": self.steps,
                "selected_regions": regions, "generator_calls": full + reduced,
                "full_image_generator_calls": full, "reduced_token_generator_calls": reduced,
                "avd_cache_reuse_steps": sum(step["avd_cache_reused"] for step in self.steps),
                "step_indexing": "zero_based"}
