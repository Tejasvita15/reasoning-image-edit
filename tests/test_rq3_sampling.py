"""Pinned native nucleus-sampling tests; no tokenizer/model downloads.

The library tests require Transformers 4.55.4 and torch. They skip in the
lightweight analysis environment; run them in the deployment environment or an
isolated temporary environment with those dependencies. CUDA adds one optional
dispatch test; CPU coverage does not claim a full GPU determinism pilot.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.rq3.backend import (
    THINKING_TOP_P_BACKEND, Step1XBackend,
    _CPUTopPLogitsWarper, _deterministic_top_p,
)

try:
    import torch
    import transformers
    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers import GenerationConfig
    from transformers.generation.logits_process import (
        LogitNormalization, LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
    )
    from transformers.generation.utils import GenerationMixin
    HAS_PINNED_LIBRARY = transformers.__version__ == "4.55.4"
except ImportError:
    torch = None
    GenerationMixin = object
    TorchDispatchMode = object
    HAS_PINNED_LIBRARY = False


class WrapperTransferTests(unittest.TestCase):
    def test_both_inputs_move_to_cpu_and_result_returns_to_sampling_device(self):
        calls = []
        cpu_ids, cpu_scores, restored = object(), object(), object()

        class Scores:
            device = "cuda:3"

            def to(self, device):
                calls.append(("scores", device))
                return cpu_scores

        class InputIDs:
            def to(self, device):
                calls.append(("input_ids", device))
                return cpu_ids

        class Output:
            def to(self, device):
                calls.append(("output", device))
                return restored

        def native(ids, scores):
            self.assertIs(ids, cpu_ids)
            self.assertIs(scores, cpu_scores)
            calls.append(("native", "cpu"))
            return Output()

        self.assertIs(_CPUTopPLogitsWarper(native)(InputIDs(), Scores()), restored)
        self.assertEqual(calls, [("input_ids", "cpu"), ("scores", "cpu"),
                                 ("native", "cpu"), ("output", "cuda:3")])


class EncoderHarness(GenerationMixin):
    """Actual HF processor builder with a tiny synthetic sampling head."""

    def __init__(self):
        self.chains = []
        self.fail_after_build = False

    def generate(self, *, input_ids, do_sample, temperature, top_p, max_new_tokens):
        config = GenerationConfig(do_sample=do_sample, temperature=temperature, top_p=top_p,
                                  top_k=50, repetition_penalty=1.05, max_new_tokens=max_new_tokens)
        processors = self._get_logits_processor(config, input_ids_seq_length=input_ids.shape[1],
                                               logits_processor=LogitsProcessorList(), device="cpu")
        self.chains.append(processors)
        if self.fail_after_build:
            raise RuntimeError("synthetic thinker failure")
        logits = torch.linspace(-2.0, 2.0, 64).reshape(1, 64)
        probabilities = processors(input_ids, logits).softmax(dim=-1)
        token = torch.multinomial(probabilities, num_samples=1)
        return torch.cat((input_ids, token), dim=1)


class CumsumDevices(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.devices = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten.cumsum.default:
            self.devices.append(args[0].device.type)
        return function(*args, **(kwargs or {}))


@unittest.skipUnless(HAS_PINNED_LIBRARY, "Requires torch and pinned transformers==4.55.4")
class NativeSamplingTests(unittest.TestCase):
    def test_native_cpu_equality_for_batches_ties_masked_logits_and_boundaries(self):
        cases = [
            torch.zeros(2, 8),
            torch.tensor([[20., 0., -20., float("-inf")], [-1000., -1000., -1000., -1000.]]),
            torch.tensor([[.10, .20, .70], [.05, .05, .90]]).log(),
            torch.arange(24, dtype=torch.float32).reshape(2, 12)[:, ::2],
        ]
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for min_tokens in (1, 2):
                for index, original in enumerate(cases):
                    with self.subTest(dtype=dtype, min_tokens=min_tokens, case=index):
                        scores = original.to(dtype=dtype)
                        before = scores.clone()
                        ids = torch.zeros((scores.shape[0], 2), dtype=torch.long)
                        native = TopPLogitsWarper(top_p=.9, min_tokens_to_keep=min_tokens)
                        rng_before = torch.get_rng_state().clone()
                        expected = native(ids, scores)
                        actual = _CPUTopPLogitsWarper(native)(ids, scores)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertTrue(torch.equal(scores, before))
                        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                        self.assertEqual(actual.dtype, scores.dtype)
                        self.assertEqual(actual.device, scores.device)

    def test_actual_generation_builder_preserves_order_objects_and_seeded_samples(self):
        encoder = EncoderHarness()
        config = GenerationConfig(do_sample=True, temperature=.7, top_k=50, top_p=.9,
                                  repetition_penalty=1.05, renormalize_logits=True)
        ids = torch.tensor([[1, 3, 5], [2, 4, 6]])
        kwargs = dict(generation_config=config, input_ids_seq_length=ids.shape[1],
                      logits_processor=LogitsProcessorList(), device="cpu")
        original_builder = encoder._get_logits_processor
        baseline = original_builder(**kwargs)
        before_wrapping = []

        def observe_native_builder(*args, **kwargs):
            processors = original_builder(*args, **kwargs)
            before_wrapping.extend(processors)
            return processors

        encoder._get_logits_processor = observe_native_builder
        with _deterministic_top_p(encoder):
            adapted = encoder._get_logits_processor(**kwargs)
            self.assertIsInstance(adapted, LogitsProcessorList)
            self.assertEqual([type(processor) for processor in adapted],
                             [RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                              TopKLogitsWarper, _CPUTopPLogitsWarper, LogitNormalization])
            for index, processor in enumerate(before_wrapping):
                if type(processor) is TopPLogitsWarper:
                    self.assertIs(adapted[index].processor, processor)
                else:
                    self.assertIs(adapted[index], processor)
            logits = torch.linspace(-3, 3, 128).reshape(2, 64)
            expected = baseline(ids, logits.clone())
            actual = adapted(ids, logits.clone())
            self.assertTrue(torch.equal(expected, actual))
            for seed in (42, 1234, 7):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    native_draws = torch.multinomial(expected.softmax(-1), num_samples=32, replacement=True)
                    torch.manual_seed(seed)
                    adapted_draws = torch.multinomial(actual.softmax(-1), num_samples=32, replacement=True)
                    self.assertTrue(torch.equal(native_draws, adapted_draws))
        self.assertIs(encoder._get_logits_processor, observe_native_builder)

    def test_context_preserves_list_identity_and_other_processors(self):
        native = TopPLogitsWarper(.9)
        before, after = object(), object()
        chain = LogitsProcessorList([before, native, after])
        encoder = SimpleNamespace(_get_logits_processor=Mock(return_value=chain))
        original = encoder._get_logits_processor
        with _deterministic_top_p(encoder):
            actual = encoder._get_logits_processor("positional", custom="keyword")
            self.assertIs(actual, chain)
            self.assertIs(actual[0], before)
            self.assertIs(actual[1].processor, native)
            self.assertIs(actual[2], after)
        self.assertIs(encoder._get_logits_processor, original)
        original.assert_called_once_with("positional", custom="keyword")

    def test_class_method_restored_after_success_and_body_failure(self):
        encoder = EncoderHarness()
        original_method = encoder._get_logits_processor
        for fail in (False, True):
            with self.subTest(fail=fail):
                try:
                    with _deterministic_top_p(encoder):
                        self.assertIn("_get_logits_processor", encoder.__dict__)
                        if fail:
                            raise LookupError("synthetic failure")
                except LookupError:
                    self.assertTrue(fail)
                self.assertNotIn("_get_logits_processor", encoder.__dict__)
                self.assertIs(encoder._get_logits_processor.__func__, original_method.__func__)
                self.assertIs(encoder._get_logits_processor.__self__, encoder)

    def test_missing_duplicate_changed_or_subclass_top_p_rejected_and_restored(self):
        class CustomTopP(TopPLogitsWarper):
            pass

        for processors in ([], [TopPLogitsWarper(.9), TopPLogitsWarper(.9)],
                           [TopPLogitsWarper(.95)], [CustomTopP(.9)]):
            with self.subTest(processors=[type(item).__name__ for item in processors]):
                builder = Mock(return_value=LogitsProcessorList(processors))
                encoder = SimpleNamespace(_get_logits_processor=builder)
                with self.assertRaisesRegex(RuntimeError, "exactly one native top_p"):
                    with _deterministic_top_p(encoder):
                        encoder._get_logits_processor()
                self.assertIs(encoder._get_logits_processor, builder)
        original = Mock(side_effect=ValueError("native builder failed"))
        encoder = SimpleNamespace(_get_logits_processor=original)
        with self.assertRaisesRegex(ValueError, "native builder failed"):
            with _deterministic_top_p(encoder):
                encoder._get_logits_processor()
        self.assertIs(encoder._get_logits_processor, original)

    def test_native_cumsum_runs_on_cpu_without_disabling_strict_determinism(self):
        was_strict = torch.are_deterministic_algorithms_enabled()
        was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            probe = CumsumDevices()
            ids = torch.tensor([[1, 2]])
            scores = torch.tensor([[1., 2., 3., 4.]])
            with probe:
                result = _CPUTopPLogitsWarper(TopPLogitsWarper(.9))(ids, scores)
            self.assertEqual(probe.devices, ["cpu"])
            self.assertEqual(result.device.type, "cpu")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        finally:
            torch.use_deterministic_algorithms(was_strict, warn_only=was_warn_only)

    def test_backend_think_uses_actual_builder_and_restores_scoped_overrides(self):
        encoder = EncoderHarness()
        decoder = SimpleNamespace(batch_decode=lambda ids, **kwargs: [f"sample-{int(ids[0, 0])}"])

        def native_think(image, instruction):
            result = encoder.generate(input_ids=torch.tensor([[1, 2]]), do_sample=True,
                                      temperature=.7, top_p=.9, max_new_tokens=128)
            return f"rewrite-{int(result[0, -1])}"

        backend = Step1XBackend.__new__(Step1XBackend)
        backend.torch = torch
        backend._seed = torch.manual_seed
        backend._sync = Mock()
        backend._resident = Mock()
        backend.pipe = SimpleNamespace(text_encoder=encoder)
        backend.thinker = SimpleNamespace(think=native_think,
                                         reflector=SimpleNamespace(client=SimpleNamespace(processor=decoder)))
        with (torch.random.fork_rng(devices=[]),
              patch.object(torch.cuda, "reset_peak_memory_stats"),
              patch.object(torch.cuda, "max_memory_allocated", return_value=123)):
            first = backend.think(None, "instruction", 42)
            second = backend.think(None, "instruction", 42)
            self.assertEqual(first["rewrite"], second["rewrite"])
            self.assertEqual(first["raw_text"], second["raw_text"])
            self.assertEqual(first["token_count"], 1)
            self.assertEqual(first["top_p_backend"], THINKING_TOP_P_BACKEND)
            for chain in encoder.chains:
                self.assertEqual([type(processor) for processor in chain],
                                 [RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                                  TopKLogitsWarper, _CPUTopPLogitsWarper])
            encoder.fail_after_build = True
            with self.assertRaisesRegex(RuntimeError, "synthetic thinker failure"):
                backend.think(None, "instruction", 42)
        self.assertNotIn("generate", encoder.__dict__)
        self.assertNotIn("_get_logits_processor", encoder.__dict__)
        self.assertEqual([call.args for call in backend._resident.call_args_list],
                         [("text_encoder",), (None,)] * 3)

    @unittest.skipUnless(HAS_PINNED_LIBRARY and torch is not None and torch.cuda.is_available(),
                         "Optional CUDA dispatch test requires an NVIDIA GPU")
    def test_cuda_input_returns_cuda_logits_but_never_runs_cuda_cumsum(self):
        was_strict = torch.are_deterministic_algorithms_enabled()
        was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            ids = torch.tensor([[1, 2]], device="cuda")
            scores = torch.tensor([[1., 2., 3., 4.]], device="cuda")
            native = TopPLogitsWarper(.9)
            expected = native(ids.cpu(), scores.cpu())
            probe = CumsumDevices()
            with probe:
                actual = _CPUTopPLogitsWarper(native)(ids, scores)
            self.assertEqual(probe.devices, ["cpu"])
            self.assertEqual(actual.device, scores.device)
            self.assertTrue(torch.equal(actual.cpu(), expected))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(was_strict, warn_only=was_warn_only)


if __name__ == "__main__":
    unittest.main()
