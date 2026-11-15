import numpy as np

from src.config import load_config
from src.dataset import get_decoded_source_images, prepare_example
from src.rq2.common import enable_determinism, load_rq2_examples, load_thought_cache
from src.seeding import set_global_seed
from src.unireason_runner import UniReasonRunner

SEED = 42
N_EXAMPLES = 5


def main():
    enable_determinism()
    config = load_config("configs/rq2/unireason.yaml")
    set_global_seed(SEED)
    examples = [prepare_example(e) for e in load_rq2_examples("data/splits.json", "k_dev")][:N_EXAMPLES]
    cache, _ = load_thought_cache(
        "outputs/rq2_unireason/thought_cache/k_dev_seed42.jsonl",
        expected_uids=[e["example_uid"] for e in examples],
        base_seed=SEED,
        model_id=config["model"]["model_id"],
    )
    runner = UniReasonRunner(config)
    all_same = True
    for e in examples:
        uid = e["example_uid"]
        images = get_decoded_source_images(e)
        for label, plan in (("thinking_off", None), ("thinking_cached", cache[uid]["rewrite"])):
            outs = [
                np.asarray(runner.run(
                    images=images,
                    instruction=e["instruction"],
                    example_uid=uid,
                    base_seed=SEED,
                    cached_thought=plan,
                )["final_image"])
                for _ in range(2)
            ]
            same = outs[0].shape == outs[1].shape and np.array_equal(outs[0], outs[1])
            all_same = all_same and same
            print(f"{uid} {label}: {'identical' if same else 'DIFFERENT'}", flush=True)
    print("RESULT: deterministic" if all_same else "RESULT: NOT deterministic", flush=True)


if __name__ == "__main__":
    main()
