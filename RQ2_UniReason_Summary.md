# RQ2 results: UniReason (RQ2.1, RQ2.2 and RQ2.3)

RQ2 asks how Step1X-style dials behave for UniReason: guidance and number of steps (RQ2.1 and RQ2.2), whether thinking changes the answer, and whether a second attempt helps (RQ2.3).

RQ2.1 and RQ2.2 are complete: generation, judging, statistics and figures. RQ2.3 is complete for all three seeds (42, 1234 and 7): generation, selection, judging and scoring. The three-seed figure and write-up are still to be finalised. The design is in the RQ2 plan (the RQ2 guidance and steps protocol).

## Headline findings

1. **Thinking did not help UniReason.** Thinking on minus off is -2.0, -3.5 and -1.6 Q at guidance 2, 4 and 6. None is above the MMD (16.77 Q), and none is statistically clear. Thinking costs about 10.6 s per image. The cached thought was checked against live thinking: identical images on 3 K-dev examples (see Common setup).
2. **Guidance 4 stays as the default.** Without thinking, guidance 6 is clearly worse than 4 (-3.8 Q, CI excludes 0). With thinking, guidance 2 beats 4 by 3.75 Q, which is below the MMD, so the default is kept.
3. **More steps help only without thinking, and only from 28 to 50.** Without thinking, 50 steps beats 28 by 3.4 Q (CI excludes 0, below the MMD). 20 and 28 steps are the same. With thinking, the step count makes no clear difference.
4. **Frozen configuration: thinking off, 50 steps, guidance 4.** It has the highest K-dev mean Q of the guidance-4 runs (48.15). On the held-out K-test (300 examples, run once) it scores Q = 45.29 [42.95, 47.67].
5. **At matched latency, thinking does not win.** Thinking on at 20 steps (29.1 s) vs thinking off at 28 steps (27.9 s): -3.66 Q, CI [-8.22, +0.90], p = 0.138. The trend favours thinking off, but it is not clear enough to claim.
6. **RQ2.3 (three seeds, provisional): both second attempts beat the first image, and feedback is not clearly better than resampling.** Averaged over seeds (n = 40), feedback minus first is +4.67 Q [+2.24, +7.52], resampling minus first is +2.05 Q [+0.28, +3.82], and feedback minus resampling is +2.62 Q [-0.19, +5.68]. All three are below the MMD, so none is claimed as a real gain. Median latency per example is about 85 s for both arms.

None of the differences in RQ2.1, RQ2.2 or RQ2.3 reaches the MMD. The effects above are trends, not decisive wins.

---

## Common setup

| Item | Setting |
|---|---|
| Model | UniReason, stage_1_align checkpoint 0030000, loaded from local weights |
| Hardware | ASPIRE2A, NVIDIA A100-SXM4-40GB, 1 GPU per job |
| Offload | `none_all_on_gpu` (whole model on GPU). This differs from Step1X's model CPU offload, so UniReason and Step1X latencies must not be compared directly |
| Thinking | The thought is generated once per example and cached (`outputs/rq2_unireason/thought_cache/k_dev_seed42.jsonl` for RQ2.1 and RQ2.2; `outputs/rq2_3_unireason/thought_cache/k_small_seed{42,1234,7}.jsonl` for RQ2.3). Thinking-on runs replay the cached thought. One thinking pass (median about 10.6 s) is charged to every thinking-on image |
| Replay check | Cached-thought replay against live thinking, seed 42, K-dev, guidance 4, 50 steps, 3 examples (`abstract_reasoning__22`, `anomaly_correction__60`, `biology__13`): MD5-identical images in all 3 pairs. Output: `outputs/equiv_check_unireason/`. Three examples is strong evidence for the replay, not proof for all examples |
| Reflection / refinement | Off in RQ2.1 and RQ2.2 (`enable_reflection_mode: false`, `max_try_cnt: 1`). RQ2.3 uses the released refinement for the feedback arm only |
| Seeds | RQ2.1 and RQ2.2: seed 42 only, per the plan. RQ2.3: seeds 42, 1234 and 7 on K-small, per the plan (S1 and S8). All three seeds are complete for RQ2.3. Starting noise is shared across guidance and step values within an example |
| Splits | K-dev 90 (tuning), K-test 300 (used once, for the frozen configuration). RQ2.3 uses K-small (40, a subset of K-test) |
| Judge | GPT-6 Astra Pro (as selected in the ChatGPT UI), prompt version `kris-chatgpt-ui-v2`. Each research question was judged in its own judging run |
| Quality score Q | Mean of the applicable component scores, 0 to 100 (KRIS aggregate) |
| Judge noise (MMD) | 16.77 Q (95th percentile of the judge's re-scoring differences, from UniReason's calibration) |
| Latency | Mean per-example latency for each run, then the median across the timing repeats (shared rule S4: median of at least 3 repeats). RQ2.2 latency is the median across repeats for each configuration |
| Statistics | Paired by example. Mean difference with a 95% paired bootstrap CI (10,000 resamples). Wilcoxon signed-rank p as a check. No multiple-comparison correction |
| Determinism (S7) | Checked on 5 K-dev examples, each generated twice with the same seed, for thinking off and thinking on. All 10 pairs were pixel-identical. RQ2.3 timed reruns: all 120 images (seed 42) pixel-identical to the originals |

**How to read the tables.** "A - B [CI]" is the mean paired difference. If the interval excludes 0, the trend is consistent across examples. It counts as meaningful only if it also exceeds the MMD. None does here.

---

## RQ2.1: Does thinking change guidance sensitivity?

**What we ran:** K-dev (90), seed 42, 50 steps, reflection off. Thinking off and on, at text guidance 2, 4 and 6. Image guidance pinned at 2.0. That is 6 configurations and 540 judged images.

### Results (mean Q, K-dev, n = 90 each)

| Thinking | Guidance 2 | Guidance 4 | Guidance 6 |
|---|---|---|---|
| Off | 50.37 | 48.15 | 44.35 |
| On | 48.40 | 44.65 | 42.80 |

Per-configuration 95% intervals (bootstrap over examples): off at guidance 4 is 48.15 [44.12, 52.13] and on at guidance 4 is 44.65 [40.23, 49.03].

| Comparison | A - B [95% CI] | Wilcoxon p |
|---|---|---|
| Thinking on - off, guidance 2 | -1.97 [-6.46, +2.69] | 0.375 |
| Thinking on - off, guidance 4 | -3.50 [-7.59, +0.56] | 0.069 |
| Thinking on - off, guidance 6 | -1.55 [-5.74, +2.69] | 0.344 |
| Guidance 2 - 4, thinking off | +2.22 [-1.11, +5.60] | 0.289 |
| Guidance 6 - 4, thinking off | **-3.80 [-6.04, -1.53]** | < 0.001 |
| Guidance 2 - 4, thinking on | +3.75 [+0.97, +6.57] | 0.014 |
| Guidance 6 - 4, thinking on | -1.85 [-4.14, +0.44] | 0.100 |
| Interaction: (on 6 - on 2) - (off 6 - off 2) | +0.42 [-3.75, +4.70] | 0.967 |

**Open:** the thinking-on minus thinking-off interval at guidance 4 and 50 steps is [-7.59, +0.56] here and [-7.55, +0.37] in RQ2.2. The same paired comparison should give one interval, so the stats file needs re-running to settle it.

### What it means

- Thinking does not reduce guidance sensitivity. The interaction is about zero.
- Guidance 6 without thinking is clearly worse than 4.
- Guidance 2 with thinking is better than 4, but by 3.75 Q, below the MMD.
- **Selected guidance for RQ2.2: 4 for both thinking modes.** The rule was the highest mean Q, keeping the default unless the winner beats it by at least the MMD. Neither alternative does.

**Note for the write-up.** The released UniReason refinement uses image guidance 1.5. This study pins image guidance at 2.0, as set in the protocol.

**Files**
- Stats: `outputs/rq2_unireason_analysis/stats_2_1_2_2.txt` (the RQ2.1 section)
- Curves (instruction following and visual consistency vs guidance): `outputs/rq2_unireason_analysis/figures/rq2_unireason_2_1_curves.png`
- Judge scores: `results/rq2_main_pass1.json` (540 rows, 0 null Q)
- Logs: `logs/rq2_unireason/`

---

## RQ2.2: Where is extra compute best spent, steps or thinking?

**What we ran**
- **Quality runs:** K-dev (90), seed 42, guidance 4, reflection off. Thinking off and on, at 20, 28 and 50 steps. The 50-step runs are reused from RQ2.1, so 360 new judged images (180 chats).
- **Frozen configuration:** run once on K-test (300).

### Latency (median seconds per image, A100-SXM4-40GB, offload `none_all_on_gpu`, n = 90 each)

| Thinking | Steps | End-to-end | Generate | Think | Peak VRAM (GB) |
|---|---|---|---|---|---|
| Off | 20 | 19.79 | 19.59 | 0 | 30.9 |
| Off | 28 | 27.90 | 27.66 | 0 | 30.9 |
| Off | 50 | 49.09 | 48.87 | 0 | 30.9 |
| On | 20 | 29.06 | 19.90 | 10.63 | 30.9 |
| On | 28 | 36.97 | 28.03 | 10.63 | 30.9 |
| On | 50 | 58.16 | 49.73 | 10.63 | 30.9 |

End-to-end = preprocess + think + generate + postprocess (median of repeats). Preprocessing is about 0 s, and postprocessing is 0.14 to 0.26 s. The end-to-end column matches the logged reconstructed total in every row.

### Quality (K-dev, guidance 4, mean Q, n = 90 each)

| Thinking | 20 steps | 28 steps | 50 steps |
|---|---|---|---|
| Off | 44.37 | 44.72 | **48.15** |
| On | 41.06 | 44.54 | 44.65 |

Per-configuration 95% intervals: off 20 steps 44.37 [40.05, 48.70]; off 28 steps 44.72 [40.93, 48.54]; on 20 steps 41.06 [36.55, 45.58]; on 28 steps 44.54 [40.02, 48.98].

### Comparisons

| Comparison | A - B [95% CI] | Wilcoxon p | Latency A - B |
|---|---|---|---|
| Thinking on - off, 20 steps | -3.31 [-7.99, +1.18] | 0.285 | +9.3 s |
| Thinking on - off, 28 steps | -0.19 [-4.56, +4.14] | 0.793 | +9.1 s |
| Thinking on - off, 50 steps | -3.50 [-7.55, +0.37] | 0.069 | +9.1 s |
| 28 - 20 steps, thinking off | +0.35 [-2.52, +3.15] | 0.620 | +8.1 s |
| 50 - 28 steps, thinking off | **+3.43 [+0.79, +6.09]** | 0.023 | +21.2 s |
| 28 - 20 steps, thinking on | +3.47 [+0.51, +6.97] | 0.114 | +7.9 s |
| 50 - 28 steps, thinking on | +0.12 [-2.34, +2.73] | 0.971 | +21.2 s |
| **Matched latency:** thinking on @ 20 - off @ 28 | -3.66 [-8.22, +0.90] | 0.138 | +1.2 s |
| Frozen config: off @ 50 - on @ 50 | +3.50 [-0.58, +7.50] | 0.069 | -9.1 s |

**Matched-latency note.** The pair that is actually matched in time is thinking on at 20 steps vs thinking off at 28 steps (29.1 s vs 27.9 s). Thinking on at 20 steps vs thinking off at 50 steps is not matched (29 s vs 49 s), so it is not used as the matched comparison.

### Frozen configuration and K-test

- **Rule:** the highest mean K-dev Q among the runs, fixed before any K-test run.
- **Selected:** thinking off, 50 steps, guidance 4, reflection off. K-dev mean 48.15, which beats the runner-up (thinking on, 50 steps, 44.65) by 3.50. That margin is below the MMD, so it is not claimed as significantly better. The rule has no MMD threshold, so this is a choice, not a demonstrated difference.
- **K-test result (300 examples, run once):**

| Subset | n | Q [95% CI] |
|---|---|---|
| All | 300 | **45.29 [42.95, 47.67]** |

K-test was run only on the chosen configuration, so it does not compare thinking off with thinking on. Per-knowledge-type breakdown for UniReason has not been computed yet. Report the K-test number separately from RQ1.1, since the settings differ.

### What it means

- Thinking costs about 9 to 10 s per image for UniReason, and it does not improve Q at any step count.
- Without thinking, going from 28 to 50 steps adds about 3.4 Q for about 21 s more per image. That is the only step effect that is clear, and it is below the MMD.
- At matched latency, the trend favours thinking off at 28 steps over thinking on at 20 steps, but the interval includes zero.
- **Comparison with the paper.** The UniReason paper's headline KrisBench numbers (68.23 with refinement) include refinement, and reflection/refinement is off in this study. Our numbers are not comparable to the paper's table. The paper's reasoning-only row (64.12) also uses different settings.

**Caveat to state:** fewer sampling steps is not distillation. This trades steps for speed on the same model.

**Files**
- Latency table (median method, use this one): `outputs/rq2_unireason_analysis/latency_table_median.txt`
- Stats: `outputs/rq2_unireason_analysis/stats_2_1_2_2.txt`. The latency table inside this file is the earlier mean-based version and is superseded by `latency_table_median.txt`
- Figure, quality vs latency (medians): `outputs/rq2_unireason_analysis/figures/rq2_unireason_2_2_quality_latency_median.png`
- Judge scores: `results/rq2_22_pass1.json` (360 rows) and `results/rq2_ktest_pass1.json` (300 rows)
- Frozen config record and logs: `logs/rq2_unireason_2/` (20 and 28 steps), `logs/rq2_unireason/` (50 steps), `logs/rq2_unireason_ktest/` (K-test)
- Replay check: `outputs/equiv_check_unireason/`, `logs/equiv_check_unireason/`

---

## RQ2.3: Is feedback better than just trying again? (UniReason)

**Status:** all three seeds are complete (generation, selection, judging, scoring). Headline numbers average the seeds within each example first, then report n = 40, as the plan requires (S8). The seed 42 detail table is kept for reference.

**What we ran.** K-small (40 examples, a subset of K-test), seed 42, thinking on with the cached plan (replay checked against live thinking), guidance 4, 50 steps, reflection off. Both arms share candidate 1, and each arm's second candidate is judged in its own run. Feedback arm: UniReason's released refinement produces candidate 2. Resampling arm: a fresh redraw from the same cached plan with a new seed. EditReward-MiMo-VL-7B picks the higher-scoring candidate in each arm. On a tie the selector keeps candidate 1 (checked in the selector code). The judge is the shared judge; each arm's images were judged in separate judging runs.

### Three-seed results (n = 40, seeds averaged within each example)

| Arm | Mean Q |
|---|---|
| First candidate only | 44.36 |
| Feedback (released refinement) | 49.03 |
| Resampling | 46.41 |

| Comparison | Mean difference Q [95% CI] | Wilcoxon p |
|---|---|---|
| Feedback minus first | +4.67 [+2.24, +7.52] | 0.001 |
| Resampling minus first | +2.05 [+0.28, +3.82] | 0.021 |
| Feedback minus resampling | +2.62 [-0.19, +5.68] | 0.247 |

Per-seed means (first / feedback / resampling): seed 42 44.11 / 47.34 / 45.78; seed 1234 44.53 / 50.05 / 45.83; seed 7 44.43 / 49.69 / 47.60.

The selector kept the second candidate in 17, 23 and 20 of 40 examples for feedback (seeds 42, 1234, 7), and in 18, 25 and 24 of 40 for resampling. No ties in any seed.

All three comparisons are below the MMD (16.77 Q), so none is claimed as a real difference.

### Seed 42 detail (n = 40)

| Measure | Feedback | Resampling | Difference (feedback - resampling), 95% CI |
|---|---|---|---|
| Mean Q, final pick | 47.34 [39.79, 55.31] | 45.78 [38.70, 53.39] | +1.56 [-2.24, +5.68] |
| First candidate only, mean Q | 44.11 [36.98, 51.61] | (same) | n/a |
| Final pick minus first candidate | +3.23 [+0.26, +6.56] | +1.67 [-1.41, +4.79] | feedback minus resampling is the row above |
| Second candidate minus first (mean Q change) | +0.57 [-4.48, +5.52] | +0.05 [-4.11, +4.22] | n/a |
| Second candidate beat / tied / lost the first | 13 / 14 / 13 | 14 / 12 / 14 | n/a |
| EditReward kept the judge-better image (non-tied pairs) | 18 of 26 | 19 of 28 | n/a |
| Latency per example, s (mean over examples, median across three timing runs) | 85.33 | 85.48 | -0.15 |
| First candidate only latency, s (same method) | 47.15 | (same) | n/a |

Wilcoxon p = 0.397 for Q (feedback minus resampling). The Q difference is well below the MMD (16.77). Feedback's gain over the first candidate (+3.23) has an interval that excludes zero, but it is still below the MMD, so no difference is claimed. Latency includes the cached thinking pass, the first candidate, the correction or redraw, and the selector time for each arm.

Latency by run (mean per example): seed 42, feedback 85.85 s and resampling 85.98 s; seed 1234, 85.33 s and 85.48 s; seed 7, 84.64 s and 84.78 s. The table reports the median of these three runs. Seed 42's values were the only ones available in earlier drafts, so the earlier 85.85 and 85.98 are replaced by the median.

**Disclosures**
- K-small is a subset of K-test, so these 40 examples were also in the K-test pool.
- The feedback arm uses UniReason's released refinement, not native reflection. It has no native internal selection, so no second view is reported.
- The exported source images differ from the official Kris-Bench copies on every row, for all three seeds and for the calibration bundle. On the seed 42 export, the mean pixel difference is about 0.1 levels out of 255, with local peaks up to 74 levels. That looks like re-encoding rather than different content. The judge and the generator both used the same HF source images. Recorded as a known issue. The checkout was verified clean at the pinned revision.
- The timed jobs requested 1 GPU, 8 CPUs and 64 GB (`jobs/rq2_3_unireason_timed.pbs`). The job logs do not record the allocation actually received.
- The UniReason checker passes for seed 42 (40 of 40 examples).
- Seeds 1234 and 7 went through the same generation, selection, judging and scoring as seed 42. Only seed 42 has the full per-seed breakdown (beat/tied/lost counts and EditReward kept-better counts) so far.

**Files**
- Selector logs (all seeds): `logs/rq2_3_unireason/selector/rq2_3_selector_k_small_seed{42,1234,7}.jsonl`
- Judge imports: `outputs/rq2_3_unireason/import_seed42/` and `outputs/rq2_3_unireason/import_seeds_1234_7/`
- Per-example pairs (seed 42): `logs/rq2_3_unireason/analysis/rq2_3_unireason_final_pairs.csv` (40 rows)
- Timing runs: `outputs/rq2_3_unireason_timed/timing_seed_{42,1234,7}.jsonl`
- Figure: to be regenerated with the updated numbers and the three-seed average
- Checker: `src/rq2/check_rq2_3_unireason.py`

---

## Open items

- **RQ2.1 and RQ2.2 interval:** the thinking-on minus thinking-off interval at guidance 4 and 50 steps needs one value. Re-run the stats file.
- **RQ2.3 figure:** regenerate with the three-seed averages and interval bars.
- **RQ2.3 write-up wording:** all comparisons are below the MMD, so describe them as trends.
- **Judge name:** GPT-6 Astra Pro, as selected in the ChatGPT UI.
- **K-small overlap.** Disclosed in the RQ2.3 section above.
