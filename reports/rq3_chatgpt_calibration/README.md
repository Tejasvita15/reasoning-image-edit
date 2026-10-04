# Manual RQ3 calibration — 2026-09-30

Both submitted passes imported successfully: 20 outputs per pass, 40 ratings.
Every paired Q value and component score agrees. The observed mean and 95th
percentile of absolute Q differences are both **0** on the 0–100 scale.

`configs/rq3.yaml` now records MMD = **0** and δ = **5** before main test
evaluation. δ is a chosen five-point acceptable-loss tolerance, not an estimate
derived from zero observed variation. Twenty matching re-scores do not establish
zero judge variability. See `decision.json` for the recorded choice and identities.

The operator rejected the session log's shared-conversation description and
authorized correction. The corrected metadata records the frozen fresh-chat
settings on that basis and the original entries' model label, **GPT-6 Astra Pro**.
The unique `operator-attested:` session IDs are local bookkeeping IDs, not
recovered ChatGPT URLs. Session independence and settings rely on that correction;
no transcripts were supplied for independent verification. Timestamps are retained
as supplied without independently verifying their chronology.

Original source-bundle files, all raw responses, scores, images, prompts, assignments
and generation identity remain unchanged. This is a retrospective, explicitly
audited metadata correction. The corrected full local bundle is at
`outputs/rq3/evidence/calibration/`; the original source bundle retains
the old metadata and should not be used for the corrected imports.

Files here are small calibration audit artifacts; no image or model files are included:

- `repeatability.json`: generated repeatability report for future main export.
- `judge_first.jsonl`, `judge_second.jsonl`: imported scores with raw responses
  and their provenance.
- `paired_differences.csv`: all 20 comparisons, including component agreement.
- `decision.json`: calibration result and the pre-test statistical settings.
- `bundle_metadata/`: corrected manifest/session log, unchanged raw responses,
  and original metadata plus the hashed audit under `private/metadata_correction/`.
- `SHA256SUMS`: checksums of the files in this directory, excluding itself.

`bundle_metadata/` alone is **not** a complete judging bundle. Combine it with a
copy of the original NSCC bundle using the dated procedure in `RQ3_experiment.md`.
That procedure verifies the original identity, imports both passes again and
compares the result with this report. Keep the original NSCC bundle until that
verification succeeds. Do not upload private metadata, scores or these reports
to a judge chat; they reveal condition assignments.

Documentation was generalized for collaborator use; `SHA256SUMS` reflects only
those Markdown edits. Scores, response files, manifests and the correction audit
remain unchanged.

The new generation protocol is
`a2001fe4612dfee6ae489c243b0fb204b1f04ba39f4877bf0f53853fe7f1438c`.
Calibration retains its archived generation protocol
`c984fa84c1c8bb0294c20f6fd337e5cc7f4aede868a64452081172970847ef52`.
The manual evaluation policy is
`095c775e5aae19c2469c5647e717d9209b14a9887ae43ec7410f50657e6af2e2`.
Run the final independent pilots under the new generation protocol; keep the
manual judge settings, batch size and code fixed for main judging.
