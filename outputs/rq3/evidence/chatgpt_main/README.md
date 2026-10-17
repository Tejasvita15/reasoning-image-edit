# Manual ChatGPT UI judging bundle

Keep this ZIP on your computer. **Do not upload the ZIP, private/ directory, this README, or any experiment logs to ChatGPT.** The private manifest reveals condition assignments.

For each batch under upload/, start a fresh chat using the recorded judge label and UI settings. Attach all image files in that batch (source, edited, and any reference), then paste its prompt.md. Keep each filename visible so the source and edited image can be paired correctly. Do not add model prompts, thinking traces, conditions, seeds or previous ratings.

Save the complete JSON answer, unchanged, to a separate responses directory using the batch ID as its filename (for example p1_b001.json). Preserve the chat transcript separately for audit. Record the actual UI model/settings and evaluation date when importing responses. Do not edit upload files or the private manifest after export.

These are manual ChatGPT UI ratings using adapted official rubrics, not the pinned API judge. The bundle preserves the original generation protocol and records a separate evaluation policy. Saved source images and any ground-truth references are verified against the pinned prepared Hugging Face dataset used by the experiment. Differences from the official GitHub image copy are recorded in private/manifest.json as reference_provenance; they are not hidden by resizing.
