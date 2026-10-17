# Manual ChatGPT UI judging bundle (RQ2)

Keep this ZIP on your computer. **Do not upload the ZIP, private/ directory, this README, or any experiment logs to ChatGPT.** The private manifest reveals condition assignments.

For each batch under upload/, start a fresh chat using the recorded judge label and UI settings. Attach all image files in that batch (source, edited, and any reference), then paste its prompt.md. Keep each filename visible so the source and edited image can be paired correctly. Do not add model prompts, thinking traces, conditions, seeds or previous ratings.

There are two independent calibration passes. Finish pass 1, then use new chats for pass 2. Do not show pass-1 answers to pass 2.

Save the complete JSON answer, unchanged, to a separate responses directory using the batch ID as its filename (for example p1_b001.json). Preserve the chat transcript separately for audit. Record the actual UI model/settings and evaluation date when importing responses. Do not edit upload files or the private manifest after export.

These are manual ChatGPT UI ratings using adapted official rubrics, not the pinned API judge, with the same judge policy and prompt renderer as RQ3. Source images, ground-truth references and instruction text are the Hugging Face records the RQ2 generator received. Differences from the official GitHub copies are recorded in private/manifest.json (reference_provenance, text_provenance); they are not substituted or hidden by resizing.
