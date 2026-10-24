# RQ1 blinded image-editing evaluation — manual Astra UI handoff

**380 main judging batches and 10 repeat batches; 1,780 individual image ratings.**
This is a manual ChatGPT UI adaptation of the KRIS-Bench rubric, not the original GPT-4o API scorer.

For EACH directory under `upload/main/`, open a NEW chat using the recorded Astra Pro mode. Attach `source.png` and every `*_edited.png` in that directory (5 or 7 images total). Paste the full `prompt.md`, including the JSON output template. Save the verbatim JSON as `responses/<batch_id>.json`.

For EACH directory under `upload/repeat/`, also use a NEW chat, independently of any earlier ratings. Attach the three PNG images and paste `prompt.md`. Save the JSON using its `repeat_bNNN` batch ID.

Judge every edited image independently using absolute 1–5 rubric scales. Do not rank edits. Mark unreadable items as unreadable with null scores rather than guessing. If a knowledge score is inapplicable, use null.

Record the exact displayed model/mode and date in `session_log_template.json`. Return the entire `responses/` folder and that session log together, in one handoff.

**Blinding:** Share only this README, `upload/` and the session log template with the judge. NEVER provide `private/manifest.json`, original labeled output directories, earlier scores or previous responses to a judging chat. In each chat, upload only the files of ONE batch; do not upload the full archive into the judge's chat.

Grouping multiple edits by source reduces uploads but can introduce context effects. Ratings are a UI adaptation and should be reported as such.
