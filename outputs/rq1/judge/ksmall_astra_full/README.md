# RQ1 Step1X: all-at-once Astra Pro manual judging

**ONE handoff includes the complete 240-output K-small experiment AND 20 independent repeat ratings for intra-judge consistency (260 ratings total).**

This is a MANUAL ChatGPT UI adaptation of official KRIS-Bench rubrics, not the pinned GPT-4o API evaluation. Actual model and settings must be recorded.

## Run once (50 short judging chats total)

1. Select GPT-6 Astra Pro if available; record the actual UI model/mode.
2. For EACH directory under `upload/main/` (40 folders): start a NEW CHAT, attach all seven images (`source.png` + six `_edited.png`), paste `prompt.md`, and request JSON in `response_template.json` shape.
3. For EACH directory under `upload/repeat/` (10 folders): use a NEW CHAT, attach the three images (`source.png` + two `_edited.png`), paste its `prompt.md`, and save JSON. Do not reveal or consult prior ratings.
4. Store verbatim responses as `responses/main_b001.json` through `responses/main_b040.json`, and `responses/repeat_b001.json` through `responses/repeat_b010.json`. Store any raw text, refusals, or unreadable items rather than inventing scores.
5. Give back the response JSON files and a short note identifying model name/mode/date. No second judging request should be needed.

## Blinding and privacy

**Never upload `private/manifest.json`, `README.md`, the entire folder, or the original labelled `ksmall_multiseed` paths to the judging chat.** Use only the files within ONE `upload/.../<batch>` directory for each chat. Never share prior scores between chats. Avoid including chat links or IDs from earlier cases in a new chat.

Blind IDs and file names do not indicate condition or random seed. All six outputs for the same original are grouped only to reduce manual uploads, so judge each item independently and avoid ranking. This grouped-prompt procedure can induce comparison bias, and is not exactly equivalent to six isolated single-item API calls.

Only the original and edited images are provided; the benchmark's two available reference images are deliberately not included in judge attachments, to avoid answer leakage.

The private manifest is for post-judging analysis only.
