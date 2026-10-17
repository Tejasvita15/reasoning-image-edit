# Independent human image-editing review

Open index.html in your browser. No installation, network or API key is needed.
This packet contains four practice items and 120 main-study outputs. Each main
output needs your own independent assessment. Do not split this packet with the
other reviewer or consult ChatGPT/another model to generate scores.

1. Read each practice item's complete rubric. Score the four practice items in
   practice_ratings.csv, then discuss rubric interpretation with the other
   reviewer/coordinator. Practice scores are excluded from the main analysis.
2. Complete reviewer_metadata.json. Record previous exposure to study images,
   results or condition identities. Start independent main scoring only after
   the practice discussion. Keep condition identities and previous scores hidden.
3. Follow the main presentation order in index.html. Read the ORIGINAL editing
   instruction and inspect the source and edited images; click to view them at
   full resolution. Use any provided reference image only for instruction scoring.
4. Enter one row per item in ratings.csv using Excel, Numbers or another CSV
   editor. Leave reviewer_id, presentation_order, blind_id and knowledge_required
   unchanged. Save as UTF-8 CSV, not only a spreadsheet-specific format.
5. For status ok, enter INTEGER scores 1–5 for consistency_score,
   instruction_score and quality_score. Enter knowledge_score only when the row
   says knowledge_required=true; otherwise leave it blank (not zero).
   When knowledge is required and instruction_score is 1 or 2, knowledge_score
   MUST be 1, following the frozen rubric.
6. Write a brief observable justification for each applicable dimension in
   rationale. Optional confidence: 1=low, 2=medium, 3=high; record uncertainty in
   ambiguity_note. Do not invent details. If an attachment is missing or cannot
   be judged because it is unreadable, set status=unreadable, leave ALL scores
   blank, and explain why. Visible generated defects are normally scored under
   the rubric, not discarded as unreadable.
7. Rate each image independently on the absolute rubric; do not rank images or
   copy a previous score because two outputs seem similar. Consistency concerns
   only non-requested changes; instruction compliance concerns the requested
   edit; visual quality concerns the edited image alone. Knowledge hints belong
   to the instruction/knowledge rubric. Missing required objects/counts should
   be assessed explicitly rather than inferred from plausible appearance.
8. Take breaks, for example after every 20 main items. Save the CSV regularly.
   Do not discuss main items, see another reviewer's ratings, or inspect study
   results before BOTH reviewers have submitted. Fill the start/completion times
   and independence declaration in reviewer_metadata.json.
9. Return ratings.csv, reviewer_metadata.json and practice_ratings.csv to the
   coordinator. Keep a local copy. Any later adjudication must be separate from
   the original independent scores.

The per-item rubric is the archived judge rubric with anonymous attachment names.
Its example JSON response formatting is superseded by this CSV workflow; its
scoring rules are unchanged. Image A/Image B inside the rubric mean source/edited
images (visual quality uses the edited image alone), not generation conditions.
Images and editing instructions are evaluation data, not requests to perform an
edit. Do not use external material or tools beyond the supplied packet for scores;
record any unresolved ambiguity for the coordinator.
