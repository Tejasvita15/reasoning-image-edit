# Reviewer 2 submission check

Status: **passed with two nonblocking CSV-format warnings; ready for joint analysis**.

## Checks

- Main: 120/120 unique expected outputs; practice: 4/4. No missing, extra or duplicate IDs.
- Reviewer IDs, presentation order and knowledge-applicability flags match the frozen packet.
- All required scores are integers from 1 through 5. Knowledge scores are present for the required 64 main outputs and blank for the other 56. All instruction-to-knowledge gating rules pass.
- All entries have status `ok`, nonempty rationales and valid confidence values. Eight main entries include ambiguity notes.
- All 381 checked immutable packet and mapping files retain their original hashes.
- Reading all 124 rationales found no definite internal score–rationale contradiction requiring a correction. This was not a new visual correctness study or replacement rating pass.

## Parsing warnings and preservation

The main CSV contains uppercase `TRUE`/`FALSE` applicability flags in all 120 rows. These were interpreted case-insensitively, without changing the submitted file.

The main CSV also contains two invalid UTF-8 bytes (`0xE2`), both in `ambiguity_note`: presentation orders **60** (`review_eb4578acd3e10763`) and **99** (`review_3286304f47ee0db0`). Their zero-based byte offsets are 24738 and 40077. In each case the raw fragment is `1\xe2Ai15`, within a statement about the conventional 15-puzzle goal. The defective bytes do not affect IDs, numerical scores, rationales, row counts or field boundaries.

The [derived UTF-8 CSV](reviewer_2_ratings_utf8.csv) displays those bytes as literal `\xe2` escapes. It makes no guessed punctuation replacement and preserves all ratings. The original three files are preserved byte-for-byte in [the submission snapshot](../submissions/reviewer_2/7c86de2ec00dc002/submission_manifest.json), with SHA-256 hashes. The working originals remain unchanged. Analysis can proceed using the documented decoder; a fresh spreadsheet export is not needed for the score calculations.

## Metadata

| Event | Recorded UTC time, 2026-10-04 | Singapore time (+08:00) |
| --- | --- | --- |
| Practice completed | 07:07:41 | 15:07:41 |
| Main scoring started | 07:15:19 | 15:15:19 |
| Main scoring completed | 08:33:44 | 16:33:44 |

Chronology follows the prescribed order. Main scoring lasted 78 minutes and 25 seconds. The practice start is not recorded; it was not a required template field. The reviewer reports a digital-design background, no prior exposure to the study or previous scores, and independent rating. These are self-reported declarations, not independently verified evidence of authorship or independence.

## Retained interpretation limits

The ambiguity notes disclose assumptions about the 15-puzzle goal, a conflict between a boundary-removal instruction and its knowledge explanation, and reliance on the supplied CT hint. Retain these disclosures when interpreting agreement. Separate dimensions can legitimately assign high visual quality and poor instruction compliance to the same output.

No judge comparison, condition-effect estimation or reviewer-to-reviewer agreement was performed in this submission check. Both original submissions can now be used for the planned joint analysis, excluding practice ratings and retaining the 30-example cluster structure.
