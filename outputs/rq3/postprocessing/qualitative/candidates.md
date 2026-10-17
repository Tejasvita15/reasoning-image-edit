# RQ3 examples for inspection

These examples were selected after scoring using the explicit deterministic rules below. They are diagnostic illustrations, not a representative sample or an independent confirmatory analysis. All 40 examples remain in quantitative estimates.

Quality scores and rationales below are the original saved ChatGPT judge outputs. They are not new human annotations. Inspect the images before writing any additional visual claim.

Each timing label is the per-example median of the three NSCC timing repetitions. Differences and speedups below use these medians.

## typical: rule-based_reasoning__52

Rule: Closest to median D-B, then closest to median charged speedup B/D.

Selected rank 1 of 40; rank-1 UID: `rule-based_reasoning__52`. Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties.

Instruction: Move two matchsticks to make the equation in the image valid.

Saved rewrite: Move the two matchsticks from the first square to the second square to form the number 12. Then, move one matchstick from the second square to the third square to form the number 2. Finally, move one matchstick from the third square to the fourth square to form the number 2.

Saved raw thinking output: Move the two matchsticks from the first square to the second square to form the number 12. Then, move one matchstick from the second square to the third square to form the number 2. Finally, move one matchstick from the third square to the fourth square to form the number 2.

Saved generated token count: 64; count includes EOS: True.

Q differences: `{"B-A": 0.0, "D-B": 0.0, "E-D": 0.0, "F-D": 0.0}`; charged B/D speedup: 1.995x.

[ABCD figure](typical__rule-based_reasoning__52__ABCD.png)

[BDEF figure](typical__rule-based_reasoning__52__BDEF.png)

[selected_regions figure](typical__rule-based_reasoning__52__selected_regions.png)

[differences figure](typical__rule-based_reasoning__52__differences.png)

**Original judge rationale, A (Q=43.75):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The main idea of "Move two matchsticks to make the equation in the image valid." appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, B (Q=43.75):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The main idea of "Move two matchsticks to make the equation in the image valid." appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, C (Q=43.75):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The main idea of "Move two matchsticks to make the equation in the image valid." appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, D (Q=43.75):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The main idea of "Move two matchsticks to make the equation in the image valid." appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, E (Q=43.75):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The main idea of "Move two matchsticks to make the equation in the image valid." appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, F (Q=43.75):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The main idea of "Move two matchsticks to make the equation in the image valid." appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

## fast_with_small_score_change: medicine__24

Rule: Highest charged speedup B/D among examples with |D-B| <= 5.

Selected rank 1 of 38; rank-1 UID: `medicine__24`. Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties.

Instruction: Remove kidney stones from the CT image

Saved rewrite: Remove the kidney stones.

Saved raw thinking output: Remove the kidney stones.

Saved generated token count: 6; count includes EOS: True.

Q differences: `{"B-A": 0.0, "D-B": 0.0, "E-D": 0.0, "F-D": 0.0}`; charged B/D speedup: 2.117x.

[ABCD figure](fast_with_small_score_change__medicine__24__ABCD.png)

[BDEF figure](fast_with_small_score_change__medicine__24__BDEF.png)

[selected_regions figure](fast_with_small_score_change__medicine__24__selected_regions.png)

[differences figure](fast_with_small_score_change__medicine__24__differences.png)

**Original judge rationale, A (Q=43.75):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The result shows limited evidence of "Remove kidney stones from the CT image", with major aspects missing or unclear. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, B (Q=43.75):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The result shows limited evidence of "Remove kidney stones from the CT image", with major aspects missing or unclear. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, C (Q=62.50):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The main idea of "Remove kidney stones from the CT image" appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, D (Q=43.75):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The result shows limited evidence of "Remove kidney stones from the CT image", with major aspects missing or unclear. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, E (Q=43.75):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The result shows limited evidence of "Remove kidney stones from the CT image", with major aspects missing or unclear. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, F (Q=43.75):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The result shows limited evidence of "Remove kidney stones from the CT image", with major aspects missing or unclear. Quality: Overall structure is coherent, with only minor visual imperfections.

## lowest_acceleration_effect: position_movement__30

Rule: Lowest D-B quality difference.

Selected rank 1 of 40; rank-1 UID: `position_movement__30`. Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties.

Instruction: Place the left cup onto the other cup.

Saved rewrite: Place the left cup on top of the right cup.

Saved raw thinking output: Place the left cup on top of the right cup. 

Saved generated token count: 13; count includes EOS: True.

Q differences: `{"B-A": 0.0, "D-B": -16.66666666666667}`; charged B/D speedup: 1.908x.

[ABCD figure](lowest_acceleration_effect__position_movement__30__ABCD.png)

[selected_regions figure](lowest_acceleration_effect__position_movement__30__selected_regions.png)

[differences figure](lowest_acceleration_effect__position_movement__30__differences.png)

**Original judge rationale, A (Q=58.33):**

> Consistency: The requested edit is accompanied by one noticeable non-target change or broader layout drift. Instruction: The image shows a clear attempt to fulfill "Place the left cup onto the other cup.", with the main requested change present but not perfectly exact in every detail. Quality: The image is understandable, but visible artifacts, distortions, or weak text rendering remain.

**Original judge rationale, B (Q=58.33):**

> Consistency: The requested edit is accompanied by one noticeable non-target change or broader layout drift. Instruction: The image shows a clear attempt to fulfill "Place the left cup onto the other cup.", with the main requested change present but not perfectly exact in every detail. Quality: The image is understandable, but visible artifacts, distortions, or weak text rendering remain.

**Original judge rationale, C (Q=58.33):**

> Consistency: The requested edit is accompanied by one noticeable non-target change or broader layout drift. Instruction: The image shows a clear attempt to fulfill "Place the left cup onto the other cup.", with the main requested change present but not perfectly exact in every detail. Quality: The image is understandable, but visible artifacts, distortions, or weak text rendering remain.

**Original judge rationale, D (Q=41.67):**

> Consistency: Multiple non-instructed elements change noticeably beyond the target edit. Instruction: The main idea of "Place the left cup onto the other cup." appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: The image is understandable, but visible artifacts, distortions, or weak text rendering remain.

## lowest_thinking_effect: count_change__48

Rule: Lowest B-A quality difference.

Selected rank 1 of 40; rank-1 UID: `count_change__48`. Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties.

Instruction: Change to six forks

Saved rewrite: Remove four forks from the image.

Saved raw thinking output: Remove four forks from the image.

Saved generated token count: 8; count includes EOS: True.

Q differences: `{"B-A": -50.0, "D-B": 0.0}`; charged B/D speedup: 2.017x.

[ABCD figure](lowest_thinking_effect__count_change__48__ABCD.png)

[selected_regions figure](lowest_thinking_effect__count_change__48__selected_regions.png)

[differences figure](lowest_thinking_effect__count_change__48__differences.png)

**Original judge rationale, A (Q=50.00):**

> Consistency: Multiple non-instructed elements change noticeably beyond the target edit. Instruction: The main idea of "Change to six forks" appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, B (Q=0.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The requested change "Change to six forks" is not successfully realized. Quality: The output is severely degraded, near-empty, or otherwise visually broken.

**Original judge rationale, C (Q=50.00):**

> Consistency: Multiple non-instructed elements change noticeably beyond the target edit. Instruction: The main idea of "Change to six forks" appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, D (Q=0.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The requested change "Change to six forks" is not successfully realized. Quality: The output is severely degraded, near-empty, or otherwise visually broken.

## highest_thinking_effect: mathematics__6

Rule: Highest B-A quality difference.

Selected rank 1 of 40; rank-1 UID: `mathematics__6`. Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties.

Instruction: Align the centers of the two circles

Saved rewrite: Move the green circle so that its center aligns with the center of the light blue circle.

Saved raw thinking output: Move the green circle so that its center aligns with the center of the light blue circle. 

Saved generated token count: 21; count includes EOS: True.

Q differences: `{"B-A": 25.0, "D-B": 0.0}`; charged B/D speedup: 2.066x.

[ABCD figure](highest_thinking_effect__mathematics__6__ABCD.png)

[selected_regions figure](highest_thinking_effect__mathematics__6__selected_regions.png)

[differences figure](highest_thinking_effect__mathematics__6__differences.png)

**Original judge rationale, A (Q=50.00):**

> Consistency: Multiple non-instructed elements change noticeably beyond the target edit. Instruction: The main idea of "Align the centers of the two circles" appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, B (Q=75.00):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The image shows a clear attempt to fulfill "Align the centers of the two circles", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, C (Q=50.00):**

> Consistency: Multiple non-instructed elements change noticeably beyond the target edit. Instruction: The main idea of "Align the centers of the two circles" appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, D (Q=75.00):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The image shows a clear attempt to fulfill "Align the centers of the two circles", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

## lowest_no_rikv_effect: count_change__47

Rule: Lowest E-D quality difference on the fixed 20-example subset.

Selected rank 1 of 20; rank-1 UID: `count_change__47`. Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties.

Instruction: Change to six interfaces

Saved rewrite: Remove two USB ports from the charger.

Saved raw thinking output: Remove two USB ports from the charger.

Saved generated token count: 9; count includes EOS: True.

Q differences: `{"B-A": 16.666666666666657, "D-B": 0.0, "E-D": -33.33333333333333, "F-D": 0.0}`; charged B/D speedup: 1.938x.

[ABCD figure](lowest_no_rikv_effect__count_change__47__ABCD.png)

[BDEF figure](lowest_no_rikv_effect__count_change__47__BDEF.png)

[selected_regions figure](lowest_no_rikv_effect__count_change__47__selected_regions.png)

[differences figure](lowest_no_rikv_effect__count_change__47__differences.png)

**Original judge rationale, A (Q=66.67):**

> Consistency: The requested edit is accompanied by one noticeable non-target change or broader layout drift. Instruction: The image shows a clear attempt to fulfill "Change to six interfaces", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, B (Q=83.33):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The edit cleanly fulfills "Change to six interfaces". Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, C (Q=66.67):**

> Consistency: The requested edit is accompanied by one noticeable non-target change or broader layout drift. Instruction: The image shows a clear attempt to fulfill "Change to six interfaces", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, D (Q=83.33):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The edit cleanly fulfills "Change to six interfaces". Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, E (Q=50.00):**

> Consistency: Multiple non-instructed elements change noticeably beyond the target edit. Instruction: The main idea of "Change to six interfaces" appears to be attempted, but the result is only partial or somewhat ambiguous. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, F (Q=83.33):**

> Consistency: Most non-instructed elements are preserved, with only minor incidental differences. Instruction: The edit cleanly fulfills "Change to six interfaces". Quality: Overall structure is coherent, with only minor visual imperfections.

## no_avd_pixel_variation: multi-instruction_execution__56

Rule: Largest F-vs-D RGB MAE on the fixed subset; descriptive image variation, not inferred quality loss.

Selected rank 1 of 20; rank-1 UID: `multi-instruction_execution__56`. Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties.

Instruction: This poster features a traditional-style painting, suitable for art exhibitions or cultural events. Keep the painting centered and surround it with light clouds and flying cranes. Use a soft blue gradient background with cloud motifs. Add a soft glow to highlight the frame. The top text 'Artistic Charm: Beauty of Tradition' should be in large elegant serif font. Add event info in handwritten style at the bottom left.

Saved rewrite: Add light clouds and flying cranes around the central painting. Apply a soft blue gradient background with cloud motifs. Enhance the frame with a soft glow effect. Update the top text to "Artistic Charm: Beauty of Tradition" in large, elegant serif font. Add event information in handwritten style at the bottom left corner.

Saved raw thinking output: Add light clouds and flying cranes around the central painting. Apply a soft blue gradient background with cloud motifs. Enhance the frame with a soft glow effect. Update the top text to "Artistic Charm: Beauty of Tradition" in large, elegant serif font. Add event information in handwritten style at the bottom left corner. 

Saved generated token count: 67; count includes EOS: True.

Q differences: `{"B-A": 0.0, "D-B": 0.0, "E-D": 0.0, "F-D": 0.0}`; charged B/D speedup: 1.858x.

[ABCD figure](no_avd_pixel_variation__multi-instruction_execution__56__ABCD.png)

[BDEF figure](no_avd_pixel_variation__multi-instruction_execution__56__BDEF.png)

[selected_regions figure](no_avd_pixel_variation__multi-instruction_execution__56__selected_regions.png)

[differences figure](no_avd_pixel_variation__multi-instruction_execution__56__differences.png)

**Original judge rationale, A (Q=50.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The image shows a clear attempt to fulfill "create an art-exhibition poster around the painting", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, B (Q=50.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The image shows a clear attempt to fulfill "create an art-exhibition poster around the painting", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, C (Q=50.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The image shows a clear attempt to fulfill "create an art-exhibition poster around the painting", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, D (Q=50.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The image shows a clear attempt to fulfill "create an art-exhibition poster around the painting", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, E (Q=50.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The image shows a clear attempt to fulfill "create an art-exhibition poster around the painting", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

**Original judge rationale, F (Q=50.00):**

> Consistency: Large parts of the image or scene differ from the source, so non-target details are not preserved. Instruction: The image shows a clear attempt to fulfill "create an art-exhibition poster around the painting", with the main requested change present but not perfectly exact in every detail. Quality: Overall structure is coherent, with only minor visual imperfections.

