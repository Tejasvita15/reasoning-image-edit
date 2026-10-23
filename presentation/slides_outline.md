# Group presentation: slide outline

**Status:** outline final. Slides still need the **[insert]** placeholders filled:

- slide 6: RQ1 results;
- slides 7–8: UniReason results.

Any change to the slides should be reflected here first.

| Slide | Title                                                  | Time     |
| ----- | ------------------------------------------------------ | -------- |
| 1     | Title                                                  | 0:10     |
| 2     | Knowing the answer isn't the same as drawing it        | 0:40     |
| 3     | How an AI image editor works                           | 0:45     |
| 4     | Editors that think and check their work                | 0:45     |
| 5     | Our project: questions and method                      | 0:45     |
| 6     | RQ1: does thinking help? Does correction help?         | 0:40     |
| 7     | RQ2: spend compute on thinking, not more steps         | 0:50     |
| 8     | RQ2: is self-correction better than just trying again? | 0:55     |
| 9     | RQ3: making it faster with RegionE                     | 0:45     |
| 10    | Judging "good editing" is hard                         | 0:40     |
| 11    | Takeaways                                              | 0:25     |
|       | **Total**                                              | **7:20** |

---

## 1. Title (0:10)

- _Does "thinking" help AI image editors?_ Knowledge-intensive image editing, AI6132 group project.
- Team names.

## 2. Hook: knowing the answer isn't the same as drawing it (0:40)

- Instruction: _"Changes in animal fur after entering summer"_, on a photo of an arctic fox in snow.
- The correct edit needs knowledge: arctic foxes turn brown in summer.
- **The model's thinking got it right in words:** _"Change the animal's fur from white to a brown color."_
- **But the image generator drew a different animal:** a red fox, still in snow.
- Point: AI editors can "know" the answer and still fail to draw it.
- **Visual:** the source image beside Step1X's edit (`biology__49`, `outputs/rq2_3/cand1/k_small/seed_42/`), with the rewrite as a caption.

## 3. Background: how an AI image editor works (0:45)

- Diffusion in one picture: start from random noise and clean it up step by step, guided by the photo and the instruction.
- Two dials we test later:
    - **Refinement steps:** more steps take longer.
    - **Guidance:** how strongly the model obeys the instruction.
- **Visual:** a simple diagram, noise → steps → edited image, with the two dials labelled.

## 4. Recent advance: editors that think and check their work (0:45)

- **Thinking:** a built-in language model rewrites a vague instruction into a concrete one, as in the fox example.
- **Reflection:** the model critiques its own result and retries if it judges it a failure. On the fox, it caught the mistake (_"it also incorrectly changed the animal's species from an arctic fox to a red fox"_), but its retry drew a red fox again.
- Models: Step1X-Edit v1.2 and UniReason (recent reasoning editors).
- Also recent:
    - **KRIS-Bench:** a benchmark of knowledge-heavy edits;
    - **RegionE:** a training-free speed-up;
    - **EditReward:** an AI model that scores edits, used on slide 8.
- **Visual:** pipeline diagram, think → edit → reflect → retry.

## 5. Our project: questions and method (0:45)

- **Benchmark:** KRIS-Bench, with factual, conceptual and procedural knowledge tasks.
- **Three questions:**
    - **RQ1:** does thinking or correction help?
    - **RQ2:** how should the dials be set, and is self-correction worth it?
    - **RQ3:** can it be made faster without losing quality?
- **Scoring:** an AI judge (a ChatGPT model using KRIS's scoring rules) rates each edit on up to 4 criteria, combined into a 0–100 quality score.
- **Fair tests:** same thought and same starting noise, so only one thing changes at a time.
- **Visual:** a 3-column question map.

## 6. RQ1: does thinking help? Does correction help? (0:40)

- Thinking on vs off; correction on vs off.
- **[RQ1 result: insert]**: give RQ1's _answer_ here. Slide 8 explains _why_ correction struggles, so don't repeat it.
- **[Example before/after: insert]**

## 7. RQ2: spend compute on thinking, not more steps (0:50)

- **Thinking:** on 90 tuning examples, a small gain of about +2 to +5 points for only about 1.4 s.
- **Steps:** going from 20 to 50 steps adds about 60 s for no measurable gain; 28 is enough.
- **Guidance:** the default (6) works best. Lowering it hurts (about −4 points), and raising it doesn't help.
- **Best setting on 300 unseen test examples:** quality 58.8.
- **Caveat:** when the AI judge re-scored the same images, its scores moved by about 5 points on average, and once by as much as 25. So these gains are small trends, not decisive wins.
- **[UniReason: insert]**
- **Visual:** `outputs/rq2_2/analysis/figures/rq2_2_quality_latency.png`.

## 8. RQ2: is self-correction better than just trying again? (0:55)

- Same first attempt. Then either the model fixes it (feedback) or simply redraws it (resampling). An AI reward model (EditReward) picks the better of the two tries.
- **Result:** feedback is no better (−1 point) and 32 s slower. Neither kind of second try beats the first attempt.
- **Example: "Insert a pencil into a glass of water"**
    - The first edit added two pencils (score 69).
    - The critique was right: _"shows two pencils… Remove the black pencil."_
    - The fix erased both pencils, leaving an empty glass (38). A plain redraw scored 75.
- **Lesson:** right diagnosis, broken repair. Out of 39 repairs of the first image, 5 improved it, 17 made it worse, and the rest made no difference.
- **[UniReason: insert]**
- **Visual:** a three-image strip from `outputs/rq2_3/{cand1,feedback_cand2,resample_cand2}/k_small/seed_42/physics__11.png`: two pencils, empty glass, one pencil.

## 9. RQ3: making it faster with RegionE (0:45)

- Why it works: most of the picture doesn't change during an edit, so RegionE skips recomputing it and reuses earlier work.
- **Result:** about 2× faster end to end (2.6× for the image generator). On average it lost less than 1 point, well inside the 5-point limit set in advance.
- **Trade-off:** peak GPU memory rises from about 24 to 35 GB.
- **Visual:** three big-number cards (≈2× faster · <1 point lost · 24 → 35 GB). This replaces the RQ3 quality–latency plot: its panels share a y-axis, so a single-panel crop loses the axis labels, and the numbers are clearer for a no-background audience.

## 10. Lesson: judging "good editing" is hard (0:40)

- **Same 30 examples, rated by two humans and the AI judge (RQ3):**
    - The humans agree with each other strongly (0.9 out of 1).
    - Each barely agrees with the AI judge (about 0.1 out of 1).
- **Same images, different verdict on thinking:** humans see a benefit (about +6 to +8 points), while the AI judge sees none (about −1 to −2).
- **Spoken line:** "Even our own AI judge gave different answers in different experiments: a small gain in RQ2, none in RQ3."
- **Takeaway:** small differences between methods need careful, multi-judge evaluation.
- **Visual:** RQ3 agreement plot (`outputs/rq3/human_validation/analysis/figures/agreement_q.pdf`).

## 11. Takeaways (0:25)

- Thinking is cheap, but its benefit is small and depends on who judges.
- More refinement steps don't buy quality.
- Self-correction isn't yet better than simply trying again: the right diagnosis can still lead to a broken repair.
- Editors can be sped up about 2× with little average quality loss.
