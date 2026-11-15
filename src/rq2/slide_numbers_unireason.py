import json, random
from src.rq2.score_rq2_3_codex import load_bundle, JUDGE, BUNDLES, SELECTOR

scores = {}
for b in BUNDLES:
    part, _ = load_bundle(JUDGE / b)
    scores.update(part)

picks = [json.loads(l) for l in SELECTOR.read_text().splitlines() if l.strip()]
picks_by_uid = {p["example_uid"]: p for p in picks}

def q(uid, cand):
    r = scores.get((uid, cand))
    return None if r is None else r["Q"]

rows = []
for uid in picks_by_uid:
    rows.append((uid, q(uid, "cand1"), q(uid, "feedback_cand2"), q(uid, "resample_cand2")))
ok = [r for r in rows if None not in r[1:]]
print(f"examples with all three scores: {len(ok)} of {len(picks_by_uid)}")

def boot(vals, n=10000, seed=0):
    rng = random.Random(seed); m = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]

# Final picks per arm (should match README: feedback 47.34, resample 45.78)
final = {}
for arm in ("feedback", "resample"):
    v = [q(uid, picks_by_uid[uid]["selection"][arm]["selected"]) for uid, *_ in ok]
    final[arm] = v
    lo, hi = boot(v)
    print(f"{arm} final pick: mean Q {sum(v)/len(v):.2f} CI [{lo:.2f}, {hi:.2f}]")
d = [a - b for a, b in zip(final["feedback"], final["resample"])]
lo, hi = boot(d)
print(f"feedback - resample: {sum(d)/len(d):+.2f} CI [{lo:+.2f}, {hi:+.2f}]")

# Second candidate vs first candidate (Step1X slide 8 table)
for name, idx in (("feedback", 2), ("resample", 3)):
    w = t = l = 0
    for r in ok:
        dd = r[idx] - r[1]
        if abs(dd) < 1e-9: t += 1
        elif dd > 0: w += 1
        else: l += 1
    print(f"{name} second vs first: beat {w} / tie {t} / lost {l}")

# Did the selector keep the judge-better image? (non-tied pairs only)
for arm, cand in (("feedback", "feedback_cand2"), ("resample", "resample_cand2")):
    kept = tot = 0
    for uid, c1, fb, rs in ok:
        q2 = fb if arm == "feedback" else rs
        if abs(q2 - c1) < 1e-9: continue
        better = cand if q2 > c1 else "cand1"
        tot += 1
        kept += picks_by_uid[uid]["selection"][arm]["selected"] == better
    print(f"{arm}: selector kept judge-better {kept} of {tot} non-tied pairs")

# Example candidates for the slide 8 strip: largest repair drop
worst = min(ok, key=lambda r: r[2] - r[1])
print(f"example for strip (feedback most worse than first): {worst[0]}  cand1={worst[1]:.1f} feedback={worst[2]:.1f} resample={worst[3]:.1f}")
c1 = [r[1] for r in ok]
lo, hi = boot(c1)
print(f"first only: mean Q {sum(c1)/len(c1):.2f} CI [{lo:.2f}, {hi:.2f}]")
for name, idx in (("feedback", 2), ("resample", 3)):
    dd = [r[idx] - r[1] for r in ok]
    lo, hi = boot(dd)
    print(f"{name} second minus first: {sum(dd)/len(dd):+.2f} CI [{lo:+.2f}, {hi:+.2f}]")
    dp = [a - r[1] for a, r in zip(final[name], ok)]
    lo, hi = boot(dp)
    print(f"{name} final pick minus first: {sum(dp)/len(dp):+.2f} CI [{lo:+.2f}, {hi:+.2f}]")
