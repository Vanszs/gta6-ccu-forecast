"""Candidate forecaster families, compared by leave-one-launch-out CV.

The final case is one launch (GTA VI) with an hour-1 print and nothing else, so every candidate maps
an hour-1 print to 23 rows of five quantiles using past launches. Loss is separable per (hour, level),
so each cell is estimated on its own: the pool's weighted empirical quantile, the fitted normal on the
log ratio, or a blend of the two.

Reported: cv42 (all launches), cv_big (targets with a big hour-1 print, the regime GTA VI is in),
practice4 (the four practice cases against the pool the grader gives them).
"""

import glob
import itertools
import json
import math
from pathlib import Path

from model import (
    LEVELS,
    Z,
    answers_from,
    clean,
    curve_from_rows,
    features,
    load_launches,
    practice_answers,
    score_case,
    weighted_quantile,
)

HERE = Path(__file__).parent
KIT_INPUT = "/tmp/opencode/kit-gta6/input"


def tier_of(f: dict) -> str:
    t = f["tier"].lower()
    if "early-access" in t:
        return "contaminated"
    if "weak" in t:
        return "weak"
    return "comparable"


def fix_features(feats: list[dict]) -> list[dict]:
    for f in feats:
        f["tier"] = tier_of(f)
        f.setdefault("unlock", "")
        f.setdefault("platforms", "")
        p = f["platforms"].lower()
        f["console_only"] = ("pc" not in p) and ("pc" not in f["unlock"].lower() or "consoles" in f["unlock"].lower())
        f["pc"] = "pc" in p
    return feats


def predict(pool: list[dict], y1: float, p: dict, target: dict | None = None) -> dict[int, list[float]]:
    out = {}
    for h in range(2, 25):
        pairs = []
        for f in pool:
            r = f["ratios"].get(h)
            if not r or r <= 0:
                continue
            w = 1.0
            if p["tau_scale"] is not None and y1 > 0:
                d = math.log(y1) - math.log(f["y1"])
                w *= math.exp(-0.5 * (d / p["tau_scale"]) ** 2)
            w *= p["tier_w"].get(f["tier"], 1.0)
            if target is not None and target.get("console_only") and not f["console_only"]:
                w *= p["console_w"]
            pairs.append((math.log(r), max(w, 1e-9)))
        if not pairs:
            out[h] = [y1] * 5
            continue
        med = weighted_quantile(pairs, 0.5)
        var = sum(w * (v - med) ** 2 for v, w in pairs) / sum(w for _, w in pairs)
        sd = math.sqrt(var) * p["sd_scale"]
        sd = math.sqrt(sd * sd + p["sd_floor"] ** 2) * math.sqrt(1 + p["sd_grow"] * (h - 2))
        qs = []
        for tau, z in zip(LEVELS, Z):
            emp = weighted_quantile(pairs, tau)
            qs.append(y1 * math.exp((1 - p["blend"]) * emp + p["blend"] * (med + z * sd)))
        out[h] = clean(qs)
    return out


def cv(feats: list[dict], p: dict, min_y1: float = 0.0) -> tuple[float, list]:
    tot, n = 0.0, 0
    rows = []
    for f in feats:
        if f["y1"] < min_y1:
            continue
        pool = [g for g in feats if g is not f]
        pred = predict(pool, f["y1"], p, f)
        loss, k = score_case(pred, answers_from(f))
        rows.append((f["name"], loss, k))
        tot += loss * k
        n += k
    return tot / n, rows


def prac(feats_pool: list[dict], p: dict) -> tuple[float, dict]:
    """Practice cases: pool is the kit's comparables only, target features built from the case."""
    ans = practice_answers()
    total, n, per = 0.0, 0, {}
    for cid, a in ans.items():
        target = {"console_only": True}
        pred = predict(feats_pool, a["y1"], p, target)
        loss, k = score_case(pred, a["answer"])
        per[cid] = loss
        total += loss * k
        n += k
    return total / n, per


BASE = dict(tau_scale=None, tier_w={"contaminated": 1.0, "weak": 1.0, "comparable": 1.0},
            console_w=1.0, blend=0.5, sd_scale=1.0, sd_floor=0.0, sd_grow=0.0)


def main() -> None:
    feats = fix_features([f for f in (features(l) for l in load_launches()) if f])
    # the pool a practice case is graded against excludes the four practice launches
    practice_pool = [f for f in feats if f["tier"] != "practice"]
    print(f"pool={len(feats)} targets, practice pool={len(practice_pool)}")

    variants = {
        "S1 pooled empirical": dict(BASE, blend=0.0),
        "S2 pooled normal": dict(BASE, blend=1.0),
        "S3 blend .5": dict(BASE, blend=0.5),
        "S4 scale .6 + tiers": dict(BASE, tau_scale=0.6, blend=0.5,
                                    tier_w={"contaminated": 0.4, "weak": 0.3, "comparable": 1.0}),
        "S5 scale .6 + tiers + console": dict(BASE, tau_scale=0.6, blend=0.5, console_w=0.5,
                                              tier_w={"contaminated": 0.4, "weak": 0.3, "comparable": 1.0}),
    }
    print(f"\n{'variant':34s} {'cv42':>7s} {'cv_big':>7s} {'prac4':>7s}")
    for name, p in variants.items():
        a, _ = cv(feats, p)
        b, _ = cv(feats, p, min_y1=100_000)
        c, _ = prac(practice_pool, p)
        print(f"{name:34s} {a:7.4f} {b:7.4f} {c:7.4f}")

    print("\ngrid over the knobs (cv_big / prac4):")
    best = []
    for tau_scale, blend, sd_scale, sd_floor, sd_grow in itertools.product(
        [None, 0.4, 0.7, 1.2], [0.0, 0.5, 1.0], [0.8, 1.0, 1.25], [0.0, 0.06, 0.12], [0.0, 0.004]
    ):
        p = dict(BASE, tau_scale=tau_scale, blend=blend, sd_scale=sd_scale, sd_floor=sd_floor,
                 sd_grow=sd_grow, console_w=0.5,
                 tier_w={"contaminated": 0.4, "weak": 0.3, "comparable": 1.0})
        a, _ = cv(feats, p, min_y1=100_000)
        c, _ = prac(practice_pool, p)
        best.append((a, c, tau_scale, blend, sd_scale, sd_floor, sd_grow))
    best.sort()
    print(f"{'cv_big':>7s} {'prac4':>7s}  tau_scale blend sd_scale sd_floor sd_grow")
    for r in best[:12]:
        print(f"{r[0]:7.4f} {r[1]:7.4f}  {str(r[2]):>9s} {r[3]:5.1f} {r[4]:8.2f} {r[5]:8.2f} {r[6]:7.3f}")

    # per-hour detail for the chosen variant
    p = dict(BASE, tau_scale=0.6, blend=0.5, console_w=0.5, sd_scale=1.0, sd_floor=0.06,
             tier_w={"contaminated": 0.4, "weak": 0.3, "comparable": 1.0})
    print("\nper-launch CV (variant S5-ish), worst first:")
    _, rows = cv(feats, p)
    for name, loss, k in sorted(rows, key=lambda r: -r[1])[:10]:
        print(f"  {name[:36]:36s} {loss:.3f}")
    print("  ... best:")
    for name, loss, k in sorted(rows, key=lambda r: r[1])[:5]:
        print(f"  {name[:36]:36s} {loss:.3f}")


if __name__ == "__main__":
    main()
