"""Second pass: use the shape of the early hours to tell unlock regimes apart, and smooth the median.

Why it matters: GTA VI is console-only and unlocks at midnight local, so at hour 1 (00:00-01:00Z) the
United States is not playing yet: it joins at hour 6 (05:00Z, east) and hour 9 (08:00Z, west). A launch
whose PC version unlocked globally at 00:00Z, by contrast, had the US live from hour 1. The two regimes
have different curves -- the global one peaks around hours 2-3, the rolling one climbs into the EU evening.

`us_onset` reads the regime off the curve itself: the step up between hours 3-5 and hours 6-8.
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


def annotate(f: dict) -> dict:
    t = f["tier"].lower()
    f["tier2"] = "contaminated" if "early-access" in t else ("weak" if "weak" in t else "comparable")
    p = f["platforms"].lower()
    f["console_only"] = "pc" not in p
    r = f["ratios"]
    early = [math.log(r[h]) for h in (3, 4, 5) if h in r]
    late = [math.log(r[h]) for h in (6, 7, 8) if h in r]
    step = (sum(late) / len(late) - sum(early) / len(early)) if early and late else 0.0
    f["us_onset"] = math.exp(step)
    return f


def smooth(xs: list[float], k: int) -> list[float]:
    """Centred moving average, edge-shrunk."""
    if k <= 1:
        return xs
    out = []
    half = k // 2
    for i in range(len(xs)):
        lo, hi = max(0, i - half), min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def predict(pool: list[dict], y1: float, p: dict, target_regime: str | None = None) -> dict[int, list[float]]:
    meds, sds, emps = [], [], []
    for h in range(2, 25):
        pairs = []
        for f in pool:
            r = f["ratios"].get(h)
            if not r or r <= 0:
                continue
            w = 1.0
            if p["tau_scale"]:
                w *= math.exp(-0.5 * ((math.log(y1) - math.log(f["y1"])) / p["tau_scale"]) ** 2)
            w *= p["tier_w"].get(f["tier2"], 1.0)
            if target_regime and p["regime_w"] < 1.0:
                same = (f["us_onset"] >= p["onset_split"]) == (target_regime == "rolling")
                if not same:
                    w *= p["regime_w"]
            if p["console_w"] < 1.0 and p.get("target_console_only") and not f["console_only"]:
                w *= p["console_w"]
            pairs.append((math.log(r), max(w, 1e-9)))
        med = weighted_quantile(pairs, 0.5)
        var = sum(w * (v - med) ** 2 for v, w in pairs) / sum(w for _, w in pairs)
        meds.append(med)
        sds.append(math.sqrt(var))
        emps.append([weighted_quantile(pairs, tau) for tau in LEVELS])
    meds = smooth(meds, p["smooth"])
    lsd = smooth([math.log(max(s, 1e-9)) for s in sds], p["smooth"])
    out = {}
    for i, h in enumerate(range(2, 25)):
        sd = math.exp(lsd[i]) * p["sd_scale"]
        sd = math.sqrt(sd * sd + p["sd_floor"] ** 2) * math.sqrt(1 + p["sd_grow"] * (h - 2))
        qs = []
        for j, (tau, z) in enumerate(zip(LEVELS, Z)):
            qs.append(y1 * math.exp((1 - p["blend"]) * emps[i][j] + p["blend"] * (meds[i] + z * sd)))
        out[h] = clean(qs)
    return out


def cv(feats, p, min_y1=0.0, regime=None):
    tot, n, rows = 0.0, 0, []
    for f in feats:
        if f["y1"] < min_y1:
            continue
        if regime == "rolling" and f["us_onset"] < p["onset_split"]:
            continue
        if regime == "global" and f["us_onset"] >= p["onset_split"]:
            continue
        pool = [g for g in feats if g is not f]
        pred = predict(pool, f["y1"], p, "rolling")
        loss, k = score_case(pred, answers_from(f))
        rows.append((f["name"], loss, k))
        tot += loss * k
        n += k
    return tot / n, rows


def prac(pool, p):
    ans = practice_answers()
    tot, n, per = 0.0, 0, {}
    for cid, a in ans.items():
        pred = predict(pool, a["y1"], dict(p, target_console_only=True), "rolling")
        loss, k = score_case(pred, a["answer"])
        per[cid] = loss
        tot += loss * k
        n += k
    return tot / n, per


BASE = dict(tau_scale=0.6, tier_w={"contaminated": 0.4, "weak": 0.3, "comparable": 1.0}, regime_w=1.0,
            onset_split=1.05, console_w=0.5, target_console_only=False, blend=0.5, sd_scale=1.0,
            sd_floor=0.06, sd_grow=0.0, smooth=1)


def main() -> None:
    feats = [annotate(f) for f in (features(l) for l in load_launches()) if f]
    practice_pool = [f for f in feats if f["tier2"] != "practice"]
    print(f"{'launch':34s} {'y1':>9s} {'surge':>7s} {'onset':>5s} {'regime':>8s}")
    for f in sorted(feats, key=lambda f: f["y1"]):
        print(f"{f['name'][:34]:34s} {f['y1']:9.0f} {(f['surge'] or 0):7.1f} {f['us_onset']:5.2f} "
              f"{'rolling' if f['us_onset'] < 1.05 else 'global':>8s}")

    print(f"\n{'variant':46s} {'cv_big':>7s} {'cv_roll':>7s} {'prac4':>7s}")
    trials = {
        "base": dict(BASE),
        "no regime": dict(BASE, regime_w=1.0),
        "regime .3": dict(BASE, regime_w=0.3),
        "regime .3 + smooth3": dict(BASE, regime_w=0.3, smooth=3),
        "regime .3 + smooth3 + no console": dict(BASE, regime_w=0.3, smooth=3, console_w=1.0),
        "regime .3 s3 sd.8": dict(BASE, regime_w=0.3, smooth=3, sd_scale=0.85),
    }
    for name, p in trials.items():
        b, _ = cv(feats, p, min_y1=100_000)
        r, _ = cv(feats, p, regime="rolling")
        c, _ = prac(practice_pool, p)
        print(f"{name:46s} {b:7.4f} {r:7.4f} {c:7.4f}")

    print("\ngrid:")
    best = []
    for tau_scale, regime_w, smooth_k, sd_scale, sd_floor, blend in itertools.product(
        [0.4, 0.6, 1.0], [1.0, 0.5, 0.25], [1, 3, 5], [0.9, 1.0, 1.15], [0.0, 0.06, 0.12], [0.3, 0.5, 0.7]
    ):
        p = dict(BASE, tau_scale=tau_scale, regime_w=regime_w, smooth=smooth_k, sd_scale=sd_scale,
                 sd_floor=sd_floor, blend=blend)
        b, _ = cv(feats, p, min_y1=100_000)
        r, _ = cv(feats, p, regime="rolling")
        c, _ = prac(practice_pool, p)
        best.append((r, b, c, tau_scale, regime_w, smooth_k, sd_scale, sd_floor, blend))
    best.sort()
    print(f"{'cv_roll':>7s} {'cv_big':>7s} {'prac4':>7s}  tau  reg  smooth sd  floor blend")
    for r in best[:15]:
        print(f"{r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f}  {r[3]:.1f} {r[4]:.2f} {r[5]:5d} {r[6]:.2f} {r[7]:.2f} {r[8]:.2f}")


if __name__ == "__main__":
    main()
