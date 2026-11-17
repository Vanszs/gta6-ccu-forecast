"""Third pass: the candidate final model, with the unlock regime read off each comparable's own text.

GTA VI unlocks at midnight local, so at hour 1 (00:00-01:00Z) the United States is not playing yet;
it joins at hour 6 (05:00Z east) and hour 9 (08:00Z west). Launches whose PC build unlocked globally
at 00:00Z had the US live from hour 1, and their curves fall where the rolling ones step up at hour 6.
The two regimes are weighted apart, judged only by what the case's own `unlock` text says.
"""

import itertools
import math
from pathlib import Path

from model import (
    LEVELS,
    Z,
    answers_from,
    clean,
    features,
    load_launches,
    practice_answers,
    score_case,
    weighted_quantile,
)

REGIME_GLOBAL = ("simultaneous", "global unlock", "single global", "worldwide")


def regime_of(unlock: str) -> str:
    u = (unlock or "").lower()
    if any(k in u for k in REGIME_GLOBAL):
        return "global"
    if "midnight local" in u or "rolling" in u:
        return "rolling"
    return "unknown"


def annotate(f: dict) -> dict:
    t = f["tier"].lower()
    if "early-access" in t:
        f["tier2"] = "contaminated"
    elif "weak" in t:
        f["tier2"] = "weak"
    elif t == "practice":
        f["tier2"] = "comparable"
    else:
        f["tier2"] = "comparable"
    f["regime"] = regime_of(f["unlock"])
    f["console_only"] = "pc" not in f["platforms"].lower()
    return f


def smooth(xs: list[float], k: int) -> list[float]:
    if k <= 1:
        return xs
    out, half = [], k // 2
    for i in range(len(xs)):
        lo, hi = max(0, i - half), min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def predict(pool: list[dict], y1: float, p: dict, target_regime: str = "rolling",
            target_console_only: bool = True) -> dict[int, list[float]]:
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
            w *= p["regime_w"].get(target_regime, {}).get(f["regime"], 1.0)
            if target_console_only and not f["console_only"]:
                w *= p["console_w"]
            pairs.append((math.log(r), max(w, 1e-9)))
        med = weighted_quantile(pairs, 0.5)
        var = sum(w * (v - med) ** 2 for v, w in pairs) / sum(w for _, w in pairs)
        meds.append(med)
        sds.append(math.log(max(math.sqrt(var), 1e-6)))
        emps.append([weighted_quantile(pairs, tau) for tau in LEVELS])
    meds = smooth(meds, p["smooth"])
    sds = smooth(sds, p["smooth"])
    out = {}
    for i, h in enumerate(range(2, 25)):
        sd = math.exp(sds[i]) * p["sd_scale"]
        sd = math.sqrt(sd * sd + p["sd_floor"] ** 2) * math.sqrt(1 + p["sd_grow"] * (h - 2))
        qs = [
            y1 * math.exp((1 - p["blend"]) * emps[i][j] + p["blend"] * (meds[i] + z * sd))
            for j, z in enumerate(Z)
        ]
        out[h] = clean(qs)
    return out


def cv(feats, p, min_y1=0.0, only_regime=None):
    tot, n, rows = 0.0, 0, []
    for f in feats:
        if f["y1"] < min_y1:
            continue
        if only_regime and f["regime"] != only_regime:
            continue
        pool = [g for g in feats if g is not f]
        pred = predict(pool, f["y1"], p, "rolling")
        loss, k = score_case(pred, answers_from(f))
        rows.append((f["name"], loss, k))
        tot += loss * k
        n += k
    return tot / n, rows


def prac(pool, p):
    tot, n, per = 0.0, 0, {}
    for cid, a in practice_answers().items():
        pred = predict(pool, a["y1"], p, "rolling")
        loss, k = score_case(pred, a["answer"])
        per[cid] = loss
        tot += loss * k
        n += k
    return tot / n, per


BASE = dict(
    tau_scale=0.7,
    tier_w={"contaminated": 0.35, "weak": 0.25, "comparable": 1.0},
    regime_w={"rolling": {"rolling": 1.0, "unknown": 0.6, "global": 0.35},
              "global": {"rolling": 0.35, "unknown": 0.6, "global": 1.0},
              "unknown": {"rolling": 0.7, "unknown": 1.0, "global": 0.5}}[  # type: ignore[index]
    "rolling"] if False else {"rolling": {"rolling": 1.0, "unknown": 0.6, "global": 0.35}},
    console_w=0.5,
    blend=0.5,
    sd_scale=1.0,
    sd_floor=0.05,
    sd_grow=0.0,
    smooth=3,
)


def main() -> None:
    feats = [annotate(f) for f in (features(l) for l in load_launches()) if f]
    practice_pool = [f for f in feats if f["tier"] != "practice"]
    from collections import Counter
    print("regimes:", Counter(f["regime"] for f in feats))
    for f in feats:
        if f["regime"] != "rolling":
            print(f"  {f['regime']:8s} {f['y1']:8.0f} {f['name'][:44]}")

    print(f"\n{'variant':52s} {'cv_big':>7s} {'cv_roll':>7s} {'prac4':>7s}")
    trials = {
        "base (regime/scale/tier/console)": dict(BASE),
        "no regime": dict(BASE, regime_w={"rolling": {"rolling": 1.0, "unknown": 1.0, "global": 1.0}}),
        "no regime, no console": dict(BASE, console_w=1.0,
                                      regime_w={"rolling": {"rolling": 1.0, "unknown": 1.0, "global": 1.0}}),
        "no scale": dict(BASE, tau_scale=None),
        "regime only": dict(BASE, tau_scale=None, console_w=1.0),
        "smooth5": dict(BASE, smooth=5),
        "sd 1.15": dict(BASE, sd_scale=1.15),
    }
    for name, p in trials.items():
        b, _ = cv(feats, p, min_y1=100_000)
        r, _ = cv(feats, p, only_regime="rolling")
        c, _ = prac(practice_pool, p)
        print(f"{name:52s} {b:7.4f} {r:7.4f} {c:7.4f}")

    print("\ngrid (ranked by cv_roll, then cv_big):")
    best = []
    for tau, cw, blend, sd_scale, sd_floor, smooth_k, g_w in itertools.product(
        [0.5, 0.7, 1.0], [0.4, 0.6, 1.0], [0.35, 0.5, 0.65], [0.95, 1.1, 1.25], [0.0, 0.06], [1, 3, 5], [0.25, 0.4]
    ):
        p = dict(BASE, tau_scale=tau, console_w=cw, blend=blend, sd_scale=sd_scale, sd_floor=sd_floor,
                 smooth=smooth_k,
                 regime_w={"rolling": {"rolling": 1.0, "unknown": 0.6, "global": g_w}})
        r, _ = cv(feats, p, only_regime="rolling")
        b, _ = cv(feats, p, min_y1=100_000)
        c, _ = prac(practice_pool, p)
        best.append((r, b, c, tau, cw, blend, sd_scale, sd_floor, smooth_k, g_w))
    best.sort()
    print(f"{'cv_roll':>7s} {'cv_big':>7s} {'prac4':>7s}  tau  cw blend sd  floor sm g_w")
    for r in best[:15]:
        print(f"{r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f}  {r[3]:.1f} {r[4]:.1f} {r[5]:.2f} {r[6]:.2f} {r[7]:.2f} {r[8]:d} {r[9]:.2f}")

    p = dict(BASE, **{})
    print("\nper-launch cv (base), best and worst:")
    _, rows = cv(feats, p, only_regime="rolling")
    for name, loss, k in sorted(rows, key=lambda r: r[1]):
        print(f"  {name[:40]:40s} {loss:.3f}")


if __name__ == "__main__":
    main()
