"""Launch table, pinball scoring and cross-validated candidate forecasters.

Loss is separable over (hour, level): the mean pinball loss on the log scale. So every (h, tau) cell is
its own estimation problem, and the only thing that couples the hours is that they share the same pool
of past launches.

Two evaluations:
  cv42       leave-one-launch-out over all 42 launches (each pool = the other 41)
  practice4  each practice case scored against the pool the grader actually gives it (the kit's 38)
"""

import glob
import json
import math
import statistics as st
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
KIT_INPUT = "/tmp/opencode/kit-gta6/input"
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


# ---------------------------------------------------------------- data


def curve_from_rows(day_start: str, rows: list[dict], key_at: str, key_val: str) -> dict[int, float]:
    day0 = datetime.fromisoformat(day_start.replace("Z", "+00:00"))
    out = {}
    for p in rows:
        at = datetime.fromisoformat(p[key_at].replace("Z", "+00:00"))
        idx = round((at - day0).total_seconds() / 3600) + 1
        v = p[key_val]
        if v is not None:
            out[idx] = float(v)
    return out


def launch_from_comparable(c: dict) -> dict:
    cv = curve_from_rows(c["day_start"], c["hourly"], "from", "viewers")
    return {
        "name": c["name"],
        "day_start": c["day_start"],
        "tier": (c.get("tier") or "").strip(),
        "platforms": c.get("platforms") or "",
        "unlock": c.get("unlock") or "",
        "curve": cv,
        "source": "kit",
    }


def load_launches() -> list[dict]:
    seen = {}
    for path in sorted(glob.glob(f"{KIT_INPUT}/*.in")):
        for c in json.load(open(path))["comparables"]:
            seen.setdefault(c["name"], launch_from_comparable(c))
    for name, blob in json.loads((HERE / "practice_hourly.json").read_text()).items():
        cv = curve_from_rows(blob["day_start"], blob["hourly"], "from", "viewers")
        seen.setdefault(
            name,
            {
                "name": name,
                "day_start": blob["day_start"],
                "tier": "practice",
                "platforms": "console",
                "unlock": "midnight local",
                "curve": cv,
                "source": "sullygnome",
            },
        )
    return list(seen.values())


def is_launch(l: dict) -> bool:
    """The GTA VI pre-release category row is not a launch: it has no release day."""
    return "curve" in l and 1 in l["curve"] and l["curve"][1] > 0 and "before release" not in l["name"]


def features(l: dict) -> dict | None:
    cv = l["curve"]
    if not is_launch(l):
        return None
    y1 = cv.get(1)
    if not y1 or y1 <= 0:
        return None
    # the same clock hour, a day earlier: how far the launch had already surged by hour 1
    prev = cv.get(-23)  # hour index 1 is 00:00Z; -23 is the same clock hour one day earlier
    ratios = {h: cv[h] / y1 for h in range(2, 25) if h in cv and cv[h] > 0}
    return {
        "name": l["name"],
        "y1": y1,
        "prev": prev,
        "surge": (y1 / prev) if prev and prev > 0 else None,
        "ratios": ratios,
        "coverage": len(ratios),
        "tier": l["tier"],
        "platforms": l["platforms"],
        "unlock": l["unlock"],
        "curve": cv,
    }


# ---------------------------------------------------------------- scoring


def pinball_item(tau: float, y: float, q: float) -> float:
    d = math.log(max(y, 1.0)) - math.log(max(q, 1e-9))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_case(pred: dict[int, list[float]], answer: dict[int, float]) -> tuple[float, int]:
    tot, n = 0.0, 0
    for h, qs in pred.items():
        if h not in answer:
            continue
        for tau, q in zip(LEVELS, qs):
            tot += pinball_item(tau, answer[h], q)
            n += 1
    return (tot / n if n else math.inf), n


# ---------------------------------------------------------------- forecasters


def weighted_quantile(pairs: list[tuple[float, float]], p: float) -> float:
    """p-quantile of values with weights (pairs = [(value, weight)])."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= p * total:
            return v
    return pairs[-1][0]


def predict(feats: list[dict], y1: float, surge: float | None, p: dict) -> dict[int, list[float]]:
    """p (params): tau_width (fraction of log-sd used for the spread), blend (0 empirical..1 normal),
    surge_tau (log-surge similarity bandwidth), tier_weight (downweight of contaminated tiers)."""
    pool = [f for f in feats if f["ratios"]]
    out = {}
    for h in range(2, 25):
        vals = [(f, f["ratios"].get(h)) for f in pool if f["ratios"].get(h)]
        if not vals:
            out[h] = [y1] * 5
            continue
        pairs = []
        for f, r in vals:
            w = 1.0
            if surge and f["surge"]:
                d = math.log(surge) - math.log(f["surge"])
                w *= math.exp(-0.5 * (d / p["surge_tau"]) ** 2)
            if "contaminated" in f["tier"]:
                w *= p["tier_weights"].get("contaminated", 1.0)
            elif f["tier"] == "weak":
                w *= p["tier_weights"].get("weak", 1.0)
            pairs.append((math.log(r), max(w, 1e-6)))
        # location: weighted median; scale: weighted sd of log ratios
        med = weighted_quantile(pairs, 0.5)
        var = sum(w * (v - med) ** 2 for v, w in pairs) / sum(w for _, w in pairs)
        sd = math.sqrt(var)
        # a floor: even a perfect analogue pool cannot pin an hour down to nothing
        sd = math.sqrt(sd * sd + p["sd_floor"] ** 2)
        sd *= p["sd_scale"]
        qs = []
        for tau, z in zip(LEVELS, Z):
            emp = weighted_quantile(pairs, tau)
            gauss = med + z * sd
            v = (1 - p["blend"]) * emp + p["blend"] * gauss
            qs.append(max(1e-6, y1 * math.exp(v)))
        out[h] = qs
    return out


def clean(xs: list[float]) -> list[float]:
    """Make non-decreasing, finite, >0, <=1e9, two decimals (as the entry must write them)."""
    xs = [min(max(x, 1e-6), 1e9) for x in xs]
    for i in range(1, len(xs)):
        xs[i] = max(xs[i], xs[i - 1])
    return [round(x, 6) for x in xs]


# ---------------------------------------------------------------- cross validation


def answers_from(f: dict) -> dict[int, float]:
    return {h: r * f["y1"] for h, r in f["ratios"].items()}


def cv_all(feats: list[dict], p: dict) -> tuple[float, list[tuple[str, float, int]]]:
    rows = []
    total_items, tot = 0, 0.0
    for f in feats:
        pool = [g for g in feats if g is not f]
        pred = {h: clean(q) for h, q in predict(pool, f["y1"], f["surge"], p).items()}
        loss, n = score_case(pred, answers_from(f))
        rows.append((f["name"], loss, n))
        tot += loss * n
        total_items += n
    return tot / total_items, rows


def practice_answers() -> dict[str, dict]:
    """The 4 practice cases: id -> {day_start, hour-1 print, the 23 answered hours}."""
    built = json.loads((HERE / "practice_hourly.json").read_text())
    out = {}
    for path in sorted(glob.glob(f"{KIT_INPUT}/*.in")):
        case = json.load(open(path))
        name = next(n for n, b in built.items() if b["day_start"] == case["day_start"])
        cv = curve_from_rows(case["day_start"], built[name]["hourly"], "from", "viewers")
        y1 = float(case["known"][0]["viewers"])
        ans = {}
        for row in case["forecast"]:
            h = row["hour"]
            pts = [cv[k] for k in (h,) if k in cv]
            if pts:
                ans[h] = pts[0]
        out[case["id"]] = {"name": name, "day_start": case["day_start"], "y1": y1, "answer": ans}
    return out


def practice_score(feats: list[dict], p: dict) -> tuple[float, dict]:
    ans = practice_answers()
    total, items, per = 0.0, 0, {}
    for cid, a in ans.items():
        # the grader's pool for a practice case is the kit's 38 comparables
        pred = {h: clean(q) for h, q in predict(feats, a["y1"], None, p).items()}
        loss, n = score_case(pred, a["answer"])
        per[cid] = (a["name"], loss, n)
        total += loss * n
        items += n
    return total / items, per


def main() -> None:
    launches = load_launches()
    feats = [f for f in (features(l) for l in launches) if f]
    print(f"{len(launches)} launch rows -> {len(feats)} usable\n")
    print(f"{'launch':34s} {'y1':>9s} {'surge':>8s} {'cov':>4s} {'r6':>6s} {'r12':>6s} {'r19':>6s} {'r24':>6s}")
    for f in sorted(feats, key=lambda f: -(f["surge"] or 0)):
        g = lambda h: math.exp(math.log(f["ratios"][h])) if h in f["ratios"] else float("nan")  # noqa: E731
        print(
            f"{f['name'][:34]:34s} {f['y1']:9.0f} {(f['surge'] or 0):8.1f} {f['coverage']:4d} "
            f"{g(6):6.2f} {g(12):6.2f} {g(19):6.2f} {g(24):6.2f}"
        )

    p = dict(surge_tau=1.0, sd_floor=0.0, sd_scale=1.0, blend=0.5,
             tier_weights={"contaminated": 0.4, "weak": 0.2})
    loss, rows = cv_all(feats, p)
    print(f"\ncv42 loss {loss:.4f}")
    for name, l, n in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"  worst {name[:34]:34s} {l:.3f}")
    pl, per = practice_score(feats, p)
    print(f"practice4 loss {pl:.4f}")
    for cid, (name, l, n) in per.items():
        print(f"  {cid} {name[:26]:26s} {l:.4f}  ({n} items)")


if __name__ == "__main__":
    main()
