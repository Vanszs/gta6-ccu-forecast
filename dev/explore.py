"""First look at the case data: comparable launch curves, release-day shape, dispersion.

Reads the kit's practice cases (outside the repo) and prints:
  - coverage of each comparable over the release day's 24 UTC hours
  - the hour-1 print and the hour-by-hour ratio to it
  - peak hour, peak/hour-1, and the cross-launch spread per hour
"""

import glob
import json
import math
import statistics as st
from datetime import datetime, timedelta, timezone

KIT_INPUT = "/tmp/opencode/kit-gta6/input"


def hour_ts(day_start: str, h: int) -> str:
    """The ISO instant that starts hour h (1..24) of the release day."""
    t = datetime.fromisoformat(day_start.replace("Z", "+00:00")) + timedelta(hours=h - 1)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def curve(comp: dict) -> dict[int, float]:
    """Hour index (1..24, and negatives for the pre-days) -> viewers."""
    day0 = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    out = {}
    for p in comp["hourly"]:
        at = datetime.fromisoformat(p["from"].replace("Z", "+00:00"))
        idx = round((at - day0).total_seconds() / 3600) + 1
        out[idx] = p["viewers"]
    return out


def main() -> None:
    cases = sorted(glob.glob(f"{KIT_INPUT}/*.in"))
    comps = {}
    for path in cases:
        case = json.load(open(path))
        for c in case["comparables"]:
            comps.setdefault(c["name"], c)

    print(f"{len(comps)} distinct comparables across {len(cases)} practice cases\n")

    rows = []
    for name, c in comps.items():
        cv = curve(c)
        missing = [h for h in range(1, 25) if h not in cv]
        if 1 not in cv:
            print(f"SKIP {name}: no hour 1")
            continue
        y1 = cv[1]
        if y1 <= 0:
            print(f"SKIP {name}: hour1={y1}")
            continue
        ratios = [cv[h] / y1 if h in cv else None for h in range(2, 25)]
        peak_h = max(range(2, 25), key=lambda h: cv.get(h, 0))
        peak = cv.get(peak_h, 0)
        rows.append(
            dict(
                name=name,
                y1=y1,
                n=len(cv),
                missing=missing,
                peak_h=peak_h,
                peak_ratio=peak / y1,
                ratios=ratios,
                tier=c.get("tier", ""),
                platforms=c.get("platforms", ""),
                unlock=(c.get("unlock") or "")[:60],
            )
        )

    print(f"{'launch':42s} {'y1':>9s} {'pts':>4s} {'miss':>4s} {'pkH':>3s} {'pk/y1':>6s} tier")
    for r in sorted(rows, key=lambda r: -r["y1"]):
        print(
            f"{r['name'][:42]:42s} {r['y1']:9.0f} {r['n']:4d} {len(r['missing']):4d} "
            f"{r['peak_h']:3d} {r['peak_ratio']:6.2f} {r['tier'][:24]}"
        )

    print("\nHour-by-hour log ratio to hour 1 (all launches with the hour):")
    print(f"{'h':>3s} {'n':>3s} {'p10':>7s} {'p25':>7s} {'med':>7s} {'p75':>7s} {'p90':>7s} {'mean':>7s} {'sd':>6s}")
    for h in range(2, 25):
        vals = [math.log(r["ratios"][h - 2]) for r in rows if r["ratios"][h - 2] is not None]
        if not vals:
            continue
        sv = sorted(vals)

        def q(p):
            i = min(len(sv) - 1, max(0, int(round(p * (len(sv) - 1)))))
            return sv[i]

        print(
            f"{h:3d} {len(vals):3d} {math.exp(q(0.10)):7.3f} {math.exp(q(0.25)):7.3f} "
            f"{math.exp(q(0.50)):7.3f} {math.exp(q(0.75)):7.3f} {math.exp(q(0.90)):7.3f} "
            f"{math.exp(st.fmean(vals)):7.3f} {st.pstdev(vals):6.3f}"
        )

    print("\nPer-launch ratio at h=6,9,12,15,18,21,24 (log-ratio table, x100):")
    hs = [2, 4, 6, 8, 10, 12, 15, 18, 21, 24]
    print(f"{'launch':34s}" + "".join(f"{h:>6d}" for h in hs))
    for r in sorted(rows, key=lambda r: -r["y1"]):
        cells = ""
        for h in hs:
            v = r["ratios"][h - 2]
            cells += f"{math.log(v):6.2f}" if v else "     ."
        print(f"{r['name'][:34]:34s}{cells}")


if __name__ == "__main__":
    main()
