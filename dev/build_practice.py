"""Turn the raw SullyGnome payloads into hourly series and validate them against the kit's own case files.

The kit's cases hold the same source (NOTICE.txt: SullyGnome hourly averages), so a practice case's
`history` is a free correctness check on this extraction: same instants, same numbers.
"""

import json
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "practice_curves.json"
OUT = HERE / "practice_hourly.json"
KIT_INPUT = "/tmp/opencode/kit-gta6/input"

RELEASE = {
    "Marvel's Wolverine": "2026-09-15",
    "Red Dead Redemption 2": "2018-10-26",
    "God of War Ragnarök": "2022-11-09",
    "Marvel's Spider-Man 2": "2023-10-20",
}


def iso(label: str) -> str:
    """SullyGnome's 'YYYY-MM-DD HH:00' label -> ISO UTC instant."""
    return datetime.strptime(label, "%Y-%m-%d %H:%M").strftime("%Y-%m-%dT%H:%M:%SZ")


def build() -> dict:
    raw = json.loads(RAW.read_text())
    out = {}
    for name, blob in raw.items():
        series = {}
        for req in blob["requests"]:
            data = req["doc"]["data"]
            want = "Average viewers" if req["kind"] == "GameViewers" else "Channels"
            ds = next(d for d in data["datasets"] if d["label"] == want)
            for label, value in zip(data["labels"], ds["data"]):
                if value is None:
                    continue
                key = iso(label)
                row = series.setdefault(key, {})
                row["viewers" if req["kind"] == "GameViewers" else "channels"] = value
        rel = RELEASE[name]
        day0 = datetime.strptime(rel, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        lo = (day0 - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hi = (day0 + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hourly = [
            {"from": k, "viewers": series[k].get("viewers"), "channels": series[k].get("channels")}
            for k in sorted(series)
            if lo <= k < hi and series[k].get("viewers") is not None
        ]
        out[name] = {"day_start": day0.strftime("%Y-%m-%dT00:00:00Z"), "hourly": hourly}
        print(f"{name:28s} release {rel}  points {len(hourly)}  "
              f"span {hourly[0]['from'] if hourly else '-'} .. {hourly[-1]['from'] if hourly else '-'}")
    return out


def validate(built: dict) -> None:
    """Compare each practice case's `history` with the series rebuilt from the raw payloads."""
    for path in sorted(glob.glob(f"{KIT_INPUT}/*.in")):
        case = json.load(open(path))
        day0 = case["day_start"]
        # find which practice launch this case is, by hour-1 viewers and day_start
        cands = [n for n, b in built.items() if b["day_start"] == day0]
        if not cands:
            print(f"{case['id']}: no rebuilt series for day_start {day0}")
            continue
        name = cands[0]
        got = {p["from"]: p["viewers"] for p in built[name]["hourly"]}
        hist = case["history_hourly"]
        same = diff = missing = 0
        worst = []
        for p in hist:
            if p["from"] in got:
                if abs(got[p["from"]] - p["viewers"]) <= 1:
                    same += 1
                else:
                    diff += 1
                    worst.append((p["from"], p["viewers"], got[p["from"]]))
            else:
                missing += 1
        print(f"{case['id']} = {name:26s} history rows {len(hist):4d}  match {same:4d}  "
              f"differ {diff:3d}  absent {missing:3d}  hour1={case['known'][0]['viewers']}")
        for row in worst[:4]:
            print(f"    {row}")


if __name__ == "__main__":
    built = build()
    OUT.write_text(json.dumps(built, indent=0), encoding="utf-8")
    print(f"\nwrote {OUT} ({OUT.stat().st_size} bytes)\n")
    validate(built)
