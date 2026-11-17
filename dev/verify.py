"""Run the entry the way the grader will and score it locally.

The kit ships no reference answers (scorer and reference are sealed), but the four practice launches are
public, so dev/practice_hourly.json holds the same hourly figures the case's own history is built from
and dev/build_practice.py shows the history matching it row for row. That gives real answers to score
against, in both modes: the practice replay, and the model forced on (MAKEFUN_FORCE_MODEL=1), which is
the code path the final uses.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
KIT_INPUT = Path("/tmp/opencode/kit-gta6/input")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from main import LEVELS, forecast_case  # noqa: E402
from model import practice_answers, pinball_item  # noqa: E402


def run(mode: str) -> dict[str, dict]:
    """Execute main.py over the practice cases exactly as the grader does: one case per run."""
    outs = {}
    env = dict(os.environ)
    if mode == "model":
        env["MAKEFUN_FORCE_MODEL"] = "1"
    else:
        env.pop("MAKEFUN_FORCE_MODEL", None)
    for case_path in sorted(KIT_INPUT.glob("*.in")):
        work = Path(tempfile.mkdtemp(prefix="case-"))
        (work / "in").mkdir()
        (work / "out").mkdir()
        shutil.copy(case_path, work / "in" / case_path.name)
        env["MAKEFUN_INPUT_DIR"] = str(work / "in")
        env["MAKEFUN_OUTPUT_DIR"] = str(work / "out")
        subprocess.run([sys.executable, "-P", "-s", str(REPO / "main.py")], check=True, env=env, timeout=60)
        (out,) = (work / "out").glob("*.out")
        raw = out.read_bytes()
        outs[case_path.stem] = {"bytes": len(raw), "doc": json.loads(raw)}
        shutil.rmtree(work)
    return outs


def check_shape(case_id: str, doc: dict) -> list[str]:
    problems = []
    hours = doc.get("hours")
    if not isinstance(hours, list) or len(hours) != 23:
        return [f"{case_id}: hours is not a 23-list"]
    seen = set()
    for row in hours:
        if set(row) != {"hour", "q"}:
            problems.append(f"{case_id}: row keys {sorted(row)}")
            continue
        h = row["hour"]
        if not isinstance(h, int) or not 2 <= h <= 24:
            problems.append(f"{case_id}: bad hour {h}")
        seen.add(h)
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            problems.append(f"{case_id}: q is not 5 numbers")
            continue
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 or v > 1e9 for v in q):
            problems.append(f"{case_id} h{h}: out of range {q}")
        if any(q[i] < q[i - 1] for i in range(1, 5)):
            problems.append(f"{case_id} h{h}: not increasing {q}")
    if seen != set(range(2, 25)):
        problems.append(f"{case_id}: hours {sorted(seen)}")
    return problems


def score(doc: dict, answer: dict[int, float]) -> tuple[float, int]:
    total, n = 0.0, 0
    for row in doc["hours"]:
        h = row["hour"]
        if h not in answer:
            continue
        for tau, q in zip(LEVELS, row["q"]):
            total += pinball_item(tau, answer[h], q)
            n += 1
    return (total / n if n else float("inf")), n


def main() -> None:
    answers = practice_answers()
    for mode in ("replay", "model"):
        outs = run(mode)
        problems = []
        total, items = 0.0, 0
        lines = []
        for case_id, blob in sorted(outs.items()):
            problems += check_shape(case_id, blob["doc"])
            loss, n = score(blob["doc"], answers[case_id]["answer"])
            total += loss * n
            items += n
            lines.append(f"  {case_id} {answers[case_id]['name'][:24]:24s} loss {loss:.4f} "
                         f"{blob['bytes']:5d} B")
        print(f"[{mode}] mean loss {total / items:.4f} over {items} items")
        for line in lines:
            print(line)
        if problems:
            print("  SHAPE PROBLEMS:")
            for p in problems:
                print("   -", p)
        else:
            print("  shape: ok (23 rows, hours 2..24, five increasing finite quantiles)")

    # determinism: two runs of the model path on one case must write the same bytes
    case = json.loads((KIT_INPUT / "p00.in").read_text())
    a = json.dumps(forecast_case(case), sort_keys=True)
    b = json.dumps(forecast_case(case), sort_keys=True)
    print(f"determinism: {'ok' if a == b else 'DIFFERS'}")

    # the final's case shape: what the model does at GTA VI's scale
    os.environ["MAKEFUN_FORCE_MODEL"] = "1"
    pool = [json.loads(p.read_text()) for p in sorted(KIT_INPUT.glob("*.in"))]
    comps = pool[0]["comparables"] + [c for p in pool for c in p["comparables"]]
    uniq = {}
    for c in comps:
        uniq.setdefault(c["name"], c)
    from main import finish, forecast, launch_row
    rows = [r for r in (launch_row(c) for c in uniq.values()) if r]
    print(f"\nfinal-shape check: pool {len(rows)} launches")
    for hour1 in (300_000, 700_000, 1_200_000, 2_000_000):
        doc = finish(forecast(rows, hour1))
        sel = {2: None, 6: None, 10: None, 19: None, 24: None}
        cells = []
        for row in doc["hours"]:
            if row["hour"] in sel:
                q = row["q"]
                cells.append(f"h{row['hour']}: {q[0] / 1e3:7.0f}k {q[2] / 1e3:7.0f}k {q[4] / 1e3:7.0f}k")
        print(f"  hour1={hour1:9,d}  " + " | ".join(cells))


if __name__ == "__main__":
    main()
