"""Fetch the four practice launches' hourly curves from SullyGnome (the source the kit names).

The final case carries all 42 launches, including these four; the practice cases deliberately leave
them out. Reading them back from the same public source the kit used gives the model the exact
comparable set the final will show, and gives us the practice answers for a local score.

Low volume and attributed: one request per game-month, a second apart, as NOTICE.txt asks.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

GAMES = [
    ("Marvel's Wolverine", 117671, [2026], [8, 9, 10]),
    ("Red Dead Redemption 2", 28264, [2018], [10]),
    ("God of War Ragnarök", 65645, [2022], [10, 11]),
    ("Marvel's Spider-Man 2", 97296, [2023], [10]),
]

OUT = Path(__file__).with_name("practice_curves.json")


def fetch(kind: str, sid: int, name: str, year: int, month: int) -> list:
    q = urllib.parse.quote(name)
    url = (
        f"https://sullygnome.com/api/charts/linecharts/getconfig/{kind}/30/0/"
        f"{sid}/{q}/%20/%20/{year}/{month}/%20/0/"
    )
    req = urllib.request.Request(
        url, headers={"user-agent": "Mozilla/5.0 (compatible; research; one-off)", "accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    doc = json.loads(body)
    # The payload wraps the series; find the list of {time/date, value} rows.
    return doc


def main() -> None:
    out = {}
    for name, sid, years, months in GAMES:
        rows = []
        for y in years:
            for m in months:
                for kind in ("GameViewers", "GameChannels"):
                    try:
                        doc = fetch(kind, sid, name, y, m)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! {name} {kind} {y}-{m}: {e}")
                        continue
                    print(f"  {name} {kind} {y}-{m}: {type(doc).__name__} "
                          f"{list(doc)[:6] if isinstance(doc, dict) else len(doc)}")
                    rows.append({"kind": kind, "year": y, "month": m, "doc": doc})
                    time.sleep(1.2)
        out[name] = {"sullygnome_id": sid, "requests": rows}
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
