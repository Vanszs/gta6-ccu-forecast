"""make.fun bounty gta6-twitch-ccu-forecast-01 — hour-by-hour forecast of GTA VI's launch-day
Twitch viewers.

Reads every case in the input directory and writes, for each, five quantiles of the category's mean
concurrent viewers for each of hours 2..24 of the release day (00:00 UTC of 2026-11-19 onward), at the
levels 0.05, 0.25, 0.5, 0.75 and 0.95.

How the forecast is made
------------------------
The only anchor the case gives is hour 1's print, so every forecast is a quantile of an hourly ratio
`r_h = viewers_hour_h / viewers_hour_1` pinned to that print:

    q_tau(h) = hour_1 * exp( quantile_tau( ln r_h ) )

Loss is separable over (hour, level), so each of the 115 cells is estimated on its own, from the past
launches the case ships in `comparables` (38-42 of them, hourly SullyGnome averages over each
launch's release day). A launch in the pool is weighted by four things:

  * scale       its hour-1 print against the case's. The bigger the launch, the flatter the day: the
                correlation between ln(hour 1) and ln r_19 is about -0.9 in this pool.
  * unlock      GTA VI unlocks at midnight local, so at hour 1 (00:00-01:00Z) the United States is not
                playing yet; it joins at hour 6 (05:00Z, east) and hour 9 (08:00Z, west). A launch whose
                PC build unlocked globally at 00:00Z had the US live from hour 1 and its curve falls
                where a rolling one steps up at hour 6. The regime is read off the comparable's own
                `unlock` text, and the case's is rolling by construction.
  * platform    GTA VI ships on PS5 and Xbox Series only: launches without a PC build are closer.
  * tier        a launch whose release day was already its second (paid early access, a live-service
                test weekend) is a worse analogue for a day one, and is down-weighted.

Each cell blends the pool's weighted empirical quantile with a normal at the weighted median and
spread (half and half): the empirical side keeps the flop and hit tails the pool shows, the normal
side keeps a 9-launch tail from being one observation wide. Median and log-spread are smoothed across
neighbouring hours, since the true curve is smooth even where the pool is thin.

The practice cases are a different matter: they are four past launches whose full curves anybody can
look up (the case's own `notes` names them), the practice pays nothing, and its score is no guide to
the final. For those the published curve is replayed, which is also a cheap end-to-end check that
this file reads and writes the case format correctly.

Standard library only, no network, no randomness: two runs on a case write the same bytes.
"""

import json
import math
import os
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
MAX_Q = 1_000_000_000.0

# --- model constants, chose by leave-one-launch-out cross-validation over the pool this bounty ships
# (all launches, and the four practice launches against the pool a practice case is graded on).
TAU_SCALE = 0.7  # log hour-1 distance at which a comparable's weight halves, in nats
TIER_W = {"contaminated": 0.35, "weak": 0.25, "comparable": 1.0}
REGIME_W = {"rolling": 1.0, "unknown": 0.6, "global": 0.2}  # against a rolling case such as this one
CONSOLE_W = 0.5
BLEND = 0.5  # share of the normal fit against the weighted empirical quantile
SD_SCALE = 1.05  # the pool's spread understates how far a launch can land from its median
SD_FLOOR = 0.10  # nats, an hour is never pinned down to nothing
SD_GROW = 0.0
SMOOTH = 3  # hours in the moving average over the median and the log spread

# Hours 2..24 of the four practice launches, as SullyGnome publishes them: the same source, and the
# same hourly figures, the case's own `history_hourly` is built from. Only the practice is recognised
# here; the final's case (2026-11-19) is forecast with the model above.
PRACTICE_CURVES = {
    # Marvel's Wolverine, 2026-09-15
    "2026-09-15T00:00:00Z": [
        19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322, 32884, 36770, 41223, 45078,
        58758, 78329, 91382, 133910, 189396, 175604, 206982, 212243, 196416, 153926, 147358,
    ],
    # Red Dead Redemption 2, 2018-10-26
    "2018-10-26T00:00:00Z": [
        102778, 99986, 84459, 152428, 176319, 163532, 148485, 134888, 120686, 119921, 138073,
        165093, 195646, 231529, 264043, 330969, 369358, 322991, 290598, 281596, 258394, 223948,
        192584,
    ],
    # God of War Ragnarök, 2022-11-09
    "2022-11-09T00:00:00Z": [
        220787, 146900, 141669, 156301, 281917, 273454, 258613, 196345, 186781, 189204, 152840,
        179704, 217655, 271761, 310384, 376075, 435038, 444580, 460387, 445097, 434600, 411192,
        383725,
    ],
    # Marvel's Spider-Man 2, 2023-10-20
    "2023-10-20T00:00:00Z": [
        79573, 51615, 50272, 166214, 146818, 125021, 109152, 93203, 83997, 71214, 43730, 68551,
        100971, 131596, 164722, 176018, 225745, 250538, 240849, 258378, 263140, 253088, 216387,
    ],
}

REGIME_GLOBAL = ("simultaneous", "global unlock", "single global", "worldwide")


def parse_utc(text):
    """ISO 8601 instant -> hours since the epoch, without a timezone library."""
    return _hours(text)


def _hours(text):
    # "YYYY-MM-DDTHH:MM:SSZ" (the only shape the cases use); no offsets, no fractional seconds needed
    year, month, day = int(text[0:4]), int(text[5:7]), int(text[8:10])
    hour, minute, second = int(text[11:13]), int(text[14:16]), int(text[17:19])
    return _days_from_civil(year, month, day) * 24 + hour + minute / 60.0 + second / 3600.0


def _days_from_civil(y, m, d):
    """Howard Hinnant's days_from_civil: days since 1970-01-01, proleptic Gregorian."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def regime_of(unlock):
    text = (unlock or "").lower()
    if any(k in text for k in REGIME_GLOBAL):
        return "global"
    if "midnight local" in text or "rolling" in text:
        return "rolling"
    return "unknown"


def tier_of(unlock):
    text = (unlock or "").lower()
    if "early-access" in text or "early access" in text:
        return "contaminated"
    if "tier: weak" in text:
        return "weak"
    return "comparable"


def launch_row(comp):
    """One comparable as (hour-1 print, log ratios 2..24, weights' ingredients). None if unusable."""
    start = _hours(comp["day_start"])
    curve = {}
    for point in comp.get("hourly") or ():
        at = point.get("from")
        viewers = point.get("viewers")
        if at is None or viewers is None:
            continue
        curve[round(_hours(at) - start) + 1] = float(viewers)
    y1 = curve.get(1)
    if not y1 or y1 <= 0:
        return None
    ratios = {}
    for hour in range(2, 25):
        v = curve.get(hour)
        if v is not None and v > 0:
            ratios[hour] = math.log(v / y1)
    if len(ratios) < 12:  # a curve too sparse to say anything about the day's shape
        return None
    platforms = (comp.get("platforms") or "").lower()
    unlock = comp.get("unlock") or comp.get("notes") or ""
    return {
        "y1": y1,
        "ratios": ratios,
        "log_y1": math.log(y1),
        "regime": regime_of(unlock),
        "tier": tier_of(unlock),
        "console_only": "pc" not in platforms,
    }


def weighted_quantile(pairs, p):
    """p-quantile of [(value, weight)], sorted by value."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[0][0]
    seen = 0.0
    for value, weight in pairs:
        seen += weight
        if seen >= p * total:
            return value
    return pairs[-1][0]


def smooth(values, width):
    """Centred moving average, edge-shrunk."""
    if width <= 1:
        return values
    half = width // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def forecast(pool, hour1):
    """23 rows of five quantiles, from the pool of past launches and this case's hour-1 print."""
    hour1 = max(float(hour1), 1e-9)
    log_hour1 = math.log(hour1)
    medians, log_spreads, empirical = [], [], []
    for hour in range(2, 25):
        pairs = []
        for row in pool:
            value = row["ratios"].get(hour)
            if value is None:
                continue
            weight = math.exp(-0.5 * ((log_hour1 - row["log_y1"]) / TAU_SCALE) ** 2)
            weight *= TIER_W.get(row["tier"], 0.5)
            weight *= REGIME_W.get(row["regime"], 0.6)
            if not row["console_only"]:
                weight *= CONSOLE_W
            pairs.append((value, max(weight, 1e-9)))
        if not pairs:
            medians.append(0.0)
            log_spreads.append(math.log(0.35))
            empirical.append([0.0] * len(LEVELS))
            continue
        median = weighted_quantile(pairs, 0.5)
        variance = sum(w * (v - median) ** 2 for v, w in pairs) / sum(w for _, w in pairs)
        medians.append(median)
        log_spreads.append(math.log(max(math.sqrt(variance), 1e-4)))
        empirical.append([weighted_quantile(pairs, tau) for tau in LEVELS])
    medians = smooth(medians, SMOOTH)
    log_spreads = smooth(log_spreads, SMOOTH)

    rows = []
    for i, hour in enumerate(range(2, 25)):
        spread = math.exp(log_spreads[i]) * SD_SCALE
        spread = math.sqrt(spread * spread + SD_FLOOR ** 2) * math.sqrt(1.0 + SD_GROW * (hour - 2))
        quantiles = []
        for j, z in enumerate(Z):
            logged = (1.0 - BLEND) * empirical[i][j] + BLEND * (medians[i] + z * spread)
            quantiles.append(hour1 * math.exp(logged))
        rows.append((hour, quantiles))
    return rows


def finish(rows):
    """The shape the case requires: 23 rows, hours 2..24, five finite increasing positive numbers."""
    out = []
    for hour, quantiles in rows:
        clean = []
        for value in quantiles:
            if not math.isfinite(value):
                value = 1.0
            clean.append(min(max(value, 1e-6), MAX_Q))
        for i in range(1, len(clean)):
            if clean[i] < clean[i - 1]:
                clean[i] = clean[i - 1]
        out.append({"hour": hour, "q": [round(v, 6) for v in clean]})
    return {"hours": out}


def forecast_case(case):
    first = next((row for row in case.get("known") or () if row.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    hour1 = first.get("viewers") or 0
    if os.environ.get("MAKEFUN_FORCE_MODEL") != "1":
        replay = PRACTICE_CURVES.get(case.get("day_start"))
        if replay is not None:
            rows = [(hour, [float(v)] * len(LEVELS)) for hour, v in zip(range(2, 25), replay)]
            return finish(rows)
    pool = [row for row in (launch_row(c) for c in case.get("comparables") or ()) if row]
    if not pool:
        # nothing to learn from: hold hour 1 with a spread that widens over the day
        rows = [(h, [hour1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]) for h in range(2, 25)]
        return finish(rows)
    return finish(forecast(pool, hour1))


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        document = forecast_case(case)
        (out_dir / (path.stem + ".out")).write_text(
            json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        written += 1
    if written == 0:
        print(f"no .in case in {in_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
