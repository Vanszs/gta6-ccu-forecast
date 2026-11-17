# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vanszs/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vanszs/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An entry to make.fun's bounty *Forecast GTA VI's launch-day Twitch viewers hour by hour*: given the Twitch
viewer count of the Grand Theft Auto VI category up to the end of the first hour of its release day
(2026-11-19, UTC), write five quantiles (5%, 25%, 50%, 75%, 95%) of that count for each of the other 23
hours, for the lowest pinball loss on the log scale.

`main.py` reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (default `/output`), standard library only, no network, deterministic.

## How it forecasts

The case's only anchor is hour 1's print, so every forecast is a quantile of an hourly ratio
`r_h = viewers_h / viewers_1`, pinned to that print. Loss is separable over (hour, level), so each of the
115 cells is estimated on its own from the 38–42 past launches the case ships in `comparables`.

A launch in the pool is weighted by four things:

| weight | why |
| --- | --- |
| **scale** — its hour-1 print against the case's | the bigger the launch, the flatter the day: `corr(ln hour1, ln r_19) ≈ -0.93` across the pool |
| **unlock regime** — read off its `unlock` text | GTA VI unlocks at midnight local, so the US is not live at hour 1 and joins at hour 6 (05:00Z) and hour 9 (08:00Z). A launch whose PC build unlocked globally at 00:00Z had the US live from hour 1 and falls where a rolling one steps up |
| **platform** — PS5/Xbox only, like GTA VI | launches without a PC build are closer analogues |
| **tier** — first day vs. an early-access second release | a launch already out for a week is a worse analogue for day one |

Each cell blends the pool's weighted empirical quantile with a normal at the weighted median and spread,
half and half: the empirical side keeps the flop and hit tails the pool shows, the normal side keeps a
nine-launch tail from being one observation wide. Median and log spread are smoothed over neighbouring
hours, since the true curve is smooth even where the pool is thin.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`dev/` holds the work behind the constants: the fetch of the four practice launches' public curves
(SullyGnome, the source the kit names), the launch table, the pinball scorer, the leave-one-launch-out
cross-validation over candidate weighting schemes, and `dev/verify.py`, which runs the entry the way the
grader does and scores it.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
