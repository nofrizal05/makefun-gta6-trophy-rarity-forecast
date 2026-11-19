# makefun-gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nofrizal05/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nofrizal05/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI trophy's Exophase earn rate at 2026-11-26T00:00:00Z, from the house's recorded reads of the game's trophy page up to the 2026-11-21T11:30:00Z deadline, as five quantiles per trophy (5%, 25%, 50%, 75%, 95%). `main.py` (Python standard library only) works in logit space: it collapses the history to its distinct snapshots (Exophase refreshes on its own schedule, so unchanged reads carry no new information), fits a per-trophy least-squares line of logit(rate) against log-time since release, and separately fits log(earners) and log(players tracked) against the same log-time axis so the forecast captures dilution — new players arriving faster than new earners pushes rates down even while earners grow. The two medians are blended in logit space, slopes are shrunk toward zero when there are few distinct reads or a noisy fit, and the quantile spread widens with horizon, fit residual, and extrapolation distance. A completely flat history (a stale page with no trend information) falls back to a wide spread around a diluted fraction of the last rate, except a trophy at 100%, which stays at 100%.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

It reads every `*.in` case in the input directory and writes one `<id>.out` JSON file per case into the output directory. No network, no randomness, no third-party packages — output is byte-identical across runs.

---

<sub>Entered on [make.fun](https://make.fun/?ref=31w76kz9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
