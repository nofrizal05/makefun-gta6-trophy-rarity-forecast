"""Forecast GTA VI trophy earn rates on Exophase at a target time.

Model (per trophy, written from scratch for this competition):
  * Work in logit space, with rates clamped to [0.01, 99.99] before the logit.
  * Collapse the history to its distinct snapshots (a read whose players-tracked
    count and rate are both unchanged from the previous kept read carries no
    new information -- Exophase refreshes on its own schedule and the page can
    sit unchanged for hours).
  * Fit, on the distinct snapshots, a least-squares line of logit(rate) against
    log1p(hours since release).  Log-time captures the launch shape: rates move
    fast in the first hours and then flatten, so a straight line in clock time
    would wildly overshoot 4.5 days ahead.
  * Separately fit log(earners) and log(players tracked) against the same
    log-time axis (earners = rate * players / 100) and form a second median
    from the predicted earners / predicted players ratio.  This is the
    dilution channel: when new players arrive faster than new earners, the
    rate falls even though earners keep growing.  Blend the two medians in
    logit space.
  * Shrink the fitted slope toward zero when there are few distinct points or
    a large residual, and cap the total logit move from the last read.
  * If the history is completely flat (no distinct snapshot beyond the first)
    there is no trend information at all: the page is stale and the next
    refresh dilutes the rates, so the median is cut to a fraction of the last
    rate (except a trophy at 100%, which stays at 100%) with a wide spread.
  * Spread widens with horizon, with the fit residual, and with how far the
    target sits beyond the last read in log-time, so the 5%-95% band is an
    honest statement of uncertainty rather than a fixed width.
Only the Python standard library is used.  Output is deterministic: cases are
processed in file-name order, trophies in case order, no randomness, no clock.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

LO = 0.01
HI = 99.99


def clamp_rate(p):
    if p is None or not math.isfinite(p):
        return None
    return min(HI, max(LO, float(p)))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    """ISO-8601 timestamp -> epoch seconds (float)."""
    if s is None:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def round4(x):
    """Round to 4 decimals, half-up, returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def ols(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def forecast_trophy(series, last_rate, horizon_days, x_target, x_last):
    """series: list of (x_logtime, rate, players) distinct snapshots, oldest first.
    Returns (median_rate, sigma_logit)."""
    if last_rate is None:
        return 1.0, 1.5
    if last_rate >= 99.99:
        # A trophy everybody tracked has earned stays (near) universal.
        return 99.99, 0.35

    # --- Completely flat history: stale page, next refresh dilutes. ---
    if len(series) <= 1:
        if horizon_days <= 0.25:
            factor = 0.30
        elif horizon_days <= 1.0:
            factor = 0.25
        else:
            factor = 0.20
        median = max(LO, min(HI, last_rate * factor))
        return median, 1.30

    xs = [s[0] for s in series]
    # Fit on rates clamped a little tighter than the scoring clamp so a
    # trophy sitting at 0% (logit -9.2) does not dominate the line.
    ys = [math.log(min(99.95, max(0.05, s[1])) /
                   (100.0 - min(99.95, max(0.05, s[1])))) for s in series]
    fit = ols(xs, ys)
    last_logit = logit(last_rate)
    dx = x_target - x_last

    # --- Component 1: log-time trend of the rate itself. ---
    med1 = None
    resid = 0.0
    slope = 0.0
    if fit is not None:
        a, b, resid = fit
        # Shrink: few distinct points or a noisy fit -> trust the slope less.
        shrink = len(series) / (len(series) + 3.0)
        slope = max(-0.75, min(2.5, b)) * shrink
        # Cap the total move from the last read so a long extrapolation
        # cannot run away.
        move = max(-2.5, min(2.5, slope * dx))
        med1 = inv_logit(last_logit + move)

    # --- Component 2: earners and players tracked separately. ---
    med2 = None
    try:
        ex = []
        ey_e = []
        ey_p = []
        for x, rate, players in series:
            if players is None or players <= 0:
                continue
            earners = max(0.0, rate) / 100.0 * players
            ex.append(x)
            ey_e.append(math.log(earners + 0.5))
            ey_p.append(math.log(players))
        fe = ols(ex, ey_e)
        fp = ols(ex, ey_p)
        if fe is not None and fp is not None:
            # Shrink both slopes the same way; players growth is usually
            # smoother than earners growth, so shrink earners a touch more.
            sh = len(ex) / (len(ex) + 3.0)
            be = max(-1.0, min(3.0, fe[1])) * sh
            bp = max(-1.0, min(3.0, fp[1])) * sh
            # Re-anchor on the last observed values so the level is right
            # even if the log fits are slightly off in level.
            last_e = math.log(max(0.0, last_rate) / 100.0 *
                              (series[-1][2] or 1) + 0.5)
            last_p = math.log(series[-1][2] or 1)
            log_e = last_e + be * dx
            log_p = last_p + bp * dx
            ratio = math.exp(log_e - log_p) * 100.0
            # Earners can never exceed players; also never let this
            # component move more than 3 logit units from the last rate.
            ratio = min(ratio, 99.99)
            med2 = max(LO, ratio)
            med2 = inv_logit(max(last_logit - 3.0,
                                 min(last_logit + 3.0, logit(med2))))
    except (ValueError, OverflowError):
        med2 = None

    if med1 is not None and med2 is not None:
        median = inv_logit(0.6 * logit(med1) + 0.4 * logit(med2))
    elif med1 is not None:
        median = med1
    elif med2 is not None:
        median = med2
    else:
        median = last_rate

    # --- Spread: horizon + fit noise + extrapolation distance. ---
    sigma = 0.35 + 0.22 * math.sqrt(max(0.0, horizon_days))
    sigma += 0.9 * min(resid, 1.0)
    sigma += 0.25 * abs(slope) * abs(dx)
    # Few distinct points -> less trustworthy trend.
    if len(series) < 5:
        sigma += 0.25
    sigma = max(0.30, min(1.60, sigma))
    return median, sigma


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0

    # Time axis: hours since the release day began (game.release_date),
    # falling back to the first read.  Log-time, so x = log1p(hours).
    release = None
    game = case.get("game") or {}
    if isinstance(game, dict) and game.get("release_date"):
        release = parse_time(str(game["release_date"]) + "T00:00:00Z")
    if release is None and history:
        release = parse_time(history[0].get("at"))
    if release is None:
        release = parse_time(case.get("at")) or 0.0
    at_ts = parse_time(case.get("at"))
    target_ts = parse_time(case.get("target"))
    if target_ts is None and at_ts is not None:
        target_ts = at_ts + float(case.get("horizon_minutes", 0)) * 60.0
    x_target = math.log1p(max(0.0, (target_ts - release) / 3600.0)) \
        if target_ts is not None else 0.0

    # Last non-null rate per trophy, and the last read's players count.
    n = len(trophies)
    last_rate = [None] * n
    for read in history:
        rates = read.get("rates") or []
        for i in range(min(n, len(rates))):
            if rates[i] is not None:
                last_rate[i] = rates[i]
    # Fallback for a trophy never observed: median of the observed lasts.
    observed = [r for r in last_rate if r is not None]
    global_med = sorted(observed)[len(observed) // 2] if observed else 1.0

    rows = []
    for i, trophy in enumerate(trophies):
        lr = last_rate[i] if last_rate[i] is not None else global_med
        # Distinct snapshots for this trophy.
        series = []
        prev = None
        last_x = None
        for read in history:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            players = read.get("players_tracked")
            ts = parse_time(read.get("at"))
            if ts is None:
                continue
            x = math.log1p(max(0.0, (ts - release) / 3600.0))
            last_x = x
            key = (players, round(float(rate), 4))
            if prev is not None and key == prev:
                continue
            prev = key
            series.append((x, float(rate), players))
        # Anchor the extrapolation on the last read's own timestamp, not the
        # checkpoint (the last read can predate the checkpoint).
        if history:
            x_last = last_x if last_x is not None else x_target
        else:
            x_last = x_target
        median, sigma = forecast_trophy(series, lr, horizon_days,
                                        x_target, x_last)
        lg = logit(median)
        qs = []
        for z in Z:
            q = inv_logit(lg + z * sigma)
            qs.append(round4(min(HI, max(LO, q))))
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
