"""
smma_screener.py
================
US SMMA Screener — runs inside SmartStock on demand.

Strategy (same as v5 backtest, ported to US markets):
  LTF : 5-min  |  SMMA(11) × SMMA(21) bullish crossover
  HTF1: 15-min |  SMMA(21) — price must be above
  HTF2: 1-hour |  SMMA(21) — price must be above
  Pre-filters : RS ≥ 65, UDTS ≥ 40, blacklist applied
  Time gates  : No entry before 10:30 AM EST
                Dead zone 13:00–13:30 EST
  Volume      : Dry-up + expansion confirmation 11:00–13:00

Output: list of current SMMA swing setups with
        entry price, SL (swing low), target (2:1 R/R)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, date, time as dtime
from collections import defaultdict
warnings.filterwarnings("ignore")

# ── Strategy constants ─────────────────────────────────────────────────────────
SMMA_FAST            = 11
SMMA_SLOW            = 21
HTF_PERIOD           = 21
MIN_RS               = 65
MIN_UDTS             = 40
SWING_LOOKBACK       = 6
HIGH_RS_THRESHOLD    = 85
HIGH_RS_LOOKBACK     = 10
MAX_SL_PCT           = 0.05
NO_TRADE_BEFORE      = dtime(10, 30)
NO_NEW_AFTER         = dtime(14, 30)
VOL_FILTER_START     = dtime(11, 0)
VOL_FILTER_END       = dtime(13, 0)
VOL_DECLINE_BARS     = 3
VOL_EXPANSION_FACTOR = 1.2
DEAD_ZONE_START      = dtime(13, 0)
DEAD_ZONE_END        = dtime(13, 30)

US_BLACKLIST = {"XOM", "CVX", "COP", "MRO", "DVN", "FCX", "NEM", "GOLD"}


# ── SMMA computation ───────────────────────────────────────────────────────────
def _smma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        out[i] = (out[i-1] * (period - 1) + arr[i]) / period
    return out


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros(len(arr))
    if len(arr) < period:
        return out
    out[period-1] = np.mean(arr[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    d  = np.diff(closes.astype(float))
    ag = float(np.mean(np.where(d > 0, d, 0.0)[-period:]))
    al = float(np.mean(np.where(d < 0, -d, 0.0)[-period:]))
    return round(100.0 - 100.0 / (1.0 + ag / al), 1) if al > 0 else 100.0


# ── UDTS score (inline) ────────────────────────────────────────────────────────
def _udts_score(closes, highs, lows, volumes, trade_date) -> float:
    n = len(closes)
    if n < 210:
        return 0.0
    c = closes.astype(float);  h = highs.astype(float)
    l = lows.astype(float);    v = volumes.astype(float)
    ema20  = _ema(c, 20);  ema200 = _ema(c, 200)
    price  = c[-1]
    vol20  = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v))
    vol_r  = v[-1] / vol20 if vol20 > 0 else 1.0
    sma20  = float(np.mean(c[-20:]));  std20 = float(np.std(c[-20:]))
    bb_w   = (4 * std20 / sma20 * 100) if sma20 > 0 else 999.0
    c15    = c[-15:];  ch = c15.max();  cl_ = c15.min()
    cons   = (ch - cl_) / ch * 100 if ch > 0 else 999.0
    vol_sp = v[-1] > vol20 * 1.5
    brkout = price > ch * 0.999 and vol_sp
    dr     = h[-1] - l[-1]
    ibs    = (c[-1] - l[-1]) / dr if dr > 0 else 0.5
    h25    = h[-25:].max() if n >= 25 else h.max()
    hl25   = float(np.mean(h[-25:] - l[-25:])) if n >= 25 else float(np.mean(h - l))
    xlp_b  = h25 - 2 * hl25
    xlp_s  = price < xlp_b and ibs > 0.4
    abv200 = price > ema200[-1] if ema200[-1] > 0 else False
    pema20 = (price - ema20[-1]) / ema20[-1] * 100 if ema20[-1] > 0 else 0.0
    rsi_v  = _rsi(c)
    dom    = trade_date.day;  dow = trade_date.weekday()
    pts = 0
    pts += 10 if abv200 else 0
    pts += 10 if -3.0 <= pema20 <= 3.0 else 0
    pts += 10 if 35 <= rsi_v <= 55 else 0
    pts += 5  if vol_r < 1.0 else 0
    pts += 10 if cons <= 10.0 else 0
    pts += 5  if bb_w <= 8.0 else 0
    pts += 15 if brkout else (7 if vol_sp else 0)
    pts += 15 if xlp_s  else (7 if ibs > 0.4 else 0)
    pts += 10 if dow in (0, 1) else 0
    pts += 10 if dom <= 3 or dom >= 27 else 0
    return round(float(pts), 1)


# ── Data fetch ────────────────────────────────────────────────────────────────
def _fetch(sym: str, interval: str) -> pd.DataFrame:
    try:
        period = "58d" if interval in ("5m", "15m") else ("2y" if interval == "1d" else "60d")
        df = yf.Ticker(sym).history(
            period=period, interval=interval,
            auto_adjust=True, prepost=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


# ── RS rating ─────────────────────────────────────────────────────────────────
def _compute_rs(daily_data: dict, today: date) -> dict:
    """Compute RS rating (0-99) for each stock relative to the universe."""
    perf = {}
    for sym, df in daily_data.items():
        sub = df[df.index.date <= today]
        if len(sub) < 20:
            continue
        lb   = min(252, len(sub) - 1)
        p_now = float(sub["Close"].iloc[-1])
        p_ago = float(sub["Close"].iloc[-1 - lb])
        if p_ago > 0:
            perf[sym] = (p_now / p_ago - 1) * 100
    if not perf:
        return {}
    vals = sorted(perf.values())
    n    = len(vals)
    return {
        sym: round((sum(1 for v in vals if v <= perf[sym]) / n) * 99)
        for sym in perf
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_smma_screen(tickers: list, log_cb=None, sessions: int = 5) -> dict:
    """
    US SMMA Screener — scans for active SMMA swing setups.

    Returns:
        {
          "results":    [ signal_dict, ... ],
          "rejections": [ reject_dict, ... ],
          "total":      int,
        }
    """
    def log(m): log_cb(m) if log_cb else None

    # Filter blacklist
    tickers = [t for t in tickers if t not in US_BLACKLIST]
    log(f"US SMMA Screen — {len(tickers)} stocks")

    # ── Phase 0: Daily data for RS + UDTS ─────────────────────────────────────
    log("Step 1: Fetching daily data for RS + UDTS filters...")
    daily_data = {}
    for sym in tickers:
        df = _fetch(sym, "1d")
        if not df.empty:
            daily_data[sym] = df

    today    = max(set(d for df in daily_data.values()
                       for d in df.index.date)) if daily_data else date.today()
    rs_today = _compute_rs(daily_data, today)

    udts_today = {}
    for sym, df in daily_data.items():
        hist = df[df.index.date < today]
        udts_today[sym] = _udts_score(
            hist["Close"].values, hist["High"].values,
            hist["Low"].values, hist["Volume"].values, today
        ) if len(hist) >= 210 else 0.0

    log(f"  RS + UDTS computed for {len(rs_today)} stocks (date: {today})")

    # ── Phase 1: Intraday signal scan ─────────────────────────────────────────
    log("Step 2: Scanning 5-min SMMA signals...")
    results    = []
    rejections = []

    for sym in tickers:
        rs   = rs_today.get(sym, 0)
        udts = udts_today.get(sym, 0.0)

        if rs < MIN_RS:
            rejections.append({"ticker": sym, "reason": f"RS {rs} < {MIN_RS}"})
            continue
        if udts < MIN_UDTS:
            rejections.append({"ticker": sym, "reason": f"UDTS {udts:.0f} < {MIN_UDTS}",
                                "rs_rating": rs})
            continue

        # Fetch intraday
        df5  = _fetch(sym, "5m")
        df15 = _fetch(sym, "15m")
        df1h = _fetch(sym, "1h")
        df1d = daily_data.get(sym)

        if df5 is None or df5.empty:
            rejections.append({"ticker": sym, "reason": "No 5-min data", "rs_rating": rs})
            continue

        # Filter to last N sessions
        last_dates = set(sorted(set(df5.index.date))[-sessions:])
        df5_f = df5[np.isin(df5.index.date, list(last_dates))].copy()
        if len(df5_f) < SMMA_SLOW + 10:
            rejections.append({"ticker": sym, "reason": "Insufficient bars",
                                "rs_rating": rs})
            continue

        # Compute LTF SMMAs
        c5      = df5_f["Close"].values.astype(float)
        s_fast  = _smma(c5, SMMA_FAST)
        s_slow  = _smma(c5, SMMA_SLOW)
        vols    = df5_f["Volume"].values.astype(float)
        lows_arr= df5_f["Low"].values.astype(float)
        times   = df5_f.index

        # Compute HTF1 (15-min SMMA) aligned to 5-min index
        htf15 = np.full(len(df5_f), np.nan)
        if df15 is not None and not df15.empty:
            df15_f = df15[np.isin(df15.index.date, list(last_dates))].copy()
            if not df15_f.empty:
                s15 = pd.Series(_smma(df15_f["Close"].values.astype(float), HTF_PERIOD),
                                 index=df15_f.index).shift(1)
                htf15 = s15.reindex(df5_f.index, method="ffill").values

        # Compute HTF2 (1-hour SMMA) aligned to 5-min index
        htf1h = np.full(len(df5_f), np.nan)
        if df1h is not None and not df1h.empty:
            df1h_f = df1h[np.isin(df1h.index.date, list(last_dates))].copy()
            if not df1h_f.empty:
                s1h = pd.Series(_smma(df1h_f["Close"].values.astype(float), HTF_PERIOD),
                                 index=df1h_f.index).shift(1)
                htf1h = s1h.reindex(df5_f.index, method="ffill").values

        # Scan for signals
        found = False
        min_i = SMMA_SLOW + VOL_DECLINE_BARS + 2

        for i in range(min_i, len(df5_f)):
            if np.isnan(s_fast[i]) or np.isnan(s_slow[i]):
                continue

            # Bullish crossover
            if not (s_fast[i-1] <= s_slow[i-1] and s_fast[i] > s_slow[i]):
                continue

            t    = times[i].time()
            # Time gates
            if t < NO_TRADE_BEFORE or t > NO_NEW_AFTER:
                continue
            if DEAD_ZONE_START <= t < DEAD_ZONE_END:
                continue

            # HTF confirmation
            if np.isnan(htf15[i]) or np.isnan(htf1h[i]):
                continue
            if not (c5[i] > htf15[i] and c5[i] > htf1h[i]):
                continue

            # Volume filter in noisy window
            if VOL_FILTER_START <= t < VOL_FILTER_END and i >= VOL_DECLINE_BARS + 1:
                vol_dec = all(
                    vols[i - VOL_DECLINE_BARS + k] > vols[i - VOL_DECLINE_BARS + k + 1]
                    for k in range(VOL_DECLINE_BARS - 1)
                )
                vol_exp = vols[i-1] > 0 and vols[i] >= vols[i-1] * VOL_EXPANSION_FACTOR
                if not (vol_dec and vol_exp):
                    continue

            # SL from swing low
            lookback = HIGH_RS_LOOKBACK if rs >= HIGH_RS_THRESHOLD else SWING_LOOKBACK
            sw_lo    = float(np.min(lows_arr[max(0, i - lookback): i + 1]))
            sl       = round(sw_lo * 0.9995, 2)
            ep       = round(float(c5[i]), 2)
            risk     = ep - sl
            if risk <= 0 or (risk / ep) > MAX_SL_PCT:
                continue

            tgt       = round(ep + risk * 2.0, 2)
            vol_ratio = round(float(vols[i]) / float(vols[i-1]), 2) if vols[i-1] > 0 else 0.0

            # Company name from yfinance info (best effort)
            name   = sym
            sector = ""
            try:
                info   = yf.Ticker(sym).fast_info
                name   = getattr(info, "name", sym) or sym
            except Exception:
                pass

            # Entry trigger: is price still above yesterday's high with volume?
            entry_triggered = False
            trigger_note    = ""
            try:
                if df1d is not None and len(df1d) >= 3:
                    prev_high = float(df1d["High"].iloc[-2])
                    cur_price = ep
                    v_avg     = float(df1d["Volume"].iloc[:-1].mean())
                    v_today   = float(df1d["Volume"].iloc[-1])
                    vr        = round(v_today / v_avg, 2) if v_avg > 0 else 0
                    entry_triggered = cur_price > prev_high and vr >= 1.2
                    trigger_note    = (f"${cur_price:.2f} > ${prev_high:.2f} + vol {vr}×"
                                       if entry_triggered
                                       else f"${cur_price:.2f} vs ${prev_high:.2f} · vol {vr}×")
            except Exception:
                pass

            results.append({
                "ticker":           sym,
                "name":             name,
                "sector":           sector,
                "signal_time":      str(times[i]),
                "signal_date":      str(times[i].date()),
                "entry":            ep,
                "sl":               sl,
                "target":           tgt,
                "risk_pct":         round(risk / ep * 100, 2),
                "rs_rating":        rs,
                "udts_score":       round(udts, 1),
                "vol_ratio":        vol_ratio,
                "htf15_smma":       round(float(htf15[i]), 2) if not np.isnan(htf15[i]) else None,
                "htf1h_smma":       round(float(htf1h[i]), 2) if not np.isnan(htf1h[i]) else None,
                "entry_triggered":  entry_triggered,
                "trigger_note":     trigger_note,
                "price_now":        ep,
            })
            log(f"  {sym} ✓ — RS:{rs} UDTS:{udts:.0f} "
                f"Entry:${ep} SL:${sl} Target:${tgt} "
                f"{'[ENTRY ACTIVE]' if entry_triggered else ''}")
            found = True
            break   # most recent signal only

        if not found:
            rejections.append({
                "ticker":    sym,
                "reason":    "No SMMA crossover in last 5 sessions",
                "rs_rating": rs,
                "udts_score": round(udts, 1),
            })

    # Sort by entry_triggered first, then RS rating
    results.sort(key=lambda x: (0 if x.get("entry_triggered") else 1,
                                 -x.get("rs_rating", 0)))
    log(f"US SMMA complete — {len(results)} setups, {len(rejections)} no-signal")
    return {"results": results, "rejections": rejections, "total": len(tickers)}
