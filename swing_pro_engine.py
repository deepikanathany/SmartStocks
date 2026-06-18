"""
swing_pro_engine.py
===================
Multi-strategy Swing Trading screener for SmartStock.

Strategies
----------
1. 20 EMA Reversion      — Trend pullback to 20 EMA, RSI reset, low volume
2. Consolidation Breakout — Tight range 8-10%, breakout on volume surge
3. XLP Mean Reversion    — IBS ratio signal (works on any sector, not just XLP)
4. Seasonal Nasdaq Mon/Tue— Day-of-week seasonal edge
5. Seasonal Turn-of-Month — First 3 / last 2 trading days of month

Each strategy is independently enabled/disabled via the `strategies` dict.
Each has its own sub-thresholds configurable from the frontend.

After quantitative scoring, results are enriched with:
  - Groq sentiment score from recent news headlines (boosts rank)
  - Groq AI verdict (Buy / Hold / Avoid + one-line reason)

Backtesting
-----------
A lightweight vectorised backtest runs over the last 6 months of daily data
to estimate win rate, avg gain, and profit factor for each passing stock
using the same strategy logic applied historically.
"""

import yfinance as yf
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Groq client (shared with the rest of the app)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from groq import Groq as _Groq
    _GROQ_KEY    = "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"
    _GROQ_MODEL  = "llama-3.3-70b-versatile"
    _groq_client = _Groq(api_key=_GROQ_KEY)
    _GROQ_OK     = True
except ImportError:
    _GROQ_OK     = False
    _groq_client = None
    _GROQ_MODEL  = ""


def _ask_groq(prompt, max_tokens=300):
    if not _GROQ_OK or not _groq_client:
        return None
    try:
        r = _groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Indicator Computation
# ─────────────────────────────────────────────────────────────────────────────
def _div_yield_pct(info):
    """Normalize Yahoo's inconsistent dividendYield field to a percentage.
    Yahoo returns dividendYield as a true decimal (0.0069=0.69%) for most
    stocks, but sometimes already as a percent expressed as a decimal
    (1.09=1.09%). A real US yield of 20%+ is essentially impossible, so
    raw >= 0.20 is treated as already-percent.
    """
    raw = info.get("dividendYield") or 0
    return round(raw, 2) if raw >= 0.20 else round(raw * 100, 2)


def _ema(arr, period):
    result = np.zeros(len(arr))
    if len(arr) < period:
        return result
    result[period - 1] = arr[:period].mean()
    k = 2.0 / (period + 1)
    for i in range(period, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.mean(gains[-period:])
    al = np.mean(losses[-period:])
    if al == 0:
        return 100.0
    return round(100.0 - 100.0 / (1 + ag / al), 1)


def _compute_indicators(closes, highs, lows, volumes):
    """
    Compute all indicators needed across all 5 strategies.
    Returns dict or None if data too short.
    """
    n = len(closes)
    if n < 210:
        return None

    # Convert to numpy arrays — critical fix: if closes/highs/lows/volumes are
    # pandas Series with a DatetimeIndex, scalar access like arr[20] raises
    # KeyError (pandas treats integers as index *labels*, not positions).
    # np.asarray strips the index so positional access works as intended.
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs,  dtype=float)
    l = np.asarray(lows,   dtype=float)
    v = np.asarray(volumes, dtype=float)

    ema20  = _ema(c, 20)
    ema50  = _ema(c, 50)
    ema200 = _ema(c, 200)

    price      = c[-1]
    vol_avg20  = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v)
    vol_ratio  = v[-1] / vol_avg20 if vol_avg20 > 0 else 1.0

    # Bollinger band width
    sma20    = np.mean(c[-20:])
    std20    = np.std(c[-20:])
    bb_width = (4 * std20 / sma20 * 100) if sma20 > 0 else 0.0  # full band width %

    # Consolidation (last 15 days)
    c15 = c[-15:] if n >= 15 else c
    cons_high = c15.max()
    cons_low  = c15.min()
    cons_pct  = (cons_high - cons_low) / cons_high * 100 if cons_high > 0 else 999.0

    # Breakout = close above 15-day range on volume spike
    vol_spike = bool(v[-1] > vol_avg20 * 1.5)
    breakout  = bool(price > cons_high * 0.999 and vol_spike)

    # IBS (Internal Bar Strength)
    daily_range = h[-1] - l[-1]
    ibs = (c[-1] - l[-1]) / daily_range if daily_range > 0 else 0.5

    # XLP signal: close below (25d high - 2×avg_hl_range) AND ibs > 0.4
    hl_range_25 = np.mean(h[-25:] - l[-25:]) if n >= 25 else np.mean(h - l)
    high_25d    = h[-25:].max() if n >= 25 else h.max()
    xlp_band    = high_25d - 2 * hl_range_25
    xlp_signal  = bool(c[-1] < xlp_band and ibs > 0.4)

    # ATR(14)
    tr   = np.array([max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                     for i in range(1, n)])
    atr14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

    # Seasonal flags
    today = datetime.today()
    dow   = today.weekday()  # 0=Mon, 1=Tue
    dom   = today.day
    is_mon_tue     = dow in (0, 1)
    is_turn_month  = (dom <= 3 or dom >= 27)

    # Price relationships
    above_200     = bool(price > ema200[-1])
    above_50      = bool(price > ema50[-1])
    pct_ema20     = ((price - ema20[-1]) / ema20[-1] * 100) if ema20[-1] > 0 else 0.0

    return {
        "price":           round(price, 2),
        "ema20":           round(float(ema20[-1]), 2),
        "ema50":           round(float(ema50[-1]), 2),
        "ema200":          round(float(ema200[-1]), 2),
        "rsi":             _rsi(c),
        "vol_ratio":       round(float(vol_ratio), 2),
        "vol_spike":       vol_spike,
        "bb_width":        round(float(bb_width), 2),
        "cons_pct":        round(float(cons_pct), 1),
        "cons_high":       round(float(cons_high), 2),
        "breakout":        breakout,
        "ibs":             round(float(ibs), 3),
        "xlp_signal":      xlp_signal,
        "atr14":           round(float(atr14), 2),
        "above_200":       above_200,
        "above_50":        above_50,
        "pct_ema20":       round(float(pct_ema20), 2),
        "is_mon_tue":      is_mon_tue,
        "is_turn_month":   is_turn_month,
        "stop_loss_atr":   round(price - 1.5 * atr14, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Scoring
# ─────────────────────────────────────────────────────────────────────────────
def _score_strategies(ind, strategies, thresh):
    """
    Score the stock across all enabled strategies.

    Returns:
        total_score  (float 0-100)
        signals      (list of str)
        strat_detail (dict: strategy -> {score, max, pct, signals})
    """
    total_pts   = 0
    total_max   = 0
    signals     = []
    strat_detail = {}

    # ── 1. 20 EMA Reversion (weight 35) ──────────────────────────────────────
    if strategies.get("ema_reversion", True):
        max_pts = 35
        pts     = 0
        sigs    = []

        if ind["above_200"]:
            pts += 10; sigs.append("Price > 200 EMA")

        prox = thresh.get("ema20_proximity_pct", 3.0)
        if -prox <= ind["pct_ema20"] <= prox:
            pts += 10; sigs.append(f"Near 20 EMA ({ind['pct_ema20']:+.1f}%)")

        rlo = thresh.get("ema_rsi_low",  40)
        rhi = thresh.get("ema_rsi_high", 55)
        if rlo <= ind["rsi"] <= rhi:
            pts += 10; sigs.append(f"RSI reset zone ({ind['rsi']:.0f})")

        vol_ceil = thresh.get("ema_vol_below", 1.0)
        if ind["vol_ratio"] < vol_ceil:
            pts += 5; sigs.append(f"Low-vol pullback ({ind['vol_ratio']:.1f}×)")

        strat_detail["ema_reversion"] = {
            "label": "20 EMA Reversion", "score": pts,
            "max": max_pts, "pct": round(pts/max_pts*100), "signals": sigs
        }
        total_pts += pts; total_max += max_pts; signals.extend(sigs)

    # ── 2. Consolidation Breakout (weight 30) ─────────────────────────────────
    if strategies.get("cons_breakout", True):
        max_pts = 30
        pts     = 0
        sigs    = []

        cr = thresh.get("cons_range_pct", 10.0)
        if ind["cons_pct"] <= cr:
            pts += 10; sigs.append(f"Tight range ({ind['cons_pct']:.1f}%)")

        bw = thresh.get("bb_width_max", 8.0)
        if ind["bb_width"] <= bw:
            pts += 5; sigs.append(f"BB squeeze ({ind['bb_width']:.1f}%)")

        if ind["breakout"]:
            pts += 15; sigs.append("Breakout on high volume")
        elif ind["vol_spike"]:
            pts += 7; sigs.append("Volume building")

        strat_detail["cons_breakout"] = {
            "label": "Consolidation Breakout", "score": pts,
            "max": max_pts, "pct": round(pts/max_pts*100), "signals": sigs
        }
        total_pts += pts; total_max += max_pts; signals.extend(sigs)

    # ── 3. XLP Mean Reversion (weight 15) ────────────────────────────────────
    if strategies.get("xlp_reversion", True):
        max_pts = 15
        pts     = 0
        sigs    = []

        if ind["xlp_signal"]:
            pts += 15; sigs.append(f"IBS mean-reversion (IBS={ind['ibs']:.2f})")
        elif ind["ibs"] > 0.4:
            pts += 7;  sigs.append(f"IBS elevated ({ind['ibs']:.2f})")

        strat_detail["xlp_reversion"] = {
            "label": "XLP Mean Reversion", "score": pts,
            "max": max_pts, "pct": round(pts/max_pts*100), "signals": sigs
        }
        total_pts += pts; total_max += max_pts; signals.extend(sigs)

    # ── 4. Seasonal Nasdaq Mon/Tue (weight 10) ────────────────────────────────
    if strategies.get("seasonal_nasdaq", True):
        max_pts = 10
        pts     = 10 if ind["is_mon_tue"] else 0
        sigs    = (["Nasdaq Mon/Tue seasonal"] if pts else [])

        strat_detail["seasonal_nasdaq"] = {
            "label": "Nasdaq Mon/Tue", "score": pts,
            "max": max_pts, "pct": round(pts/max_pts*100) if max_pts else 0, "signals": sigs
        }
        total_pts += pts; total_max += max_pts; signals.extend(sigs)

    # ── 5. Seasonal Turn-of-Month (weight 10) ─────────────────────────────────
    if strategies.get("seasonal_tom", True):
        max_pts = 10
        pts     = 10 if ind["is_turn_month"] else 0
        sigs    = (["Turn-of-month window"] if pts else [])

        strat_detail["seasonal_tom"] = {
            "label": "Turn-of-Month", "score": pts,
            "max": max_pts, "pct": round(pts/max_pts*100) if max_pts else 0, "signals": sigs
        }
        total_pts += pts; total_max += max_pts; signals.extend(sigs)

    total_score = round(total_pts / total_max * 100, 1) if total_max > 0 else 0.0
    return total_score, signals, strat_detail


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Backtest (vectorised, last 6 months)
# ─────────────────────────────────────────────────────────────────────────────
def _backtest_stock(closes, highs, lows, volumes, strategies, thresh,
                    hold_days=5, target_pct=2.0, stop_pct=1.5):
    """
    Simulate the strategy on the last 6 months of daily data.
    For each day where signals fired, track outcome after `hold_days`.
    Returns dict: {trades, wins, losses, avg_gain_pct, profit_factor}
    """
    n = len(closes)
    if n < 252:
        return None

    # Same fix as _compute_indicators — ensure numpy positional indexing
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs,  dtype=float)
    l = np.asarray(lows,   dtype=float)
    v = np.asarray(volumes, dtype=float)

    # Only backtest last 126 trading days (~6 months)
    start = max(210, n - 126)

    trades = []
    for i in range(start, n - hold_days - 1):
        # Compute indicators up to day i
        sub_c = c[:i+1]; sub_h = h[:i+1]; sub_l = l[:i+1]; sub_v = v[:i+1]

        ind = _compute_indicators(
            sub_c, sub_h, sub_l, sub_v
        )
        if not ind:
            continue

        score, signals, _ = _score_strategies(ind, strategies, thresh)
        min_score = thresh.get("min_score", 30)
        if score < min_score or not signals:
            continue

        # Entry at next day open (approximated as close)
        entry  = c[i]
        target = entry * (1 + target_pct / 100)
        stop   = entry - 1.5 * ind["atr14"]

        # Outcome over hold_days
        outcome_prices = c[i+1:i+1+hold_days]
        outcome_high   = h[i+1:i+1+hold_days]
        outcome_low    = l[i+1:i+1+hold_days]

        # Check if stop hit or target hit
        hit_stop   = any(outcome_low[j] <= stop   for j in range(len(outcome_low)))
        hit_target = any(outcome_high[j] >= target for j in range(len(outcome_high)))

        if hit_target and not hit_stop:
            gain = target_pct
        elif hit_stop:
            gain = round((stop - entry) / entry * 100, 2)
        else:
            gain = round((outcome_prices[-1] - entry) / entry * 100, 2)

        trades.append(gain)

    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0,
                "avg_gain_pct": 0, "profit_factor": 0, "win_rate": 0}

    wins   = [g for g in trades if g > 0]
    losses = [g for g in trades if g <= 0]
    avg_g  = round(sum(wins)   / len(wins),   2) if wins   else 0
    avg_l  = round(sum(losses) / len(losses), 2) if losses else 0
    pf     = round(abs(sum(wins) / sum(losses)), 2) if sum(losses) != 0 else 99.0
    wr     = round(len(wins) / len(trades) * 100, 1)
    avg_gain = round(sum(trades) / len(trades), 2)

    return {
        "trades":         len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       wr,
        "avg_gain_pct":   avg_gain,
        "avg_win_pct":    avg_g,
        "avg_loss_pct":   avg_l,
        "profit_factor":  pf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Groq Sentiment Score
# ─────────────────────────────────────────────────────────────────────────────
def _sentiment_score(ticker, name):
    """
    Fetch recent news headlines via yfinance and ask Groq to score sentiment 1-100.
    Returns (score, summary) or (50, "N/A").
    """
    if not _GROQ_OK:
        return 50, "Groq not available"

    try:
        news_items = yf.Ticker(ticker).news or []
        if not news_items:
            return 50, "No recent news"

        headlines = "\n".join([
            f"- {item.get('title','')}"
            for item in news_items[:8]
        ])

        prompt = f"""You are a financial news sentiment analyst.

Stock: {name} ({ticker})

Recent headlines:
{headlines}

Score the overall sentiment for a SHORT-TERM SWING TRADER (1-2 week horizon).
Consider: earnings surprises, analyst upgrades/downgrades, macro risks, product news.

Respond ONLY in this JSON format, no markdown:
{{"score": 72, "summary": "One sentence max 20 words explaining the sentiment."}}"""

        raw = _ask_groq(prompt, max_tokens=100)
        if not raw:
            return 50, "N/A"

        raw    = raw.replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        score  = max(1, min(100, int(parsed.get("score", 50))))
        summ   = parsed.get("summary", "")
        return score, summ

    except Exception:
        return 50, "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Groq AI Verdict
# ─────────────────────────────────────────────────────────────────────────────
def _ai_verdict(ticker, name, sector, score, signals, ind, sentiment_score):
    """
    Ask Groq for a Buy/Hold/Avoid verdict based on the swing pro score + signals.
    Returns dict: {verdict, reason, error}
    """
    if not _GROQ_OK:
        return {"verdict": "N/A", "reason": "Groq not available", "error": True}

    sigs_text = "; ".join(signals) if signals else "None"
    prompt = f"""You are a swing trading analyst. A stock has passed a multi-strategy screen.

Stock:    {name} ({ticker})
Sector:   {sector}
Score:    {score}/100
Signals:  {sigs_text}
Price:    ${ind['price']}
RSI:      {ind['rsi']}
Sentiment score: {sentiment_score}/100 (1=very negative, 100=very positive)
ATR stop: ${ind['stop_loss_atr']}
Above 200 EMA: {'Yes' if ind['above_200'] else 'No'}

For a 1-2 week swing trade, give a verdict: Strong Buy / Buy / Hold / Avoid
One sentence max 20 words explaining why, referencing the key signal.

Respond ONLY in JSON, no markdown:
{{"verdict": "Buy", "reason": "One sentence here."}}"""

    try:
        raw    = _ask_groq(prompt, max_tokens=80)
        if not raw:
            return {"verdict": "N/A", "reason": "No response", "error": True}
        raw    = raw.replace("```json","").replace("```","").strip()
        parsed = json.loads(raw)
        return {
            "verdict": parsed.get("verdict", "Hold"),
            "reason":  parsed.get("reason", ""),
            "error":   False,
        }
    except Exception as e:
        return {"verdict": "N/A", "reason": str(e)[:60], "error": True}


# ─────────────────────────────────────────────────────────────────────────────
# Single Stock Screener
# ─────────────────────────────────────────────────────────────────────────────
def screen_stock_swing_pro(symbol, strategies, thresh, allowed_sectors=None,
                            run_backtest=True, run_sentiment=True, run_verdict=True):
    """
    Full Swing Pro screen for one stock.
    Returns (result_dict | None, rejection_dict | None)
    """
    rejection = {
        "ticker": symbol, "name": symbol, "sector": "—",
        "price_now": None, "52w_high": None, "52w_low": None,
        "failed_filters": [], "scorecard": [],
    }

    def _rej(reason):
        """Return a rejection with a single descriptive filter entry."""
        rejection["failed_filters"] = [reason]
        rejection["scorecard"] = [{
            "filter": reason, "passed": False, "active": True,
            "actual": "N/A", "threshold": "—",
        }]
        return None, rejection

    try:
        tk = yf.Ticker(symbol)

        # ── Fetch info with retry + fast_info fallback ────────────────────────
        info = {}
        for _attempt in range(3):
            try:
                info = tk.info or {}
                if info and len(info) >= 5:
                    break
                time.sleep(1 + _attempt)
            except Exception:
                time.sleep(2 + _attempt * 2)

        # fast_info fallback — lighter endpoint, less prone to rate limiting
        if not info or len(info) < 5:
            try:
                fi = tk.fast_info
                info = {
                    "longName":              getattr(fi, "description", symbol),
                    "shortName":             symbol,
                    "sector":                getattr(fi, "sector", "—"),
                    "regularMarketPrice":    getattr(fi, "last_price", None),
                    "fiftyTwoWeekHigh":      getattr(fi, "year_high", None),
                    "fiftyTwoWeekLow":       getattr(fi, "year_low", None),
                    "marketCap":             getattr(fi, "market_cap", None),
                    "trailingPE":            getattr(fi, "pe_forward", None),
                    "exchange":              getattr(fi, "exchange", "—"),
                    "_from_fast_info":       True,
                }
                # fast_info doesn't give sector — mark as unknown but don't reject
                if not info.get("sector"):
                    info["sector"] = "—"
            except Exception:
                pass

        if not info or not info.get("regularMarketPrice") and not info.get("_from_fast_info"):
            return _rej("No Data (yfinance timeout — retry later)")

        name   = info.get("longName") or info.get("shortName") or symbol
        sector = info.get("sector", "—")

        # Populate rejection with basic data immediately so table columns show values
        rejection["name"]          = name
        rejection["sector"]        = sector
        rejection["price_now"]     = info.get("regularMarketPrice") or info.get("currentPrice")
        rejection["52w_high"]      = info.get("fiftyTwoWeekHigh")
        rejection["52w_low"]       = info.get("fiftyTwoWeekLow")
        rejection["pe"]            = info.get("trailingPE") or info.get("forwardPE")
        rejection["div_yield_pct"] = _div_yield_pct(info)
        rejection["market_cap_b"]  = round((info.get("marketCap") or 0) / 1e9, 2)

        # Only apply sector filter if a non-empty list was explicitly passed
        if allowed_sectors and len(allowed_sectors) > 0 and sector and sector not in allowed_sectors:
            return _rej(f"Sector: {sector}")

        # ── Fetch price history ───────────────────────────────────────────────
        hist = None
        for _attempt in range(3):
            try:
                hist = tk.history(period="2y", interval="1d", auto_adjust=True)
                if hist is not None and len(hist) >= 210:
                    break
                time.sleep(1)
                hist = tk.history(period="3y", interval="1d", auto_adjust=True)
                if hist is not None and len(hist) >= 210:
                    break
            except Exception as _he:
                time.sleep(2)

        if hist is None or len(hist) < 210:
            days = len(hist) if hist is not None else 0
            return _rej(f"Insufficient History ({days} days)")

        # Normalise column names
        if "Adj Close" in hist.columns and "Close" not in hist.columns:
            hist = hist.rename(columns={"Adj Close": "Close"})

        ind = _compute_indicators(hist["Close"], hist["High"], hist["Low"], hist["Volume"])
        if not ind:
            return _rej("Indicator Error")

        min_score = thresh.get("min_score", 30)
        score, signals, strat_detail = _score_strategies(ind, strategies, thresh)

        # Build rich scorecard with actual indicator values per strategy
        scorecard = []
        for strat_id, sd in strat_detail.items():
            # Main strategy row
            scorecard.append({
                "filter":    sd["label"],
                "passed":    sd["score"] > 0,
                "active":    True,
                "actual":    f"{sd['score']}/{sd['max']} ({sd['pct']}%)",
                "threshold": "score > 0",
            })
            # Individual signal sub-rows
            for sig in sd.get("signals", []):
                scorecard.append({
                    "filter":    f"  ✓ {sig}",
                    "passed":    True,
                    "active":    True,
                    "actual":    "",
                    "threshold": "",
                })

        # Add key indicator values as informational rows
        scorecard.extend([
            {"filter": "RSI(14)",       "passed": 40<=ind["rsi"]<=55,       "active": True,
             "actual": f"{ind['rsi']:.0f}",            "threshold": "40–55"},
            {"filter": "EMA20 proximity","passed": abs(ind["pct_ema20"])<=3, "active": True,
             "actual": f"{ind['pct_ema20']:+.1f}%",    "threshold": "within ±3%"},
            {"filter": "Above 200 EMA", "passed": ind["above_200"],          "active": True,
             "actual": f"${ind['ema200']:.2f}",        "threshold": "price above"},
            {"filter": "Vol Ratio",     "passed": ind["vol_ratio"] >= 1.5,   "active": True,
             "actual": f"{ind['vol_ratio']:.1f}×",     "threshold": "≥ 1.5×"},
            {"filter": "BB Width",      "passed": ind["bb_width"] <= 8,      "active": True,
             "actual": f"{ind['bb_width']:.1f}%",      "threshold": "≤ 8%"},
            {"filter": "Cons. Range",   "passed": ind["cons_pct"] <= 10,     "active": True,
             "actual": f"{ind['cons_pct']:.1f}%",      "threshold": "≤ 10%"},
            {"filter": "IBS",           "passed": ind["ibs"] > 0.4,          "active": True,
             "actual": f"{ind['ibs']:.2f}",            "threshold": "> 0.4"},
            {"filter": "Overall Score", "passed": score >= min_score,        "active": True,
             "actual": f"{score:.0f}/100",             "threshold": f"≥ {min_score}"},
        ])

        rejection["scorecard"]       = scorecard
        rejection["failed_filters"]  = [s["filter"] for s in scorecard
                                         if not s["passed"] and "  ✓" not in s["filter"]]
        rejection["swing_pro_score"] = round(score, 1)
        rejection["swing_score"]     = round(score / 10, 1)  # map 0-100 to 0-10 for ⚡ column
        rejection["rsi"]             = ind["rsi"]
        rejection["price_now"]       = rejection["price_now"] or ind["price"]

        if score < min_score:
            return None, rejection

        # ── Enrichment — each step isolated so failures don't kill the result ──
        price  = ind["price"]
        high52 = info.get("fiftyTwoWeekHigh") or float(hist["High"].max())
        low52  = info.get("fiftyTwoWeekLow")  or float(hist["Low"].min())

        sent_score, sent_summary = 50, "N/A"
        if run_sentiment:
            try:
                sent_score, sent_summary = _sentiment_score(symbol, name)
            except Exception:
                sent_score, sent_summary = 50, "Sentiment unavailable"

        verdict_data = {"verdict": "—", "reason": "", "error": False}
        if run_verdict:
            try:
                verdict_data = _ai_verdict(symbol, name, sector, score, signals,
                                           ind, sent_score)
            except Exception:
                verdict_data = {"verdict": "N/A", "reason": "Verdict unavailable", "error": True}

        bt = None
        if run_backtest:
            try:
                bt = _backtest_stock(
                    hist["Close"].values, hist["High"].values,
                    hist["Low"].values,   hist["Volume"].values,
                    strategies, thresh,
                )
            except Exception:
                bt = None

        # Composite rank: 60% swing score + 40% sentiment
        rank_score = round(score * 0.6 + sent_score * 0.4, 1)

        close_series = hist["Close"]
        monthly_chg = round((price / float(close_series.iloc[-21]) - 1) * 100, 2) if len(close_series) >= 21 else None
        weekly_chg  = round((price / float(close_series.iloc[-5])  - 1) * 100, 2) if len(close_series) >= 5  else None

        result = {
            # Identity
            "ticker":            symbol,
            "name":              name,
            "sector":            sector,
            "exchange":          info.get("exchange", "—"),
            # Price data
            "price_now":         round(price, 2),
            "52w_high":          round(float(high52), 2),
            "52w_low":           round(float(low52),  2),
            "monthly_chg_pct":   monthly_chg,
            "weekly_chg_pct":    weekly_chg,
            "market_cap_b":      round((info.get("marketCap") or 0) / 1e9, 2),
            "pe":                info.get("trailingPE") or info.get("forwardPE"),
            "div_yield_pct":     _div_yield_pct(info),
            # Strategy scores
            "swing_pro_score":   score,
            "rank_score":        rank_score,
            "swing_pro_signals": signals,
            "strat_detail":      strat_detail,
            # Indicators
            "rsi":               ind["rsi"],
            "ema20":             ind["ema20"],
            "ema50":             ind["ema50"],
            "ema200":            ind["ema200"],
            "vol_ratio":         ind["vol_ratio"],
            "bb_width":          ind["bb_width"],
            "cons_pct":          ind["cons_pct"],
            "ibs":               ind["ibs"],
            "atr14":             ind["atr14"],
            "stop_loss_atr":     ind["stop_loss_atr"],
            "above_200":         ind["above_200"],
            "pct_ema20":         ind["pct_ema20"],
            "breakout":          ind["breakout"],
            # Sentiment
            "sentiment_score":   sent_score,
            "sentiment_summary": sent_summary,
            # AI verdict
            "verdict":           verdict_data["verdict"],
            "verdict_reason":    verdict_data["reason"],
            # Backtest
            "backtest":          bt,
            # Scorecard for rejections
            "scorecard":         scorecard,
            "failed_filters":    [],
        }
        return result, None

    except Exception as _ex:
        import traceback as _tb
        err_msg = str(_ex)[:200]
        tb_msg  = _tb.format_exc()[-300:]
        rejection["failed_filters"] = [f"Error: {err_msg}"]
        rejection["scorecard"] = [
            {"filter": "Processing Error", "passed": False, "active": True,
             "actual": err_msg, "threshold": "—"},
            {"filter": "Traceback", "passed": False, "active": True,
             "actual": tb_msg, "threshold": "—"},
        ]
        return None, rejection


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_swing_pro(tickers, strategies, thresh, allowed_sectors=None,
                  run_backtest=True, run_sentiment=True, run_verdict=True,
                  log_cb=None, reject_cb=None):
    """
    Run the full Swing Pro screen across all tickers.

    Parameters
    ----------
    tickers          : list of ticker symbols from the selected universe
    strategies       : dict of strategy enable flags
    thresh           : dict of threshold values
    allowed_sectors  : list of sector strings to restrict to (None = all)
    run_backtest     : whether to run the 6-month backtest per stock
    run_sentiment    : whether to fetch Groq sentiment scores
    run_verdict      : whether to fetch Groq AI verdicts
    log_cb           : optional logging callback

    Returns
    -------
    { "results": [...], "rejections": [...], "universe_size": int }
    """
    def log(m):
        if log_cb: log_cb(m)

    total      = len(tickers)
    results    = []
    rejections = []

    log(f"Swing Pro — {total} stocks in universe")
    log(f"Strategies: {[k for k,v in strategies.items() if v]}")

    # ── Adaptive parallel processing ─────────────────────────────────────────
    # Workers scale with universe size — small universes get more workers
    # A semaphore limits concurrent yfinance calls to prevent rate limiting
    import threading as _threading

    if total <= 150:
        workers = 4     # Top 100/200 — fast enough, low risk
    elif total <= 600:
        workers = 3     # NASDAQ 100, sector universes
    else:
        workers = 2     # NYSE, full NASDAQ — conservative to avoid bans

    # Semaphore limits how many stocks hit yfinance simultaneously
    _yf_sem = _threading.Semaphore(workers)

    log(f"Processing {total} stocks with {workers} workers...")

    _lock    = _threading.Lock()
    _done    = [0]

    def _screen_one(sym):
        with _yf_sem:          # at most `workers` concurrent yfinance calls
            try:
                result, rejection = screen_stock_swing_pro(
                    sym, strategies, thresh, allowed_sectors,
                    run_backtest, run_sentiment, run_verdict,
                )
            except Exception as _e:
                rejection = {
                    "ticker": sym, "name": sym, "sector": "—",
                    "price_now": None, "52w_high": None, "52w_low": None,
                    "failed_filters": [f"Unexpected: {str(_e)[:80]}"],
                    "scorecard": [{"filter": "Unexpected Error", "passed": False,
                                   "active": True, "actual": str(_e)[:80], "threshold": "—"}],
                }
                result = None

            # Brief per-stock sleep inside semaphore to pace yfinance requests
            time.sleep(0.3)
            return sym, result, rejection

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_screen_one, sym): sym for sym in tickers}
        for future in as_completed(futures):
            sym, result, rejection = future.result()

            with _lock:
                _done[0] += 1
                pct = int(_done[0] / total * 100)

            log(f"__progress__{pct}__{sym}__{len(results)}")

            if result:
                with _lock:
                    results.append(result)
                log(f"PASS {sym} — score:{result['swing_pro_score']} "
                    f"sentiment:{result['sentiment_score']} "
                    f"verdict:{result['verdict']}")
            elif rejection:
                with _lock:
                    rejections.append(rejection)
                if reject_cb:
                    try: reject_cb(rejection)
                    except Exception: pass

    # Sort by composite rank score descending
    results.sort(key=lambda r: r.get("rank_score", 0), reverse=True)

    log(f"Swing Pro complete — {total} screened · "
        f"{len(results)} passed · {len(rejections)} rejected")

    return {
        "results":       results,
        "rejections":    rejections,
        "universe_size": total,
    }