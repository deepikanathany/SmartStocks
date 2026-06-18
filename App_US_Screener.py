"""
app_us_screener.py
==================
US S&P 500 Intraday Screener  —  Port 5007
Run: python app_us_screener.py  →  http://localhost:5007

Backtest findings hardcoded (75,400 trades, 100 S&P 500 stocks, 29 days):
  · Friday = SHORT only (60.6% WR).  Long WR Friday = 35% — never long.
  · Monday = lean LONG (55.8% WR)
  · Regime: SPY today close vs open  →  BULL / BEAR
  · Bull day best LONG:  10:00 PT (63.2% WR), 11:00 (62.8%), 07:00 (62.4%)
  · Bear day best SHORT: 10:00 PT (64.2% WR), 08:00 (62.7%), 09:00 (60.5%)
  · Top long  combos: AAPL/FCX 12:00 PT (72.4%), MSFT/GOOGL 11:00 (69%), BAC 07:00 (69%)
  · Top short combos: DUK 10:00 PT (75.9%), NOC 11:00 PT (72.4%), CB/PG/COP (69%)
  · Top short sectors: Utilities 10:00 PT (64.7%), Energy 09:00 PT (59.8%), Real Estate 10:00 PT (58.6%)

Scoring 0-9 per stock:
  Technical (0-6): SPY regime, 1d return, 5d return, intraday vs open, near 52w high/low, EMA trend
  News (0-3):      Groq reads headlines, BULLISH/BEARISH/NEUTRAL + confidence
"""

from flask import Flask, Response, jsonify, render_template_string, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz, warnings, time, json, threading, uuid, requests
import xml.etree.ElementTree as ET
from groq import Groq

warnings.filterwarnings("ignore")

app = Flask(__name__)
PT_TZ = pytz.timezone("America/Los_Angeles")

GROQ_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"
STOP_PCT = 0.01
TARGET_PCT = 0.02
MAX_W = 8

# ── BACKTEST FINDINGS ─────────────────────────────────────────────────────────
DOW_BIAS = {
    0: {"dir": "LONG", "wr": 55.8, "note": "Monday — lean long (55.8% WR)"},
    1: {"dir": "NEUTRAL", "wr": 46.8, "note": "Tuesday — neutral, follow SPY regime"},
    2: {"dir": "SHORT", "wr": 51.3, "note": "Wednesday — lean short (51.3% WR)"},
    3: {"dir": "NEUTRAL", "wr": 51.1, "note": "Thursday — balanced, follow SPY regime"},
    4: {"dir": "SHORT", "wr": 60.6, "note": "Friday — SHORT ONLY (60.6% WR). Never long."},
}

BULL_LONG_TIMES = [
    {"slot": "10:00", "wr_bull": 63.2, "avg": 0.214},
    {"slot": "11:00", "wr_bull": 62.8, "avg": 0.210},
    {"slot": "07:00", "wr_bull": 62.4, "avg": 0.426},
    {"slot": "08:00", "wr_bull": 62.1, "avg": 0.344},
]
BEAR_SHORT_TIMES = [
    {"slot": "10:00", "wr_bear": 64.2, "avg": 0.218},
    {"slot": "08:00", "wr_bear": 62.7, "avg": 0.288},
    {"slot": "09:00", "wr_bear": 60.5, "avg": 0.188},
    {"slot": "11:00", "wr_bear": 60.2, "avg": 0.123},
]

# Per-stock best entry times (PT) from backtest top combos
# LONG — from top 15 stock+slot long combos
TOP_LONG_SLOTS = {
    "AAPL": "12:00", "MSFT": "12:00", "ADI": "12:00",  # Tech: 15:00 ET = 12:00 PT
    "FCX": "11:00", "NEM": "11:00", "SHW": "11:00", "LIN": "11:00",  # Materials: 14:00 ET
    "GOOGL": "11:00", "NFLX": "08:00", "META": "11:00", "DIS": "09:00",  # Comm: 14:00/11:00 ET
    "BAC": "07:00", "JPM": "07:00", "WFC": "07:00", "C": "07:00",  # Financials open: 10:00 ET
    "GS": "11:00", "MS": "11:00", "AXP": "11:00",  # Financials: 14:00 ET
    "AMZN": "09:00", "HD": "09:00", "MCD": "10:00", "COST": "10:00",  # Cons Disc: midday
    "LLY": "09:00", "UNH": "09:00", "ISRG": "09:00",  # Healthcare: 12:00 ET
    "ABBV": "10:00", "MRK": "10:00", "AMGN": "10:00",  # Healthcare: 13:00 ET
    "XOM": "07:00", "CVX": "07:00", "COP": "07:00",  # Energy: open
    "CAT": "07:00", "GE": "07:00", "HON": "09:00", "DE": "09:00",  # Industrials
    "NEE": "07:00", "SO": "07:00", "DUK": "10:00",  # Utilities
    "PLD": "07:00", "EQIX": "07:00", "AMT": "09:00",  # Real Estate
    "WMT": "10:00", "PG": "11:00", "KO": "10:00", "PEP": "10:00",  # Cons Staples
    "NVDA": "08:00", "AMD": "08:00", "QCOM": "09:00", "TXN": "09:00",  # Tech semis
    "TSLA": "08:00", "NKE": "09:00", "BKNG": "10:00",  # Cons Disc
}

# SHORT — from top 15 stock+slot short combos
TOP_SHORT_SLOTS = {
    "DUK": "10:00", "SO": "10:00", "NEE": "10:00", "AEP": "10:00",  # Utilities: 13:00 ET
    "NOC": "11:00", "RTX": "11:00", "GE": "11:00", "CAT": "11:00",  # Industrials: 14:00 ET
    "CB": "08:00", "AXP": "08:00", "MS": "08:00",  # Financials: 11:00 ET
    "AMGN": "06:30", "PG": "06:30", "KO": "06:30", "PEP": "06:30",  # Cons Staples: open
    "COP": "09:00", "XOM": "09:00", "CVX": "09:00", "SLB": "09:00",  # Energy: 12:00 ET
    "PLD": "10:00", "EQIX": "10:00", "AMT": "10:00", "PSA": "10:00",  # Real Estate: 13:00 ET
    "LLY": "10:00", "UNH": "10:00", "JNJ": "10:00", "ABBV": "10:00",  # Healthcare: 13:00 ET
    "JPM": "10:00", "BAC": "10:00", "WFC": "10:00", "GS": "10:00",  # Financials: 13:00 ET
    "GOOGL": "09:00", "META": "09:00", "NFLX": "09:00",  # Comm: 12:00 ET
    "AAPL": "09:00", "MSFT": "09:00", "NVDA": "09:00", "AMD": "09:00",  # Tech: 12:00 ET
    "TSLA": "08:00", "AMZN": "08:00", "HD": "08:00",  # Cons Disc: 11:00 ET
}

# Sector-level best times (PT) — fallback when stock not in above dicts
SECTOR_LONG_TIMES = {
    "Technology": {"slot": "10:00", "wr": 62.4},  # 13:00 ET bull
    "Financials": {"slot": "07:00", "wr": 62.4},  # 10:00 ET open
    "Healthcare": {"slot": "09:00", "wr": 61.8},  # 12:00 ET
    "Consumer Cyclical": {"slot": "09:00", "wr": 61.8},  # 12:00 ET
    "Communication Services": {"slot": "11:00", "wr": 62.8},  # 14:00 ET
    "Consumer Staples": {"slot": "07:00", "wr": 62.4},  # 10:00 ET
    "Energy": {"slot": "07:00", "wr": 62.4},  # 10:00 ET open
    "Industrials": {"slot": "07:00", "wr": 62.4},  # 10:00 ET
    "Utilities": {"slot": "10:00", "wr": 51.1},  # 13:00 ET
    "Real Estate": {"slot": "07:00", "wr": 62.4},  # 10:00 ET
    "Basic Materials": {"slot": "11:00", "wr": 62.8},  # 14:00 ET
}

SECTOR_SHORT_TIMES = {
    "Utilities": {"slot": "10:00", "wr": 64.7},  # 13:00 ET
    "Energy": {"slot": "09:00", "wr": 59.8},  # 12:00 ET
    "Real Estate": {"slot": "10:00", "wr": 58.6},  # 13:00 ET
    "Consumer Staples": {"slot": "11:00", "wr": 58.2},  # 14:00 ET
    "Industrials": {"slot": "12:00", "wr": 55.5},  # 15:00 ET
    "Healthcare": {"slot": "10:00", "wr": 53.2},  # 13:00 ET
    "Communication Services": {"slot": "09:00", "wr": 51.1},  # 12:00 ET
    "Financials": {"slot": "10:00", "wr": 49.0},  # 13:00 ET
    "Technology": {"slot": "09:00", "wr": 49.2},  # 12:00 ET
    "Consumer Cyclical": {"slot": "08:00", "wr": 50.0},  # 11:00 ET
    "Basic Materials": {"slot": "11:00", "wr": 52.0},  # 14:00 ET
}

# ── UNIVERSE ──────────────────────────────────────────────────────────────────
SP500 = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "ACN", "AMD",
    "TXN", "QCOM", "INTU", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "FTNT",
    "MCHP", "ADI", "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "BKNG", "TJX",
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "BRK-B", "JPM", "V", "MA",
    "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "MCO", "CB",
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN", "ISRG",
    "VRTX", "REGN", "BSX", "GILD", "WMT", "PG", "KO", "PEP", "COST", "PM",
    "MO", "MDLZ", "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "GE", "CAT",
    "HON", "UNP", "RTX", "DE", "ETN", "EMR", "NOC", "ITW", "NEE", "SO",
    "DUK", "AEP", "PLD", "EQIX", "AMT", "PSA", "LIN", "FCX", "NEM", "SHW",
]

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology",
    "ORCL": "Technology", "CSCO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "ACN": "Technology", "AMD": "Technology", "TXN": "Technology", "QCOM": "Technology",
    "INTU": "Technology", "AMAT": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "SNPS": "Technology", "CDNS": "Technology", "PANW": "Technology", "FTNT": "Technology",
    "MCHP": "Technology", "ADI": "Technology",
    "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", "HD": "Consumer Cyclical",
    "MCD": "Consumer Cyclical", "NKE": "Consumer Cyclical", "LOW": "Consumer Cyclical",
    "BKNG": "Consumer Cyclical", "TJX": "Consumer Cyclical",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BAC": "Financials", "WFC": "Financials", "GS": "Financials", "MS": "Financials",
    "C": "Financials", "AXP": "Financials", "BLK": "Financials", "SPGI": "Financials",
    "MCO": "Financials", "CB": "Financials",
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "AMGN": "Healthcare", "ISRG": "Healthcare", "VRTX": "Healthcare", "REGN": "Healthcare",
    "BSX": "Healthcare", "GILD": "Healthcare",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy", "SLB": "Energy", "MPC": "Energy",
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "RTX": "Industrials", "DE": "Industrials", "ETN": "Industrials", "EMR": "Industrials",
    "NOC": "Industrials", "ITW": "Industrials",
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "AEP": "Utilities",
    "PLD": "Real Estate", "EQIX": "Real Estate", "AMT": "Real Estate", "PSA": "Real Estate",
    "LIN": "Basic Materials", "FCX": "Basic Materials", "NEM": "Basic Materials", "SHW": "Basic Materials",
}

_jobs = {};
_lock = threading.Lock()


# ── DATA HELPERS ──────────────────────────────────────────────────────────────

def clean_d(h):
    if h is None or h.empty: return None
    try:
        h.index = pd.to_datetime(h.index).tz_localize(None)
    except:
        h.index = pd.to_datetime(h.index).tz_convert(None)
    return h


def fetch_daily(sym):
    try:
        h = yf.Ticker(sym).history(period="60d", interval="1d")
        return sym, clean_d(h)
    except:
        return sym, None


def fetch_intra(sym):
    istart = (date.today() - timedelta(days=57)).strftime("%Y-%m-%d")
    iend = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
    for iv in ["30m", "1h"]:
        try:
            h = yf.Ticker(sym).history(start=istart, end=iend, interval=iv)
            if h is None or h.empty: continue
            try:
                h.index = h.index.tz_localize(PT_TZ)
            except:
                h.index = h.index.tz_convert(PT_TZ)
            h["_d"] = h.index.strftime("%Y-%m-%d")
            h["_t"] = h.index.strftime("%H:%M")
            if len(h) > 5: return sym, h
        except:
            pass
    return sym, None


# ── REGIME ────────────────────────────────────────────────────────────────────

def get_spy_regime():
    """SPY today: BULL if close > open, BEAR otherwise."""
    try:
        info = yf.Ticker("SPY").info or {}
        price = float(info.get("regularMarketPrice") or info.get("currentPrice") or 0)
        open_ = float(info.get("regularMarketOpen") or 0)
        if price and open_:
            return ("BULL" if price > open_ else "BEAR"), round(price, 2), round(open_, 2)
        h = clean_d(yf.Ticker("SPY").history(period="5d", interval="1d"))
        if h is not None and len(h) >= 1:
            cl = float(h["Close"].iloc[-1]);
            op = float(h["Open"].iloc[-1])
            return ("BULL" if cl > op else "BEAR"), round(cl, 2), round(op, 2)
    except:
        pass
    return "UNKNOWN", 0, 0


# ── DAY RECOMMENDATION ────────────────────────────────────────────────────────

def get_day_rec():
    now = datetime.now(PT_TZ)
    dow = now.weekday()
    bias = DOW_BIAS.get(dow, {"dir": "NEUTRAL", "wr": 50, "note": "—"})
    regime, spy_price, spy_open = get_spy_regime()

    if dow == 4:  # Friday — always short
        direction = "SHORT"
        times = BEAR_SHORT_TIMES
        note = bias["note"]
    elif dow == 0:  # Monday — lean long
        direction = "LONG"
        times = BULL_LONG_TIMES
        note = bias["note"] + (f" · SPY {regime}")
    elif regime == "BEAR":
        direction = "SHORT"
        times = BEAR_SHORT_TIMES
        note = f"{bias['note']} · SPY BEAR today"
    else:
        direction = "LONG"
        times = BULL_LONG_TIMES
        note = f"{bias['note']} · SPY BULL today"

    return {
        "dow": now.strftime("%A"),
        "dow_n": dow,
        "regime": regime,
        "spy_price": spy_price,
        "spy_open": spy_open,
        "direction": direction,
        "note": note,
        "day_wr": bias["wr"],
        "entry_times": times,
        "best_time": times[0]["slot"] if times else "10:00",
        "best_wr": times[0].get("wr_bull") or times[0].get("wr_bear", "?") if times else "?",
    }


# ── TECHNICAL SCORE 0-6 ───────────────────────────────────────────────────────

def tech_score(sym, dh, ih, direction, regime):
    score = 0;
    det = {}

    # T1: SPY regime aligned
    if (direction == "SHORT" and regime == "BEAR") or (direction == "LONG" and regime == "BULL"):
        score += 1;
        det["t1"] = "✓"
    else:
        det["t1"] = "✗"

    if dh is None or len(dh) < 6:
        return score, det

    cl = dh["Close"];
    hi = dh["High"];
    lo_d = dh["Low"]
    price = round(float(cl.iloc[-1]), 4)
    w52h = round(float(hi.max()), 2)
    w52l = round(float(lo_d.min()), 2)
    det.update({"price": price, "w52h": w52h, "w52l": w52l})

    # T2: 1-day return
    if len(cl) >= 2:
        r1 = float(cl.iloc[-1] / cl.iloc[-2] - 1)
        det["r1d"] = round(r1 * 100, 2)
        if (direction == "SHORT" and r1 < 0) or (direction == "LONG" and r1 > 0):
            score += 1;
            det["t2"] = "✓"
        else:
            det["t2"] = "✗"

    # T3: 5-day return
    if len(cl) >= 6:
        r5 = float(cl.iloc[-1] / cl.iloc[-6] - 1)
        det["r5d"] = round(r5 * 100, 2)
        if (direction == "SHORT" and r5 < 0) or (direction == "LONG" and r5 > 0):
            score += 1;
            det["t3"] = "✓"
        else:
            det["t3"] = "✗"

    # T4/T5/T6 from intraday
    if ih is not None:
        today = str(date.today())
        day = ih[ih["_d"] == today]
        open9 = day[day["_t"] == "06:30"]

        if not day.empty and not open9.empty:
            pnow = float(day.iloc[-1]["Close"])
            popen = float(open9.iloc[0]["Open"])
            ri = pnow / popen - 1
            det["r_intra"] = round(ri * 100, 2)
            det["price_now"] = round(pnow, 2)
            if (direction == "SHORT" and ri < 0) or (direction == "LONG" and ri > 0):
                score += 1;
                det["t4"] = "✓"
            else:
                det["t4"] = "✗"

        # T5: EMA trend
        if len(cl) >= 50:
            ema20 = float(cl.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(cl.ewm(span=50, adjust=False).mean().iloc[-1])
            if direction == "LONG" and price > ema20 > ema50:
                score += 1; det["t5"] = "✓"
            elif direction == "SHORT" and price < ema20 < ema50:
                score += 1; det["t5"] = "✓"
            else:
                det["t5"] = "✗"

        # T6: near 52w high (short) or 52w low (long)
        if price and w52h and w52l:
            dist_high = (w52h - price) / w52h * 100
            dist_low = (price - w52l) / w52l * 100
            det["dist_high"] = round(dist_high, 1);
            det["dist_low"] = round(dist_low, 1)
            if direction == "SHORT" and dist_high <= 5:
                score += 1;
                det["t6"] = "✓"
            elif direction == "LONG" and dist_low <= 10:
                score += 1;
                det["t6"] = "✓"
            else:
                det["t6"] = "✗"

    return score, det


# ── NEWS SCORE 0-3 ────────────────────────────────────────────────────────────

def news_score(sym, name, direction):
    headlines = []
    try:
        after = (date.today() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = (f"https://news.google.com/rss/search?q="
               f"{requests.utils.quote(name + ' stock NYSE NASDAQ after:' + after)}"
               f"&hl=en-US&gl=US&ceid=US:en")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.ok:
            for item in ET.fromstring(r.content).findall(".//item")[:5]:
                t2 = item.findtext("title", "").strip()
                if len(t2) > 10: headlines.append(t2)
    except:
        pass
    if not headlines:
        try:
            for n in (yf.Ticker(sym).news or [])[:4]:
                if n.get("title"): headlines.append(n["title"])
        except:
            pass
    if not headlines:
        return {"sentiment": "NEUTRAL", "conf": 5, "summary": "No news found", "score": 1}
    block = "\n".join(f"- {h}" for h in headlines[:5])
    try:
        raw = Groq(api_key=GROQ_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"US Stock: {name} ({sym}) S&P 500\nRecent headlines:\n{block}\n"
                f"JSON only, no markdown: {{\"sentiment\":\"BULLISH\",\"confidence\":7,\"summary\":\"max 8 words\"}}"}],
            temperature=0, max_tokens=80,
        ).choices[0].message.content.strip()
        p = json.loads(raw.replace("```json", "").replace("```", "").strip())
        sent = p.get("sentiment", "NEUTRAL").upper()
        conf = int(p.get("confidence", 5))
        summ = p.get("summary", "")[:55]
        ns = 2 if (direction == "SHORT" and sent == "BEARISH") or (direction == "LONG" and sent == "BULLISH") \
            else 1 if sent == "NEUTRAL" else 0
        if conf >= 8: ns = min(ns + 1, 3)
        return {"sentiment": sent, "conf": conf, "summary": summ, "score": ns}
    except:
        return {"sentiment": "NEUTRAL", "conf": 0, "summary": "Parse error", "score": 1}


# ── BACKGROUND JOB ────────────────────────────────────────────────────────────

def _log(jid, msg, kind="info"):
    with _lock: _jobs[jid]["log"].append({"msg": msg, "kind": kind})


def _td(d, k): return d.get(k, "?") if d else "?"


def _run(jid):
    try:
        _log(jid, "=" * 54, "info")
        _log(jid, "US S&P 500 SCREENER", "title")

        _log(jid, "STEP 1 — Day & SPY regime", "phase")
        day_rec = get_day_rec()
        direction = day_rec["direction"]
        regime = day_rec["regime"]
        dow = day_rec["dow"]
        best_t = day_rec["best_time"]
        best_wr = day_rec["best_wr"]

        _log(jid, f"  {dow}  |  SPY {regime} (${day_rec['spy_price']} vs open ${day_rec['spy_open']})", "info")
        _log(jid, f"  Direction: {direction}", "match" if direction == "LONG" else "error")
        _log(jid, f"  {day_rec['note']}", "info")
        _log(jid, f"  Best entry: {best_t} PT  ({best_wr}% WR from backtest)", "match")

        if day_rec["dow_n"] == 4:
            _log(jid, "  ★ FRIDAY — SHORT ONLY (60.6% WR). Do NOT go long today.", "error")

        with _lock:
            _jobs[jid]["day_rec"] = day_rec

        _log(jid, "=" * 54, "info")
        _log(jid, "STEP 2 — Fetching 100 S&P 500 stocks", "phase")
        daily_data = {};
        intra_data = {}
        with ThreadPoolExecutor(max_workers=MAX_W) as ex:
            fd = {ex.submit(fetch_daily, s): s for s in SP500}
            fi = {ex.submit(fetch_intra, s): s for s in SP500}
            for f in as_completed(fd):
                sym, h = f.result()
                if h is not None: daily_data[sym] = h
            for f in as_completed(fi):
                sym, h = f.result()
                if h is not None: intra_data[sym] = h
        _log(jid, f"  Daily: {len(daily_data)}/100  |  Intraday: {len(intra_data)}/100", "info")

        _log(jid, "=" * 54, "info")
        _log(jid, "STEP 3 — Scoring + Groq news", "phase")

        scored = []
        for sym in SP500:
            dh = daily_data.get(sym);
            ih = intra_data.get(sym)
            sector = SECTOR_MAP.get(sym, "Unknown")
            _log(jid, f"  {sym}...", "info")
            try:
                # Score BOTH directions — user can toggle in UI
                ts_long, tdet_long = tech_score(sym, dh, ih, "LONG", regime)
                ts_short, tdet_short = tech_score(sym, dh, ih, "SHORT", regime)
                ns_long = news_score(sym, sym, "LONG")
                ns_short = news_score(sym, sym, "SHORT")
                time.sleep(0.2)

                total_long = ts_long + ns_long["score"]
                total_short = ts_short + ns_short["score"]

                s_long = SECTOR_LONG_TIMES.get(sector, {})
                s_short = SECTOR_SHORT_TIMES.get(sector, {})
                bt_long = TOP_LONG_SLOTS.get(sym, s_long.get("slot", BULL_LONG_TIMES[0]["slot"]))
                bt_short = TOP_SHORT_SLOTS.get(sym, s_short.get("slot", BEAR_SHORT_TIMES[0]["slot"]))

                price = tdet_long.get("price_now") or tdet_long.get("price")
                if price:
                    p = round(price, 2)
                    entry_long = p;
                    stop_long = round(p * (1 - STOP_PCT), 2);
                    target_long = round(p * (1 + TARGET_PCT), 2)
                    entry_short = p;
                    stop_short = round(p * (1 + STOP_PCT), 2);
                    target_short = round(p * (1 - TARGET_PCT), 2)
                else:
                    p = entry_long = stop_long = target_long = entry_short = stop_short = target_short = None

                scored.append({
                    "sym": sym, "sector": sector, "price": p,
                    "r1d": tdet_long.get("r1d"), "r5d": tdet_long.get("r5d"),
                    "r_intra": tdet_long.get("r_intra"),
                    "w52h": tdet_long.get("w52h"), "w52l": tdet_long.get("w52l"),
                    "regime": regime,
                    # LONG
                    "total_long": total_long, "tech_long": ts_long,
                    "news_sc_long": ns_long["score"], "sentiment_long": ns_long["sentiment"],
                    "conf_long": ns_long["conf"], "news_summary_long": ns_long["summary"],
                    "entry_long": entry_long, "stop_long": stop_long, "target_long": target_long,
                    "bt_time_long": bt_long,
                    "t1l": _td(tdet_long, "t1"), "t2l": _td(tdet_long, "t2"), "t3l": _td(tdet_long, "t3"),
                    "t4l": _td(tdet_long, "t4"), "t5l": _td(tdet_long, "t5"), "t6l": _td(tdet_long, "t6"),
                    # SHORT
                    "total_short": total_short, "tech_short": ts_short,
                    "news_sc_short": ns_short["score"], "sentiment_short": ns_short["sentiment"],
                    "conf_short": ns_short["conf"], "news_summary_short": ns_short["summary"],
                    "entry_short": entry_short, "stop_short": stop_short, "target_short": target_short,
                    "bt_time_short": bt_short,
                    "t1s": _td(tdet_short, "t1"), "t2s": _td(tdet_short, "t2"), "t3s": _td(tdet_short, "t3"),
                    "t4s": _td(tdet_short, "t4"), "t5s": _td(tdet_short, "t5"), "t6s": _td(tdet_short, "t6"),
                })
            except Exception as se:
                _log(jid, f"  {sym} skipped — {str(se)[:60]}", "error")
                continue

        scored_long = sorted(scored, key=lambda x: (-x["total_long"], -(x.get("r_intra") or 0)))
        scored_short = sorted(scored, key=lambda x: (-x["total_short"], (x.get("r_intra") or 0)))

        _log(jid, "=" * 54, "info")
        _log(jid, "TOP 10 LONG", "done_msg")
        for i, s2 in enumerate(scored_long[:10], 1):
            ri = f"{s2['r_intra']:+.1f}%" if s2["r_intra"] is not None else "—"
            _log(jid,
                 f"  #{i} {s2['sym']:<8} LongScore:{s2['total_long']}/9  Intra:{ri}  Entry:{s2['bt_time_long']} PT",
                 "match" if i <= 3 else "info")
        _log(jid, "TOP 10 SHORT", "done_msg")
        for i, s2 in enumerate(scored_short[:10], 1):
            ri = f"{s2['r_intra']:+.1f}%" if s2["r_intra"] is not None else "—"
            _log(jid,
                 f"  #{i} {s2['sym']:<8} ShortScore:{s2['total_short']}/9  Intra:{ri}  Entry:{s2['bt_time_short']} PT",
                 "match" if i <= 3 else "info")

        with _lock:
            _jobs[jid]["results_long"] = scored_long
            _jobs[jid]["results_short"] = scored_short
            _jobs[jid]["done"] = True

    except Exception as e:
        _log(jid, f"Error: {e}", "error")
        with _lock:
            _jobs[jid]["done"] = True


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index(): return render_template_string(HTML)


@app.route("/api/run", methods=["POST"])
def run_scan():
    jid = str(uuid.uuid4())[:8]
    with _lock: _jobs[jid] = {"log": [], "results_long": [], "results_short": [], "day_rec": {}, "done": False}
    threading.Thread(target=_run, args=(jid,), daemon=True).start()
    return jsonify({"job_id": jid})


# ── TICKER SEARCH — company name → symbol suggestions ─────────────────────────

@app.route("/api/search_ticker", methods=["POST"])
def search_ticker():
    """
    Takes a company name (e.g. "Apple" or "Bank of America") and returns
    up to 5 matching US stock symbols with company names and current price.
    Uses Yahoo Finance search API.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query or len(query) < 2:
        return jsonify({"results": [], "error": "Query too short"})

    try:
        # Yahoo Finance search endpoint
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}&quotesCount=8&newsCount=0&listsCount=0"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        raw = resp.json()
        quotes = raw.get("quotes", [])

        results = []
        for q in quotes:
            # Only US equity — filter out ETFs, mutual funds, indices
            qtype = q.get("quoteType", "")
            exchange = q.get("exchange", "")
            sym = q.get("symbol", "")
            name = q.get("longname") or q.get("shortname") or sym

            if qtype not in ("EQUITY",):
                continue
            # Only NYSE/NASDAQ
            if exchange not in ("NYQ", "NMS", "NGM", "NCM", "NYSEArca"):
                continue
            if not sym or "." in sym:  # skip foreign listings
                continue

            # Get current price
            try:
                price = round(float(yf.Ticker(sym).fast_info.get("lastPrice", 0) or 0), 2)
            except Exception:
                price = None

            results.append({
                "symbol":   sym,
                "name":     name,
                "exchange": exchange,
                "price":    price,
            })

            if len(results) >= 5:
                break

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"results": [], "error": str(e)[:80]})


# ── NOVICE SCREENER — swing + value analysis for one stock ────────────────────

def _run_novice(jid, symbol, company_name):
    """
    Runs swing (1 week–1 month) and value (6 months–1 year) analysis
    on a single stock and generates teen-friendly Groq explanations.
    """
    def log(msg, kind="info"):
        _log(jid, msg, kind)

    try:
        log("=" * 50, "info")
        log(f"NOVICE SCREENER — {company_name} ({symbol})", "title")
        log("=" * 50, "info")

        # ── Fetch data ────────────────────────────────────────────────────────
        log("Fetching stock data...", "phase")
        dh = fetch_daily(symbol)[1]
        ih = fetch_intra(symbol)[1]

        if dh is None or len(dh) < 10:
            log(f"Could not load data for {symbol}. Try another stock.", "error")
            with _lock:
                _jobs[jid]["done"] = True
            return

        cl   = dh["Close"]
        hi   = dh["High"]
        lo   = dh["Low"]
        vol  = dh["Volume"]
        price= round(float(cl.iloc[-1]), 2)

        log(f"  {company_name} ({symbol}) — Current price: ${price}", "match")

        # ── Basic indicators ──────────────────────────────────────────────────
        r1d  = round(float(cl.iloc[-1] / cl.iloc[-2] - 1) * 100, 2) if len(cl) >= 2 else 0
        r5d  = round(float(cl.iloc[-1] / cl.iloc[-6] - 1) * 100, 2) if len(cl) >= 6 else 0
        r1m  = round(float(cl.iloc[-1] / cl.iloc[-22] - 1) * 100, 2) if len(cl) >= 22 else 0
        r3m  = round(float(cl.iloc[-1] / cl.iloc[-66] - 1) * 100, 2) if len(cl) >= 66 else 0
        r6m  = round(float(cl.iloc[-1] / cl.iloc[-126] - 1) * 100, 2) if len(cl) >= 126 else 0

        w52h = round(float(hi.max()), 2)
        w52l = round(float(lo.min()), 2)
        dist_high = round((w52h - price) / w52h * 100, 1)
        dist_low  = round((price - w52l) / w52l * 100, 1)

        # EMA trend
        ema20 = round(float(cl.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
        ema50 = round(float(cl.ewm(span=50, adjust=False).mean().iloc[-1]), 2)
        ema200= round(float(cl.ewm(span=200, adjust=False).mean().iloc[-1]), 2) if len(cl) >= 200 else None

        # RSI
        delta = cl.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-9)
        rsi   = round(float(100 - 100 / (1 + rs.iloc[-1])), 1)

        # Volume trend
        avg_vol_20 = round(float(vol.iloc[-20:].mean()), 0) if len(vol) >= 20 else 0
        last_vol   = int(vol.iloc[-1])
        vol_ratio  = round(last_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

        # PE ratio and dividend yield from yfinance info
        try:
            info     = yf.Ticker(symbol).info or {}
            pe       = info.get("trailingPE") or info.get("forwardPE")
            pe       = round(float(pe), 1) if pe else None
            div_yield= round(float(info.get("dividendYield", 0) or 0) * 100, 2)
            mktcap   = info.get("marketCap")
            mktcap_b = round(mktcap / 1e9, 1) if mktcap else None
            sector   = info.get("sector", "Unknown")
            industry = info.get("industry", "")
            eps      = info.get("trailingEps")
        except Exception:
            pe = div_yield = mktcap_b = eps = None
            sector = "Unknown"; industry = ""

        log(f"  RSI: {rsi}  |  EMA20: ${ema20}  |  EMA50: ${ema50}", "info")
        log(f"  1d: {r1d:+.1f}%  5d: {r5d:+.1f}%  1m: {r1m:+.1f}%  6m: {r6m:+.1f}%", "info")
        log(f"  52w range: ${w52l} — ${w52h}  |  PE: {pe}  |  Div: {div_yield}%", "info")

        # ── SWING SCORE (1 week – 1 month) ───────────────────────────────────
        log("Calculating swing score (1 week – 1 month)...", "phase")

        swing_score   = 0
        swing_signals = []
        swing_flags   = []  # things that look bad

        # Price above both EMAs — uptrend
        if price > ema20 > ema50:
            swing_score += 2
            swing_signals.append("Price is above its short-term and medium-term moving averages — the trend is going up")
        elif price > ema20:
            swing_score += 1
            swing_signals.append("Price is above its short-term moving average — slight upward momentum")
        else:
            swing_flags.append("Price is below its moving averages — the stock might keep going down short term")

        # RSI — not overbought, not oversold
        if 40 <= rsi <= 65:
            swing_score += 2
            swing_signals.append(f"RSI is {rsi} — the stock isn't too hot or too cold, good zone to buy")
        elif rsi < 40:
            swing_score += 1
            swing_signals.append(f"RSI is {rsi} — the stock has sold off a lot, could bounce back")
            swing_flags.append("RSI below 40 — the stock is weak right now, wait for it to stabilise")
        elif rsi > 70:
            swing_flags.append(f"RSI is {rsi} — the stock is overbought, might pull back soon")

        # 5-day momentum
        if r5d > 1:
            swing_score += 1
            swing_signals.append(f"Up {r5d:.1f}% this week — good short-term momentum")
        elif r5d < -3:
            swing_flags.append(f"Down {abs(r5d):.1f}% this week — not a great time to jump in for a short trade")

        # Volume confirmation
        if vol_ratio >= 1.2:
            swing_score += 1
            swing_signals.append(f"Volume is {vol_ratio:.1f}x the normal level — more people are buying/selling, which means the move is real")
        elif vol_ratio < 0.7:
            swing_flags.append("Volume is very low — nobody is interested right now, the price move might not last")

        # Distance from 52w high
        if dist_high <= 8:
            swing_score += 1
            swing_signals.append(f"Only {dist_high}% below its 52-week high — the stock is in strong territory")
        elif dist_high >= 30:
            swing_flags.append(f"Down {dist_high}% from its yearly high — it's struggled a lot recently")

        # 1-month trend
        if r1m > 2:
            swing_score += 1
            swing_signals.append(f"Up {r1m:.1f}% this month — the trend is your friend")
        elif r1m < -5:
            swing_flags.append(f"Down {abs(r1m):.1f}% this month — short-term trend is not great")

        swing_score = min(swing_score, 10)

        # Verdict
        if swing_score >= 7:
            swing_verdict = "GOOD"
        elif swing_score >= 4:
            swing_verdict = "MIXED"
        else:
            swing_verdict = "RISKY"

        # ── VALUE SCORE (6 months – 1 year) ──────────────────────────────────
        log("Calculating value score (6 months – 1 year)...", "phase")

        value_score   = 0
        value_signals = []
        value_flags   = []

        # PE ratio
        if pe is not None:
            if pe < 15:
                value_score += 2
                value_signals.append(f"PE ratio is {pe} — this means you pay ${pe} for every $1 the company earns. Under 15 is a bargain!")
            elif pe < 25:
                value_score += 1
                value_signals.append(f"PE ratio is {pe} — reasonably priced for this kind of company")
            elif pe > 40:
                value_flags.append(f"PE ratio is {pe} — this is expensive. You're paying a lot compared to what the company earns right now")
        else:
            value_flags.append("No PE ratio available — could mean the company isn't profitable yet")

        # Dividend
        if div_yield >= 2:
            value_score += 2
            value_signals.append(f"Pays a {div_yield}% dividend — the company actually pays YOU just for owning the stock!")
        elif div_yield >= 0.5:
            value_score += 1
            value_signals.append(f"Pays a small {div_yield}% dividend — a little bonus for holding it")
        else:
            value_flags.append("No dividend — the company keeps all its profits to grow the business (not always bad)")

        # EMA200 — long term trend
        if ema200 is not None:
            if price > ema200:
                value_score += 2
                value_signals.append(f"Price is above its 200-day average (${ema200}) — the long-term trend is still up")
            else:
                value_flags.append(f"Price is below its 200-day average (${ema200}) — the long-term trend is pointing down")

        # 6-month return
        if r6m > 5:
            value_score += 1
            value_signals.append(f"Up {r6m:.1f}% in the last 6 months — growing nicely")
        elif r6m < -15:
            value_flags.append(f"Down {abs(r6m):.1f}% in 6 months — the stock has had a tough stretch")

        # Distance from 52w low — is it beaten down?
        if dist_low <= 15:
            value_flags.append(f"Only {dist_low}% above its yearly low — the stock is near its worst price of the year")
        elif dist_low >= 50:
            value_score += 1
            value_signals.append(f"Well above its 52-week low — the company has been growing steadily")

        # Market cap — stability
        if mktcap_b and mktcap_b >= 50:
            value_score += 1
            value_signals.append(f"Market cap ${mktcap_b}B — this is a large, established company (more stable than a small startup)")
        elif mktcap_b and mktcap_b < 5:
            value_flags.append(f"Market cap only ${mktcap_b}B — small company, higher risk but higher potential reward")

        value_score = min(value_score, 10)

        if value_score >= 7:
            value_verdict = "GOOD"
        elif value_score >= 4:
            value_verdict = "MIXED"
        else:
            value_verdict = "RISKY"

        # ── PIVOT POINT ANALYSIS ──────────────────────────────────────────────
        log("Checking weekly, monthly and yearly pivots...", "phase")

        def _calc_pivot(h, l, c):
            """Standard pivot: P = (H+L+C)/3, S1/R1 around it."""
            p  = round((h + l + c) / 3, 2)
            r1 = round(2 * p - l, 2)
            s1 = round(2 * p - h, 2)
            r2 = round(p + (h - l), 2)
            s2 = round(p - (h - l), 2)
            return {"pivot": p, "r1": r1, "r2": r2, "s1": s1, "s2": s2}

        def _near(price, level, pct=1.5):
            """True if price is within pct% of a pivot level."""
            return abs(price - level) / level * 100 <= pct

        def _avg_vol(vol_series, n=5):
            return float(vol_series.iloc[-n:].mean()) if len(vol_series) >= n else float(vol_series.mean())

        pivots = {}

        # Weekly pivot — last 5 trading days
        if len(hi) >= 5:
            wh = float(hi.iloc[-5:].max())
            wl = float(lo.iloc[-5:].min())
            wc = float(cl.iloc[-5])          # close at start of week
            pivots["weekly"] = _calc_pivot(wh, wl, wc)

        # Monthly pivot — last 22 trading days
        if len(hi) >= 22:
            mh = float(hi.iloc[-22:].max())
            ml = float(lo.iloc[-22:].min())
            mc = float(cl.iloc[-22])
            pivots["monthly"] = _calc_pivot(mh, ml, mc)

        # Yearly pivot — last 252 trading days
        if len(hi) >= 252:
            yh = float(hi.iloc[-252:].max())
            yl = float(lo.iloc[-252:].min())
            yc = float(cl.iloc[-252])
            pivots["yearly"] = _calc_pivot(yh, yl, yc)

        # Check if current price has rebounded from any pivot level
        pivot_hits = []
        avg_vol_5  = _avg_vol(vol, 5)

        for timeframe, pv in pivots.items():
            for label, level in [("Pivot", pv["pivot"]),
                                  ("Support 1 (S1)", pv["s1"]),
                                  ("Support 2 (S2)", pv["s2"]),
                                  ("Resistance 1 (R1)", pv["r1"]),
                                  ("Resistance 2 (R2)", pv["r2"])]:
                if not _near(price, level):
                    continue

                # Volume check — is recent vol above 5-day average?
                vol_strong = vol_ratio >= 1.15

                # Direction — did price bounce UP from support or hold at resistance?
                at_support    = label.startswith("Support") or label == "Pivot"
                at_resistance = label.startswith("Resistance")
                bounced_up    = r1d > 0 and at_support
                held_resist   = r1d < 0 and at_resistance

                if bounced_up or held_resist or _near(price, level, pct=0.8):
                    pivot_hits.append({
                        "timeframe":  timeframe,
                        "level_name": label,
                        "level":      round(level, 2),
                        "price":      price,
                        "dist_pct":   round(abs(price - level) / level * 100, 2),
                        "vol_strong": vol_strong,
                        "vol_ratio":  vol_ratio,
                        "bounced_up": bounced_up,
                        "r1d":        r1d,
                    })

        # Build plain-text pivot summary for Groq
        if pivot_hits:
            pivot_lines = []
            for h_ in pivot_hits:
                action = "bounced UP from" if h_["bounced_up"] else "sitting near"
                vol_note = f"volume {h_['vol_ratio']:.1f}x normal" if h_["vol_strong"] else "average volume"
                pivot_lines.append(
                    f"- {h_['timeframe'].capitalize()} {h_['level_name']} at ${h_['level']}: "
                    f"price is {action} this level ({h_['dist_pct']:.1f}% away), {vol_note}"
                )
            pivot_text = "\n".join(pivot_lines)
            log(f"  Pivot hits found: {len(pivot_hits)}", "match")
            for h_ in pivot_hits:
                log(f"    {h_['timeframe']} {h_['level_name']} ${h_['level']} "
                    f"{'↑ BOUNCE' if h_['bounced_up'] else '~near'} "
                    f"vol {h_['vol_ratio']:.1f}x", "match")
        else:
            pivot_text = "- Price is not near any key weekly, monthly or yearly pivot level right now."
            log("  No significant pivot touches at current price", "info")

        # ── GROQ — teen-friendly explanation ─────────────────────────────────
        log("Asking Groq AI for a plain-English explanation...", "phase")

        # Fetch recent headlines for context
        headlines = []
        try:
            after = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
            url   = (f"https://news.google.com/rss/search?q="
                     f"{requests.utils.quote(company_name + ' stock after:' + after)}"
                     f"&hl=en-US&gl=US&ceid=US:en")
            resp  = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
            if resp.ok:
                for item in ET.fromstring(resp.content).findall(".//item")[:4]:
                    t = item.findtext("title", "").strip()
                    if len(t) > 10:
                        headlines.append(t)
        except Exception:
            pass
        if not headlines:
            try:
                for n in (yf.Ticker(symbol).news or [])[:3]:
                    if n.get("title"):
                        headlines.append(n["title"])
            except Exception:
                pass

        news_block  = "\n".join(f"- {h}" for h in headlines[:4]) or "No recent headlines found."

        swing_sigs_text  = "\n".join(f"- {s}" for s in swing_signals)  or "- No strong signals"
        swing_flags_text = "\n".join(f"- {f}" for f in swing_flags)    or "- None"
        value_sigs_text  = "\n".join(f"- {v}" for v in value_signals)  or "- No strong signals"
        value_flags_text = "\n".join(f"- {v}" for v in value_flags)    or "- None"

        groq_prompt = f"""You are explaining stocks to a teenager who is learning about investing for the first time.

Company: {company_name} ({symbol})
Current price: ${price}
Sector: {sector}

SWING ANALYSIS (should I buy for 1 week to 1 month?):
Score: {swing_score}/10  |  Verdict: {swing_verdict}
Good signs:
{swing_sigs_text}
Warning signs:
{swing_flags_text}

VALUE ANALYSIS (should I buy and hold for 6 months to 1 year?):
Score: {value_score}/10  |  Verdict: {value_verdict}
Good signs:
{value_sigs_text}
Warning signs:
{value_flags_text}

PIVOT POINT ANALYSIS (key price levels the stock is near):
{pivot_text}

Recent news:
{news_block}

Write THREE short paragraphs using simple, friendly language (like talking to a smart 15-year-old):

1. SWING (1 week to 1 month): Should they buy for a short trade? Why or why not? 3-4 sentences.
2. VALUE (6 months to 1 year): Should they buy and hold? Mention PE ratio, dividend if any, long-term trend. 3-4 sentences.
3. PIVOTS: Explain in plain words what a pivot point is (like a floor or ceiling the stock price bounces off). Then explain what the pivot data above means for this stock right now — is the stock near a support level (good place to bounce up) or a resistance level (wall that might stop it going higher)? Does the volume make this more or less convincing? Keep it to 3-4 sentences. If there are no pivot hits, say the stock is not near any key level right now and that's okay.

Rules:
- No jargon without explaining it in plain words
- Use real numbers but always say what they mean
- Be honest — if it looks risky, say so clearly
- Sound encouraging even if the answer is "not right now"
- End with one sentence that is the single most important takeaway for a young investor
- Return as JSON only, no markdown:
{{"swing_explanation": "...", "value_explanation": "...", "pivot_explanation": "...", "key_takeaway": "..."}}"""

        try:
            raw_groq = Groq(api_key=GROQ_KEY).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": groq_prompt}],
                temperature=0.4,
                max_tokens=900,
            ).choices[0].message.content.strip()

            parsed = json.loads(
                raw_groq.replace("```json", "").replace("```", "").strip()
            )
            swing_explanation = parsed.get("swing_explanation", "")
            value_explanation = parsed.get("value_explanation", "")
            pivot_explanation = parsed.get("pivot_explanation", "")
            key_takeaway      = parsed.get("key_takeaway", "")
        except Exception as eg:
            swing_explanation = f"Swing score: {swing_score}/10. " + " ".join(swing_signals[:2])
            value_explanation = f"Value score: {value_score}/10. " + " ".join(value_signals[:2])
            pivot_explanation = pivot_text if pivot_hits else "The stock is not near any key pivot level right now."
            key_takeaway      = "Always do your own research before investing!"

        log("Groq explanation ready!", "match")

        # ── Store result ──────────────────────────────────────────────────────
        result = {
            "symbol":             symbol,
            "company_name":       company_name,
            "price":              price,
            "sector":             sector,
            "industry":           industry,
            "mktcap_b":           mktcap_b,
            "pe":                 pe,
            "div_yield":          div_yield,
            "rsi":                rsi,
            "ema20":              ema20,
            "ema50":              ema50,
            "ema200":             ema200,
            "r1d":                r1d,
            "r5d":                r5d,
            "r1m":                r1m,
            "r6m":                r6m,
            "w52h":               w52h,
            "w52l":               w52l,
            "vol_ratio":          vol_ratio,
            "swing_score":        swing_score,
            "swing_verdict":      swing_verdict,
            "swing_signals":      swing_signals,
            "swing_flags":        swing_flags,
            "swing_explanation":  swing_explanation,
            "value_score":        value_score,
            "value_verdict":      value_verdict,
            "value_signals":      value_signals,
            "value_flags":        value_flags,
            "value_explanation":  value_explanation,
            "pivot_hits":         pivot_hits,
            "pivots":             pivots,
            "pivot_explanation":  pivot_explanation,
            "key_takeaway":       key_takeaway,
            "headlines":          headlines,
        }

        log(f"SWING: {swing_score}/10 — {swing_verdict}", "match" if swing_verdict=="GOOD" else "info")
        log(f"VALUE: {value_score}/10 — {value_verdict}", "match" if value_verdict=="GOOD" else "info")
        log(f"Key takeaway: {key_takeaway}", "match")

        with _lock:
            _jobs[jid]["novice_result"] = result
            _jobs[jid]["done"] = True

    except Exception as e:
        _log(jid, f"Error: {e}", "error")
        with _lock:
            _jobs[jid]["done"] = True


@app.route("/api/novice", methods=["POST"])
def novice_scan():
    """Start a novice screener analysis for one stock."""
    from flask import request as freq
    data   = freq.get_json() or {}
    symbol = data.get("symbol", "").upper().strip()
    name   = data.get("name", symbol)
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    jid = str(uuid.uuid4())[:8]
    with _lock:
        _jobs[jid] = {"log": [], "novice_result": None, "done": False}
    threading.Thread(target=_run_novice, args=(jid, symbol, name), daemon=True).start()
    return jsonify({"job_id": jid})


@app.route("/api/novice_progress/<jid>")
def novice_progress(jid):
    """SSE stream for novice screener progress."""
    def gen():
        sent = 0
        while True:
            with _lock:
                job = _jobs.get(jid, {})
            if not job:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"; break
            while sent < len(job.get("log", [])):
                yield f"data: {json.dumps(job['log'][sent])}\n\n"
                sent += 1
            if job.get("done"):
                yield f"data: {json.dumps({'done': True, 'novice_result': job.get('novice_result')})}\n\n"
                break
            time.sleep(0.2)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/progress/<jid>")
def progress(jid):
    def gen():
        sent = 0
        while True:
            with _lock:
                job = _jobs.get(jid, {})
            if not job: yield f"data: {json.dumps({'error': 'not found'})}\n\n"; break
            while sent < len(job.get("log", [])):
                yield f"data: {json.dumps(job['log'][sent])}\n\n";
                sent += 1
            if job.get("done"):
                yield f"data: {json.dumps({'done': True, 'results_long': job['results_long'], 'results_short': job['results_short'], 'day_rec': job['day_rec']})}\n\n"
                break
            time.sleep(0.2)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>US Screener · SmartStock</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#08090a;--surface:#0f1215;--surface2:#141a1e;
  --border:#1d2730;--border2:#243040;
  --accent:#00d68f;--accent2:#0095ff;--red:#ff4444;--gold:#ffb700;
  --text:#dde4ec;--muted:#4a6070;--muted2:#2a3a48;
  --mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh}
.app{display:grid;grid-template-columns:290px 1fr;min-height:100vh}

/* ── SIDEBAR ── */
.sidebar{background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:var(--border2)}
.logo-bar{padding:18px 18px 14px;border-bottom:1px solid var(--border);flex-shrink:0}
.logo{font-size:1.3rem;font-weight:800;letter-spacing:1px}
.logo .acc{color:var(--accent)}
.logo-tag{font-family:var(--mono);font-size:0.55rem;color:var(--muted);letter-spacing:3px;text-transform:uppercase;margin-top:3px}
.sb-body{padding:14px;flex:1}
.sh{font-family:var(--mono);font-size:0.55rem;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin:18px 0 9px;padding-bottom:5px;border-bottom:1px solid var(--border)}
.sh:first-child{margin-top:0}

/* Day grid */
.day-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:3px;margin-bottom:10px}
.dc{padding:5px 2px;border:1px solid var(--border);border-radius:2px;text-align:center;cursor:default}
.dc.today{border-color:var(--gold);background:rgba(255,183,0,.05)}
.dc-n{font-family:var(--mono);font-size:7px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-bottom:2px}
.dc-v{font-family:var(--mono);font-size:11px;font-weight:600}
.dc-d{font-family:var(--mono);font-size:7px;margin-top:1px}

/* Timing chips */
.t-row{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px}
.tc{font-family:var(--mono);font-size:9px;font-weight:600;padding:3px 8px;border-radius:2px;border:1px solid;cursor:default}
.tc-gold{border-color:var(--gold);color:var(--gold);background:rgba(255,183,0,.07)}
.tc-red{border-color:var(--red);color:var(--red);background:rgba(255,68,68,.07)}
.tc-dim{border-color:var(--border2);color:var(--muted)}

/* Scan btn */
.scan-wrap{padding:0 14px 14px}
.scan-btn{width:100%;padding:11px;font-family:var(--sans);font-size:0.82rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;background:transparent;border:1px solid var(--accent);color:var(--accent);border-radius:3px;cursor:pointer;transition:all .2s}
.scan-btn:hover{background:rgba(0,214,143,.08);box-shadow:0 0 18px rgba(0,214,143,.15)}
.scan-btn:disabled{border-color:var(--border2);color:var(--muted);cursor:not-allowed;box-shadow:none}
@keyframes pulse-btn{0%,100%{opacity:1}50%{opacity:.5}}
.scan-btn.scanning{animation:pulse-btn 1.1s infinite;border-color:var(--gold);color:var(--gold)}
.view-btn{flex:1;padding:7px 4px;font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:1px;background:transparent;border:1px solid var(--border2);color:var(--muted);cursor:pointer;border-radius:3px;transition:all .15s;text-align:center}
.view-btn:hover{color:var(--text);border-color:var(--text)}
.view-btn.active.long-mode{background:rgba(0,214,143,.1);border-color:var(--accent);color:var(--accent)}
.view-btn.active.short-mode{background:rgba(255,68,68,.1);border-color:var(--red);color:var(--red)}

/* Log */
.log-hdr{padding:6px 14px;border-top:1px solid var(--border);border-bottom:1px solid var(--border);background:var(--surface2);font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
.log-body{flex:1;overflow-y:auto;padding:7px 12px;font-family:var(--mono);font-size:10px;line-height:1.9;min-height:180px;max-height:280px}
.log-body::-webkit-scrollbar{width:2px}
.log-body::-webkit-scrollbar-thumb{background:var(--border2)}
.l-info{color:var(--muted2)}.l-phase{color:var(--accent2);font-weight:600}
.l-match{color:var(--accent)}.l-error{color:var(--gold)}.l-done_msg{color:var(--gold);font-weight:700}
.l-title{color:var(--text);font-weight:700;font-size:12px}

/* ── MAIN ── */
.main{display:flex;flex-direction:column;min-height:100vh;overflow:hidden}

/* Topbar */
.topbar{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid var(--border);background:var(--surface);flex-shrink:0;gap:10px;flex-wrap:wrap}
.tb-left{display:flex;align-items:center;gap:12px}
.tb-title{font-size:0.82rem;font-weight:700;letter-spacing:1px}
.badge{font-family:var(--mono);font-size:9px;font-weight:600;padding:3px 9px;border-radius:2px;letter-spacing:2px;text-transform:uppercase}
.b-bull{background:rgba(0,214,143,.1);color:var(--accent);border:1px solid rgba(0,214,143,.25)}
.b-bear{background:rgba(255,68,68,.1);color:var(--red);border:1px solid rgba(255,68,68,.25)}
.b-neu{background:rgba(74,96,112,.15);color:var(--muted);border:1px solid var(--border2)}
.b-long{background:rgba(0,214,143,.1);color:var(--accent);border:1px solid rgba(0,214,143,.25)}
.b-short{background:rgba(255,68,68,.1);color:var(--red);border:1px solid rgba(255,68,68,.25)}
.b-warn{background:rgba(255,183,0,.1);color:var(--gold);border:1px solid rgba(255,183,0,.25)}
.tb-right{font-family:var(--mono);font-size:10px;color:var(--muted)}
.mkt-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--muted);margin-right:5px;vertical-align:middle}
.mkt-dot.open{background:var(--accent);box-shadow:0 0 6px var(--accent)}

/* Status strip */
.status-strip{display:flex;border-bottom:1px solid var(--border);background:var(--surface2);overflow-x:auto;flex-shrink:0}
.ss-block{padding:8px 16px;border-right:1px solid var(--border);flex-shrink:0}
.ss-lbl{font-family:var(--mono);font-size:8px;color:var(--muted2);letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px}
.ss-val{font-family:var(--mono);font-size:12px;font-weight:600}
.sv-g{color:var(--accent)}.sv-r{color:var(--red)}.sv-gold{color:var(--gold)}.sv-dim{color:var(--muted)}

/* Content */
.content{flex:1;overflow-y:auto;padding:20px 24px}

/* Result header */
.res-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px}
.res-title{font-size:0.95rem;font-weight:700;letter-spacing:.5px}
.res-sub{font-family:var(--mono);font-size:10px;color:var(--muted)}
.sort-row{display:flex;gap:6px;flex-wrap:wrap}
.sort-btn{font-family:var(--mono);font-size:9px;padding:3px 9px;background:transparent;border:1px solid var(--border2);color:var(--muted);cursor:pointer;border-radius:2px;transition:all .12s}
.sort-btn:hover{color:var(--text);border-color:var(--text)}
.sort-btn.active{color:var(--accent);border-color:var(--accent);background:rgba(0,214,143,.06)}

/* Cards grid */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:14px;transition:border-color .15s;cursor:default}
.card:hover{border-color:var(--border2)}
.card.rank1{border-color:rgba(255,183,0,.35)}
.card.rank2{border-color:rgba(0,214,143,.2)}
.card.rank3{border-color:rgba(0,149,255,.2)}

/* Card internals */
.card-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}
.c-sym{font-size:1rem;font-weight:800;color:var(--text);letter-spacing:.5px}
.c-sec{font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:2px}
.score-ring{width:42px;height:42px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;border:2px solid}
.sr-hi{border-color:var(--accent);background:rgba(0,214,143,.07)}
.sr-md{border-color:var(--accent2);background:rgba(0,149,255,.07)}
.sr-lo{border-color:var(--border2);background:transparent}
.sc-num{font-family:var(--mono);font-size:15px;font-weight:700;line-height:1}
.sc-den{font-family:var(--mono);font-size:8px;color:var(--muted);margin-top:1px}

/* Levels */
.levels{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:10px}
.lv{background:var(--surface2);border:1px solid var(--border);border-radius:3px;padding:5px 7px;text-align:center}
.lv-lbl{font-family:var(--mono);font-size:8px;color:var(--muted);margin-bottom:2px}
.lv-val{font-family:var(--mono);font-size:11px;font-weight:600}
.lv-e{color:var(--accent2)}.lv-s{color:var(--red)}.lv-t{color:var(--accent)}

/* Returns */
.rets{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}
.rc{font-family:var(--mono);font-size:9px;padding:1px 5px;border-radius:2px}
.up{background:rgba(0,214,143,.1);color:var(--accent)}
.dn{background:rgba(255,68,68,.1);color:var(--red)}
.nu{background:var(--surface2);color:var(--muted)}

/* Tech signals */
.sigs{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:8px}
.sig{font-family:var(--mono);font-size:9px;font-weight:600;padding:1px 5px;border-radius:2px}
.sig-y{background:rgba(0,214,143,.1);color:var(--accent)}
.sig-n{background:rgba(255,68,68,.06);color:rgba(255,68,68,.5)}

/* News */
.news-row{display:flex;align-items:flex-start;gap:6px;margin-bottom:8px}
.ns{font-family:var(--mono);font-size:9px;font-weight:700;padding:2px 6px;border-radius:2px;flex-shrink:0}
.ns-b{background:rgba(0,214,143,.1);color:var(--accent)}
.ns-r{background:rgba(255,68,68,.1);color:var(--red)}
.ns-n{background:var(--surface2);color:var(--muted)}
.ns-txt{font-family:var(--mono);font-size:9px;color:var(--muted);line-height:1.5}

/* Entry time */
.et-row{display:flex;align-items:center;gap:7px;padding-top:8px;border-top:1px solid var(--border)}
.et-lbl{font-family:var(--mono);font-size:9px;color:var(--muted)}
.et-val{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--gold);background:rgba(255,183,0,.08);padding:2px 7px;border-radius:2px;border:1px solid rgba(255,183,0,.2)}
.et-wr{font-family:var(--mono);font-size:9px;color:var(--muted2)}

/* Empty */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;color:var(--muted2);gap:10px;text-align:center}
.empty-icon{font-size:28px;opacity:.2}
.empty-txt{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase}

/* Score bar */
.score-bar-wrap{display:flex;align-items:center;gap:6px;margin-bottom:3px}
.score-track{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.score-fill{height:100%;border-radius:2px;transition:width .4s}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.card{animation:fadeUp .35s ease both}
.card:nth-child(1){animation-delay:.04s}.card:nth-child(2){animation-delay:.08s}
.card:nth-child(3){animation-delay:.12s}.card:nth-child(4){animation-delay:.16s}
.card:nth-child(5){animation-delay:.2s}.card:nth-child(6){animation-delay:.24s}
.card:nth-child(7){animation-delay:.28s}.card:nth-child(8){animation-delay:.32s}
.card:nth-child(9){animation-delay:.36s}.card:nth-child(10){animation-delay:.4s}
@media(max-width:860px){.app{grid-template-columns:1fr}.sidebar{position:relative;height:auto}.log-body{max-height:160px}}
/* Novice screener */
.nv-card{background:#141a1e;border:1px solid #1d2730;border-radius:10px;padding:14px 16px}
.nv-stat{background:#141a1e;border:1px solid #1d2730;border-radius:8px;padding:8px 12px;min-width:80px;text-align:center}
.nv-stat-lbl{font-family:'JetBrains Mono',monospace;font-size:8px;color:#4a6070;letter-spacing:1px;margin-bottom:3px}
.nv-stat-val{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#dde4ec}
</style>
</head>
<body>
<div class="app">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="logo-bar">
    <div class="logo">Smart<span class="acc">Stock</span></div>
    <div class="logo-tag">US Intraday Screener · S&P 500</div>
  </div>

  <div class="sb-body">
    <div class="sh">Weekly bias (backtest)</div>
    <div class="day-grid" id="dayGrid"></div>

    <div class="sh">Best entry times</div>
    <div class="t-row" id="timingRow">
      <span style="font-family:var(--mono);font-size:10px;color:var(--muted2)">Run scan to load</span>
    </div>

    <div class="sh">Backtest rules</div>
    <div style="font-family:var(--mono);font-size:9px;color:var(--muted);line-height:1.9">
      Friday = SHORT only (60.6% WR)<br>
      Monday = lean LONG (55.8% WR)<br>
      Bear day: 10:00 PT short (64.2% WR)<br>
      Bull day: 10:00 PT long (63.2% WR)<br>
      07:00 PT bull long: +0.43% avg return<br>
      Best short: Utilities, Energy, REITs
    </div>
  </div>

  <div class="scan-wrap">
    <button class="scan-btn" id="scanBtn" onclick="startScan()">▶ SCAN NOW</button>
    <div style="display:flex;gap:6px;margin-top:8px">
      <button class="view-btn" id="vb-long" onclick="setView('LONG')">▲ LONG</button>
      <button class="view-btn" id="vb-short" onclick="setView('SHORT')">▼ SHORT</button>
    </div>
    <div style="font-family:var(--mono);font-size:9px;color:var(--muted2);margin-top:6px;text-align:center" id="scanTs"></div>

    <!-- NOVICE SCREENER — inline in sidebar -->
    <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border)">
      <div style="font-family:var(--mono);font-size:9px;letter-spacing:2px;color:#7c3aed;
                  margin-bottom:8px">🎓 NOVICE SCREENER</div>

      <!-- Company name search input -->
      <input id="nvSearch" type="text"
        placeholder="Type company name..."
        oninput="nvSearchDebounce()" autocomplete="off"
        style="width:100%;background:var(--surface2);border:1px solid var(--border2);
               border-radius:4px;padding:8px 10px;color:var(--text);
               font-family:var(--mono);font-size:11px;outline:none;margin-bottom:4px">

      <!-- Autocomplete dropdown -->
      <div id="nvSuggestions" style="display:none;background:#141a1e;border:1px solid #1d2730;
           border-radius:4px;overflow:hidden;margin-bottom:6px"></div>

      <!-- Selected stock pill -->
      <div id="nvSelected" style="display:none;background:rgba(0,214,143,.07);
           border:1px solid rgba(0,214,143,.2);border-radius:4px;padding:6px 10px;
           margin-bottom:6px;font-family:var(--mono);font-size:10px;
           display:none;justify-content:space-between;align-items:center">
        <span id="nvSelLabel" style="color:var(--accent);font-weight:700"></span>
        <span id="nvSelPrice" style="color:var(--gold)"></span>
      </div>

      <!-- Analyse button -->
      <button id="nvRunBtn" onclick="runNoviceInline()" disabled
        style="width:100%;padding:9px;background:transparent;border:1px solid #7c3aed;
               color:#7c3aed;border-radius:4px;font-family:var(--sans);font-size:0.75rem;
               font-weight:800;letter-spacing:2px;cursor:not-allowed;opacity:0.4;
               transition:all .2s">
        ANALYSE
      </button>
    </div>
  </div>

  <div class="log-hdr">Scan log</div>
  <div class="log-body" id="logBody">
    <span class="l-info">Click SCAN NOW to begin...</span>
  </div>
</div>

<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div class="tb-left">
      <span class="tb-title">US INTRADAY SCREENER</span>
      <span class="badge b-neu" id="dowBadge">—</span>
      <span class="badge" id="regimeBadge">—</span>
      <span class="badge" id="dirBadge">—</span>
      <span id="warnBadge"></span>
    </div>
    <div class="tb-right">
      <span class="mkt-dot" id="mktDot"></span>
      <span id="mktStatus">—</span>
      &nbsp;&nbsp;
      <span id="clock">—</span>
    </div>
  </div>

  <div class="status-strip">
    <div class="ss-block"><div class="ss-lbl">SPY</div><div class="ss-val sv-dim" id="ssSPY">—</div></div>
    <div class="ss-block"><div class="ss-lbl">Direction</div><div class="ss-val sv-dim" id="ssDir">—</div></div>
    <div class="ss-block"><div class="ss-lbl">Best entry</div><div class="ss-val sv-gold" id="ssEntry">—</div></div>
    <div class="ss-block"><div class="ss-lbl">Backtest WR</div><div class="ss-val sv-dim" id="ssWR">—</div></div>
    <div class="ss-block" style="flex:1"><div class="ss-lbl">Note</div><div class="ss-val sv-dim" id="ssNote" style="font-size:10px">—</div></div>
  </div>

  <div class="content">
    <div class="res-hdr">
      <div>
        <div class="res-title">Top 10 stocks <span id="dirLabel"></span></div>
        <div class="res-sub" id="resSub">Score 0–9 = tech(6) + news(3) · Stop 1% · Target 2%</div>
      </div>
      <div class="sort-row">
        <button class="sort-btn active" id="sb-total" onclick="doSort('total')">Score</button>
        <button class="sort-btn" id="sb-r_intra" onclick="doSort('r_intra')">Intraday</button>
        <button class="sort-btn" id="sb-r1d" onclick="doSort('r1d')">1-day</button>
        <button class="sort-btn" id="sb-r5d" onclick="doSort('r5d')">5-day</button>
      </div>
    </div>
    <div class="cards" id="cardsDiv">
      <div class="empty">
        <div class="empty-icon">◈</div>
        <div class="empty-txt">Run screener to see today's S&P 500 trades</div>
      </div>
    </div>

    <!-- NOVICE SCREENER RESULTS — shown in main area when Analyse is clicked -->
    <div id="novicePanel" style="display:none;padding:0 0 24px">

      <!-- Header bar -->
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--border)">
        <div>
          <div style="font-size:14px;font-weight:800;letter-spacing:2px;color:#7c3aed">
            🎓 NOVICE SCREENER
          </div>
          <div id="nv-header-sub" style="font-family:var(--mono);font-size:10px;
               color:var(--muted);margin-top:3px"></div>
        </div>
        <button onclick="closeNovicePanel()"
          style="background:none;border:1px solid var(--border2);border-radius:4px;
                 color:var(--muted);padding:5px 12px;cursor:pointer;font-family:var(--mono);
                 font-size:10px;letter-spacing:1px">
          ✕ BACK TO SCREENER
        </button>
      </div>

      <!-- Novice log -->
      <div id="nvLog" style="font-family:var(--mono);font-size:9px;background:var(--surface);
           border:1px solid var(--border);border-radius:4px;padding:10px;max-height:100px;
           overflow-y:auto;margin-bottom:16px;display:none;color:var(--muted)"></div>

      <!-- Stats row -->
      <div style="display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap">
        <div class="nv-stat" id="nv-price"></div>
        <div class="nv-stat" id="nv-rsi"></div>
        <div class="nv-stat" id="nv-pe"></div>
        <div class="nv-stat" id="nv-div"></div>
        <div class="nv-stat" id="nv-mktcap"></div>
        <div class="nv-stat" id="nv-sector"></div>
      </div>

      <!-- Three analysis cards in a grid -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">

        <!-- Swing card -->
        <div class="nv-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div>
              <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#00d68f">
                ⚡ SWING  (1 week – 1 month)
              </div>
              <div style="font-size:9px;color:#4a6070;font-family:var(--mono);margin-top:2px">
                Short-term trade
              </div>
            </div>
            <div id="nv-swing-badge" style="font-size:9px;font-weight:700;padding:3px 10px;
                 border-radius:99px;font-family:var(--mono)"></div>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <div style="font-family:var(--mono);font-size:9px;color:#4a6070;width:36px">Score</div>
            <div style="flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden">
              <div id="nv-swing-bar" style="height:100%;border-radius:3px;transition:width .6s"></div>
            </div>
            <div id="nv-swing-score" style="font-family:var(--mono);font-size:10px;
                 color:#00d68f;width:32px;text-align:right"></div>
          </div>
          <p id="nv-swing-text" style="font-size:12px;line-height:1.75;color:var(--text);
             margin-bottom:10px"></p>
          <div id="nv-swing-sigs" style="font-size:10.5px;color:var(--muted);line-height:1.9"></div>
        </div>

        <!-- Value card -->
        <div class="nv-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div>
              <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#ffb700">
                💰 VALUE  (6 months – 1 year)
              </div>
              <div style="font-size:9px;color:#4a6070;font-family:var(--mono);margin-top:2px">
                Buy and hold
              </div>
            </div>
            <div id="nv-value-badge" style="font-size:9px;font-weight:700;padding:3px 10px;
                 border-radius:99px;font-family:var(--mono)"></div>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <div style="font-family:var(--mono);font-size:9px;color:#4a6070;width:36px">Score</div>
            <div style="flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden">
              <div id="nv-value-bar" style="height:100%;border-radius:3px;transition:width .6s"></div>
            </div>
            <div id="nv-value-score" style="font-family:var(--mono);font-size:10px;
                 color:#ffb700;width:32px;text-align:right"></div>
          </div>
          <p id="nv-value-text" style="font-size:12px;line-height:1.75;color:var(--text);
             margin-bottom:10px"></p>
          <div id="nv-value-sigs" style="font-size:10.5px;color:var(--muted);line-height:1.9"></div>
        </div>

      </div>

      <!-- Pivot card — full width -->
      <div class="nv-card" style="margin-bottom:14px;border-color:rgba(0,149,255,.2)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <div>
            <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#0095ff">
              📐 PIVOT LEVELS  (weekly · monthly · yearly)
            </div>
            <div style="font-size:9px;color:#4a6070;font-family:var(--mono);margin-top:2px">
              Key price floors and ceilings the stock bounces off
            </div>
          </div>
          <div id="nv-pivot-badge" style="font-size:9px;font-weight:700;padding:3px 10px;
               border-radius:99px;font-family:var(--mono);
               background:rgba(0,149,255,.12);color:#0095ff">CHECKING</div>
        </div>
        <div id="nv-pivot-hits" style="margin-bottom:10px"></div>
        <p id="nv-pivot-text" style="font-size:12px;line-height:1.75;color:var(--text)"></p>
      </div>

      <!-- Key takeaway — full width -->
      <div style="background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.3);
           border-radius:6px;padding:14px 16px">
        <div style="font-size:9px;font-weight:700;letter-spacing:2px;color:#7c3aed;
                    font-family:var(--mono);margin-bottom:6px">★ KEY TAKEAWAY FOR YOUR SON</div>
        <p id="nv-takeaway" style="font-size:13px;color:var(--text);line-height:1.75;
           font-weight:600"></p>
      </div>

    </div><!-- /novicePanel -->
  </div><!-- /content -->
</div><!-- /main -->

<script>
var allLong=[], allShort=[], currentView='LONG', sortCol='total_long', sortMult=-1;

function pad(n){return String(n).padStart(2,'0')}
function tick(){
  var now=new Date(), et=new Date(now.toLocaleString('en-US',{timeZone:'America/Los_Angeles'}));
  var h=et.getHours(),m=et.getMinutes(),s=et.getSeconds();
  document.getElementById('clock').textContent=pad(h)+':'+pad(m)+':'+pad(s)+' PT';
  var wk=et.getDay(), isOpen=(wk>=1&&wk<=5&&(h>9||(h===9&&m>=30))&&h<16);
  var dot=document.getElementById('mktDot');
  dot.className='mkt-dot'+(isOpen?' open':'');
  document.getElementById('mktStatus').textContent=isOpen?'Market open':'Market closed';
}
tick(); setInterval(tick,1000);

var DAY_DATA=[
  {n:'MON',lwr:55.8,swr:39.6,bias:'LONG'},
  {n:'TUE',lwr:46.8,swr:48.1,bias:'NEU'},
  {n:'WED',lwr:44.1,swr:51.3,bias:'SHORT'},
  {n:'THU',lwr:51.1,swr:44.3,bias:'NEU'},
  {n:'FRI',lwr:35.0,swr:60.6,bias:'SHORT'},
];
function buildDayGrid(){
  var now=new Date(), et=new Date(now.toLocaleString('en-US',{timeZone:'America/Los_Angeles'}));
  var today=et.getDay()-1; if(today<0)today=6;
  var html='';
  DAY_DATA.forEach(function(d,i){
    var isT=i===today;
    var col=d.bias==='LONG'?'var(--accent)':d.bias==='SHORT'?'var(--red)':'var(--muted)';
    var primary=d.bias==='LONG'?d.lwr:d.bias==='SHORT'?d.swr:Math.max(d.lwr,d.swr);
    html+='<div class="dc'+(isT?' today':'')+'">'
      +'<div class="dc-n" style="color:'+(isT?'var(--gold)':'var(--muted)')+'">'+d.n+'</div>'
      +'<div class="dc-v" style="color:'+col+'">'+primary+'%</div>'
      +'<div class="dc-d" style="color:'+col+'">'+d.bias+'</div>'
    +'</div>';
  });
  document.getElementById('dayGrid').innerHTML=html;
}
buildDayGrid();

// Set default LONG/SHORT button based on today's backtest bias
(function(){
  var now=new Date(), et=new Date(now.toLocaleString('en-US',{timeZone:'America/Los_Angeles'}));
  var today=et.getDay()-1; if(today<0)today=6; // 0=Mon
  var bias=today>=0&&today<5?DAY_DATA[today].bias:'NEU';
  var defaultView=bias==='SHORT'?'SHORT':'LONG';
  var bl=document.getElementById('vb-long');
  var bs=document.getElementById('vb-short');
  if(bl) bl.className='view-btn'+(defaultView==='LONG'?' active long-mode':'');
  if(bs) bs.className='view-btn'+(defaultView==='SHORT'?' active short-mode':'');
  currentView=defaultView;
})();

async function startScan(){
  var btn=document.getElementById('scanBtn');
  btn.disabled=true; btn.textContent='SCANNING...'; btn.classList.add('scanning');
  document.getElementById('logBody').innerHTML='';
  document.getElementById('cardsDiv').innerHTML='<div class="empty"><div class="empty-icon" style="animation:pulse-btn 1s infinite">◈</div><div class="empty-txt">Fetching 100 stocks + Groq news...</div></div>';
  document.getElementById('dirLabel').innerHTML='';
  allLong=[]; allShort=[];
  try{
    var r=await fetch('/api/run',{method:'POST'});
    var d=await r.json();
    if(d.error){addLog('Error: '+d.error,'error');reset();return;}
    listen(d.job_id);
  }catch(e){addLog('Error: '+e.message,'error');reset();}
}

function listen(jid){
  var es=new EventSource('/api/progress/'+jid);
  es.onmessage=function(e){
    var d=JSON.parse(e.data);
    if(d.error){addLog('Error: '+d.error,'error');es.close();reset();return;}
    if(d.msg) addLog(d.msg,d.kind||'info');
    if(d.done){
      es.close(); reset();
      document.getElementById('scanTs').textContent='Last: '+new Date().toLocaleTimeString('en-US',{timeZone:'America/Los_Angeles',hour12:false});
      if(d.day_rec) updateStatus(d.day_rec);
      allLong=d.results_long||[];
      allShort=d.results_short||[];
      currentView=d.day_rec?d.day_rec.direction:'LONG';
      setView(currentView);
    }
  };
  es.onerror=function(){es.close();reset();addLog('Stream closed','error');};
}

function reset(){
  var btn=document.getElementById('scanBtn');
  btn.disabled=false; btn.textContent='▶ SCAN NOW'; btn.classList.remove('scanning');
}

function addLog(msg,kind){
  var b=document.getElementById('logBody');
  var et=new Date().toLocaleTimeString('en-US',{timeZone:'America/Los_Angeles',hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  var s=document.createElement('span'); s.className='l-'+(kind||'info');
  s.textContent='['+et+'] '+msg;
  b.appendChild(s); b.appendChild(document.createElement('br'));
  b.scrollTop=b.scrollHeight;
}

function updateStatus(dr){
  document.getElementById('dowBadge').textContent=dr.dow;
  var reg=document.getElementById('regimeBadge');
  reg.textContent='SPY '+dr.regime+' $'+dr.spy_price;
  reg.className='badge '+(dr.regime==='BULL'?'b-bull':'b-bear');
  var db=document.getElementById('dirBadge');
  db.textContent=dr.direction;
  db.className='badge '+(dr.direction==='LONG'?'b-long':'b-short');
  var wb=document.getElementById('warnBadge');
  wb.innerHTML=dr.dow_n===4?'<span class="badge b-warn">★ FRIDAY — SHORT ONLY</span>':'';
  var spy=document.getElementById('ssSPY');
  spy.textContent='SPY '+dr.regime+' $'+dr.spy_price;
  spy.className='ss-val '+(dr.regime==='BULL'?'sv-g':'sv-r');
  var sd=document.getElementById('ssDir');
  sd.textContent=dr.direction;
  sd.className='ss-val '+(dr.direction==='LONG'?'sv-g':'sv-r');
  document.getElementById('ssEntry').textContent=dr.best_time+' PT';
  var sw=document.getElementById('ssWR');
  sw.textContent=dr.best_wr+'%';
  sw.className='ss-val '+(dr.best_wr>=60?'sv-g':dr.best_wr>=55?'sv-gold':'sv-r');
  document.getElementById('ssNote').textContent=dr.note;

  var times=dr.entry_times||[];
  var dir=dr.direction;
  var html='';
  times.forEach(function(t,i){
    var wr=t.wr_bull||t.wr_bear||'?';
    var cls=i===0?(dir==='SHORT'?'tc tc-red':'tc tc-gold'):'tc tc-dim';
    html+='<span class="'+cls+'">'+t.slot+(i===0?' ★':'')+' · '+wr+'%</span>';
  });
  document.getElementById('timingRow').innerHTML=html;
}

function setView(v){
  currentView=v;
  var vbl=document.getElementById('vb-long');
  var vbs=document.getElementById('vb-short');
  if(vbl) vbl.className='view-btn'+(v==='LONG'?' active long-mode':'');
  if(vbs) vbs.className='view-btn'+(v==='SHORT'?' active short-mode':'');
  sortCol=v==='LONG'?'total_long':'total_short';
  document.querySelectorAll('.sort-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('sb-total').classList.add('active');
  sortMult=-1;
  renderCards(v==='LONG'?allLong:allShort, v);
}

function doSort(col){
  document.querySelectorAll('.sort-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('sb-'+col).classList.add('active');
  var mc=currentView==='SHORT'&&col==='total'?'total_short':col==='total'?'total_long':col;
  if(sortCol===mc) sortMult*=-1; else{sortCol=mc;sortMult=-1;}
  renderCards(currentView==='LONG'?allLong:allShort, currentView);
}

function renderCards(results, dir){
  if(!results||!results.length) return;
  var isShort=dir==='SHORT';
  document.getElementById('dirLabel').innerHTML=
    '<span class="badge '+(isShort?'b-short':'b-long')+'" style="font-size:11px;margin-left:6px">'
    +(isShort?'▼ SHORT':'▲ LONG')+'</span>';
  document.getElementById('resSub').textContent=
    'Score 0–9 · tech(6) + news(3) · Stop 1% · Target 2% · '+(isShort?'most bearish':'most bullish');

  var sorted=[...results].sort(function(a,b){
    var av=a[sortCol],bv=b[sortCol];
    if(av==null)av=-9999; if(bv==null)bv=-9999;
    return sortMult*(av-bv);
  });

  function rc(v,lbl){
    if(v==null) return '';
    var c=v>0?'up':v<0?'dn':'nu';
    return '<span class="rc '+c+'">'+lbl+' '+(v>0?'+':'')+v.toFixed(1)+'%</span>';
  }
  function sig(v,lbl){
    if(!v||v==='?') return '';
    return '<span class="sig '+(v==='✓'?'sig-y':'sig-n')+'">'+lbl+'</span>';
  }

  var html='';
  sorted.slice(0,10).forEach(function(r,i){
    var rank=i+1;
    var rCls=rank===1?'rank1':rank===2?'rank2':rank===3?'rank3':'';
    var sc=isShort?(r.total_short||0):(r.total_long||0);
    var tech=isShort?r.tech_short:r.tech_long;
    var nsc=isShort?r.news_sc_short:r.news_sc_long;
    var sent=isShort?r.sentiment_short:r.sentiment_long;
    var summ=isShort?r.news_summary_short:r.news_summary_long;
    var entry=isShort?r.entry_short:r.entry_long;
    var stop=isShort?r.stop_short:r.stop_long;
    var tgt=isShort?r.target_short:r.target_long;
    var bt=isShort?r.bt_time_short:r.bt_time_long;
    var t1=isShort?r.t1s:r.t1l,t2=isShort?r.t2s:r.t2l,t3=isShort?r.t3s:r.t3l;
    var t4=isShort?r.t4s:r.t4l,t5=isShort?r.t5s:r.t5l,t6=isShort?r.t6s:r.t6l;
    var fillW=Math.round(sc/9*100);
    var fillC=isShort?(sc>=7?'var(--red)':sc>=5?'#ff7055':'var(--muted)')
                     :(sc>=7?'var(--accent)':sc>=5?'var(--accent2)':'var(--muted)');
    var srCls=sc>=7?'sr-hi':sc>=5?'sr-md':'sr-lo';
    var ns=sent==='BULLISH'?'ns-b':sent==='BEARISH'?'ns-r':'ns-n';
    var e=entry?'$'+entry.toFixed(2):'—';
    var s=stop?'$'+stop.toFixed(2):'—';
    var t=tgt?'$'+tgt.toFixed(2):'—';
    html+='<div class="card '+rCls+'">'
      +'<div class="card-top">'
        +'<div>'
          +'<div class="c-sym">'+rank+'. '+r.sym+'</div>'
          +'<div class="c-sec">'+r.sector+'</div>'
          +'<div class="score-bar-wrap" style="margin-top:5px">'
            +'<div class="score-track"><div class="score-fill" style="width:'+fillW+'%;background:'+fillC+'"></div></div>'
          +'</div>'
        +'</div>'
        +'<div class="score-ring '+srCls+'">'
          +'<div class="sc-num" style="color:'+fillC+'">'+sc+'</div>'
          +'<div class="sc-den">T:'+tech+' N:'+nsc+'</div>'
        +'</div>'
      +'</div>'
      +'<div class="levels">'
        +'<div class="lv"><div class="lv-lbl">Entry</div><div class="lv-val lv-e">'+e+'</div></div>'
        +'<div class="lv"><div class="lv-lbl">Stop</div><div class="lv-val lv-s">'+s+'</div></div>'
        +'<div class="lv"><div class="lv-lbl">Target</div><div class="lv-val lv-t">'+t+'</div></div>'
      +'</div>'
      +'<div class="rets">'+rc(r.r1d,'1d')+rc(r.r5d,'5d')+rc(r.r_intra,'intra')+'</div>'
      +'<div class="sigs">'+sig(t1,'REGIME')+sig(t2,'1D')+sig(t3,'5D')+sig(t4,'INTRA')+sig(t5,'EMA')+sig(t6,'LEVEL')+'</div>'
      +'<div class="news-row">'
        +'<span class="ns '+ns+'">'+sent+'</span>'
        +'<span class="ns-txt">'+summ+'</span>'
      +'</div>'
      +'<div class="et-row">'
        +'<span class="et-lbl">Backtest entry:</span>'
        +'<span class="et-val">'+bt+' PT</span>'
        +(rank===1?'<span class="et-wr">highest WR</span>':'')
      +'</div>'
    +'</div>';
  });
  document.getElementById('cardsDiv').innerHTML=html;
}

// ═══════════════════════════════════════════════════════════
// NOVICE SCREENER — inline in main panel
// ═══════════════════════════════════════════════════════════
var _nvSym = '', _nvName = '', _nvTimer = null;

function nvSearchDebounce() {
  clearTimeout(_nvTimer);
  _nvTimer = setTimeout(nvDoSearch, 320);
}

async function nvDoSearch() {
  var q = document.getElementById('nvSearch').value.trim();
  var box = document.getElementById('nvSuggestions');
  if (q.length < 2) { box.style.display = 'none'; return; }
  try {
    var r = await fetch('/api/search_ticker', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q})
    });
    var data = await r.json();
    var results = data.results || [];
    if (!results.length) { box.style.display = 'none'; return; }
    var html = '';
    results.forEach(function(res) {
      var price = res.price ? '$' + res.price : '';
      html += '<div onclick="nvSelect(\'' + res.symbol + '\',\'' +
        res.name.replace(/\'/g, "&#39;") + '\',' + (res.price || 0) + ')" ' +
        'style="padding:8px 10px;cursor:pointer;border-bottom:1px solid #1d2730;' +
        'display:flex;justify-content:space-between;align-items:center;font-size:11px" ' +
        'onmouseover="this.style.background=\'#1d2730\'" onmouseout="this.style.background=\'\'">' +
        '<div><span style="font-family:var(--mono);font-weight:700;color:var(--accent)">' +
        res.symbol + '</span>' +
        '<span style="color:var(--muted);margin-left:8px">' + res.name + '</span></div>' +
        '<span style="font-family:var(--mono);color:var(--gold)">' + price + '</span>' +
        '</div>';
    });
    box.innerHTML = html;
    box.style.display = 'block';
  } catch(e) { box.style.display = 'none'; }
}

function nvSelect(sym, name, price) {
  _nvSym = sym; _nvName = name;
  document.getElementById('nvSuggestions').style.display = 'none';
  document.getElementById('nvSearch').value = name;
  var sel = document.getElementById('nvSelected');
  sel.style.display = 'flex';
  document.getElementById('nvSelLabel').textContent = sym + ' — ' + name;
  document.getElementById('nvSelPrice').textContent = price ? '$' + price : '';
  var btn = document.getElementById('nvRunBtn');
  btn.disabled = false; btn.style.opacity = '1'; btn.style.cursor = 'pointer';
}

async function runNoviceInline() {
  if (!_nvSym) return;
  var btn = document.getElementById('nvRunBtn');
  btn.disabled = true; btn.textContent = 'ANALYSING...'; btn.style.opacity = '0.6';
  document.getElementById('cardsDiv').style.display = 'none';
  document.getElementById('novicePanel').style.display = 'block';
  document.getElementById('nv-header-sub').textContent = _nvName + ' (' + _nvSym + ') — fetching...';
  var logDiv = document.getElementById('nvLog');
  logDiv.innerHTML = ''; logDiv.style.display = 'block';
  function addLog(msg, kind) {
    var col = kind === 'match' ? 'var(--accent)' : kind === 'error' ? 'var(--red)' :
              kind === 'phase' ? 'var(--accent2)' : kind === 'title' ? '#7c3aed' : 'var(--muted)';
    logDiv.innerHTML += '<div style="color:' + col + ';margin-bottom:1px">' + msg + '</div>';
    logDiv.scrollTop = logDiv.scrollHeight;
  }
  ['nv-swing-text','nv-swing-sigs','nv-value-text','nv-value-sigs',
   'nv-pivot-text','nv-pivot-hits','nv-takeaway'].forEach(function(id){
    var el = document.getElementById(id); if(el) el.textContent='';
  });
  try {
    var r = await fetch('/api/novice', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({symbol: _nvSym, name: _nvName})
    });
    var d = await r.json();
    var es = new EventSource('/api/novice_progress/' + d.job_id);
    es.onmessage = function(e) {
      var data = JSON.parse(e.data);
      if (data.msg) addLog(data.msg, data.kind || 'info');
      if (data.done) {
        es.close();
        btn.disabled = false; btn.textContent = 'ANALYSE'; btn.style.opacity = '1';
        logDiv.style.display = 'none';
        if (data.novice_result) {
          renderNoviceInline(data.novice_result);
          document.getElementById('nv-header-sub').textContent =
            data.novice_result.company_name + ' (' + data.novice_result.symbol + ')' +
            '  ·  $' + data.novice_result.price + '  ·  ' + (data.novice_result.sector || '');
        }
      }
    };
    es.onerror = function() {
      es.close(); addLog('Connection error', 'error');
      btn.disabled = false; btn.textContent = 'ANALYSE'; btn.style.opacity = '1';
    };
  } catch(e) {
    addLog('Error: ' + e, 'error');
    btn.disabled = false; btn.textContent = 'ANALYSE'; btn.style.opacity = '1';
  }
}

function closeNovicePanel() {
  document.getElementById('novicePanel').style.display = 'none';
  document.getElementById('cardsDiv').style.display = 'block';
  var btn = document.getElementById('nvRunBtn');
  btn.disabled = true; btn.textContent = 'ANALYSE'; btn.style.opacity = '0.4';
}

function nvStat(id, label, val, color) {
  var el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '<div class="nv-stat-lbl">' + label + '</div>' +
    '<div class="nv-stat-val" style="color:' + (color || 'var(--text)') + '">' + val + '</div>';
}

function renderNoviceInline(res) {
  nvStat('nv-price',  'PRICE',    '$' + res.price);
  nvStat('nv-rsi',    'RSI',      res.rsi,
    res.rsi < 40 ? 'var(--red)' : res.rsi > 70 ? 'var(--gold)' : 'var(--accent)');
  nvStat('nv-pe',     'P/E',      res.pe || 'N/A',
    res.pe && res.pe < 20 ? 'var(--accent)' : res.pe && res.pe > 35 ? 'var(--red)' : 'var(--text)');
  nvStat('nv-div',    'DIVIDEND', res.div_yield ? res.div_yield + '%' : 'None',
    res.div_yield >= 2 ? 'var(--accent)' : 'var(--text)');
  nvStat('nv-mktcap', 'MKT CAP', res.mktcap_b ? '$' + res.mktcap_b + 'B' : 'N/A');
  nvStat('nv-sector', 'SECTOR',   res.sector || '—', 'var(--muted)');

  var sc = res.swing_score, sv = res.swing_verdict;
  var scol = sv==='GOOD'?'var(--accent)':sv==='MIXED'?'var(--gold)':'var(--red)';
  document.getElementById('nv-swing-score').textContent = sc + '/10';
  document.getElementById('nv-swing-bar').style.width = (sc*10)+'%';
  document.getElementById('nv-swing-bar').style.background = scol;
  var swb = document.getElementById('nv-swing-badge');
  swb.textContent = sv; swb.style.color = scol; swb.style.background = 'rgba(0,0,0,0.2)';
  document.getElementById('nv-swing-text').textContent = res.swing_explanation;
  var sd = '';
  (res.swing_signals||[]).forEach(function(s){sd+='<div style="color:var(--accent);margin-bottom:2px">✓ '+s+'</div>';});
  (res.swing_flags||[]).forEach(function(f){sd+='<div style="color:var(--red);margin-bottom:2px">⚠ '+f+'</div>';});
  document.getElementById('nv-swing-sigs').innerHTML = sd;

  var vc = res.value_score, vv = res.value_verdict;
  var vcol = vv==='GOOD'?'var(--accent)':vv==='MIXED'?'var(--gold)':'var(--red)';
  document.getElementById('nv-value-score').textContent = vc + '/10';
  document.getElementById('nv-value-bar').style.width = (vc*10)+'%';
  document.getElementById('nv-value-bar').style.background = vcol;
  var vb = document.getElementById('nv-value-badge');
  vb.textContent = vv; vb.style.color = vcol; vb.style.background = 'rgba(0,0,0,0.2)';
  document.getElementById('nv-value-text').textContent = res.value_explanation;
  var vd = '';
  (res.value_signals||[]).forEach(function(s){vd+='<div style="color:var(--gold);margin-bottom:2px">✓ '+s+'</div>';});
  (res.value_flags||[]).forEach(function(f){vd+='<div style="color:var(--red);margin-bottom:2px">⚠ '+f+'</div>';});
  document.getElementById('nv-value-sigs').innerHTML = vd;

  var hits = res.pivot_hits || [];
  var pb = document.getElementById('nv-pivot-badge');
  var ph = document.getElementById('nv-pivot-hits');
  if (!hits.length) {
    pb.textContent='NOT NEAR A PIVOT'; pb.style.color='var(--muted)'; pb.style.background='rgba(42,58,72,.5)';
    ph.innerHTML='<div style="font-size:10px;color:var(--muted);font-style:italic">Not near any key level right now.</div>';
  } else {
    pb.textContent=hits.length+' HIT'+(hits.length>1?'S':''); pb.style.color='var(--accent2)';
    var hh='';
    hits.forEach(function(h){
      var lc=h.level_name.indexOf('Support')!==-1||h.level_name==='Pivot'?'var(--accent)':'var(--red)';
      var vn=h.vol_strong?'<span style="color:var(--accent)">✓ vol '+h.vol_ratio.toFixed(1)+'x</span>':'<span style="color:var(--muted)">vol '+h.vol_ratio.toFixed(1)+'x</span>';
      hh+='<div style="display:flex;align-items:center;gap:12px;padding:6px 10px;background:var(--surface);border-radius:4px;margin-bottom:4px;font-size:11px">'+
        '<span style="font-family:var(--mono);font-size:8px;color:var(--muted);width:55px">'+h.timeframe.toUpperCase()+'</span>'+
        '<span style="color:var(--muted);width:85px">'+h.level_name+'</span>'+
        '<span style="font-family:var(--mono);font-weight:700;color:'+lc+'">\$'+h.level+'</span>'+
        '<span style="color:'+(h.bounced_up?'var(--accent)':'var(--gold)')+';font-size:10px">'+(h.bounced_up?'↑ bounce':'~ near')+'</span>'+
        '<span style="margin-left:auto">'+vn+'</span>'+
        '</div>';
    });
    ph.innerHTML=hh;
  }
  document.getElementById('nv-pivot-text').textContent = res.pivot_explanation || '';
  document.getElementById('nv-takeaway').textContent = res.key_takeaway;
}

</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║  US S&P 500 Screener  →  http://localhost:5007       ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print("\n  Backtest findings (75,400 trades):")
    print("  · Friday    = SHORT only (60.6% WR)")
    print("  · Monday    = lean LONG  (55.8% WR)")
    print("  · Bull day  = 10:00 PT long  (63.2% WR)")
    print("  · Bear day  = 10:00 PT short (64.2% WR)")
    print("  · Top short = DUK 10:00 PT (75.9%), NOC 11:00 PT (72.4%)\n")
    app.run(debug=False, port=5007, threaded=True)