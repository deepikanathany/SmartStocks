"""
us_timing_bt.py
===============
S&P 500 Intraday Timing Backtest — Last 42 Trading Days (~60 calendar days)

Same structure as banknifty_bt.py but for US markets:
  - Universe   : S&P 500 top 100 by market cap (configurable via N_STOCKS)
  - Session    : Regular hours 9:30 AM – 4:00 PM ET
  - Slots      : Every 30-min candle 9:30 → 15:30 (13 slots)
  - Stop       : 1%   |  Target: 2%  |  Exit: 4 PM ET close
  - Regime     : SPY today close vs open (BULL / BEAR)
  - Timezone   : All times in US Eastern (ET)

Outputs:
  1. Best day of week  (Mon–Fri WR for LONG and SHORT)
  2. Best entry slot   (all 13 slots, both directions, bull/bear split)
  3. Full heatmap      (every slot side by side)
  4. Sector breakdown  (best slot per sector)
  5. Top stock+slot    (top 15 combos, min 20 trades)
  6. Bear-day shorts   (top slots when SPY is red)
  7. Bull-day longs    (top slots when SPY is green)

Runtime estimate:
  100 stocks × 13 slots × 2 directions × ~42 days ≈ 109,200 simulations
  Parallel fetch → ~5–8 min total
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz, warnings, time, sys
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
ET      = pytz.timezone("America/New_York")
N_DAYS  = 42          # calendar days back (~42 trading days)
STOP    = 0.01        # 1% stop
TARGET  = 0.02        # 2% target
MAX_W   = 10          # parallel workers
N_STOCKS= 100         # top N by market cap from S&P 500 list
SESSION_START = "09:30"
SESSION_END   = "16:00"

# ── S&P 500 TOP 100 BY MARKET CAP ─────────────────────────────────────────────
# Ordered roughly by market cap, spread across all 11 GICS sectors
SP500_TOP100 = [
    # Technology (22)
    "AAPL","MSFT","NVDA","AVGO","ORCL","CSCO","ADBE","CRM","ACN","AMD",
    "TXN","QCOM","INTU","AMAT","LRCX","KLAC","SNPS","CDNS","PANW","FTNT",
    "MCHP","ADI",
    # Consumer Discretionary (8)
    "AMZN","TSLA","HD","MCD","NKE","LOW","BKNG","TJX",
    # Communication Services (6)
    "GOOGL","META","NFLX","DIS","CMCSA","T",
    # Financials (14)
    "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","C","AXP",
    "BLK","SPGI","MCO","CB",
    # Healthcare (14)
    "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","DHR","AMGN","ISRG",
    "VRTX","REGN","BSX","GILD",
    # Consumer Staples (8)
    "WMT","PG","KO","PEP","COST","PM","MO","MDLZ",
    # Energy (6)
    "XOM","CVX","COP","EOG","SLB","MPC",
    # Industrials (10)
    "GE","CAT","HON","UNP","RTX","DE","ETN","EMR","NOC","ITW",
    # Utilities (4)
    "NEE","SO","DUK","AEP",
    # Real Estate (4)
    "PLD","EQIX","AMT","PSA",
    # Materials (4)
    "LIN","FCX","NEM","SHW",
][:N_STOCKS]

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AVGO":"Technology",
    "ORCL":"Technology","CSCO":"Technology","ADBE":"Technology","CRM":"Technology",
    "ACN":"Technology","AMD":"Technology","TXN":"Technology","QCOM":"Technology",
    "INTU":"Technology","AMAT":"Technology","LRCX":"Technology","KLAC":"Technology",
    "SNPS":"Technology","CDNS":"Technology","PANW":"Technology","FTNT":"Technology",
    "MCHP":"Technology","ADI":"Technology",
    "AMZN":"Cons. Disc.","TSLA":"Cons. Disc.","HD":"Cons. Disc.","MCD":"Cons. Disc.",
    "NKE":"Cons. Disc.","LOW":"Cons. Disc.","BKNG":"Cons. Disc.","TJX":"Cons. Disc.",
    "GOOGL":"Comm. Svc","META":"Comm. Svc","NFLX":"Comm. Svc",
    "DIS":"Comm. Svc","CMCSA":"Comm. Svc","T":"Comm. Svc",
    "BRK-B":"Financials","JPM":"Financials","V":"Financials","MA":"Financials",
    "BAC":"Financials","WFC":"Financials","GS":"Financials","MS":"Financials",
    "C":"Financials","AXP":"Financials","BLK":"Financials","SPGI":"Financials",
    "MCO":"Financials","CB":"Financials",
    "LLY":"Healthcare","UNH":"Healthcare","JNJ":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","TMO":"Healthcare","ABT":"Healthcare","DHR":"Healthcare",
    "AMGN":"Healthcare","ISRG":"Healthcare","VRTX":"Healthcare","REGN":"Healthcare",
    "BSX":"Healthcare","GILD":"Healthcare",
    "WMT":"Cons. Staples","PG":"Cons. Staples","KO":"Cons. Staples","PEP":"Cons. Staples",
    "COST":"Cons. Staples","PM":"Cons. Staples","MO":"Cons. Staples","MDLZ":"Cons. Staples",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","SLB":"Energy","MPC":"Energy",
    "GE":"Industrials","CAT":"Industrials","HON":"Industrials","UNP":"Industrials",
    "RTX":"Industrials","DE":"Industrials","ETN":"Industrials","EMR":"Industrials",
    "NOC":"Industrials","ITW":"Industrials",
    "NEE":"Utilities","SO":"Utilities","DUK":"Utilities","AEP":"Utilities",
    "PLD":"Real Estate","EQIX":"Real Estate","AMT":"Real Estate","PSA":"Real Estate",
    "LIN":"Materials","FCX":"Materials","NEM":"Materials","SHW":"Materials",
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def last_n_tdays(n_calendar):
    """Return last N calendar days worth of trading days."""
    out = []
    d   = date.today() - timedelta(days=1)
    cutoff = date.today() - timedelta(days=n_calendar)
    while d >= cutoff:
        if d.weekday() < 5: out.append(d)
        d -= timedelta(days=1)
    return sorted(out)

def pb(c, t):
    p = int(c/t*40)
    sys.stdout.write(f"\r  [{'█'*p}{'░'*(40-p)}] {c}/{t}    ")
    sys.stdout.flush()
    if c == t: print()

def all_slots():
    """30-min slots from 9:30 to 15:30 ET (13 slots)."""
    slots = []
    t = datetime(2000, 1, 1, 9, 30)
    end = datetime(2000, 1, 1, 16, 0)
    while t < end:
        slots.append(t.strftime("%H:%M"))
        t += timedelta(minutes=30)
    return slots

def fetch_intra(sym):
    """Fetch 30-min intraday for last 60 days. Returns (sym, df) with ET date/time columns."""
    istart = (date.today() - timedelta(days=65)).strftime("%Y-%m-%d")
    iend   = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
    for iv in ["30m", "1h"]:
        try:
            h = yf.Ticker(sym).history(start=istart, end=iend, interval=iv)
            if h is None or h.empty: continue
            # Convert to ET
            try:    h.index = h.index.tz_localize(ET)
            except: h.index = h.index.tz_convert(ET)
            h["_d"] = h.index.strftime("%Y-%m-%d")
            h["_t"] = h.index.strftime("%H:%M")
            if len(h) > 10: return sym, h, iv
        except: pass
    return sym, None, None

def spy_regime(spy_daily, scan):
    """SPY today: BULL if close > open, BEAR if close < open."""
    h = spy_daily[spy_daily.index.date <= scan]
    if h.empty: return "UNKNOWN"
    return "BULL" if float(h["Close"].iloc[-1]) > float(h["Open"].iloc[-1]) else "BEAR"

def simulate(day_bars, slot, direction):
    """Enter at open of slot candle. Stop 1%, target 2%, exit at EOD close."""
    at = day_bars[day_bars["_t"] == slot]
    if at.empty:
        after = day_bars[day_bars["_t"] > slot]
        if after.empty: return None
        at = after.iloc[[0]]

    entry = round(float(at.iloc[0]["Open"]), 4)
    if entry <= 0: return None

    stop_p   = round(entry*(1+STOP),  4) if direction=="SHORT" else round(entry*(1-STOP),  4)
    target_p = round(entry*(1-TARGET),4) if direction=="SHORT" else round(entry*(1+TARGET), 4)

    window = day_bars[(day_bars["_t"] >= slot) & (day_bars["_t"] <= SESSION_END)]
    if window.empty: return None

    ep = er = None
    for i, (_, row) in enumerate(window.iterrows()):
        hi = float(row["High"]); lo = float(row["Low"]); cl = float(row["Close"])
        if direction == "SHORT": sh = hi >= stop_p; th = lo <= target_p
        else:                    sh = lo <= stop_p; th = hi >= target_p
        if sh and th:            ep = stop_p;   er = "STOP";   break
        elif th:                 ep = target_p; er = "TARGET"; break
        elif sh:                 ep = stop_p;   er = "STOP";   break
        elif i == len(window)-1: ep = round(cl,4); er = "EOD"; break

    if ep is None: return None
    res = round((entry-ep)/entry*100, 3) if direction=="SHORT" else round((ep-entry)/entry*100, 3)
    return {"entry":entry, "exit":ep, "exit_reason":er, "result_pct":res,
            "outcome": "WIN" if res>0 else "LOSS" if res<0 else "FLAT"}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    days  = last_n_tdays(N_DAYS)
    s, e  = days[0], days[-1]
    slots = all_slots()

    print("\n" + "="*72)
    print("S&P 500 INTRADAY TIMING BACKTEST")
    print(f"Period  : {s}  →  {e}  ({len(days)} trading days)")
    print(f"Stocks  : {len(SP500_TOP100)} (top {N_STOCKS} S&P 500 by market cap)")
    print(f"Slots   : {len(slots)} (30-min, {slots[0]}–{slots[-1]} ET)")
    print(f"Stop    : {STOP*100}%  |  Target: {TARGET*100}%  |  Exit: 4 PM ET")
    print(f"Regime  : SPY today close vs open")
    print(f"Total sims ≈ {len(SP500_TOP100)*len(slots)*2*len(days):,}")
    print("="*72)

    t0 = time.time()

    # Fetch SPY regime
    print("\n[1/2] SPY daily regime...")
    spy = None
    try:
        spy = yf.Ticker("SPY").history(
            start=(date.today()-timedelta(days=100)).strftime("%Y-%m-%d"),
            end=(date.today()+timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="1d")
        try:    spy.index = pd.to_datetime(spy.index).tz_localize(None)
        except: spy.index = pd.to_datetime(spy.index).tz_convert(None)
        print(f"  SPY rows={len(spy)}")
    except Exception as e:
        print(f"  SPY fetch failed: {e}")

    # Fetch intraday
    print(f"[2/2] {len(SP500_TOP100)} stocks 30-min intraday (parallel)...")
    idata = {}; ivused = {}
    with ThreadPoolExecutor(max_workers=MAX_W) as ex:
        fs = {ex.submit(fetch_intra, sym): sym for sym in SP500_TOP100}
        done = 0
        for f in as_completed(fs):
            done += 1; pb(done, len(SP500_TOP100))
            sym, h, iv = f.result()
            if h is not None: idata[sym] = h; ivused[sym] = iv

    sample = next((idata[s2] for s2 in idata), None)
    if sample is not None:
        dates_avail = sorted(sample["_d"].unique())
        print(f"  ET dates in data: {dates_avail[-5:]}")
        print(f"  Intervals used: 30m={sum(1 for v in ivused.values() if v=='30m')}, "
              f"1h={sum(1 for v in ivused.values() if v=='1h')}")
    print(f"  OK: {len(idata)}/{len(SP500_TOP100)}  ({round(time.time()-t0,1)}s)")

    # ── SIMULATE ──────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("Running simulations...\n")

    overall  = {sl: {"LONG":[], "SHORT":[]} for sl in slots}
    by_bull  = {sl: {"LONG":[], "SHORT":[]} for sl in slots}
    by_bear  = {sl: {"LONG":[], "SHORT":[]} for sl in slots}
    by_dow   = {i: {"LONG":[], "SHORT":[]} for i in range(5)}
    by_sector= {}
    by_stock = {}
    all_trades = []

    dates_used = set()
    for sym in SP500_TOP100:
        if sym not in idata: continue
        ih     = idata[sym]
        sector = SECTOR_MAP.get(sym, "Unknown")
        if sector not in by_sector:
            by_sector[sector] = {sl: {"LONG":[], "SHORT":[]} for sl in slots}
        if sym not in by_stock:
            by_stock[sym] = {sl: {"LONG":[], "SHORT":[]} for sl in slots}

        for scan in days:
            scan_str = str(scan)
            day_bars = ih[ih["_d"] == scan_str]
            if len(day_bars) < 4: continue
            dates_used.add(scan_str)

            regime = spy_regime(spy, scan) if spy is not None else "UNKNOWN"
            dow    = scan.weekday()

            for sl in slots:
                for direction in ["LONG", "SHORT"]:
                    r = simulate(day_bars, sl, direction)
                    if r is None: continue
                    res = r["result_pct"]

                    overall[sl][direction].append(res)
                    by_dow[dow][direction].append(res)
                    if regime == "BULL": by_bull[sl][direction].append(res)
                    elif regime == "BEAR": by_bear[sl][direction].append(res)
                    by_sector[sector][sl][direction].append(res)
                    by_stock[sym][sl][direction].append(res)

                    all_trades.append({
                        "sym":sym, "sector":sector, "date":scan_str,
                        "dow":scan.strftime("%A"), "dow_n":dow,
                        "slot":sl, "direction":direction, "regime":regime, **r
                    })

    n_days_actual = len(dates_used)
    bull_days = len(set(t["date"] for t in all_trades if t["regime"]=="BULL"))
    bear_days = len(set(t["date"] for t in all_trades if t["regime"]=="BEAR"))
    print(f"  {len(all_trades):,} trades  ·  {len(idata)} stocks  ·  {n_days_actual} days")
    print(f"  Regime: Bull={bull_days}  Bear={bear_days}")

    if not all_trades:
        print("  No trades — check ET dates vs scan dates"); return

    # ── HELPER ────────────────────────────────────────────────────────────────
    def wr(lst): return round(sum(1 for r in lst if r>0)/len(lst)*100,1) if lst else 0.0
    def av(lst): return round(sum(lst)/len(lst),3) if lst else 0.0

    # ── 1. BEST DAY OF WEEK ───────────────────────────────────────────────────
    print("\n" + "="*72)
    print("BEST DAY OF WEEK")
    dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    print(f"  {'Day':<12}{'Long Trd':>9}{'Long WR':>9}{'Short Trd':>10}{'Short WR':>10}  Verdict")
    print("  " + "-"*55)
    for i, dname in enumerate(dow_names):
        lt = by_dow[i]["LONG"]; st = by_dow[i]["SHORT"]
        lwr = wr(lt); swr = wr(st)
        best = "LONG ★" if lwr >= 60 and lwr > swr else "SHORT ★" if swr >= 60 and swr > lwr else "—"
        print(f"  {dname:<12}{len(lt):>9}{lwr:>8.1f}%{len(st):>10}{swr:>9.1f}%  {best}")

    # ── 2. BEST LONG SLOTS ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("TOP 10 — LONG ENTRY TIMES (all stocks combined)")
    print(f"  {'Slot':<8}{'Trd':>7}{'WR%':>7}{'AvgRet':>9}  {'WR(Bull)':>9}{'WR(Bear)':>9}")
    print("  " + "-"*52)
    long_rows = [(sl, overall[sl]["LONG"], by_bull[sl]["LONG"], by_bear[sl]["LONG"])
                 for sl in slots if overall[sl]["LONG"]]
    long_rows.sort(key=lambda x: -wr(x[1]))
    for sl, lt, lb, lr in long_rows[:10]:
        mk = " ◀" if wr(lt) == wr(long_rows[0][1]) else ""
        print(f"  {sl:<8}{len(lt):>7}{wr(lt):>6.1f}%{av(lt):>+9.3f}%  "
              f"{wr(lb):>8.1f}%{wr(lr):>9.1f}%{mk}")

    # ── 3. BEST SHORT SLOTS ───────────────────────────────────────────────────
    print(f"\n  TOP 10 — SHORT ENTRY TIMES")
    print(f"  {'Slot':<8}{'Trd':>7}{'WR%':>7}{'AvgRet':>9}  {'WR(Bull)':>9}{'WR(Bear)':>9}")
    print("  " + "-"*52)
    short_rows = [(sl, overall[sl]["SHORT"], by_bull[sl]["SHORT"], by_bear[sl]["SHORT"])
                  for sl in slots if overall[sl]["SHORT"]]
    short_rows.sort(key=lambda x: -wr(x[1]))
    for sl, st, sb, sr in short_rows[:10]:
        mk = " ◀" if wr(st) == wr(short_rows[0][1]) else ""
        print(f"  {sl:<8}{len(st):>7}{wr(st):>6.1f}%{av(st):>+9.3f}%  "
              f"{wr(sb):>8.1f}%{wr(sr):>9.1f}%{mk}")

    # ── 4. FULL HEATMAP ───────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("FULL HEATMAP — ALL SLOTS (ET)")
    print(f"  {'Slot':<8}  {'─── LONG ───':^30}  {'─── SHORT ───':^30}")
    print(f"  {'':8}  {'WR%':>6}{'Avg':>8}{'Bull':>8}{'Bear':>8}"
          f"  {'WR%':>6}{'Avg':>8}{'Bull':>8}{'Bear':>8}")
    print("  " + "-"*72)
    for sl in slots:
        lt=overall[sl]["LONG"]; st=overall[sl]["SHORT"]
        lb=by_bull[sl]["LONG"]; lr=by_bear[sl]["LONG"]
        sb=by_bull[sl]["SHORT"];sr=by_bear[sl]["SHORT"]
        def fmt(lst,b,r):
            if not lst: return " "*30
            return f"{wr(lst):>5.1f}%{av(lst):>+8.3f}%{wr(b):>8.1f}%{wr(r):>8.1f}%"
        lm = " L◀" if lt and wr(lt)>=65 else "   "
        sm = " S◀" if st and wr(st)>=65 else "   "
        print(f"  {sl:<8}  {fmt(lt,lb,lr)}{lm}  {fmt(st,sb,sr)}{sm}")

    # ── 5. SECTOR BREAKDOWN ───────────────────────────────────────────────────
    print("\n" + "="*72)
    print("BEST SLOT PER SECTOR — LONG")
    print(f"  {'Sector':<16}{'Best slot':<10}{'WR%':>7}{'AvgRet':>9}  {'WR(Bull)':>9}")
    print("  " + "-"*52)
    for sector in sorted(by_sector.keys()):
        best_wr=-1; best_sl="—"; best_av=0; best_bull=0
        for sl in slots:
            lt=by_sector[sector][sl]["LONG"]
            lb=by_bull[sl]["LONG"]
            if not lt: continue
            w2=wr(lt)
            if w2 > best_wr:
                best_wr=w2; best_sl=sl; best_av=av(lt)
                best_bull=wr([r for i,r in enumerate(lt)
                               if i<len(by_bull[sl]["LONG"])])
        bw = wr(by_bull[best_sl]["LONG"]) if by_bull.get(best_sl,{}).get("LONG") else 0
        print(f"  {sector:<16}{best_sl:<10}{round(best_wr,1):>6.1f}%{best_av:>+9.3f}%  {bw:>8.1f}%")

    print(f"\n  BEST SLOT PER SECTOR — SHORT")
    print(f"  {'Sector':<16}{'Best slot':<10}{'WR%':>7}{'AvgRet':>9}  {'WR(Bear)':>9}")
    print("  " + "-"*52)
    for sector in sorted(by_sector.keys()):
        best_wr=-1; best_sl="—"; best_av=0
        for sl in slots:
            st=by_sector[sector][sl]["SHORT"]
            if not st: continue
            w2=wr(st)
            if w2 > best_wr: best_wr=w2; best_sl=sl; best_av=av(st)
        bw = wr(by_bear[best_sl]["SHORT"]) if by_bear.get(best_sl,{}).get("SHORT") else 0
        print(f"  {sector:<16}{best_sl:<10}{round(best_wr,1):>6.1f}%{best_av:>+9.3f}%  {bw:>8.1f}%")

    # ── 6. BEAR DAY BEST SHORT SLOTS ─────────────────────────────────────────
    print("\n" + "="*72)
    print("BEAR DAYS ONLY — TOP SHORT SLOTS (SPY red)")
    bear_rows = [(sl, by_bear[sl]["SHORT"]) for sl in slots if by_bear[sl]["SHORT"]]
    bear_rows.sort(key=lambda x: -wr(x[1]))
    print(f"  {'Slot':<8}{'WR(bear)':>9}{'WR(all)':>9}{'Avg':>9}{'Trades':>8}")
    for sl, br in bear_rows[:8]:
        print(f"  {sl:<8}{wr(br):>8.1f}%{wr(overall[sl]['SHORT']):>8.1f}%"
              f"{av(br):>+9.3f}%{len(br):>8}")

    print(f"\n  BULL DAYS ONLY — TOP LONG SLOTS (SPY green)")
    bull_rows = [(sl, by_bull[sl]["LONG"]) for sl in slots if by_bull[sl]["LONG"]]
    bull_rows.sort(key=lambda x: -wr(x[1]))
    print(f"  {'Slot':<8}{'WR(bull)':>9}{'WR(all)':>9}{'Avg':>9}{'Trades':>8}")
    for sl, bl in bull_rows[:8]:
        print(f"  {sl:<8}{wr(bl):>8.1f}%{wr(overall[sl]['LONG']):>8.1f}%"
              f"{av(bl):>+9.3f}%{len(bl):>8}")

    # ── 7. TOP STOCK+SLOT COMBOS ──────────────────────────────────────────────
    print("\n" + "="*72)
    print("TOP 15 STOCK+SLOT — LONG (min 20 trades)")
    combos = []
    for sym, slotdata in by_stock.items():
        for sl, dirs in slotdata.items():
            lt = dirs["LONG"]
            if len(lt) < 20: continue
            combos.append({"sym":sym,"sector":SECTOR_MAP.get(sym,"?"),
                           "slot":sl,"n":len(lt),"wr":wr(lt),"avg":av(lt)})
    combos.sort(key=lambda x: -x["wr"])
    print(f"  {'Stock':<10}{'Sector':<14}{'Slot':<8}{'Trd':>6}{'WR%':>7}{'AvgRet':>9}")
    for c in combos[:15]:
        print(f"  {c['sym']:<10}{c['sector']:<14}{c['slot']:<8}"
              f"{c['n']:>6}{c['wr']:>6.1f}%{c['avg']:>+9.3f}%")

    print(f"\n  TOP 15 STOCK+SLOT — SHORT (min 20 trades)")
    combos = []
    for sym, slotdata in by_stock.items():
        for sl, dirs in slotdata.items():
            st = dirs["SHORT"]
            if len(st) < 20: continue
            combos.append({"sym":sym,"sector":SECTOR_MAP.get(sym,"?"),
                           "slot":sl,"n":len(st),"wr":wr(st),"avg":av(st)})
    combos.sort(key=lambda x: -x["wr"])
    print(f"  {'Stock':<10}{'Sector':<14}{'Slot':<8}{'Trd':>6}{'WR%':>7}{'AvgRet':>9}")
    for c in combos[:15]:
        print(f"  {c['sym']:<10}{c['sector']:<14}{c['slot']:<8}"
              f"{c['n']:>6}{c['wr']:>6.1f}%{c['avg']:>+9.3f}%")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    import pandas as pd
    summary_rows = []
    for sl in slots:
        for d2 in ["LONG","SHORT"]:
            lt=overall[sl][d2]; lb=by_bull[sl][d2]; lr=by_bear[sl][d2]
            summary_rows.append({
                "slot":sl,"direction":d2,"trades":len(lt),
                "wr":wr(lt),"avg_ret":av(lt),
                "wr_bull":wr(lb),"wr_bear":wr(lr),
            })
    pd.DataFrame(summary_rows).to_csv("us_timing_summary.csv", index=False)
    pd.DataFrame(all_trades[:200000]).to_csv("us_timing_trades.csv", index=False)

    print(f"\n  Files: us_timing_summary.csv  us_timing_trades.csv")
    print(f"  Runtime: {round(time.time()-t0,1)}s")
    print("\n" + "="*72)
    print("DONE — share the output to build the live screener")
    print()


if __name__ == "__main__":
    run()