"""
usic_screener.py
================
US Investing Championship Screener

Synthesises the common playbook across the last 5 USIC winners:
  2020 Oliver Kell    — Cycle of Price Action + CAN SLIM
  2021 Mark Minervini — SEPA Trend Template + VCP
  2022 Maziyar Yousefizad — Risk management + derivatives discipline
  2023 Tanmay Khandelwal  — B.E.S.T.: Earnings acceleration, Sector strength
  2024 J Law          — Growth momentum, sector leadership

Adds on top of SEPA:
  Stage 4 — EPS Acceleration  : forward growth must exceed trailing (Kell/Khandelwal/Minervini)
  Stage 5 — EPS Surprise      : estimated beat signal (soft filter, affects champion score)
  Stage 6 — Sector RS Gate    : stock's sector must outperform SPY (Kell, Khandelwal, J Law)
  Stage 7 — Champion Score    : composite 0-10 score from all 5 winner criteria

Called by app.py → POST /api/run_usic
"""

import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from screener_engine import (
    _get_sma,
    _sma_trending_up,
    compute_rs_raw,
    _detect_vcp,
)

# ── Sector ETF map ─────────────────────────────────────────────────────────────
SECTOR_ETF = {
    "Technology":             "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Financial Services":     "XLF",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
}

DEFAULT_THRESHOLDS = {
    "sma200_min_days":        20,
    "min_pct_above_52w_low":  30,
    "max_pct_below_52w_high": 25,
    "min_eps_growth_pct":     25,
    "min_rev_growth_pct":     20,
    "vcp_contractions":       2,
    "eps_accel_required":     True,
    "surprise_required":      False,
    "sector_rs_required":     True,
    "min_champion_score":     5.0,
    "min_rs_rating":          75,
}


# ── Stage 4 helper: EPS Acceleration ─────────────────────────────────────────
def _get_eps_acceleration(info):
    eps_trailing  = info.get("trailingEps")
    eps_forward   = info.get("forwardEps")
    earn_growth   = info.get("earningsGrowth")

    trailing_pct = round(float(earn_growth) * 100, 1) if earn_growth is not None else None
    forward_pct  = None
    if eps_trailing and eps_forward and float(eps_trailing) != 0:
        forward_pct = round((float(eps_forward) - float(eps_trailing)) / abs(float(eps_trailing)) * 100, 1)

    accelerating = False
    note         = "Insufficient data"

    if trailing_pct is not None and forward_pct is not None:
        accelerating = forward_pct > trailing_pct
        note = f"Trailing {trailing_pct:+.1f}% → Forward {forward_pct:+.1f}%"
    elif forward_pct is not None:
        accelerating = forward_pct > 20
        note = f"Forward {forward_pct:+.1f}% (trailing N/A)"
    elif trailing_pct is not None:
        accelerating = trailing_pct > 25
        note = f"Trailing {trailing_pct:+.1f}% (forward N/A)"

    return {
        "trailing_growth_pct": trailing_pct,
        "forward_growth_pct":  forward_pct,
        "accelerating":        accelerating,
        "note":                note,
    }


# ── Stage 5 helper: EPS Surprise ─────────────────────────────────────────────
def _get_eps_surprise(info):
    eps_actual   = info.get("trailingEps")
    eps_estimate = info.get("epsCurrentYear") or info.get("epsForward")
    eg           = info.get("earningsGrowth")
    rg           = info.get("revenueGrowth")

    surprise_pct = None
    beat         = False
    note         = "N/A"

    if eps_actual and eps_estimate and float(eps_estimate) != 0:
        surprise_pct = round((float(eps_actual) - float(eps_estimate)) / abs(float(eps_estimate)) * 100, 1)
        beat         = surprise_pct >= 0
        note         = f"{surprise_pct:+.1f}% vs estimate"
    elif eg is not None and rg is not None:
        gap  = round((float(eg) - float(rg)) * 100, 1)
        beat = gap > 3
        note = f"EPS/Rev spread: {gap:+.1f}%"
        surprise_pct = gap

    return {"surprise_pct": surprise_pct, "beat": beat, "note": note}


# ── Stage 6 helper: Sector RS ─────────────────────────────────────────────────
def _get_sector_rs(sector, market_closes, lookback_days=63):
    etf = SECTOR_ETF.get(sector)
    if not etf:
        return {"sector_etf": "—", "sector_return": None, "spy_return": None,
                "outperforming": True, "note": "No ETF mapped"}

    try:
        end   = datetime.today()
        start = end - timedelta(days=lookback_days + 10)

        etf_hist = yf.Ticker(etf).history(
            start=start.strftime("%Y-%m-%d"), interval="1d"
        )
        if etf_hist is None or len(etf_hist) < 10:
            return {"sector_etf": etf, "sector_return": None, "spy_return": None,
                    "outperforming": True, "note": "ETF data unavailable"}

        sector_ret = round(
            (float(etf_hist["Close"].iloc[-1]) / float(etf_hist["Close"].iloc[0]) - 1) * 100, 2
        )

        spy_ret = None
        if market_closes is not None and len(market_closes) >= lookback_days:
            spy_ret = round(
                (float(market_closes.iloc[-1]) / float(market_closes.iloc[-lookback_days]) - 1) * 100, 2
            )

        outperforming = spy_ret is None or sector_ret >= spy_ret
        note = f"{etf} {sector_ret:+.1f}% vs SPY {spy_ret:+.1f}%" if spy_ret is not None else f"{etf} {sector_ret:+.1f}%"

        return {
            "sector_etf":    etf,
            "sector_return": sector_ret,
            "spy_return":    spy_ret,
            "outperforming": outperforming,
            "note":          note,
        }

    except Exception as ex:
        return {"sector_etf": etf, "sector_return": None, "spy_return": None,
                "outperforming": True, "note": f"Error: {str(ex)[:40]}"}


# ── Stage 7 helper: Champion Score ────────────────────────────────────────────
def _compute_champion_score(rs_rating, eps_accel, eps_surprise, sector_rs, info, vcp):
    """
    Composite USIC Champion Score 0–10.
    RS Strength       0-3   (institutional accumulation — all 5 winners)
    EPS Acceleration  0-2   (growth speeding up — Kell, Khandelwal, Minervini)
    EPS Surprise      0-1   (beating estimates — fuel for continuation)
    Sector Leadership 0-1   (sector outperforming — Kell, J Law, Khandelwal)
    VCP Quality       0-2   (tight base = low risk entry — Minervini, Kell)
    Fundamental Grade 0-1   (ROE proxy)
    """
    score = 0.0

    # RS Strength (0-3)
    if rs_rating is not None:
        if rs_rating >= 90:   score += 3.0
        elif rs_rating >= 80: score += 2.0
        elif rs_rating >= 75: score += 1.0
        else:                 score += 0.5

    # EPS Acceleration (0-2)
    fwd = eps_accel.get("forward_growth_pct") or 0
    if eps_accel.get("accelerating"):
        score += 2.0 if fwd >= 40 else 1.5 if fwd >= 25 else 1.0
    elif fwd > 20:
        score += 0.5

    # EPS Surprise (0-1)
    if eps_surprise.get("beat"):
        sp = eps_surprise.get("surprise_pct") or 0
        score += 1.0 if sp >= 10 else 0.5

    # Sector Leadership (0-1)
    if sector_rs.get("outperforming"):
        score += 1.0

    # VCP Quality (0-2)
    if vcp:
        n = vcp.get("contractions", 0)
        score += 1.0 if n >= 3 else 0.5 if n >= 2 else 0
        if vcp.get("vol_dry_up"):  score += 0.5
        if vcp.get("near_pivot"):  score += 0.5

    # Fundamental Grade (0-1)
    roe = info.get("returnOnEquity")
    if roe:
        roe = float(roe)
        if roe > 0.20:   score += 1.0
        elif roe > 0.12: score += 0.5

    return round(min(score, 10.0), 1)


# ── Main per-stock screener ────────────────────────────────────────────────────
def screen_stock_usic(symbol, thresholds, market_closes, all_rs_raw):
    t = thresholds

    rejection = {
        "ticker": symbol, "name": symbol, "sector": "—",
        "price_now": None, "failed_filters": [], "scorecard": [],
        "stage_failed": None,
    }

    try:
        tk   = yf.Ticker(symbol)
        info = tk.info or {}
        if not info or len(info) < 5:
            return None, None

        name   = info.get("longName") or info.get("shortName") or symbol
        sector = info.get("sector", "—")
        rejection["name"]   = name
        rejection["sector"] = sector

        hist = tk.history(period="18mo", interval="1d")
        if hist is None or len(hist) < 200:
            return None, None

        closes  = hist["Close"]
        volumes = hist["Volume"]
        price   = float(closes.iloc[-1])
        rejection["price_now"] = price

        sc = []

        def chk(label, passed, actual="", threshold=""):
            sc.append({"filter": label, "passed": passed,
                       "actual": str(actual), "threshold": str(threshold), "active": True})
            return passed

        # ── Stage 1: Trend Template ───────────────────────────────────────────
        sma50  = _get_sma(closes, 50)
        sma150 = _get_sma(closes, 150)
        sma200 = _get_sma(closes, 200)
        if not all([sma50, sma150, sma200]):
            return None, None

        high52 = info.get("fiftyTwoWeekHigh") or float(closes.max())
        low52  = info.get("fiftyTwoWeekLow")  or float(closes.min())
        pct_above_low  = (price - low52)  / low52  * 100 if low52  else 0
        pct_below_high = (high52 - price) / high52 * 100 if high52 else 0
        sma200_rising  = _sma_trending_up(closes, 200, t.get("sma200_min_days", 20))

        t1 = chk("Price > SMA150 & SMA200", price > sma150 and price > sma200,
                  f"${price:.2f}", f">${sma150:.2f} & >${sma200:.2f}")
        t2 = chk("SMA150 > SMA200",         sma150 > sma200, f"{sma150:.2f}", f">{sma200:.2f}")
        t3 = chk("SMA200 Rising",           sma200_rising,
                  "Yes" if sma200_rising else "No", f"≥{t.get('sma200_min_days',20)}d")
        t4 = chk("SMA50 > SMA150 & SMA200", sma50 > sma150 and sma50 > sma200,
                  f"{sma50:.2f}", f">{sma150:.2f}")
        t5 = chk("Price > SMA50",           price > sma50, f"${price:.2f}", f">${sma50:.2f}")
        t6 = chk(f"≥{t.get('min_pct_above_52w_low',30)}% above 52W Low",
                  pct_above_low >= t.get("min_pct_above_52w_low", 30),
                  f"+{pct_above_low:.1f}%", f"≥{t.get('min_pct_above_52w_low',30)}%")
        t7 = chk(f"≤{t.get('max_pct_below_52w_high',25)}% below 52W High",
                  pct_below_high <= t.get("max_pct_below_52w_high", 25),
                  f"-{pct_below_high:.1f}%", f"≤{t.get('max_pct_below_52w_high',25)}%")

        rs_raw = compute_rs_raw(closes, market_closes)
        if rs_raw is not None:
            all_rs_raw[symbol] = rs_raw
        chk("RS Rating (pending)", rs_raw is not None,
            f"raw={rs_raw:.3f}" if rs_raw else "N/A", "> threshold")

        if not all([t1, t2, t3, t4, t5, t6, t7]):
            rejection["scorecard"]      = sc
            rejection["failed_filters"] = [s["filter"] for s in sc if not s["passed"]]
            rejection["stage_failed"]   = 1
            return None, rejection

        # ── Stage 2: Fundamentals ─────────────────────────────────────────────
        eps_now    = info.get("trailingEps")
        eps_fwd    = info.get("forwardEps")
        rev_growth = info.get("revenueGrowth")
        eps_growth = None
        if eps_now and eps_fwd and float(eps_now) != 0:
            eps_growth = (float(eps_fwd) - float(eps_now)) / abs(float(eps_now)) * 100

        min_eps = t.get("min_eps_growth_pct", 25)
        min_rev = t.get("min_rev_growth_pct", 20)
        f1 = chk("EPS Growth", eps_growth is not None and eps_growth >= min_eps,
                  f"{eps_growth:.1f}%" if eps_growth is not None else "N/A", f"≥{min_eps}%")
        f2 = chk("Revenue Growth",
                  rev_growth is not None and float(rev_growth) * 100 >= min_rev,
                  f"{float(rev_growth)*100:.1f}%" if rev_growth is not None else "N/A",
                  f"≥{min_rev}%")

        if not (f1 and f2):
            rejection["scorecard"]      = sc
            rejection["failed_filters"] = [s["filter"] for s in sc if not s["passed"]]
            rejection["stage_failed"]   = 2
            return None, rejection

        # ── Stage 3: VCP ─────────────────────────────────────────────────────
        vcp = _detect_vcp(closes, volumes, t.get("vcp_contractions", 2))
        chk("VCP ≥ 2 Contractions", vcp["contractions"] >= 2,
            f"{vcp['contractions']}", "≥ 2")
        chk("Volume Dry-up",        vcp["vol_dry_up"],
            "Yes" if vcp["vol_dry_up"] else "No", "< 50% avg")

        # ── Stage 4: EPS Acceleration ─────────────────────────────────────────
        eps_accel = _get_eps_acceleration(info)
        a1 = chk("EPS Accelerating", eps_accel["accelerating"],
                  eps_accel["note"], "Forward > Trailing")
        if t.get("eps_accel_required", True) and not a1:
            rejection["scorecard"]      = sc
            rejection["failed_filters"] = ["EPS Acceleration — " + eps_accel["note"]]
            rejection["stage_failed"]   = 4
            return None, rejection

        # ── Stage 5: EPS Surprise ─────────────────────────────────────────────
        eps_surprise = _get_eps_surprise(info)
        chk("EPS Beat", eps_surprise["beat"], eps_surprise["note"], "Beat estimate")
        if t.get("surprise_required", False) and not eps_surprise["beat"]:
            rejection["scorecard"]      = sc
            rejection["failed_filters"] = ["EPS Surprise — " + eps_surprise["note"]]
            rejection["stage_failed"]   = 5
            return None, rejection

        # ── Stage 6: Sector RS Gate ───────────────────────────────────────────
        sector_rs = _get_sector_rs(sector, market_closes)
        s1 = chk("Sector Outperforming SPY", sector_rs["outperforming"],
                  sector_rs["note"], "Sector ≥ SPY 63d")
        if t.get("sector_rs_required", True) and not s1:
            rejection["scorecard"]      = sc
            rejection["failed_filters"] = ["Sector RS — " + sector_rs["note"]]
            rejection["stage_failed"]   = 6
            return None, rejection

        # ── Stage 7: Champion Score ───────────────────────────────────────────
        champ = _compute_champion_score(
            rs_rating=75, eps_accel=eps_accel, eps_surprise=eps_surprise,
            sector_rs=sector_rs, info=info, vcp=vcp
        )
        min_cs = t.get("min_champion_score", 5.0)
        chk(f"Champion Score ≥ {min_cs}", champ >= min_cs, f"{champ}/10", f"≥{min_cs}")
        if champ < min_cs:
            rejection["scorecard"]      = sc
            rejection["failed_filters"] = [f"Champion Score {champ}/10 < {min_cs}"]
            rejection["stage_failed"]   = 7
            return None, rejection

        # ── Build result ──────────────────────────────────────────────────────
        monthly_chg   = round((price / float(closes.iloc[-21]) - 1) * 100, 2) if len(closes) >= 21 else None
        weekly_chg    = round((price / float(closes.iloc[-5])  - 1) * 100, 2) if len(closes) >= 5  else None
        gross_margin  = info.get("grossMargins")
        profit_margin = info.get("profitMargins")
        inst_pct      = info.get("institutionPercentHeld") or info.get("institutionsPercentHeld")

        return {
            "ticker":             symbol,
            "name":               name,
            "sector":             sector,
            "exchange":           info.get("exchange", "—"),
            "price_now":          price,
            "52w_high":           high52,
            "52w_low":            low52,
            "sma50":              round(sma50, 2),
            "sma150":             round(sma150, 2),
            "sma200":             round(sma200, 2),
            "pct_above_52w_low":  round(pct_above_low, 1),
            "pct_below_52w_high": round(pct_below_high, 1),
            "rs_raw":             rs_raw,
            "rs_rating":          None,
            "champion_score":     champ,
            "eps_growth_pct":     round(eps_growth, 1) if eps_growth is not None else None,
            "rev_growth_pct":     round(float(rev_growth)*100,1) if rev_growth else None,
            "eps_trailing_pct":   eps_accel.get("trailing_growth_pct"),
            "eps_forward_pct":    eps_accel.get("forward_growth_pct"),
            "eps_accelerating":   eps_accel.get("accelerating", False),
            "eps_accel_note":     eps_accel.get("note", ""),
            "eps_surprise_pct":   eps_surprise.get("surprise_pct"),
            "eps_beat":           eps_surprise.get("beat", False),
            "eps_surprise_note":  eps_surprise.get("note", ""),
            "sector_etf":         sector_rs.get("sector_etf"),
            "sector_return_pct":  sector_rs.get("sector_return"),
            "spy_return_pct":     sector_rs.get("spy_return"),
            "sector_outperforming": sector_rs.get("outperforming", True),
            "sector_rs_note":     sector_rs.get("note", ""),
            "vcp":                vcp.get("vcp", False),
            "vcp_contractions":   vcp.get("contractions", 0),
            "max_pullback_pct":   vcp.get("max_pullback_pct"),
            "last_pullback_pct":  vcp.get("last_pullback_pct"),
            "vol_dry_up":         vcp.get("vol_dry_up", False),
            "near_pivot":         vcp.get("near_pivot", False),
            "gross_margin_pct":   round(float(gross_margin)*100,1) if gross_margin else None,
            "profit_margin_pct":  round(float(profit_margin)*100,1) if profit_margin else None,
            "inst_pct":           round(float(inst_pct)*100,1) if inst_pct else None,
            "pe":                 info.get("trailingPE") or info.get("forwardPE"),
            "market_cap_b":       round(info.get("marketCap",0)/1e9,2),
            "monthly_chg_pct":    monthly_chg,
            "weekly_chg_pct":     weekly_chg,
            "scorecard":          sc,
            "failed_filters":     [],
            "stage_failed":       None,
            "ai_score":           None,
        }, None

    except Exception:
        return None, None


def run_usic_screen(tickers, thresholds, log_cb=None):
    """
    Orchestrate the full USIC screen. Returns (results, rejections).
    Same contract as run_sepa_screen.
    """
    def log(m):
        if log_cb: log_cb(m)

    log("USIC Phase 0 — Fetching SPY benchmark...")
    try:
        spy_hist      = yf.Ticker("SPY").history(period="18mo", interval="1d")
        market_closes = spy_hist["Close"] if spy_hist is not None and not spy_hist.empty else None
    except Exception:
        market_closes = None

    all_rs_raw = {}
    results    = []
    rejections = []
    total      = len(tickers)

    log(f"USIC Phases 1-7 — Screening {total} stocks...")

    def screen_one(sym):
        return sym, screen_stock_usic(sym, thresholds, market_closes, all_rs_raw)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(screen_one, sym): sym for sym in tickers}
        done    = 0
        for future in as_completed(futures):
            done += 1
            pct  = int(done / total * 100)
            sym, out = future.result()
            result, rejection = out if out else (None, None)
            log(f"__usic_progress__{pct}__{sym}__{len(results)}")
            if result:
                results.append(result)
            elif rejection:
                rejections.append(rejection)

    # ── RS percentile pass ────────────────────────────────────────────────────
    if all_rs_raw:
        scores  = sorted(all_rs_raw.values())
        n       = len(scores)
        min_rs  = thresholds.get("min_rs_rating", 75)

        for r in results:
            raw = r.get("rs_raw")
            r["rs_rating"] = round(sum(1 for s in scores if s <= raw) / n * 99 + 1) if raw else None

        before  = len(results)
        results = [r for r in results if r.get("rs_rating") is not None and r["rs_rating"] >= min_rs]
        if before - len(results):
            log(f"RS filter: removed {before-len(results)} stocks below RS {min_rs}")

        # Recompute champion score with real RS
        for r in results:
            vcp_proxy = {"contractions": r.get("vcp_contractions",0),
                         "vol_dry_up": r.get("vol_dry_up",False),
                         "near_pivot": r.get("near_pivot",False)}
            r["champion_score"] = _compute_champion_score(
                rs_rating    = r["rs_rating"],
                eps_accel    = {"accelerating": r.get("eps_accelerating",False),
                                "forward_growth_pct": r.get("eps_forward_pct")},
                eps_surprise = {"beat": r.get("eps_beat",False),
                                "surprise_pct": r.get("eps_surprise_pct")},
                sector_rs    = {"outperforming": r.get("sector_outperforming",True)},
                info         = {},
                vcp          = vcp_proxy,
            )

    results.sort(key=lambda r: r.get("champion_score",0), reverse=True)
    log(f"USIC complete — {total} screened · {len(results)} passed all stages · {len(rejections)} rejected")
    return results, rejections
