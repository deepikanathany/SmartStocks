"""
minervini_ranker.py
===================
Value-based ranking engine for Minervini SEPA results.

Takes a list of tickers that have already passed the SEPA screen,
fetches their value/quality metrics from Yahoo Finance, and returns
a ranked list using a weighted scoring system designed to find the
top 3-5 highest-conviction picks from the SEPA universe.

Weightage design rationale:
  SEPA stocks already have the technical timing right (Stage 2 uptrend,
  RS ≥ 70, EPS/Rev growth confirmed). The value filters below assess
  QUALITY and RELATIVE CHEAPNESS within that technically strong group.

Filter weights (total = 100):
  ROCE              20%  — best single indicator of business quality / moat
  P/E vs Industry   18%  — relative cheapness vs sector peers
  ROE               15%  — profitability of shareholder equity
  P/E absolute      12%  — absolute valuation (lower = more room to run)
  Debt/Equity       12%  — financial stability under pressure
  Monthly Trend      8%  — confirms momentum is recent not stale
  Dividend Yield     7%  — management confidence in cash generation
  Analyst Rating     5%  — institutional consensus confirmation
  ATR% Volatility    3%  — risk-per-share (position sizing consideration)

Knockout rules (disqualify regardless of score):
  - Debt/Equity > 80%   — high leverage kills SEPA stocks in corrections
  - ROCE < 8%           — minimum business quality bar

Called by app.py → POST /api/rank_sepa
"""

import yfinance as yf
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Scoring tables ────────────────────────────────────────────────────────────
def _score_roce(v):
    if v is None: return 0
    if v >= 30:  return 5
    if v >= 20:  return 4
    if v >= 15:  return 3
    if v >= 10:  return 2
    return 1

def _score_pe_vs_industry(v):
    if v is None: return 2   # neutral — no benchmark available
    if v <= 0.5:  return 5
    if v <= 0.75: return 4
    if v <= 1.0:  return 3
    if v <= 1.5:  return 2
    return 1

def _score_roe(v):
    if v is None: return 0
    if v >= 30:  return 5
    if v >= 20:  return 4
    if v >= 15:  return 3
    if v >= 10:  return 2
    return 1

def _score_pe(v):
    if v is None: return 2   # neutral — growth stocks often lack trailing PE
    if v <= 10:   return 5
    if v <= 15:   return 4
    if v <= 20:   return 3
    if v <= 30:   return 2
    return 1

def _score_de(v):
    """v = debt/equity as a percentage (e.g. 45 = 45%)"""
    if v is None: return 3   # neutral
    if v <= 5:    return 5
    if v <= 20:   return 4
    if v <= 40:   return 3
    if v <= 60:   return 2
    return 1

def _score_monthly(v):
    if v is None: return 2
    if v >= 5:    return 5
    if v >= 2:    return 4
    if v >= 0:    return 3
    if v >= -2:   return 2
    return 1

def _score_dividend(v):
    """v = dividend yield %"""
    if v is None or v == 0: return 1
    if v >= 3:  return 5
    if v >= 2:  return 4
    if v >= 1:  return 3
    if v >= 0.5: return 2
    return 1

def _score_analyst(v):
    """v = analyst mean (1=Strong Buy, 5=Strong Sell)"""
    if v is None: return 2
    if v <= 1.5:  return 5
    if v <= 2.5:  return 4
    if v <= 3.5:  return 2
    return 0

def _score_atr(v):
    """v = ATR as % of price — we prefer 1.5-4% (not too volatile)"""
    if v is None: return 3
    if 1.5 <= v <= 3: return 5
    if 3 < v <= 5:    return 4
    if 5 < v <= 7:    return 3
    if v > 7:         return 2
    return 2   # < 1.5% = too illiquid for Minervini


# ── Weights ───────────────────────────────────────────────────────────────────
WEIGHTS = {
    "roce":            0.20,
    "pe_vs_industry":  0.18,
    "roe":             0.15,
    "pe":              0.12,
    "debt_equity":     0.12,
    "monthly_trend":   0.08,
    "dividend":        0.07,
    "analyst":         0.05,
    "atr":             0.03,
}

SECTOR_PE_FALLBACK = {
    "Technology": 35.0, "Communication Services": 22.0,
    "Consumer Cyclical": 25.0, "Consumer Defensive": 22.0,
    "Healthcare": 28.0, "Financials": 14.0, "Financial Services": 14.0,
    "Industrials": 22.0, "Basic Materials": 18.0,
    "Energy": 12.0, "Utilities": 18.0, "Real Estate": 35.0,
}


def _fetch_value_metrics(ticker: str) -> dict:
    """
    Fetch all value metrics for a single ticker from Yahoo Finance.
    Returns a dict of raw metric values (not scores).
    Designed to be run in a thread pool.
    """
    result = {
        "ticker":          ticker,
        "name":            ticker,
        "sector":          None,
        "pe":              None,
        "pe_vs_industry":  None,
        "debt_equity":     None,  # as percentage
        "dividend":        None,  # as percentage
        "roe":             None,
        "roce":            None,
        "monthly_trend":   None,
        "analyst":         None,
        "atr":             None,
        "price":           None,
        "market_cap_b":    None,
        "rs_rating":       None,  # passed through from SEPA if available
        "eps_growth_pct":  None,
        "rev_growth_pct":  None,
        "error":           None,
    }

    try:
        t    = yf.Ticker(ticker)
        info = {}
        for attempt in range(3):
            try:
                info = t.info or {}
                if info.get("symbol") or info.get("shortName"):
                    break
            except Exception:
                time.sleep(0.5)

        result["name"]   = info.get("shortName") or info.get("longName") or ticker
        result["sector"] = info.get("sector", "")

        # P/E
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe:
            try: result["pe"] = round(float(pe), 2)
            except: pass

        # P/E vs Industry
        sector = result["sector"] or ""
        industry = info.get("industry", "")
        benchmark_pe = SECTOR_PE_FALLBACK.get(sector)
        if result["pe"] and benchmark_pe and benchmark_pe > 0:
            result["pe_vs_industry"] = round(result["pe"] / benchmark_pe, 3)

        # Debt/Equity
        de = info.get("debtToEquity")
        if de is not None:
            try: result["debt_equity"] = round(float(de), 2)   # yfinance gives this as ratio × 100
            except: pass

        # Dividend yield
        dy = info.get("dividendYield")
        if dy is not None:
            try:
                dy_f = float(dy)
                result["dividend"] = round(dy_f * 100 if dy_f < 1 else dy_f, 2)
            except: pass

        # ROE
        roe = info.get("returnOnEquity")
        if roe is not None:
            try: result["roe"] = round(float(roe) * 100, 2)
            except: pass

        # ROCE — compute from balance sheet + income stmt
        try:
            bs  = t.balance_sheet
            inc = t.income_stmt
            if bs is not None and not bs.empty and inc is not None and not inc.empty:
                ta = cl = ebit = None
                for lbl in bs.index:
                    ll = str(lbl).lower()
                    if "total assets"        in ll: ta   = float(bs.loc[lbl].iloc[0])
                    if "current liabilities" in ll: cl   = float(bs.loc[lbl].iloc[0])
                for lbl in inc.index:
                    ll = str(lbl).lower()
                    if "ebit" in ll or "operating income" in ll:
                        ebit = float(inc.loc[lbl].iloc[0])
                if ta and cl and ebit and (ta - cl) > 0:
                    result["roce"] = round(ebit / (ta - cl) * 100, 2)
        except Exception:
            pass

        # Monthly trend
        try:
            end   = datetime.today()
            start = end - timedelta(days=60)
            hist  = t.history(start=start.strftime("%Y-%m-%d"),
                              end=end.strftime("%Y-%m-%d"), interval="1mo")
            if hist is not None and len(hist) >= 2:
                p_now  = float(hist["Close"].iloc[-1])
                p_prev = float(hist["Close"].iloc[-2])
                result["monthly_trend"] = round((p_now - p_prev) / p_prev * 100, 2)
                result["price"] = round(p_now, 2)
        except Exception:
            pass

        if result["price"] is None:
            p = info.get("regularMarketPrice") or info.get("currentPrice")
            if p: result["price"] = round(float(p), 2)

        # Analyst rating
        am = info.get("recommendationMean")
        if am is not None:
            try: result["analyst"] = round(float(am), 2)
            except: pass
        result["analyst_label"] = (info.get("recommendationKey") or "N/A").title()

        # ATR%
        try:
            hist14 = t.history(period="30d", interval="1d")
            if hist14 is not None and len(hist14) >= 14:
                closes = hist14["Close"].values
                highs  = hist14["High"].values
                lows   = hist14["Low"].values
                trs    = []
                for i in range(1, len(closes)):
                    tr = max(highs[i] - lows[i],
                             abs(highs[i] - closes[i-1]),
                             abs(lows[i]  - closes[i-1]))
                    trs.append(tr)
                atr14 = sum(trs[-14:]) / 14
                if closes[-1] > 0:
                    result["atr"] = round(atr14 / closes[-1] * 100, 2)
        except Exception:
            pass

        # Market cap
        mc = info.get("marketCap")
        if mc:
            try: result["market_cap_b"] = round(float(mc) / 1e9, 2)
            except: pass

    except Exception as e:
        result["error"] = str(e)[:80]

    return result


def _compute_weighted_score(metrics: dict) -> dict:
    """
    Apply the scoring tables and weights to raw metrics.
    Returns scores dict + final weighted score.
    """
    scores = {
        "roce":           _score_roce(metrics.get("roce")),
        "pe_vs_industry": _score_pe_vs_industry(metrics.get("pe_vs_industry")),
        "roe":            _score_roe(metrics.get("roe")),
        "pe":             _score_pe(metrics.get("pe")),
        "debt_equity":    _score_de(metrics.get("debt_equity")),
        "monthly_trend":  _score_monthly(metrics.get("monthly_trend")),
        "dividend":       _score_dividend(metrics.get("dividend")),
        "analyst":        _score_analyst(metrics.get("analyst")),
        "atr":            _score_atr(metrics.get("atr")),
    }

    weighted = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    final    = round(weighted, 3)
    max_possible = 5.0   # max score per filter × 1.0 weight sum

    # Knockout rules
    knocked_out = False
    knockout_reason = ""
    de = metrics.get("debt_equity")
    roce = metrics.get("roce")
    if de is not None and de > 80:
        knocked_out = True
        knockout_reason = f"D/E {de}% > 80% knockout"
    elif roce is not None and roce < 8:
        knocked_out = True
        knockout_reason = f"ROCE {roce}% < 8% knockout"

    return {
        "scores":          scores,
        "final_score":     final,
        "max_score":       max_possible,
        "score_pct":       round(final / max_possible * 100, 1),
        "knocked_out":     knocked_out,
        "knockout_reason": knockout_reason,
    }


def rank_sepa_results(
    tickers: list,
    sepa_data: dict = None,
    log_cb=None,
    max_workers: int = 6,
) -> dict:
    """
    Main entry point. Fetches value metrics for all tickers,
    scores and ranks them, returns full ranked list.

    Parameters:
        tickers   : list of ticker strings e.g. ["AAPL", "MSFT", ...]
        sepa_data : optional dict of ticker → SEPA result (for rs_rating, eps_growth etc.)
                    If provided, merges SEPA fields into the ranked output.
        log_cb    : optional logging callback(msg, kind)
        max_workers: thread pool size

    Returns:
        {
            "ranked":    [ {...stock data + value_score...}, ... ],  # sorted best first
            "top5":      [ ...top 5 non-knocked-out stocks... ],
            "knocked":   [ ...disqualified stocks... ],
            "total":     int,
            "scored":    int,
            "weights":   WEIGHTS dict,
        }
    """
    def log(msg, kind="info"):
        if log_cb: log_cb(msg, kind)

    sepa_data = sepa_data or {}

    log("━" * 48, "info")
    log(f"Value Ranker — scoring {len(tickers)} SEPA stocks", "phase")
    log("Fetching value metrics from Yahoo Finance...", "info")

    all_metrics = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_value_metrics, t): t for t in tickers}
        done    = 0
        for future in as_completed(futures):
            done += 1
            metrics = future.result()
            sym     = metrics["ticker"]
            log(f"  [{done}/{len(tickers)}] {sym} — "
                f"ROCE:{metrics.get('roce','N/A')}%  "
                f"ROE:{metrics.get('roe','N/A')}%  "
                f"PE:{metrics.get('pe','N/A')}×", "info")
            all_metrics.append(metrics)

    log("Scoring and ranking...", "phase")

    ranked   = []
    knocked  = []

    for m in all_metrics:
        scoring  = _compute_weighted_score(m)
        # Merge SEPA data if available
        sepa_row = sepa_data.get(m["ticker"], {})

        row = {
            # Identity
            "ticker":          m["ticker"],
            "name":            m["name"],
            "sector":          m["sector"],
            "price":           m["price"],
            "market_cap_b":    m["market_cap_b"],

            # Value metrics (raw)
            "roce":            m.get("roce"),
            "roe":             m.get("roe"),
            "pe":              m.get("pe"),
            "pe_vs_industry":  m.get("pe_vs_industry"),
            "debt_equity":     m.get("debt_equity"),
            "monthly_trend":   m.get("monthly_trend"),
            "dividend":        m.get("dividend"),
            "analyst":         m.get("analyst"),
            "analyst_label":   m.get("analyst_label", "N/A"),
            "atr":             m.get("atr"),

            # Scoring
            "scores":          scoring["scores"],
            "value_score":     scoring["final_score"],
            "score_pct":       scoring["score_pct"],
            "knocked_out":     scoring["knocked_out"],
            "knockout_reason": scoring["knockout_reason"],

            # SEPA fields (pass-through)
            "rs_rating":       sepa_row.get("rs_rating"),
            "eps_growth_pct":  sepa_row.get("eps_growth_pct"),
            "rev_growth_pct":  sepa_row.get("rev_growth_pct"),
            "vcp":             sepa_row.get("vcp", False),
            "near_pivot":      sepa_row.get("near_pivot", False),

            "error":           m.get("error"),
        }

        if scoring["knocked_out"]:
            knocked.append(row)
        else:
            ranked.append(row)

    # Sort by value_score descending
    ranked.sort(key=lambda r: r["value_score"], reverse=True)
    knocked.sort(key=lambda r: r["value_score"], reverse=True)

    top5 = ranked[:5]

    log("━" * 48, "info")
    log(f"✓ Ranking complete — {len(ranked)} stocks scored · {len(knocked)} knocked out", "match")
    for i, r in enumerate(top5, 1):
        log(f"  #{i} {r['ticker']} — score: {r['score_pct']}% "
            f"(ROCE:{r.get('roce','N/A')}% ROE:{r.get('roe','N/A')}% PE:{r.get('pe','N/A')}×)", "match")

    return {
        "ranked":  ranked,
        "top5":    top5,
        "knocked": knocked,
        "total":   len(tickers),
        "scored":  len(ranked),
        "weights": {k: round(v * 100) for k, v in WEIGHTS.items()},
    }
