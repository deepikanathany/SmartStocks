"""
golden_combo.py
===============
5 AI agents that power the Golden Combo Flip Analysis tab.
Called by app.py — no Flask code here.

Agent 1 — Screen stocks with N+ passing filters  (pure Python)
Agent 2 — Fetch real prices from Yahoo Finance    (yfinance)
Agent 3 — Re-evaluate filters for a target date  (Groq AI)
Agent 4 — Calculate gainers and losers            (pure Python math)
Agent 5 — Analyze which filters predicted gains   (Groq AI)

Works for BOTH SmartStock (US) and StocksIndia (NSE/BSE).
Pass market="IN" for India stocks (adds .NS suffix to yfinance lookups).
Pass market="US" for US stocks (no suffix needed).
"""

import yfinance as yf
from groq import Groq
import json
from datetime import datetime, timedelta

# ── Groq API config ────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# Create the Groq client once at module level — reused for all calls
groq_client = Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Ask Groq a question and get the reply as text
# ─────────────────────────────────────────────────────────────────────────────
def _ask_groq(prompt: str, max_tokens: int = 1000) -> str:
    """
    Send a prompt to Groq and return the raw text response.

    Parameters:
        prompt     : the question/instruction we send to the AI
        max_tokens : maximum length of the AI's reply (1 token ≈ 1 word)

    Returns:
        The AI's reply as a plain string.
    """
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Build the Yahoo Finance ticker symbol
# ─────────────────────────────────────────────────────────────────────────────
def _yf_symbol(symbol: str, market: str) -> str:
    """
    Convert a bare symbol to the yfinance format.
    - India (market="IN"): adds .NS suffix, falls back to .BO
    - US    (market="US"): symbol unchanged
    """
    if market == "IN":
        return symbol + ".NS"
    return symbol  # US stocks need no suffix


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Build scorecard-based filter dict from a result dict
# ─────────────────────────────────────────────────────────────────────────────
def _filters_from_scorecard(stock: dict) -> dict:
    """
    Convert the screener engine's 'scorecard' list into the dict format
    that golden_combo expects:
        { "Filter Name": {"st": "PASS" | "FAIL", "v": "display value"} }

    This bridges the screener output to Agent 1's input.
    Works identically for SmartStock and StocksIndia since both use
    the same scorecard format.
    """
    filters = {}
    for entry in stock.get("scorecard", []):
        f_name = entry.get("filter", "")
        passed = entry.get("passed", False)
        active = entry.get("active", True)
        if not f_name:
            continue
        filters[f_name] = {
            "st": "PASS" if passed else "FAIL",
            "v":  entry.get("actual", ""),
            "active": active,
        }
    return filters


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — Screen stocks with N+ passing filters
# ─────────────────────────────────────────────────────────────────────────────
def agent1_screen(all_stocks: list, min_filters: int = 7) -> list:
    """
    Filter the stock list to only those passing min_filters or more ACTIVE filters.

    Parameters:
        all_stocks  : list of stock dicts from the screener (passed OR rejected results)
                      Each dict must have a 'scorecard' key OR a 'filters' key.
        min_filters : minimum number of passing active filters (default 7)

    Returns:
        List of qualifying stock dicts, sorted best score first.
        Each dict gains two new keys:
            pass_count    (int)  — how many active filters passed
            passed_filters (list) — names of the passing filters
    """
    qualified = []

    for stock in all_stocks:
        # Support both screener format (scorecard) and raw format (filters dict)
        if "scorecard" in stock and stock["scorecard"]:
            filters = _filters_from_scorecard(stock)
        else:
            filters = stock.get("filters", {})

        pass_count  = 0
        passed_list = []

        for filter_name, filter_data in filters.items():
            # Skip inactive filters — they don't count toward the score
            if isinstance(filter_data, dict):
                if not filter_data.get("active", True):
                    continue
                if filter_data.get("st") == "PASS":
                    pass_count += 1
                    passed_list.append(filter_name)
            elif filter_data == "PASS":
                pass_count += 1
                passed_list.append(filter_name)

        if pass_count >= min_filters:
            stock["pass_count"]     = pass_count
            stock["passed_filters"] = passed_list
            qualified.append(stock)

    # Sort by pass_count descending — best stocks first
    qualified.sort(key=lambda s: s["pass_count"], reverse=True)
    return qualified


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — Fetch real prices from Yahoo Finance
# ─────────────────────────────────────────────────────────────────────────────
def agent2_fetch_prices(stocks: list, target_date: str, market: str = "IN") -> dict:
    """
    Fetch the closing price of each stock on or near a target date.

    Parameters:
        stocks      : list of qualified stock dicts from Agent 1
        target_date : date string "YYYY-MM-DD"
        market      : "IN" for NSE India (adds .NS), "US" for US stocks

    Returns:
        A dict mapping symbol → price  e.g. {"RELIANCE": 2450.0, "TCS": 3810.5}
    """
    prices    = {}
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")

    # Fetch a small window because markets close on weekends/holidays
    start_dt = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
    end_dt   = (target_dt + timedelta(days=5)).strftime("%Y-%m-%d")

    for stock in stocks:
        # strip .NS/.BO suffix if already present
        raw_sym = stock.get("ticker") or stock.get("symbol", "")
        symbol  = raw_sym.replace(".NS", "").replace(".BO", "")
        if not symbol:
            continue

        yf_sym = _yf_symbol(symbol, market)

        try:
            ticker  = yf.Ticker(yf_sym)
            history = ticker.history(start=start_dt, end=end_dt, interval="1d")

            if history.empty and market == "IN":
                # Fallback: try BSE
                ticker  = yf.Ticker(symbol + ".BO")
                history = ticker.history(start=start_dt, end=end_dt, interval="1d")

            if not history.empty:
                price          = round(float(history["Close"].iloc[-1]), 2)
                prices[symbol] = price

        except Exception:
            pass  # Skip silently — Agent 4 handles missing prices

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — Ask Groq to re-evaluate which filters would pass on target date
# ─────────────────────────────────────────────────────────────────────────────
def agent3_reeval_filters(
    stocks: list,
    prices: dict,
    target_date: str,
    market: str = "IN",
) -> dict:
    """
    For each stock, ask Groq which of the passing filters would likely
    still pass on the target date, given the price and baseline data.

    All stocks sent in a single Groq call to save time and API quota.

    Parameters:
        stocks      : qualified stocks from Agent 1
        prices      : price dict from Agent 2
        target_date : the date we are re-evaluating for
        market      : "IN" for India context, "US" for US market context

    Returns:
        A dict mapping symbol → list of passing filter names
    """
    stock_summaries = []
    for stock in stocks:
        raw_sym = stock.get("ticker") or stock.get("symbol", "")
        symbol  = raw_sym.replace(".NS", "").replace(".BO", "")
        price   = prices.get(symbol)
        if not price:
            continue

        stock_summaries.append({
            "symbol":          symbol,
            "sector":          stock.get("sector", ""),
            "baseline_passed": stock.get("passed_filters", []),
            "baseline_score":  stock.get("pass_count", 0),
            "price_on_date":   price,
        })

    if not stock_summaries:
        return {}

    market_ctx = "NSE India" if market == "IN" else "US equity markets (NYSE/NASDAQ)"

    prompt = f"""You are a financial analyst reviewing {market_ctx} stocks.

For each stock below, determine which filters from its baseline_passed list
would STILL be passing on {target_date}, given its price on that date.

Rules:
- Fundamental filters (Market Cap, Dividend Yield, P/E Ratio, Debt/Equity,
  ROCE, ROE) change slowly — assume they still pass if they passed at baseline.
- Momentum/technical filters (Monthly Trend, Weekly Trend, EMA Trend,
  MACD Signal, Volume Ratio, RSI(14) Range) depend on recent price action.
  Use the price_on_date and sector context to estimate if they still pass.
- ATR% (Volatility) is usually stable — assume it passes if it passed at baseline.
- Analyst Rating changes slowly — assume it still passes.
- P/E vs Industry: assume it still passes if it passed at baseline.

Stocks to analyse:
{json.dumps(stock_summaries, indent=2)}

Return ONLY a JSON object. No explanation, no markdown, no extra text.
Format: {{"SYMBOL": ["Filter1", "Filter2", ...], "SYMBOL2": [...]}}"""

    raw_reply = _ask_groq(prompt, max_tokens=1500)

    try:
        clean  = raw_reply.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result
    except json.JSONDecodeError:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — Calculate price changes and identify gainers/losers
# ─────────────────────────────────────────────────────────────────────────────
def agent4_find_gainers(stocks: list, prices: dict, reeval: dict) -> dict:
    """
    Compare target-date prices against baseline prices to find gainers.

    Parameters:
        stocks  : qualified stocks from Agent 1
        prices  : target-date prices from Agent 2
        reeval  : re-evaluated filter lists from Agent 3

    Returns:
        {"gainers": [...], "losers": [...]}
    """
    gainers = []
    losers  = []

    for stock in stocks:
        raw_sym      = stock.get("ticker") or stock.get("symbol", "")
        symbol       = raw_sym.replace(".NS", "").replace(".BO", "")
        target_price = prices.get(symbol)

        if not target_price:
            continue

        baseline_price = stock.get("price_now")
        if not baseline_price:
            continue

        pct_change = round(
            ((target_price - baseline_price) / baseline_price) * 100, 2
        )

        row = {
            "symbol":           symbol,
            "company":          stock.get("name", symbol),
            "sector":           stock.get("sector", ""),
            "baseline_price":   round(float(baseline_price), 2),
            "target_price":     round(float(target_price), 2),
            "pct_change":       pct_change,
            "baseline_score":   stock.get("pass_count", 0),
            "baseline_filters": stock.get("passed_filters", []),
            "target_filters":   reeval.get(symbol, []),
            "target_score":     len(reeval.get(symbol, [])),
        }

        if pct_change > 0:
            gainers.append(row)
        else:
            losers.append(row)

    gainers.sort(key=lambda x: x["pct_change"], reverse=True)
    losers.sort(key=lambda x: x["pct_change"])

    return {"gainers": gainers, "losers": losers}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — Ask Groq to analyze which filters predicted price gains
# ─────────────────────────────────────────────────────────────────────────────
def agent5_analyze_filters(gainers_data: dict, market: str = "IN") -> dict:
    """
    Ask Groq to analyze which baseline filters were most common
    among gaining stocks vs losing stocks, and rank their predictive weight.

    Parameters:
        gainers_data : the output from Agent 4
        market       : "IN" for India context, "US" for US market context

    Returns:
        A dict with filter rankings, key insights, and a summary.
    """
    gainers = gainers_data.get("gainers", [])
    losers  = gainers_data.get("losers",  [])

    gainer_filter_data = [
        {"symbol": g["symbol"], "pct_change": g["pct_change"], "filters": g["baseline_filters"]}
        for g in gainers
    ]
    loser_filter_data = [
        {"symbol": l["symbol"], "pct_change": l["pct_change"], "filters": l["baseline_filters"]}
        for l in losers
    ]

    market_label = "NSE India" if market == "IN" else "US equity markets (NYSE/NASDAQ)"

    prompt = f"""You are a quantitative analyst studying {market_label} stocks.

I ran a screen and found {len(gainers)} stocks that GAINED in price
and {len(losers)} stocks that DECLINED or stayed flat.

Below is the list of filters that were passing for each stock
at the time of screening (before the price move).

GAINING STOCKS:
{json.dumps(gainer_filter_data, indent=2)}

DECLINING STOCKS:
{json.dumps(loser_filter_data, indent=2)}

Your task:
1. Count how often each filter appears in gainers vs decliners
2. Rank all filters by their predictive power for price gains (score 1-10)
3. Identify the top 3 most predictive filters
4. Write 3 key insights about what these filters tell us
5. Write a 2-sentence summary of the overall findings

The possible filters include:
Market Cap, Dividend Yield, P/E Ratio, Debt/Equity, Monthly Trend,
Weekly Trend, Analyst Rating, ROCE, ROE, P/E vs Industry,
RSI(14) Range, EMA Trend, Volume Ratio, MACD Signal, ATR% (Volatility)

Return ONLY valid JSON, no markdown, no explanation:
{{
  "filter_rankings": [
    {{
      "filter": "Monthly Trend",
      "weight": 9,
      "gainer_freq_pct": 85,
      "loser_freq_pct": 40,
      "insight": "one sentence"
    }}
  ],
  "top_predictors": ["Filter1", "Filter2", "Filter3"],
  "key_insights": ["insight1", "insight2", "insight3"],
  "summary": "Two sentence summary here."
}}"""

    raw_reply = _ask_groq(prompt, max_tokens=1500)

    try:
        clean  = raw_reply.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        return result
    except json.JSONDecodeError:
        return {
            "filter_rankings": [],
            "top_predictors":  [],
            "key_insights":    ["Analysis could not be parsed. Please retry."],
            "summary":         "Analysis unavailable.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — Run all 5 agents in sequence
# ─────────────────────────────────────────────────────────────────────────────
def run_golden_combo(
    all_stocks: list,
    target_date: str,
    min_filters: int = 7,
    market: str = "IN",
) -> dict:
    """
    Run all 5 agents in sequence and return the complete result.

    Parameters:
        all_stocks  : full list of stock result dicts (passed + rejected) from screener run
        target_date : "YYYY-MM-DD" — the date to look up historical prices
        min_filters : minimum passing filters to qualify (default 7)
        market      : "IN" = India (NSE/BSE), "US" = SmartStock (NYSE/NASDAQ)

    Returns:
        {
            "qualified_count": int,
            "gainers": [...],
            "losers": [...],
            "analysis": { filter_rankings, top_predictors, key_insights, summary },
            "error": None | str,
        }
    """
    try:
        # Agent 1 — filter by pass_count
        qualified = agent1_screen(all_stocks, min_filters=min_filters)
        if not qualified:
            return {
                "qualified_count": 0,
                "gainers": [],
                "losers": [],
                "analysis": {},
                "error": f"No stocks passed {min_filters}+ filters. Try lowering the minimum.",
            }

        # Agent 2 — fetch historical prices
        prices = agent2_fetch_prices(qualified, target_date, market=market)
        if not prices:
            return {
                "qualified_count": len(qualified),
                "gainers": [],
                "losers": [],
                "analysis": {},
                "error": f"Could not fetch prices for {target_date}. The date may be too recent or a market holiday.",
            }

        # Agent 3 — re-evaluate filters on target date
        reeval = agent3_reeval_filters(qualified, prices, target_date, market=market)

        # Agent 4 — find gainers and losers
        gainers_data = agent4_find_gainers(qualified, prices, reeval)

        # Agent 5 — analyze filters
        analysis = agent5_analyze_filters(gainers_data, market=market)

        return {
            "qualified_count": len(qualified),
            "gainers":         gainers_data["gainers"],
            "losers":          gainers_data["losers"],
            "analysis":        analysis,
            "error":           None,
        }

    except Exception as e:
        return {
            "qualified_count": 0,
            "gainers": [],
            "losers": [],
            "analysis": {},
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run directly: python golden_combo.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Agent 1 — filter screener...")

    test_stocks = [
        {
            "ticker": "TESTSTOCK_A",
            "name":   "Test Company A",
            "sector": "Technology",
            "price_now": 100.0,
            "scorecard": [
                {"filter": "Market Cap",     "passed": True,  "active": True,  "actual": "$100B"},
                {"filter": "Dividend Yield", "passed": True,  "active": True,  "actual": "2%"},
                {"filter": "P/E Ratio",      "passed": True,  "active": True,  "actual": "18x"},
                {"filter": "Debt/Equity",    "passed": True,  "active": True,  "actual": "20%"},
                {"filter": "Monthly Trend",  "passed": True,  "active": True,  "actual": "+3%"},
                {"filter": "Weekly Trend",   "passed": True,  "active": True,  "actual": "+1%"},
                {"filter": "EMA Trend",      "passed": True,  "active": True,  "actual": "Bullish"},
                {"filter": "Volume Ratio",   "passed": False, "active": True,  "actual": "0.8x"},
                {"filter": "RSI(14) Range",  "passed": False, "active": True,  "actual": "70"},
            ],
        },
        {
            "ticker": "TESTSTOCK_B",
            "name":   "Test Company B",
            "sector": "Healthcare",
            "price_now": 50.0,
            "scorecard": [
                {"filter": "Market Cap",     "passed": True,  "active": True,  "actual": "$50B"},
                {"filter": "Dividend Yield", "passed": False, "active": True,  "actual": "0%"},
                {"filter": "P/E Ratio",      "passed": False, "active": True,  "actual": "45x"},
            ],
        },
    ]

    result = agent1_screen(test_stocks, min_filters=5)
    print(f"Stocks with 5+ filters: {len(result)}")
    for s in result:
        print(f"  {s['ticker']} — score: {s['pass_count']} — passed: {s['passed_filters']}")

    print("\nAll functions defined. File loaded successfully.")
