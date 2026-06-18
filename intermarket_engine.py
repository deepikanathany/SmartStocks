"""
intermarket_engine.py
=====================
Cross-market correlation engine for SMMA screeners.
Works for both US (SmartStock) and NSE (StocksIndia).

How it works:
  1. Fetch basket of correlated global markets ONCE per screener run
  2. Compute lagged correlations (lag 1-3 days) per qualifying stock
  3. Weight each market by its historical predictive power for that stock
  4. Score = sum(weight × today's market return) → -100 to +100

Score interpretation:
  > 40   : HIGH confidence  — strong intermarket confirmation
  15-40  : MEDIUM           — moderate confirmation
  0-15   : LOW              — weak, trade with caution
  < 0    : BEARISH          — intermarket contradiction, skip

Usage:
  from intermarket_engine import IntermarketEngine
  ie     = IntermarketEngine(market="US")  # or "NSE"
  basket = ie.fetch_basket()               # call ONCE per run
  for result in qualifying_stocks:
      ticker = result["ticker"]
      score  = ie.score_stock(ticker, basket)
      result["im_score"]      = score["score"]
      result["im_confidence"] = score["confidence"]
      result["im_drivers"]    = score["drivers"]
      result["im_conflicts"]  = score["conflicts"]
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Market baskets ─────────────────────────────────────────────────────────────

US_BASKET = {
    # Broad market
    "SPY":   "S&P 500",
    "QQQ":   "NASDAQ",
    "IWM":   "Russell 2000",
    # Bonds / credit
    "TLT":   "20yr Bonds",
    "HYG":   "High Yield",
    # Volatility
    "^VIX":  "VIX",
    # Dollar
    "UUP":   "US Dollar",
    # Commodities
    "GLD":   "Gold",
    "USO":   "Oil",
    # Sector ETFs
    "XLK":   "Technology",
    "XLF":   "Financials",
    "XLV":   "Healthcare",
    "XLE":   "Energy",
    "XLI":   "Industrials",
    "XLU":   "Utilities",
    # Global
    "EEM":   "Emerging Mkts",
}

NSE_BASKET = {
    # Domestic indices
    "^NSEI":    "Nifty 50",
    "^NSEBANK": "Bank Nifty",
    # Global influence
    "SPY":      "S&P 500 (US)",
    "QQQ":      "NASDAQ (US)",
    "^VIX":     "VIX",
    # Currency
    "USDINR=X": "USD/INR",
    # Commodities (critical for India)
    "GLD":      "Gold",
    "BZ=F":     "Brent Crude",
    # Emerging markets
    "EEM":      "Emerging Mkts",
    "EWJ":      "Japan",
}

LOOKBACK_DAYS = 252   # 1 year of daily data for correlation
LAGS          = [1, 2, 3]   # predict 1, 2, 3 days ahead
MIN_CORR      = 0.10   # minimum |correlation| to be considered predictive
TOP_N_MARKETS = 8      # use top N most predictive markets per stock


class IntermarketEngine:
    def __init__(self, market: str = "US"):
        """
        market: "US" for SmartStock, "NSE" for StocksIndia
        """
        self.market  = market.upper()
        self.basket  = US_BASKET if self.market == "US" else NSE_BASKET
        self.suffix  = "" if self.market == "US" else ".NS"
        self._basket_data  = None   # cached after fetch_basket()
        self._basket_today = None   # today's returns (last row)

    # ── Step 1: Fetch basket ────────────────────────────────────────────────

    def fetch_basket(self) -> dict:
        """
        Fetch 1 year of daily returns for all basket markets.
        Call ONCE per screener run — reuse the result for all stocks.

        Returns:
            { symbol: pd.Series of daily returns (252 bars) }
        """
        print(f"  [IM] Fetching {self.market} market basket "
              f"({len(self.basket)} markets)...")
        data    = {}
        success = 0

        for sym, name in self.basket.items():
            try:
                df = yf.Ticker(sym).history(
                    period="1y", interval="1d",
                    auto_adjust=True, prepost=False
                )
                if df is None or df.empty or len(df) < 60:
                    continue
                # Timezone-strip
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                returns = df["Close"].pct_change().dropna()
                data[sym] = {"name": name, "returns": returns}
                success += 1
            except Exception:
                pass

        self._basket_data  = data
        self._basket_today = self._get_today_returns(data)
        print(f"  [IM] Basket ready — {success}/{len(self.basket)} markets loaded")
        return data

    def _get_today_returns(self, basket_data: dict) -> dict:
        """Extract the most recent day's return for each market."""
        today = {}
        for sym, d in basket_data.items():
            returns = d["returns"]
            if len(returns) >= 1:
                today[sym] = float(returns.iloc[-1])
        return today

    # ── Step 2: Score one stock ─────────────────────────────────────────────

    def score_stock(self, ticker: str,
                    basket_data: dict = None,
                    stock_df: pd.DataFrame = None) -> dict:
        """
        Compute intermarket score for one stock.

        ticker     : e.g. "AAPL" (US) or "RELIANCE" (NSE, without .NS)
        basket_data: result of fetch_basket() — pass in to avoid re-fetching
        stock_df   : optional pre-fetched daily DataFrame for this stock
                     (pass if already fetched during screening to save time)

        Returns:
            {
              "score":      int (-100 to 100),
              "confidence": "HIGH" | "MEDIUM" | "LOW" | "BEARISH",
              "drivers":    ["SPY +0.8% → bullish", ...],
              "conflicts":  ["VIX rising → bearish"],
              "top_markets":{ sym: corr },
            }
        """
        if basket_data is None:
            basket_data = self._basket_data
        if basket_data is None:
            return self._neutral(ticker, "No basket data")

        # Fetch stock daily data if not provided
        if stock_df is None:
            stock_df = self._fetch_stock(ticker)
        if stock_df is None or len(stock_df) < 60:
            return self._neutral(ticker, "Insufficient history")

        stock_returns = stock_df["Close"].pct_change().dropna()
        if len(stock_returns) < 60:
            return self._neutral(ticker, "Insufficient returns")

        # Compute lagged correlations for each basket market
        correlations = {}
        for sym, d in basket_data.items():
            market_returns = d["returns"]
            corr = self._lagged_correlation(stock_returns, market_returns)
            if abs(corr) >= MIN_CORR:
                correlations[sym] = {"corr": corr, "name": d["name"]}

        if not correlations:
            return self._neutral(ticker, "No significant correlations")

        # Keep top N most predictive (by |correlation|)
        top = dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]["corr"]),
            reverse=True
        )[:TOP_N_MARKETS])

        # Directional score = sum(corr × today_return) weighted
        today = self._basket_today or self._get_today_returns(basket_data)
        raw_score = 0.0
        total_weight = 0.0
        drivers   = []
        conflicts = []

        for sym, info in top.items():
            if sym not in today:
                continue
            corr      = info["corr"]
            name      = info["name"]
            ret_today = today[sym]
            contrib   = corr * ret_today   # positive = bullish confirmation

            weight = abs(corr)
            raw_score    += contrib
            total_weight += weight

            ret_str = f"{ret_today*100:+.2f}%"

            # Classify driver vs conflict
            if contrib > 0.0001:
                drivers.append(f"{name} {ret_str} → bullish")
            elif contrib < -0.0001:
                conflicts.append(f"{name} {ret_str} → bearish")

        # Normalise to -100 → +100
        if total_weight > 0:
            normalised = (raw_score / total_weight) * 100
        else:
            normalised = 0.0

        # Scale to a more intuitive range (raw values are small)
        score = int(round(np.clip(normalised * 15, -100, 100)))

        # Confidence label
        if score >= 40:
            confidence = "HIGH"
        elif score >= 15:
            confidence = "MEDIUM"
        elif score >= 0:
            confidence = "LOW"
        else:
            confidence = "BEARISH"

        return {
            "score":       score,
            "confidence":  confidence,
            "drivers":     drivers[:4],
            "conflicts":   conflicts[:3],
            "top_markets": {s: round(v["corr"], 3) for s, v in top.items()},
        }

    def _lagged_correlation(self, stock_rets: pd.Series,
                             market_rets: pd.Series) -> float:
        """
        Compute the mean correlation across lags 1, 2, 3 days.
        lag=1: market today predicts stock tomorrow
        """
        corrs = []
        for lag in LAGS:
            try:
                # Shift market returns forward — market at t predicts stock at t+lag
                shifted = market_rets.shift(lag)
                combined = pd.concat([stock_rets, shifted], axis=1).dropna()
                if len(combined) < 30:
                    continue
                c = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                if not np.isnan(c):
                    corrs.append(c)
            except Exception:
                pass
        return float(np.mean(corrs)) if corrs else 0.0

    def _fetch_stock(self, ticker: str) -> pd.DataFrame:
        """Fetch 1 year of daily OHLCV for a stock."""
        try:
            sym = ticker + self.suffix if not ticker.endswith(self.suffix) else ticker
            df  = yf.Ticker(sym).history(
                period="1y", interval="1d",
                auto_adjust=True, prepost=False
            )
            if df is None or df.empty:
                return None
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception:
            return None

    def _neutral(self, ticker: str, reason: str) -> dict:
        return {
            "score": 50, "confidence": "NEUTRAL",
            "drivers": [reason], "conflicts": [],
            "top_markets": {},
        }

    # ── Step 3: Batch score a list of results ───────────────────────────────

    def score_results(self, results: list,
                      basket_data: dict = None,
                      log_cb=None) -> list:
        """
        Add IM scores to a list of result dicts in place.
        Each result must have a "ticker" or "symbol" key.

        Returns the same list with im_score, im_confidence,
        im_drivers, im_conflicts added to each item.
        """
        if basket_data is None:
            basket_data = self._basket_data
        if basket_data is None:
            if log_cb:
                log_cb("IM: No basket loaded — call fetch_basket() first")
            return results

        if log_cb:
            log_cb(f"IM: Scoring {len(results)} stocks...")

        for r in results:
            ticker = (r.get("ticker") or r.get("symbol") or "").replace(".NS", "")
            if not ticker:
                continue
            try:
                scored = self.score_stock(ticker, basket_data)
                r["im_score"]      = scored["score"]
                r["im_confidence"] = scored["confidence"]
                r["im_drivers"]    = "; ".join(scored["drivers"][:3])
                r["im_conflicts"]  = "; ".join(scored["conflicts"][:2])
                r["im_top"]        = scored["top_markets"]
                if log_cb:
                    log_cb(f"  {ticker}: IM={scored['score']} "
                           f"({scored['confidence']})")
            except Exception as e:
                r["im_score"]      = 50
                r["im_confidence"] = "NEUTRAL"
                r["im_drivers"]    = ""
                r["im_conflicts"]  = str(e)[:60]

        if log_cb:
            high   = sum(1 for r in results if r.get("im_score",0) >= 40)
            medium = sum(1 for r in results if 15 <= r.get("im_score",0) < 40)
            low    = sum(1 for r in results if 0 <= r.get("im_score",0) < 15)
            bear   = sum(1 for r in results if r.get("im_score",0) < 0)
            log_cb(f"IM done — HIGH:{high} MED:{medium} LOW:{low} BEARISH:{bear}")

        return results