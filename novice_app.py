"""
novice_app.py  —  SmartStocks Novice Screener
Standalone app — separated from SmartStocks to allow independent growth.
Run with:  python novice_app.py
Then open: http://localhost:5008
"""

import time
import json
import threading
from flask import Flask, Response, jsonify, render_template_string, request
import yfinance as yf
import numpy as np

app = Flask(__name__)

# ── Groq config ──────────────────────────────────────────────────────────────
try:
    from groq import Groq as _Groq
    GROQ_API_KEY   = "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"
    GROQ_MODEL     = "llama-3.3-70b-versatile"
    _groq_client   = _Groq(api_key=GROQ_API_KEY)
    _groq_available = True
except ImportError:
    _groq_available = False
    _groq_client    = None
    GROQ_MODEL      = ""

# ── Job store ────────────────────────────────────────────────────────────────
_novice_jobs = {}
_novice_lock = threading.Lock()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/novice_search", methods=["POST"])
def novice_search():
    """Company name → up to 5 matching US stock symbols.
    Price is fetched quickly via fast_info after symbol selection, not during search.
    """
    import requests as _req
    query = (request.get_json() or {}).get("query", "").strip()
    if not query or len(query) < 2:
        return jsonify({"results": []})
    try:
        url  = (f"https://query2.finance.yahoo.com/v1/finance/search"
                f"?q={_req.utils.quote(query)}&quotesCount=10&newsCount=0&listsCount=0")
        resp = _req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        quotes = resp.json().get("quotes", [])
        results = []
        for q in quotes:
            if q.get("quoteType") != "EQUITY": continue
            if q.get("exchange") not in ("NYQ","NMS","NGM","NCM","NYSEArca"): continue
            sym = q.get("symbol", "")
            if not sym or "." in sym: continue
            # Get price from the search result itself (regularMarketPrice field)
            # This avoids a separate yfinance call per symbol which is very slow
            price = None
            try:
                raw_price = q.get("regularMarketPrice") or q.get("regularMarketPreviousClose")
                if raw_price:
                    price = round(float(raw_price), 2)
            except Exception:
                price = None
            results.append({
                "symbol": sym,
                "name":   q.get("longname") or q.get("shortname") or sym,
                "price":  price,
            })
            if len(results) >= 5: break
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)[:80]})


@app.route("/api/novice", methods=["POST"])
def novice_run():
    data   = request.get_json() or {}
    sym    = data.get("symbol", "").upper().strip()
    name   = data.get("name", sym)
    if not sym:
        return jsonify({"error": "symbol required"}), 400
    jid = __import__("uuid").uuid4().hex[:8]
    with _novice_lock:
        _novice_jobs[jid] = {"log": [], "result": None, "done": False}
    threading.Thread(target=_run_novice_job, args=(jid, sym, name), daemon=True).start()
    return jsonify({"job_id": jid})


@app.route("/api/novice_progress/<jid>")
def novice_progress(jid):
    def gen():
        sent = 0
        while True:
            with _novice_lock:
                job = _novice_jobs.get(jid, {})
            if not job:
                yield f"data:{json.dumps({'error':'not found'})}\n\n"; break
            while sent < len(job.get("log", [])):
                yield f"data:{json.dumps(job['log'][sent])}\n\n"
                sent += 1
            if job.get("done"):
                yield f"data:{json.dumps({'done':True,'result':job.get('result')})}\n\n"
                break
            time.sleep(0.2)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Analysis engine ───────────────────────────────────────────────────────────

def _run_novice_job(jid, symbol, company_name):
    from datetime import date, timedelta
    import requests as _req
    import xml.etree.ElementTree as ET

    def log(msg, kind="info"):
        with _novice_lock:
            _novice_jobs[jid]["log"].append({"msg": msg, "kind": kind})

    try:
        log(f"Fetching data for {company_name} ({symbol})…", "phase")

        df = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 10:
            log(f"No data found for {symbol}.", "error")
            with _novice_lock: _novice_jobs[jid]["done"] = True
            return

        cl  = df["Close"]; hi = df["High"]; lo = df["Low"]; vol = df["Volume"]
        price = round(float(cl.iloc[-1]), 2)

        def _pct(a, b): return round((a / b - 1) * 100, 2) if b else 0
        r1d = _pct(cl.iloc[-1], cl.iloc[-2])  if len(cl) >= 2   else 0
        r5d = _pct(cl.iloc[-1], cl.iloc[-6])  if len(cl) >= 6   else 0
        r1m = _pct(cl.iloc[-1], cl.iloc[-22]) if len(cl) >= 22  else 0
        r6m = _pct(cl.iloc[-1], cl.iloc[-126])if len(cl) >= 126 else 0

        w52h   = round(float(hi.max()), 2)
        w52l   = round(float(lo.min()), 2)
        ema20  = round(float(cl.ewm(span=20,  adjust=False).mean().iloc[-1]), 2)
        ema50  = round(float(cl.ewm(span=50,  adjust=False).mean().iloc[-1]), 2)
        ema200 = round(float(cl.ewm(span=200, adjust=False).mean().iloc[-1]), 2) if len(cl) >= 200 else None

        delta  = cl.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi    = round(float(100 - 100 / (1 + gain.iloc[-1] / max(loss.iloc[-1], 1e-9))), 1)

        avg_vol20 = float(vol.iloc[-20:].mean()) if len(vol) >= 20 else float(vol.mean())
        vol_ratio = round(float(vol.iloc[-1]) / avg_vol20, 2) if avg_vol20 > 0 else 1.0

        try:
            info      = yf.Ticker(symbol).info or {}
            pe        = round(float(v), 1) if (v := info.get("trailingPE") or info.get("forwardPE")) else None
            raw_div   = float(info.get("dividendYield", 0) or 0)
            div_yield = round(raw_div * 100, 2) if raw_div < 1 else round(raw_div, 2)
            mktcap_b  = round(info.get("marketCap", 0) / 1e9, 1) if info.get("marketCap") else None
            sector    = info.get("sector", "Unknown")
            eps       = float(info.get("trailingEps") or 0)
        except Exception:
            pe = div_yield = mktcap_b = None; sector = "Unknown"; eps = 0

        # Sector PE lookup
        SECTOR_PE = {
            "Technology": 28, "Communication Services": 22,
            "Consumer Discretionary": 25, "Consumer Staples": 20,
            "Health Care": 22, "Financials": 14, "Industrials": 21,
            "Energy": 12, "Utilities": 17, "Real Estate": 35, "Materials": 18,
        }
        sector_pe_avg = SECTOR_PE.get(sector, 20)

        pe_vs_sector = ""
        if pe and sector_pe_avg:
            diff_pct = round((pe - sector_pe_avg) / sector_pe_avg * 100, 1)
            if diff_pct > 20:
                pe_vs_sector = (f"PE {pe} vs {sector} sector average {sector_pe_avg} — "
                                f"stock is {diff_pct}% MORE expensive than its peers")
            elif diff_pct < -15:
                pe_vs_sector = (f"PE {pe} vs {sector} sector average {sector_pe_avg} — "
                                f"stock is {abs(diff_pct)}% CHEAPER than its peers (potential bargain)")
            else:
                pe_vs_sector = (f"PE {pe} vs {sector} sector average {sector_pe_avg} — "
                                f"stock is priced in line with its peers")

        log(f"  ${price}  RSI:{rsi}  EMA20:{ema20}  PE:{pe}  SectorPE:{sector_pe_avg}", "info")

        # ── Swing score ───────────────────────────────────────────────────────
        sw_score = 0; sw_good = []; sw_warn = []
        if price > ema20 > ema50:
            sw_score += 2; sw_good.append("Price above both moving averages — uptrend")
        elif price > ema20:
            sw_score += 1; sw_good.append("Price above short-term moving average — mild uptrend")
        else:
            sw_warn.append("Price below moving averages — downtrend short term")

        if 40 <= rsi <= 65:   sw_score += 2; sw_good.append(f"RSI {rsi} in sweet spot (not too hot/cold)")
        elif rsi < 40:        sw_score += 1; sw_good.append(f"RSI {rsi} — oversold, possible bounce")
        elif rsi > 70:        sw_warn.append(f"RSI {rsi} — overbought, may pull back")

        if r5d > 1:           sw_score += 1; sw_good.append(f"Up {r5d:.1f}% this week — good momentum")
        elif r5d < -3:        sw_warn.append(f"Down {abs(r5d):.1f}% this week — weak short-term")

        if vol_ratio >= 1.2:  sw_score += 1; sw_good.append(f"Volume {vol_ratio:.1f}x normal — strong activity")
        elif vol_ratio < 0.7: sw_warn.append("Very low volume — limited conviction")

        dist_high = round((w52h - price) / w52h * 100, 1)
        if dist_high <= 8:    sw_score += 1; sw_good.append(f"Only {dist_high}% below 52-week high")
        elif dist_high >= 30: sw_warn.append(f"Down {dist_high}% from yearly high")

        if r1m > 2:           sw_score += 1; sw_good.append(f"Up {r1m:.1f}% this month")
        elif r1m < -5:        sw_warn.append(f"Down {abs(r1m):.1f}% this month")

        sw_score   = min(sw_score, 10)
        sw_verdict = "GOOD" if sw_score >= 7 else "MIXED" if sw_score >= 4 else "RISKY"

        # Swing buy zone
        swing_buy_zone = None
        if sw_verdict in ("RISKY", "MIXED"):
            candidates = []
            if ema20  < price: candidates.append(("EMA20",  round(ema20,  2)))
            if ema50  < price: candidates.append(("EMA50",  round(ema50,  2)))
            if ema200 and ema200 < price: candidates.append(("EMA200", round(ema200, 2)))
            valid = [(l, v) for l, v in candidates if (price - v) / price * 100 >= 0.5]
            if valid:
                valid.sort(key=lambda x: price - x[1])
                lbl, lv = valid[0]
                swing_buy_zone = {"price": lv, "reason": lbl,
                    "desc": f"${lv} ({lbl}) — this moving average has acted as a floor before"}
            elif w52l > 0:
                fb = round(w52l * 1.05, 2)
                swing_buy_zone = {"price": fb, "reason": "52w low zone",
                    "desc": f"${fb} — near the 52-week low of ${w52l}, a level where buyers tend to step in"}

        # ── Value score ───────────────────────────────────────────────────────
        va_score = 0; va_good = []; va_warn = []
        if pe is not None:
            if pe < 15:    va_score += 2; va_good.append(f"PE {pe} — bargain (you pay ${pe} for every $1 the company earns)")
            elif pe < 25:  va_score += 1; va_good.append(f"PE {pe} — fairly priced (you pay ${pe} for every $1 earned)")
            elif pe <= 40: va_warn.append(f"PE {pe} — a bit pricey (paying ${pe} per $1 earned — needs strong growth)")
            else:          va_warn.append(f"PE {pe} — expensive (paying ${pe} per $1 earned — very high expectation baked in)")
        else:
            va_warn.append("No PE ratio — company may not yet be profitable")

        if div_yield >= 2:    va_score += 2; va_good.append(f"Dividend {div_yield}% — company pays you {div_yield}% per year just for holding!")
        elif div_yield >= 0.5: va_score += 1; va_good.append(f"Small dividend {div_yield}% — a little bonus for holding")
        else:                 va_warn.append("No dividend — company keeps profits to reinvest in growth")

        if ema200:
            if price > ema200: va_score += 2; va_good.append(f"Above 200-day average ${ema200} — long-term uptrend")
            else:              va_warn.append(f"Below 200-day average ${ema200} — long-term trend down")

        if r6m > 5:    va_score += 1; va_good.append(f"Up {r6m:.1f}% in 6 months — growing")
        elif r6m < -15: va_warn.append(f"Down {abs(r6m):.1f}% in 6 months — tough stretch")

        if mktcap_b and mktcap_b >= 50:
            va_score += 1; va_good.append(f"Market cap ${mktcap_b}B — large stable company")
        elif mktcap_b and mktcap_b < 5:
            va_warn.append(f"Small cap ${mktcap_b}B — higher risk and reward")

        va_score   = min(va_score, 10)
        va_verdict = "GOOD" if va_score >= 7 else "MIXED" if va_score >= 4 else "RISKY"

        # Value buy prices
        value_buy = None
        if va_verdict in ("RISKY", "MIXED"):
            vb_tech = vb_valuation = None
            tech_candidates = []
            if ema200 and ema200 < price: tech_candidates.append(("200-day average", round(ema200, 2)))
            if ema50  and ema50  < price: tech_candidates.append(("50-day average",  round(ema50,  2)))
            if tech_candidates:
                tech_candidates.sort(key=lambda x: price - x[1])
                t_lbl, t_lv = tech_candidates[0]
                vb_tech = {"price": t_lv,
                    "desc": f"${t_lv} ({t_lbl}) — a long-term price floor where patient buyers often enter"}
            if eps and eps > 0 and sector_pe_avg:
                fair_val = round(eps * sector_pe_avg, 2)
                pct_diff = round(abs(fair_val - price) / price * 100, 1)
                vb_valuation = {"price": fair_val,
                    "desc": (f"${fair_val} — at this price the PE would equal the {sector} "
                             f"sector average of {sector_pe_avg}× ({pct_diff}% "
                             f"{'below' if fair_val < price else 'above'} current price)")}
            value_buy = {"technical": vb_tech, "valuation": vb_valuation}

        # ── Pivots ────────────────────────────────────────────────────────────
        def _pv(h, l, c):
            p = round((h + l + c) / 3, 2)
            return {"pivot": p, "r1": round(2*p-l, 2), "r2": round(p+(h-l), 2),
                    "s1": round(2*p-h, 2), "s2": round(p-(h-l), 2)}

        def _near(px, lv, pct=1.5):
            return abs(px - lv) / lv * 100 <= pct if lv else False

        pivots = {}
        if len(hi) >= 5:
            pivots["weekly"]  = _pv(float(hi.iloc[-5:].max()),   float(lo.iloc[-5:].min()),   float(cl.iloc[-5]))
        if len(hi) >= 22:
            pivots["monthly"] = _pv(float(hi.iloc[-22:].max()),  float(lo.iloc[-22:].min()),  float(cl.iloc[-22]))
        if len(hi) >= 252:
            pivots["yearly"]  = _pv(float(hi.iloc[-252:].max()), float(lo.iloc[-252:].min()), float(cl.iloc[-252]))

        pivot_hits = []
        for tf, pv in pivots.items():
            for lbl, lv in [("Pivot", pv["pivot"]), ("Support 1", pv["s1"]),
                            ("Support 2", pv["s2"]), ("Resistance 1", pv["r1"]), ("Resistance 2", pv["r2"])]:
                if not _near(price, lv): continue
                is_sup  = lbl.startswith("Support") or lbl == "Pivot"
                bounced = r1d > 0 and is_sup
                pivot_hits.append({
                    "timeframe": tf, "level_name": lbl, "level": round(lv, 2),
                    "price": price, "dist_pct": round(abs(price - lv) / lv * 100, 2),
                    "vol_strong": vol_ratio >= 1.15, "vol_ratio": vol_ratio,
                    "bounced_up": bounced, "r1d": r1d,
                })

        ptxt = ("\n".join(
            f"- {h['timeframe'].capitalize()} {h['level_name']} at ${h['level']}: "
            f"price {'bounced UP from' if h['bounced_up'] else 'near'} this level, "
            f"vol {h['vol_ratio']:.1f}x normal"
            for h in pivot_hits
        ) if pivot_hits else "Price is not near any key weekly/monthly/yearly pivot level right now.")

        # ── News ──────────────────────────────────────────────────────────────
        headlines = []
        try:
            import requests as _req2
            import xml.etree.ElementTree as ET2
            after = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
            url   = (f"https://news.google.com/rss/search?q="
                     f"{_req2.utils.quote(company_name + ' stock after:' + after)}"
                     f"&hl=en-US&gl=US&ceid=US:en")
            r = _req2.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
            if r.ok:
                for item in ET2.fromstring(r.content).findall(".//item")[:4]:
                    t = item.findtext("title", "").strip()
                    if len(t) > 10: headlines.append(t)
        except Exception:
            pass
        if not headlines:
            try:
                for n in (yf.Ticker(symbol).news or [])[:3]:
                    if n.get("title"): headlines.append(n["title"])
            except Exception:
                pass
        news_block = "\n".join(f"- {h}" for h in headlines[:4]) or "No recent news."

        # ── Groq ──────────────────────────────────────────────────────────────
        log("Asking Groq AI for plain-English explanation…", "phase")

        def _buy_txt(buy_obj):
            if not buy_obj: return "Not applicable (verdict is GOOD)."
            lines = []
            if buy_obj.get("technical"):
                lines.append(f"  Technical support zone: {buy_obj['technical']['desc']}")
            if buy_obj.get("valuation"):
                lines.append(f"  Valuation fair value:   {buy_obj['valuation']['desc']}")
            return "\n".join(lines) if lines else "Not applicable."

        swing_buy_txt = _buy_txt(swing_buy_zone) if swing_buy_zone else "Not applicable (verdict is GOOD)."
        value_buy_txt = _buy_txt(value_buy)      if value_buy      else "Not applicable (verdict is GOOD)."

        prompt = f"""You are explaining stocks to a teenager or young adult learning to invest. Be their cool older sibling who knows about money — honest, encouraging, using real-life comparisons.

Company: {company_name} ({symbol})  |  Price: ${price}  |  Sector: {sector}

SWING SCORE (1 week – 1 month): {sw_score}/10  →  {sw_verdict}
Good signs: {'; '.join(sw_good) or 'none'}
Warnings:   {'; '.join(sw_warn) or 'none'}
SWING BUY ZONE (if waiting is better):
{swing_buy_txt}

VALUE SCORE (6 months – 1 year): {va_score}/10  →  {va_verdict}
Good signs: {'; '.join(va_good) or 'none'}
Warnings:   {'; '.join(va_warn) or 'none'}
VALUE BUY PRICE (if waiting is better):
{value_buy_txt}

PE vs SECTOR:
  Stock PE: {pe or 'N/A'}
  {sector} sector average PE: {sector_pe_avg}
  Comparison: {pe_vs_sector or 'N/A'}

KEY PRICE LEVELS (pivots):
{ptxt}

Recent news:
{news_block}

Write FOUR paragraphs (3-4 sentences each).

TONE RULES:
- Talk like a friendly older sibling, not a finance textbook
- Say "only 3 out of a possible 10 points" not "a score of 3/10"
- For price moves say "the price has gone up by X%" not "the stock appreciated X%"
- Use real-world analogies when they help (especially for pivots and PE)
- PE analogy: "paying $X for every $1 the company earns — like a lemonade stand that earns $1/day, you're paying $X to own it"
- If verdict is RISKY or MIXED, mention the buy zone/price naturally in the text

PARAGRAPH 1 — SWING: Should I buy for a quick trade (1 week–1 month)?
PARAGRAPH 2 — VALUE: Should I buy and hold (6 months–1 year)? ALWAYS compare PE to the {sector} sector average of {sector_pe_avg}. Mention dividend if any.
PARAGRAPH 3 — KEY PRICE LEVELS: Explain pivot/support/resistance with a real-world analogy, then explain what the specific levels mean for this stock. Does volume confirm it?
PARAGRAPH 4 — GROQ'S OWN TAKE: One paragraph where you share your honest overall view of this stock for a young investor starting out. Be real — is it a learning opportunity, a hold-and-wait, or something to skip for now?

End with one KEY TAKEAWAY sentence — the single most important thing for a young investor to know about this stock right now.

Return ONLY valid JSON, no markdown:
{{"swing_explanation":"...","value_explanation":"...","pivot_explanation":"...","overall_take":"...","key_takeaway":"..."}}"""

        sw_expl = va_expl = pv_expl = overall = takeaway = ""
        if _groq_available:
            try:
                raw = _groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4, max_tokens=1100
                ).choices[0].message.content.strip()
                p = json.loads(raw.replace("```json", "").replace("```", "").strip())
                sw_expl  = p.get("swing_explanation", "")
                va_expl  = p.get("value_explanation", "")
                pv_expl  = p.get("pivot_explanation", "")
                overall  = p.get("overall_take", "")
                takeaway = p.get("key_takeaway", "")
                log("Groq explanation ready!", "ok")
            except Exception as eg:
                log(f"Groq error: {eg}", "error")
                sw_expl  = f"Swing score {sw_score}/10 ({sw_verdict}). " + " ".join(sw_good[:2])
                va_expl  = f"Value score {va_score}/10 ({va_verdict}). " + " ".join(va_good[:2])
                pv_expl  = ptxt
                overall  = "Groq explanation unavailable."
                takeaway = "Always research before you invest!"
        else:
            sw_expl  = f"Swing: {sw_score}/10 ({sw_verdict}). " + " ".join(sw_good[:2])
            va_expl  = f"Value: {va_score}/10 ({va_verdict}). " + " ".join(va_good[:2])
            pv_expl  = ptxt
            overall  = "Install the groq package to get AI explanations: pip install groq"
            takeaway = "Always research before you invest!"

        result = dict(
            symbol=symbol, company_name=company_name, price=price, sector=sector,
            mktcap_b=mktcap_b, pe=pe, div_yield=div_yield, rsi=rsi,
            ema20=ema20, ema50=ema50, ema200=ema200,
            r1d=r1d, r5d=r5d, r1m=r1m, r6m=r6m, w52h=w52h, w52l=w52l, vol_ratio=vol_ratio,
            sector_pe_avg=sector_pe_avg, pe_vs_sector=pe_vs_sector,
            swing_score=sw_score, swing_verdict=sw_verdict,
            swing_signals=sw_good, swing_flags=sw_warn, swing_explanation=sw_expl,
            swing_buy_zone=swing_buy_zone,
            value_score=va_score, value_verdict=va_verdict,
            value_signals=va_good, value_flags=va_warn, value_explanation=va_expl,
            value_buy=value_buy,
            pivot_hits=pivot_hits, pivot_explanation=pv_expl,
            overall_take=overall, key_takeaway=takeaway, headlines=headlines,
        )

        log(f"Swing {sw_score}/10 {sw_verdict} · Value {va_score}/10 {va_verdict} · "
            f"{len(pivot_hits)} pivot hit(s)", "ok")

        with _novice_lock:
            _novice_jobs[jid]["result"] = result
            _novice_jobs[jid]["done"]   = True

    except Exception as e:
        with _novice_lock:
            _novice_jobs[jid]["log"].append({"msg": f"Error: {e}", "kind": "error"})
            _novice_jobs[jid]["done"] = True


# ── HTML ─────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartStocks — Novice Screener</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --sans: 'Inter', system-ui, sans-serif;
    --mono: 'JetBrains Mono', monospace;
    --bg:   #f5f5f5;
    --card: #ffffff;
    --border: #e5e7eb;
    --text:   #111827;
    --muted:  #6b7280;
    --faint:  #9ca3af;
    --grad:   linear-gradient(90deg, #c026d3, #e11d48, #f97316);
  }

  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ── Top nav ── */
  .nav {
    background: #fff;
    border-bottom: 1px solid var(--border);
    padding: 0 32px;
    height: 56px;
    display: flex;
    align-items: center;
    gap: 10px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .nav-logo {
    font-size: 1rem;
    font-weight: 800;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.5px;
  }
  .nav-sub {
    font-size: 0.7rem;
    color: var(--muted);
    border-left: 1px solid var(--border);
    padding-left: 10px;
    margin-left: 4px;
  }

  /* ── Layout ── */
  .page {
    max-width: 900px;
    margin: 0 auto;
    padding: 32px 24px 64px;
  }

  /* ── Hero search ── */
  .hero {
    text-align: center;
    margin-bottom: 32px;
  }
  .hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
  }
  .hero-sub {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 24px;
  }
  .search-wrap {
    position: relative;
    max-width: 520px;
    margin: 0 auto;
  }
  .search-input {
    width: 100%;
    padding: 14px 20px;
    font-size: 0.95rem;
    font-family: var(--sans);
    background: #fff;
    border: 2px solid var(--border);
    border-radius: 14px;
    outline: none;
    color: var(--text);
    transition: border-color .2s, box-shadow .2s;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
  }
  .search-input:focus {
    border-color: #c026d3;
    box-shadow: 0 0 0 4px rgba(192,38,211,.1);
  }
  .search-input::placeholder { color: var(--faint); }

  /* Dropdown */
  #nvDropdown {
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    left: 0; right: 0;
    background: #fff;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(0,0,0,.12);
    z-index: 200;
  }
  .dd-item {
    padding: 11px 16px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #f9fafb;
    transition: background .15s;
  }
  .dd-item:last-child { border-bottom: none; }
  .dd-item:hover { background: #f9fafb; }
  .dd-sym { font-family: var(--mono); font-weight: 700; color: #c026d3; font-size: 0.75rem; }
  .dd-name { color: var(--muted); font-size: 0.72rem; margin-left: 10px; }
  .dd-price { font-family: var(--mono); font-size: 0.7rem; color: #d97706; font-weight: 600; }

  /* Selected pill */
  #nvPill {
    display: none;
    max-width: 520px;
    margin: 10px auto 0;
    background: #f0fdf4;
    border: 1.5px solid #bbf7d0;
    border-radius: 8px;
    padding: 8px 14px;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
  }
  .pill-sym  { font-family: var(--mono); font-weight: 700; color: #16a34a; }
  .pill-price{ font-family: var(--mono); color: #d97706; font-weight: 600; }

  /* Analyse button */
  .btn-analyse {
    display: block;
    max-width: 520px;
    margin: 12px auto 0;
    width: 100%;
    padding: 13px;
    border: none;
    border-radius: 12px;
    background: var(--grad);
    color: #fff;
    font-family: var(--sans);
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 2px;
    cursor: not-allowed;
    opacity: .38;
    transition: opacity .2s, transform .1s;
  }
  .btn-analyse:not(:disabled) { cursor: pointer; opacity: 1; }
  .btn-analyse:not(:disabled):hover { transform: translateY(-1px); }
  .btn-analyse:not(:disabled):active { transform: translateY(0); }

  /* Progress log */
  #nvLog {
    display: none;
    max-width: 520px;
    margin: 14px auto 0;
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    max-height: 80px;
    overflow-y: auto;
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--muted);
  }

  /* ── Results panel ── */
  #nvPanel { display: none; }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
  }
  .result-company { font-size: 1.2rem; font-weight: 800; color: var(--text); }
  .result-sub { font-size: 0.72rem; color: var(--muted); margin-top: 3px; }
  .btn-back {
    background: #f9fafb; border: 1.5px solid var(--border);
    border-radius: 8px; color: var(--muted); padding: 7px 16px;
    cursor: pointer; font-size: 0.68rem; letter-spacing: 1px;
    transition: background .15s;
  }
  .btn-back:hover { background: #f3f4f6; }

  /* Stats strip */
  .stats-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-pill {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 12px;
    text-align: center;
    flex: 1;
    min-width: 70px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }
  .stat-lbl { font-family: var(--mono); font-size: 0.45rem; color: var(--faint); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px; }
  .stat-val { font-family: var(--mono); font-size: 0.78rem; font-weight: 700; color: var(--text); }

  /* Cards */
  .cards-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
  @media(max-width:640px){ .cards-grid { grid-template-columns: 1fr; } }

  .nv-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
  }
  .nv-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--grad);
  }
  .card-pivot::before { background: linear-gradient(90deg,#0ea5e9,#6366f1); }
  .card-overall::before { background: linear-gradient(90deg,#c026d3,#f97316); }

  .card-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .ct-swing  { background: linear-gradient(90deg,#7c3aed,#2563eb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .ct-value  { background: linear-gradient(90deg,#d97706,#f97316); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .ct-pivot  { background: linear-gradient(90deg,#0ea5e9,#6366f1); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .ct-overall{ background: linear-gradient(90deg,#c026d3,#f97316); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

  .nv-badge {
    font-family: var(--mono); font-size: 0.55rem; font-weight: 800;
    padding: 3px 10px; border-radius: 99px; letter-spacing: 1px;
  }
  .b-good  { background:#dcfce7; color:#16a34a; border:1px solid #bbf7d0; }
  .b-mixed { background:#fef9c3; color:#ca8a04; border:1px solid #fde68a; }
  .b-risky { background:#fee2e2; color:#dc2626; border:1px solid #fecaca; }

  .score-row { display:flex; align-items:center; gap:8px; margin-bottom:14px; }
  .score-lbl { font-family:var(--mono); font-size:0.48rem; color:var(--faint); letter-spacing:1px; width:36px; }
  .score-track { flex:1; height:6px; background:#f3f4f6; border-radius:99px; overflow:hidden; }
  .score-fill  { height:100%; border-radius:99px; transition:width .6s; }
  .fill-swing  { background:linear-gradient(90deg,#7c3aed,#2563eb); }
  .fill-value  { background:linear-gradient(90deg,#d97706,#f97316); }
  .score-num { font-family:var(--mono); font-size:0.65rem; font-weight:700; width:32px; text-align:right; }

  .nv-explain { font-size:0.8rem; line-height:1.85; color:#1f2937; margin-bottom:12px; }

  .sig-good { color:#16a34a; font-size:0.68rem; line-height:1.75; margin-bottom:3px; }
  .sig-warn { color:#ea580c; font-size:0.68rem; line-height:1.75; margin-bottom:3px; }

  /* Buy zone boxes */
  .buy-box-swing {
    display:none; margin-top:12px;
    background:#eef2ff; border:1.5px solid #c7d2fe; border-radius:10px; padding:10px 12px;
  }
  .buy-box-value {
    display:none; margin-top:12px;
    background:#fffbeb; border:1.5px solid #fde68a; border-radius:10px; padding:10px 12px;
  }
  .buy-box-label {
    font-size:0.5rem; font-weight:700; letter-spacing:1.5px; margin-bottom:6px;
  }
  .buy-box-swing .buy-box-label { color:#4f46e5; }
  .buy-box-value .buy-box-label { color:#b45309; }
  .buy-line { font-size:0.68rem; color:#374151; line-height:1.75; margin-bottom:6px; }
  .buy-sub-lbl { font-size:0.48rem; color:var(--faint); letter-spacing:1px; margin-bottom:2px; }

  /* Pivot rows */
  .pivot-row {
    display:flex; align-items:center; gap:10px;
    padding:7px 10px; background:#f9fafb; border:1px solid #f3f4f6;
    border-radius:8px; margin-bottom:6px; font-size:0.65rem;
  }
  .pivot-tf   { font-size:0.48rem; color:var(--faint); width:56px; text-transform:uppercase; letter-spacing:1px; }
  .pivot-type { color:var(--muted); font-size:0.62rem; width:100px; }
  .pivot-price{ font-family:var(--mono); font-weight:700; font-size:0.72rem; }
  .pivot-dir  { font-size:0.65rem; }
  .pivot-vol  { margin-left:auto; font-size:0.6rem; }

  /* Takeaway */
  .takeaway-box {
    background: linear-gradient(135deg,#fdf4ff,#fff7ed);
    border: 1.5px solid rgba(192,38,211,.2);
    border-radius: 14px;
    padding: 18px 20px;
    margin-top: 14px;
  }
  .takeaway-label {
    display:flex; align-items:center; gap:8px;
    font-size:0.58rem; font-weight:800; letter-spacing:2px;
    background:var(--grad); -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:8px;
  }
  .takeaway-text {
    font-size:0.88rem; color:#111827; line-height:1.8; font-weight:600;
  }
</style>
</head>
<body>

<nav class="nav">
  <span class="nav-logo">SmartStocks</span>
  <span class="nav-sub">🎓 Novice Screener</span>
</nav>

<div class="page">

  <!-- Hero + search -->
  <div class="hero" id="heroSection">
    <h1 class="hero-title">Novice Screener</h1>
    <p class="hero-sub">Type any company name — get a plain-English swing, value &amp; pivot analysis</p>

    <div class="search-wrap">
      <input id="nvSearch" class="search-input" type="text"
        placeholder="e.g. Apple, Nike, Microsoft, Tesla…"
        oninput="nvDebounce()" autocomplete="off">
      <div id="nvDropdown" style="display:none"></div>
    </div>

    <div id="nvPill" style="display:none">
      <span class="pill-sym" id="nvPillSym"></span>
      <span class="pill-price" id="nvPillPrice"></span>
    </div>

    <button id="nvRunBtn" class="btn-analyse" onclick="nvRun()" disabled>
      ▶ ANALYSE THIS STOCK
    </button>

    <div id="nvLog"></div>
  </div>

  <!-- Results panel -->
  <div id="nvPanel">

    <div class="result-header">
      <div>
        <div class="result-company" id="nvCompanyName">—</div>
        <div class="result-sub" id="nvHeaderSub">—</div>
      </div>
      <button class="btn-back" onclick="nvClose()">✕ NEW SEARCH</button>
    </div>

    <!-- Stats strip -->
    <div class="stats-strip">
      <div class="stat-pill" id="nvStatPrice"></div>
      <div class="stat-pill" id="nvStatRsi"></div>
      <div class="stat-pill" id="nvStatPe"></div>
      <div class="stat-pill" id="nvStatSectorPe"></div>
      <div class="stat-pill" id="nvStatDiv"></div>
      <div class="stat-pill" id="nvStatMktcap"></div>
      <div class="stat-pill" id="nvStatSector"></div>
    </div>

    <!-- Swing + Value -->
    <div class="cards-grid">

      <div class="nv-card">
        <div class="card-title-row">
          <span class="card-title ct-swing">⚡ SWING &nbsp;1 week – 1 month</span>
          <span id="nvSwingBadge" class="nv-badge"></span>
        </div>
        <div class="score-row">
          <span class="score-lbl">SCORE</span>
          <div class="score-track"><div id="nvSwingBar" class="score-fill fill-swing" style="width:0%"></div></div>
          <span id="nvSwingScore" class="score-num" style="color:#7c3aed"></span>
        </div>
        <p id="nvSwingText" class="nv-explain"></p>
        <div id="nvSwingSigs"></div>
        <div class="buy-box-swing" id="nvSwingBuyBox">
          <div class="buy-box-label">⏳ BETTER ENTRY ZONE</div>
          <div class="buy-line" id="nvSwingBuyTech"></div>
        </div>
      </div>

      <div class="nv-card">
        <div class="card-title-row">
          <span class="card-title ct-value">💰 VALUE &nbsp;6 months – 1 year</span>
          <span id="nvValueBadge" class="nv-badge"></span>
        </div>
        <div class="score-row">
          <span class="score-lbl">SCORE</span>
          <div class="score-track"><div id="nvValueBar" class="score-fill fill-value" style="width:0%"></div></div>
          <span id="nvValueScore" class="score-num" style="color:#d97706"></span>
        </div>
        <p id="nvValueText" class="nv-explain"></p>
        <div id="nvValueSigs"></div>
        <div class="buy-box-value" id="nvValueBuyBox">
          <div class="buy-box-label">⏳ BETTER BUY PRICES</div>
          <div id="nvValueBuyTech" style="display:none">
            <div class="buy-sub-lbl">📐 TECHNICAL SUPPORT</div>
            <div class="buy-line" id="nvValueBuyTechTxt"></div>
          </div>
          <div id="nvValueBuyVal" style="display:none">
            <div class="buy-sub-lbl">💹 FAIR VALUE (sector PE)</div>
            <div class="buy-line" id="nvValueBuyValTxt"></div>
          </div>
        </div>
      </div>

    </div>

    <!-- Pivot — full width -->
    <div class="nv-card card-pivot" style="margin-bottom:14px">
      <div class="card-title-row">
        <span class="card-title ct-pivot">📐 KEY PRICE LEVELS &nbsp;(weekly · monthly · yearly)</span>
        <span id="nvPivotBadge" class="nv-badge" style="background:#e0f2fe;color:#0369a1;border-color:#bae6fd">—</span>
      </div>
      <div id="nvPivotHits" style="margin-bottom:10px"></div>
      <p id="nvPivotText" class="nv-explain" style="margin-bottom:0;color:#374151"></p>
    </div>

    <!-- Overall take — full width -->
    <div class="nv-card card-overall" style="margin-bottom:14px">
      <div class="card-title-row">
        <span class="card-title ct-overall">🤔 GROQ'S OVERALL TAKE</span>
      </div>
      <p id="nvOverallText" class="nv-explain" style="margin-bottom:0"></p>
    </div>

    <!-- Key takeaway -->
    <div class="takeaway-box">
      <div class="takeaway-label"><span>⭐</span> KEY TAKEAWAY</div>
      <p id="nvTakeaway" class="takeaway-text"></p>
    </div>

  </div><!-- /nvPanel -->
</div><!-- /page -->

<script>
var _nvSym='', _nvName='', _nvTimer=null;

function nvDebounce(){ clearTimeout(_nvTimer); _nvTimer=setTimeout(nvSearch,320); }

async function nvSearch(){
  var q=document.getElementById('nvSearch').value.trim();
  var box=document.getElementById('nvDropdown');
  if(q.length<2){ box.style.display='none'; return; }

  // Show loading state immediately
  box.innerHTML='<div style="padding:10px 16px;font-size:0.72rem;color:#9ca3af">Searching…</div>';
  box.style.display='block';

  try{
    var r=await fetch('/api/novice_search',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q})
    });
    var d=await r.json();
    var results=d.results||[];
    if(!results.length){
      box.innerHTML='<div style="padding:10px 16px;font-size:0.72rem;color:#9ca3af">No US stocks found for "'+q+'" — try the ticker symbol (e.g. AAPL)</div>';
      return;
    }
    box.innerHTML='';
    results.forEach(function(res){
      var div=document.createElement('div');
      div.className='dd-item';
      div.innerHTML='<div><span class="dd-sym">'+res.symbol+'</span>'+
        '<span class="dd-name">'+res.name+'</span></div>'+
        '<span class="dd-price">'+(res.price?'$'+res.price:'')+'</span>';
      div.onclick=function(){ nvPick(res.symbol, res.name, res.price||0); };
      box.appendChild(div);
    });
    box.style.display='block';
  }catch(e){
    box.innerHTML='<div style="padding:10px 16px;font-size:0.72rem;color:#dc2626">Search error — check your connection</div>';
  }
}

function nvPick(sym,name,price){
  _nvSym=sym; _nvName=name;
  document.getElementById('nvDropdown').style.display='none';
  document.getElementById('nvSearch').value=name;
  var pill=document.getElementById('nvPill');
  pill.style.display='flex';
  document.getElementById('nvPillSym').textContent=sym+' — '+name;
  document.getElementById('nvPillPrice').textContent=price?'$'+price:'';
  var btn=document.getElementById('nvRunBtn');
  btn.disabled=false;
}

async function nvRun(){
  if(!_nvSym) return;
  var btn=document.getElementById('nvRunBtn');
  btn.disabled=true; btn.textContent='ANALYSING…';

  var log=document.getElementById('nvLog');
  log.innerHTML=''; log.style.display='block';
  document.getElementById('nvPanel').style.display='none';

  function addLog(msg,kind){
    var col=kind==='ok'?'#16a34a':kind==='error'?'#dc2626':
             kind==='phase'?'#7c3aed':'#9ca3af';
    log.innerHTML+='<div style="color:'+col+';margin-bottom:1px">'+msg+'</div>';
    log.scrollTop=log.scrollHeight;
  }

  try{
    var r=await fetch('/api/novice',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbol:_nvSym,name:_nvName})});
    var d=await r.json();
    var es=new EventSource('/api/novice_progress/'+d.job_id);
    es.onmessage=function(e){
      var data=JSON.parse(e.data);
      if(data.msg) addLog(data.msg,data.kind||'info');
      if(data.done){
        es.close();
        btn.disabled=false; btn.textContent='▶ ANALYSE THIS STOCK';
        log.style.display='none';
        if(data.result) nvRender(data.result);
      }
    };
    es.onerror=function(){
      es.close(); addLog('Connection error — try again','error');
      btn.disabled=false; btn.textContent='▶ ANALYSE THIS STOCK';
    };
  }catch(e){
    addLog('Error: '+e,'error');
    btn.disabled=false; btn.textContent='▶ ANALYSE THIS STOCK';
  }
}

function nvClose(){
  document.getElementById('nvPanel').style.display='none';
  document.getElementById('nvSearch').value='';
  document.getElementById('nvPill').style.display='none';
  document.getElementById('nvLog').style.display='none';
  _nvSym=''; _nvName='';
  var btn=document.getElementById('nvRunBtn');
  btn.disabled=true; btn.textContent='▶ ANALYSE THIS STOCK';
}

function setStat(id,label,val,color){
  var el=document.getElementById(id); if(!el) return;
  el.innerHTML='<div class="stat-lbl">'+label+'</div>'+
    '<div class="stat-val" style="color:'+(color||'#111827')+'">'+val+'</div>';
}

function badge(id,verdict){
  var el=document.getElementById(id); if(!el) return;
  el.textContent=verdict;
  el.className='nv-badge '+(verdict==='GOOD'?'b-good':verdict==='MIXED'?'b-mixed':'b-risky');
}

function nvRender(res){
  document.getElementById('nvCompanyName').textContent=res.company_name+' ('+res.symbol+')';
  document.getElementById('nvHeaderSub').textContent=
    '$'+res.price+'  ·  '+res.sector+'  ·  Sector avg PE: '+res.sector_pe_avg;

  // Stats
  setStat('nvStatPrice','PRICE','$'+res.price);
  setStat('nvStatRsi','RSI',res.rsi,
    res.rsi<40?'#dc2626':res.rsi>70?'#d97706':'#16a34a');
  var peCol='#111827';
  if(res.pe&&res.sector_pe_avg){
    var diff=(res.pe-res.sector_pe_avg)/res.sector_pe_avg*100;
    peCol=diff>20?'#dc2626':diff<-15?'#16a34a':'#d97706';
  }
  setStat('nvStatPe','STOCK P/E',res.pe||'N/A',peCol);
  setStat('nvStatSectorPe','SECTOR P/E',res.sector_pe_avg||'N/A','#9ca3af');
  setStat('nvStatDiv','DIVIDEND',res.div_yield?res.div_yield+'%':'None',
    res.div_yield>=2?'#16a34a':'#111827');
  setStat('nvStatMktcap','MKT CAP',res.mktcap_b?'$'+res.mktcap_b+'B':'N/A');
  setStat('nvStatSector','SECTOR',res.sector||'—','#9ca3af');

  // Swing
  badge('nvSwingBadge',res.swing_verdict);
  document.getElementById('nvSwingScore').textContent=res.swing_score+'/10';
  document.getElementById('nvSwingBar').style.width=(res.swing_score*10)+'%';
  document.getElementById('nvSwingText').textContent=res.swing_explanation;
  var sd='';
  (res.swing_signals||[]).forEach(function(s){sd+='<div class="sig-good">✓ '+s+'</div>';});
  (res.swing_flags||[]).forEach(function(f){sd+='<div class="sig-warn">⚠ '+f+'</div>';});
  document.getElementById('nvSwingSigs').innerHTML=sd;
  var swBox=document.getElementById('nvSwingBuyBox');
  if((res.swing_verdict==='RISKY'||res.swing_verdict==='MIXED')&&res.swing_buy_zone){
    document.getElementById('nvSwingBuyTech').innerHTML=
      '📍 <strong style="color:#4f46e5">$'+res.swing_buy_zone.price+'</strong> — '+res.swing_buy_zone.desc;
    swBox.style.display='block';
  } else { swBox.style.display='none'; }

  // Value
  badge('nvValueBadge',res.value_verdict);
  document.getElementById('nvValueScore').textContent=res.value_score+'/10';
  document.getElementById('nvValueBar').style.width=(res.value_score*10)+'%';
  document.getElementById('nvValueText').textContent=res.value_explanation;
  var vd='';
  (res.value_signals||[]).forEach(function(s){vd+='<div class="sig-good">✓ '+s+'</div>';});
  (res.value_flags||[]).forEach(function(f){vd+='<div class="sig-warn">⚠ '+f+'</div>';});
  document.getElementById('nvValueSigs').innerHTML=vd;
  var vBox=document.getElementById('nvValueBuyBox');
  if((res.value_verdict==='RISKY'||res.value_verdict==='MIXED')&&res.value_buy){
    var vb=res.value_buy; var hasAny=false;
    var tEl=document.getElementById('nvValueBuyTech');
    var tTxt=document.getElementById('nvValueBuyTechTxt');
    if(vb.technical){
      tTxt.innerHTML='📍 <strong style="color:#b45309">$'+vb.technical.price+'</strong> — '+vb.technical.desc;
      tEl.style.display='block'; hasAny=true;
    } else { tEl.style.display='none'; }
    var vEl=document.getElementById('nvValueBuyVal');
    var vTxt=document.getElementById('nvValueBuyValTxt');
    if(vb.valuation){
      vTxt.innerHTML='📍 <strong style="color:#b45309">$'+vb.valuation.price+'</strong> — '+vb.valuation.desc;
      vEl.style.display='block'; hasAny=true;
    } else { vEl.style.display='none'; }
    vBox.style.display=hasAny?'block':'none';
  } else { vBox.style.display='none'; }

  // Pivots
  var hits=res.pivot_hits||[];
  var pb=document.getElementById('nvPivotBadge');
  var ph=document.getElementById('nvPivotHits');
  if(!hits.length){
    pb.textContent='NO KEY LEVELS NEARBY';
    pb.style.cssText='background:#f3f4f6;color:#6b7280;border:1px solid #e5e7eb';
    ph.innerHTML='<div style="font-size:0.68rem;color:#9ca3af;font-style:italic;padding:4px 0">'+
      'Not near any key price level right now — that\'s fine, it means the price can move freely.</div>';
  } else {
    pb.textContent=hits.length+' KEY LEVEL'+(hits.length>1?'S':'')+' NEARBY';
    pb.style.cssText='background:#e0f2fe;color:#0369a1;border:1px solid #bae6fd';
    var hh='';
    hits.forEach(function(h){
      var isS=h.level_name.indexOf('Support')!==-1||h.level_name==='Pivot';
      var lc=isS?'#16a34a':'#dc2626';
      var tl=isS?'🟢 Support floor':'🔴 Resistance ceiling';
      var vn=h.vol_strong
        ?'<span style="color:#16a34a">vol '+h.vol_ratio.toFixed(1)+'× ✓</span>'
        :'<span style="color:#9ca3af">vol '+h.vol_ratio.toFixed(1)+'×</span>';
      hh+='<div class="pivot-row">'+
        '<span class="pivot-tf">'+h.timeframe+'</span>'+
        '<span class="pivot-type">'+tl+'</span>'+
        '<span class="pivot-price" style="color:'+lc+'">$'+h.level+'</span>'+
        '<span class="pivot-dir" style="color:'+(h.bounced_up?'#16a34a':'#d97706')+'">'+
        (h.bounced_up?'↑ bounced up':'~ nearby')+'</span>'+
        '<span class="pivot-vol">'+vn+'</span></div>';
    });
    ph.innerHTML=hh;
  }
  document.getElementById('nvPivotText').textContent=res.pivot_explanation||'';

  // Overall take
  document.getElementById('nvOverallText').textContent=res.overall_take||'';

  // Takeaway
  document.getElementById('nvTakeaway').textContent=res.key_takeaway;

  // Show panel, hide hero search
  document.getElementById('heroSection').style.display='none';
  document.getElementById('nvPanel').style.display='block';
  window.scrollTo(0,0);
}

// Close dropdown on outside click
document.addEventListener('click',function(e){
  if(!e.target.closest('.search-wrap'))
    document.getElementById('nvDropdown').style.display='none';
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║   SmartStocks Novice Screener                    ║")
    print("  ║   http://localhost:5008                          ║")
    print("  ╚══════════════════════════════════════════════════╝\n")
    print("  Groq AI:", "✓ available" if _groq_available else "✗ not available (pip install groq)")
    print()
    app.run(debug=False, port=5008, threaded=True)