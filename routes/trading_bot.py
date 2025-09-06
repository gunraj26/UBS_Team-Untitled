from flask import request, jsonify

from routes import app

# --- Simple keyword sentiment lexicon (very lightweight) ---
BULLISH_WORDS = {
    "approval","adopt","partnership","launch","bullish","record",
    "increase","upgrades","reserve","buyback","investment","hiring",
    "surge","growth","win","positive","raises","tops","beat","beats",
    "support","signs","approves","allocates","backs","funds"
}
BEARISH_WORDS = {
    "ban","hack","breach","lawsuit","probe","investigation","bearish",
    "selloff","cuts","cut","downgrade","liquidation","exploit","exit",
    "halt","halts","suspends","suspend","block","blocks","restrict",
    "tax","taxes","negative","fine","fines","bankrupt","insolvency",
    "outage","bug","vulnerability"
}

def text_sentiment(title: str) -> int:
    """Very naive sentiment: +1 bullish, -1 bearish, 0 neutral."""
    t = (title or "").lower()
    bull = any(w in t for w in BULLISH_WORDS)
    bear = any(w in t for w in BEARISH_WORDS)
    if bull and not bear:
        return 1
    if bear and not bull:
        return -1
    return 0

def last_previous_candle(ev):
    pcs = ev.get("previous_candles") or []
    return pcs[-1] if pcs else None

def first_obs_candle(ev):
    ocs = ev.get("observation_candles") or []
    return ocs[0] if ocs else None

def pre_news_momentum(ev):
    """Return simple momentum sign from last two previous candles (close - open sum)."""
    pcs = ev.get("previous_candles") or []
    if len(pcs) >= 2:
        m1 = (pcs[-1]["close"] - pcs[-1]["open"])
        m2 = (pcs[-2]["close"] - pcs[-2]["open"])
        return 1 if (m1 + m2) > 0 else -1 if (m1 + m2) < 0 else 0
    if len(pcs) == 1:
        m = pcs[-1]["close"] - pcs[-1]["open"]
        return 1 if m > 0 else -1 if m < 0 else 0
    return 0

def candle_strength(c):
    """Return body and wick metrics for a single candle dict."""
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
    body = cl - o
    rng = max(h - l, 1e-9)
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    body_pct = body / max(o, 1e-9)
    rng_pct = rng / max(o, 1e-9)
    upper_wick_ratio = upper_wick / rng
    lower_wick_ratio = lower_wick / rng
    return body, body_pct, rng_pct, upper_wick_ratio, lower_wick_ratio

def decide_for_event(ev):
    """
    Produce (decision, confidence_score).
    Heuristic:
      1) Use first observation candle shape + pre-news momentum for mean-reversion vs trend.
      2) Fall back to title sentiment.
      3) Fall back to momentum.
    Confidence is based on candle range % and breakout size.
    """
    obs = first_obs_candle(ev)
    prev = last_previous_candle(ev)
    if not obs or not prev:
        # if missing data, skip (very low confidence default)
        return "LONG", 0.0

    # Features
    pmom = pre_news_momentum(ev)
    body, body_pct, rng_pct, uwr, lwr = candle_strength(obs)

    # Breakout vs last previous high/low
    breakout_up = (obs["high"] - prev["high"]) / max(prev["high"], 1e-9)
    breakout_dn = (prev["low"] - obs["low"]) / max(prev["low"], 1e-9)

    # Title sentiment
    senti = text_sentiment(ev.get("title", ""))

    # Core rules
    decision = None

    # Strong bearish engulf / rejection after upside momentum -> SHORT (mean reversion)
    if body < 0 and uwr > 0.6 and pmom > 0:
        decision = "SHORT"
    # Strong bullish with long lower wick after downside momentum -> LONG (mean reversion)
    elif body > 0 and lwr > 0.6 and pmom < 0:
        decision = "LONG"
    # Trend-follow if strong body without long wick
    elif body > 0 and uwr < 0.4:
        decision = "LONG"
    elif body < 0 and lwr < 0.4:
        decision = "SHORT"

    # If undecided, use sentiment
    if decision is None:
        if senti > 0:
            decision = "LONG"
        elif senti < 0:
            decision = "SHORT"

    # If still undecided, follow momentum
    if decision is None:
        decision = "LONG" if pmom >= 0 else "SHORT"

    # Confidence: larger candles + clean breakout => higher
    confidence = abs(body_pct) + rng_pct + max(breakout_up, breakout_dn, 0)
    return decision, float(confidence)

@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        events = request.get_json(force=True)
        if not isinstance(events, list):
            return jsonify({"error": "Payload must be a JSON array of events."}), 400

        decisions = []
        scored = []
        for ev in events:
            # Guard missing id
            ev_id = ev.get("id")
            if ev_id is None:
                continue
            decision, conf = decide_for_event(ev)
            scored.append((conf, {"id": ev_id, "decision": decision}))

        # Pick top 50 by confidence
        scored.sort(key=lambda x: x[0], reverse=True)
        top_50 = [item for _, item in scored[:50]]

        # If fewer than 50 valid, just return what we have
        return jsonify(top_50), 200

    except Exception as e:
        # Fail-safe: never crash the evaluator
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the app (for local testing)
    # In production, serve via gunicorn/uvicorn, etc.
    app.run(host="0.0.0.0", port=8000)
