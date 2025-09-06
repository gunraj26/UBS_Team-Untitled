# routes/trading_bot.py

import json
import logging
from flask import request
from routes import app

logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------

POSITIVE_TERMS = {
    "bull", "bullish", "surge", "rally", "pump", "breakout", "record", "ath",
    "adopt", "adoption", "approve", "approval", "approved", "greenlight",
    "partnership", "integrate", "integration", "support", "back", "backs",
    "invest", "investment", "buy", "buys", "accumulate", "long", "reserve",
    "liquidity", "stimulus", "etf", "spot etf", "cut rates", "rate cut",
    "inflation cools", "easing", "halving", "merge", "upgrade", "optimism",
    "positive", "beat", "beats", "above expectations", "surprise",
}
NEGATIVE_TERMS = {
    "bear", "bearish", "dump", "crash", "plunge", "selloff", "sell-off",
    "sell", "sells", "liquidation", "liquidations", "liquidated",
    "ban", "bans", "prohibit", "restrict", "restriction", "lawsuit", "sue",
    "investigation", "probe", "hack", "exploit", "outage", "halt", "suspend",
    "delay", "delays", "reject", "rejection", "fraud", "scam", "rug", "down",
    "downgrade", "defaults", "default", "bankrupt", "insolvent",
    "tightening", "rate hike", "hike rates", "inflation rises", "fine", "fines",
    "negative", "miss", "misses", "below expectations", "tax", "taxes",
}
NOISY_SOURCES = {"twitter", "x", "reddit", "tg", "telegram"}
CURATED_SOURCES = {"coindesk", "cointelegraph", "bloomberg", "reuters", "ft", "wsj"}

W_MOMENTUM = 0.40
W_CANDLE_RET = 0.25
W_VOL_SPIKE = 0.15
W_SENTIMENT = 0.20

# -----------------------------
# Helpers
# -----------------------------

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def parse_candle(c):
    return {
        "open": safe_float(c.get("open")),
        "high": safe_float(c.get("high")),
        "low": safe_float(c.get("low")),
        "close": safe_float(c.get("close")),
        "volume": safe_float(c.get("volume")),
        "timestamp": int(c.get("timestamp", 0)),
    }

def headline_sentiment(title, source):
    if not isinstance(title, str):
        return 0.0
    t = title.lower()
    pos_hits = sum(1 for term in POSITIVE_TERMS if term in t)
    neg_hits = sum(1 for term in NEGATIVE_TERMS if term in t)
    base = (pos_hits - neg_hits) / (1.0 + pos_hits + neg_hits)

    base += min(t.count("!"), 3) * 0.05
    if (len(title) >= 8 and sum(ch.isupper() for ch in title) / max(len(title), 1) > 0.35):
        base += 0.05

    s = (source or "").lower()
    if s in NOISY_SOURCES:
        base *= 0.7
    elif s in CURATED_SOURCES:
        base *= 1.2

    return max(min(base, 1.0), -1.0)

def zscore_list(values):
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    std = var ** 0.5
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]

def compute_features(event):
    try:
        event_id = event.get("id")
        prev = event.get("previous_candles") or []
        obs = event.get("observation_candles") or []
        if not prev or not obs:
            return None

        prev_candles = [parse_candle(c) for c in prev if isinstance(c, dict)]
        obs_candles = [parse_candle(c) for c in obs if isinstance(c, dict)]
        if not prev_candles or not obs_candles:
            return None

        last_prev = prev_candles[-1]
        first_obs = obs_candles[0]

        if last_prev["close"] <= 0 or first_obs["open"] <= 0:
            return None

        momentum = (first_obs["close"] - last_prev["close"]) / last_prev["close"]
        candle_ret = (first_obs["close"] - first_obs["open"]) / first_obs["open"]
        avg_prev_vol = sum(c["volume"] for c in prev_candles) / max(len(prev_candles), 1)
        vol_spike = (first_obs["volume"] / avg_prev_vol - 1.0) if avg_prev_vol > 0 else 0.0
        sentiment = headline_sentiment(str(event.get("title", "")), str(event.get("source", "")))

        return {
            "id": event_id,
            "momentum": momentum,
            "candle_ret": candle_ret,
            "vol_spike": vol_spike,
            "sentiment": sentiment,
        }
    except Exception as e:
        logger.error(f"compute_features error for id={event.get('id')}: {e}")
        return None

def score_event(f, zctx):
    mz = zctx["momentum"].get(f["id"], 0.0)
    cz = zctx["candle_ret"].get(f["id"], 0.0)
    vz = zctx["vol_spike"].get(f["id"], 0.0)
    return (W_MOMENTUM * mz) + (W_CANDLE_RET * cz) + (W_VOL_SPIKE * vz) + (W_SENTIMENT * f["sentiment"])

# -----------------------------
# Route
# -----------------------------

@app.route('/trading-bot', methods=['POST'])
def trading_bot():
    payload = request.get_json(silent=True)
    if not isinstance(payload, list):
        return json.dumps({"error": "Body must be a JSON array of events"}), 400

    features = [compute_features(ev) for ev in payload if isinstance(ev, dict)]
    features = [f for f in features if f]

    if not features:
        return json.dumps({"error": "No valid events found"}), 400

    momentum_z = zscore_list([f["momentum"] for f in features])
    candle_ret_z = zscore_list([f["candle_ret"] for f in features])
    vol_spike_z = zscore_list([f["vol_spike"] for f in features])

    zctx = {
        "momentum": {f["id"]: z for f, z in zip(features, momentum_z)},
        "candle_ret": {f["id"]: z for f, z in zip(features, candle_ret_z)},
        "vol_spike": {f["id"]: z for f, z in zip(features, vol_spike_z)},
    }

    scored = []
    for f in features:
        s = score_event(f, zctx)
        decision = "LONG" if s > 0 else "SHORT"
        scored.append((f["id"], s, decision))

    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    top = scored[: min(50, len(scored))]

    return json.dumps([{"id": _id, "decision": decision} for (_id, _score, decision) in top])
