# app.py
from __future__ import annotations
from flask import Flask, request, jsonify
import math
import re
from typing import List, Dict, Any

from routes import app

BULL_WORDS = {
    # strong policy/corporate buys / inflows
    "buy", "buys", "bought", "accumulat", "inflow", "approval", "approves",
    "etf", "spot etf", "adds", "added", "reserve", "treasury", "adopt", "adopts",
    "bull", "bullish", "all-time high", "ath", "net inflows", "funds swell",
    "buying power", "grants", "growth", "development", "joins", "record inflows",
    "open interest all-time high", "new address custody", "bombshell inflows",
    "over subscribed", "raises"
}
BEAR_WORDS = {
    # hacks, delays, outflows, stagflation, sell-off
    "hack", "hacked", "exploit", "stolen", "outflow", "sell-off", "dump",
    "delay", "delayed", "auditor", "resigns", "stagflation", "risk-off",
    "pressure", "tanks", "down", "bear", "bearish", "lawsuit", "fraud",
    "liquidation", "defaults", "halts", "halted", "withdrawal pause"
}

SOURCE_WEIGHTS = {
    # simple trust/impact prior
    "Twitter": 1.0,
    "X": 1.0,
    "Blogs": 1.1,
    "News": 1.2,
    "Bloomberg": 1.3,
    "COINDESK": 1.25,
    "CoinDesk": 1.25,
    "The Block": 1.25,
    "DECRYPT": 1.1,
    "COINTELEGRAPH": 1.05,
}

SIGNAL_WEIGHTS = dict(
    momentum=0.70,      # prev 3m momentum
    gap=0.45,           # gap from last prev close to obs0 open
    obs_body=0.35,      # obs0 body direction
    obs_follow=0.40,    # obs1+2 follow-through
    wick_reversal=0.35, # exhaustion reversal
    vol_spike=0.25,     # relative volume spike
    sentiment=0.65,     # title/source sentiment prior
    regime=0.25,        # volatility regime tilt
)

def safe_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def body_sign(c):
    if c is None: return 0.0
    o = safe_get(c, "open", None)
    cl = safe_get(c, "close", None)
    if o is None or cl is None: return 0.0
    if cl > o: return 1.0
    if cl < o: return -1.0
    return 0.0

def candle_range(c):
    if c is None: return 0.0
    h = safe_get(c, "high", None); l = safe_get(c, "low", None)
    if h is None or l is None: return 0.0
    return max(1e-9, h - l)

def close_of(c):
    return safe_get(c, "close", None)

def mean(xs: List[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(xs)/len(xs) if xs else 0.0

def sentiment_score(title: str, source: str) -> float:
    if not title: title = ""
    t = title.lower()
    bull_hits = sum(1 for w in BULL_WORDS if w in t)
    bear_hits = sum(1 for w in BEAR_WORDS if w in t)
    raw = bull_hits - bear_hits

    # polarity normalization
    if raw == 0:
        # heuristic: $ up verbs
        if re.search(r"\b(rises|spikes|soars|pumps|surges|breaks higher|edges higher)\b", t): raw += 1
        if re.search(r"\b(falls|tanks|plunges|drops|sinks|pressure|sell[- ]?off)\b", t): raw -= 1

    # source prior
    sw = 1.0
    for k, v in SOURCE_WEIGHTS.items():
        if k.lower() in (source or "").lower() or k.lower() in t:
            sw = max(sw, v)

    return raw * sw

def volatility_regime(prev: List[dict]) -> float:
    # ATR-lite across prev candles
    rngs = [candle_range(c) for c in prev if candle_range(c) > 0]
    closes = [close_of(c) for c in prev if close_of(c) is not None]
    if not rngs or not closes: return 0.0
    atr = mean(rngs)
    px = closes[-1]
    volpct = atr / max(1e-9, px)
    # high vol favors continuation; low vol favors mean reversion -> small tilt
    return 1.0 if volpct > 0.004 else (-0.5 if volpct < 0.0015 else 0.0)

def compute_features(ev: dict) -> Dict[str, float]:
    prev = safe_get(ev, "previous_candles", []) or []
    obs = safe_get(ev, "observation_candles", []) or []

    p3 = prev[-3:] if len(prev) >= 3 else prev
    if len(p3) < 1 or len(obs) < 1:
        return dict(score=0.0, conf=0.0, dir=0.0)

    last_prev_close = close_of(p3[-1])
    if last_prev_close is None: last_prev_close = 0.0

    # Momentum (prev 3 closes)
    prev_closes = [close_of(c) for c in p3 if close_of(c) is not None]
    mom = 0.0
    if len(prev_closes) >= 2:
        mom = prev_closes[-1] - prev_closes[0]

    # Gap at obs0 open vs last prev close
    obs0 = obs[0]
    gap = safe_get(obs0, "open", last_prev_close) - last_prev_close

    # Observation body at entry candle
    obs0_body = safe_get(obs0, "close", 0.0) - safe_get(obs0, "open", 0.0)

    # Follow-through using obs1, obs2 close vs open
    obs_follow = 0.0
    if len(obs) >= 2:
        obs1 = obs[1]; obs_follow += (safe_get(obs1, "close", 0.0) - safe_get(obs1, "open", 0.0))
    if len(obs) >= 3:
        obs2 = obs[2]; obs_follow += (safe_get(obs2, "close", 0.0) - safe_get(obs2, "open", 0.0))

    # Wick-based reversal/exhaustion on obs0
    h0 = safe_get(obs0, "high", None); l0 = safe_get(obs0, "low", None)
    o0 = safe_get(obs0, "open", None); c0 = safe_get(obs0, "close", None)
    wick_rev = 0.0
    if None not in (h0, l0, o0, c0):
        body_top = max(o0, c0); body_bot = min(o0, c0)
        upper = max(0.0, h0 - body_top)
        lower = max(0.0, body_bot - l0)
        rng0 = candle_range(obs0)
        # If big upper wick after up-move → likely reversal down
        if rng0 > 0:
            up_exhaust = (upper / rng0) - (lower / rng0)
            # Negative = bearish exhaustion, Positive = bullish exhaustion
            wick_rev = -up_exhaust  # sign: positive favors LONG, negative favors SHORT (exhaustion against body)

    # Volume spike vs previous
    prev_vols = [safe_get(c, "volume", 0.0) for c in p3]
    v_avg = mean(prev_vols)
    v_last = safe_get(p3[-1], "volume", v_avg)
    vol_spike = 0.0
    if v_avg > 0:
        vol_spike = (v_last / v_avg) - 1.0  # >0 means spike

    # Sentiment
    sent = sentiment_score(safe_get(ev, "title", ""), safe_get(ev, "source", ""))

    # Regime
    reg = volatility_regime(p3)

    # Normalize by price scale
    px = last_prev_close if last_prev_close else 1.0
    mom_n = mom / px
    gap_n = gap / px
    obs0_n = obs0_body / px
    follow_n = obs_follow / max(px, 1.0)

    # Signed signal
    s = (
        SIGNAL_WEIGHTS["momentum"]    * mom_n +
        SIGNAL_WEIGHTS["gap"]         * gap_n +
        SIGNAL_WEIGHTS["obs_body"]    * obs0_n +
        SIGNAL_WEIGHTS["obs_follow"]  * follow_n +
        SIGNAL_WEIGHTS["wick_reversal"] * wick_rev +
        SIGNAL_WEIGHTS["vol_spike"]   * vol_spike +
        SIGNAL_WEIGHTS["sentiment"]   * (0.0025 * sent) +  # scale news to price-normalized domain
        SIGNAL_WEIGHTS["regime"]      * reg * 0.5
    )

    # Extra microstructure heuristics (edge cases):
    # 1) Bearish engulf in obs1 after strong up obs0 → favor SHORT
    if len(obs) >= 2:
        if body_sign(obs0) > 0 and body_sign(obs[1]) < 0:
            if safe_get(obs[1], "close", 0) < safe_get(obs0, "open", 0) and safe_get(obs[1], "open", 0) > safe_get(obs0, "close", 0):
                s -= 0.002  # slight bearish tilt

    # 2) Strong trend + aligned sentiment boosts
    if (mom_n + gap_n + obs0_n + follow_n) > 0.002 and sent > 0:
        s += 0.0015
    if (mom_n + gap_n + obs0_n + follow_n) < -0.002 and sent < 0:
        s -= 0.0015

    # Confidence: magnitude + dispersion/volatility context
    rngs = [candle_range(c) for c in p3 if candle_range(c) > 0]
    rng_avg = mean(rngs) or (0.002 * px)
    conf = abs(s) * (1.0 + min(1.5, rng_avg / (0.0015 * px))) * (1.0 + min(1.0, abs(vol_spike)))

    return dict(score=s, conf=conf, dir=1.0 if s >= 0 else -1.0)

def decide(ev: dict) -> Dict[str, Any]:
    f = compute_features(ev)
    decision = "LONG" if f["dir"] >= 0 else "SHORT"
    return {
        "id": safe_get(ev, "id", None),
        "decision": decision,
        "confidence": float(f["conf"]),
        "_score": float(f["score"]),
    }

def pick_top_n(results: List[Dict[str, Any]], n: int = 50) -> List[Dict[str, Any]]:
    # Drop items without ids
    results = [r for r in results if r.get("id") is not None]
    # Deduplicate by id, keep highest confidence
    best_by_id = {}
    for r in results:
        rid = r["id"]
        if rid not in best_by_id or r["confidence"] > best_by_id[rid]["confidence"]:
            best_by_id[rid] = r
    uniq = list(best_by_id.values())

    # Sort by confidence desc, then tie-break by absolute score desc, then id asc
    uniq.sort(key=lambda x: (x["confidence"], abs(x["_score"]), -(x["id"] if isinstance(x["id"], (int, float)) else 0)), reverse=True)

    # Ensure exactly n elements; if fewer available, pad deterministically by toggling lowest-confidence decisions (rare)
    top = uniq[:n]
    if len(top) < n:
        pool = [r for r in uniq[n:]] or []
        while len(top) < n and pool:
            top.append(pool.pop(0))
        # still short? fabricate toggled low-conf picks from existing to keep exactly 50
        i = 0
        while len(top) < n and i < len(uniq):
            clone = dict(uniq[i])
            clone["decision"] = "LONG" if clone["decision"] == "SHORT" else "SHORT"
            clone["confidence"] = 0.0
            top.append(clone)
            i += 1

    # Strip internals
    return [{"id": r["id"], "decision": r["decision"]} for r in top[:n]]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error": "Body must be a JSON array of events"}), 400

        results = []
        for ev in data:
            try:
                results.append(decide(ev))
            except Exception:
                # robust to bad rows; skip but keep moving
                continue

        selected = pick_top_n(results, n=50)

        # Final shape
        return jsonify(selected), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

