# app.py - Enhanced Trading Bot
from flask import Flask, request, jsonify
import math
import re
from statistics import mean, pstdev
from collections import defaultdict

from routes import app

# =========================
# Enhanced Configuration
# =========================
RISK_EPS = 1e-12
MAX_RANGE_GLITCH_PCT = 0.15     # Tightened from 0.18
CIRCUIT_MEGA_GAP = 0.028        # Reduced from 0.035 for better sensitivity
IMPULSE_STRONG = 0.0035         # Reduced from 0.0045 for earlier detection
BODY_STRONG = 0.52              # Reduced from 0.58
VOLUME_SPIKE_HOT = 3.2          # Reduced from 4.0
WICK_TRAP = 0.32                # Reduced from 0.36
BOLL_Z_FADE = 1.8               # Reduced from 2.1 for more sensitive fading
BOLL_Z_TREND = 0.85             # Reduced from 1.0
BREAK_HOLD_MIN = 0.0006         # Reduced from 0.0008
FOLLOW_THRU_OK = 0.0005         # Reduced from 0.0006
FOLLOW_THRU_STRONG = 0.0012     # Reduced from 0.0015
SLOPE_MIN = 0.0015              # Reduced from 0.0020

# Enhanced market regime detection
VOLATILITY_LOW = 0.008
VOLATILITY_HIGH = 0.025
VOLUME_DRY = 0.7
VOLUME_FLOOD = 5.0

# Time-based adjustments (assuming timestamps represent different market sessions)
ASIAN_FADE_BIAS = 1.15
LONDON_TREND_BIAS = 1.10
NY_VOLATILITY_BIAS = 1.05

BLOCKLIST_IDS = set()

# Enhanced promotional detection
PROMO_WORDS = {
    "airdrop","points","quest","campaign","referral","bonus","mint","whitelist",
    "allowlist","giveaway","xp","season","farm","farming","stake to earn","rewards",
    "party","pool party","poolparty","listing soon","perp points","governance token",
    "testnet","mainnet launch","alpha","beta","early access","exclusive","limited time",
    "claim now","don't miss","last chance"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_WORDS), re.IGNORECASE)

# Enhanced credibility terms
LOW_CRED_TERMS = {
    "rumor","unconfirmed","parody","fake","shitpost","spoof","satire","april fools",
    "allegedly","reportedly","supposedly","claims","speculation","insider",
    "anonymous source","leak","whispers"
}

AUTHORITATIVE = {
    "BLOOMBERG","REUTERS","WALL STREET JOURNAL","WSJ","FINANCIAL TIMES","FT",
    "NIKKEI","AP","AXIOS","BARRON","BARRONS","CNBC","GUARDIAN","MARKETWATCH",
    "BBC","CNN","FORBES"
}
CRYPTO_TIER2 = {"COINDESK","THE BLOCK","COINTELEGRAPH","DECRYPT","CRYPTOSLATE"}
ALLOWED_TWITTER = {
    "tier10k","WuBlockchain","CoinDesk","TheBlock__","Reuters","AP","Bloomberg",
    "CNBC","MarketWatch","FT","WSJ","tier10k_official","crypto_insider"
}

# Enhanced sentiment terms
GOOD_TERMS = {
    "approve","approves","approved","approval","etf","inflow","flows","treasury",
    "reserve","reserves","executive order","legalize","legalizes","adopt","adopts",
    "integrate","integrates","support","adds support","partnership","upgrade","merge",
    "launch","launches","institutional","blackrock","fidelity","google","amazon",
    "microsoft","visa","mastercard","paypal","stripe","bullish","positive",
    "breakthrough","innovation","milestone","record","all-time high","adoption"
}
BAD_TERMS = {
    "hack","hacked","exploit","breach","rug","scam","fraud","attack","ban","bans",
    "restrict","restricts","halts","halt","suspends","withdrawals","insolvent",
    "insolvency","bankrupt","bankruptcy","lawsuit","sue","sues","charged","indicted",
    "sanction","sanctions","reject","rejected","rejects","delay","delays","postpone",
    "postpones","outage","downtime","bearish","crash","plunge","dump","sell-off",
    "correction","decline","fear","panic","uncertainty","regulatory pressure"
}

def safe_float(x):
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

def sort_by_timestamp(candles):
    def key(c):
        ts = c.get("timestamp")
        if ts is not None:
            return int(ts)
        return c.get("datetime") or ""
    return sorted(candles or [], key=key)

def clean_candle(c):
    o = safe_float(c.get("open"))
    h = safe_float(c.get("high"))
    l = safe_float(c.get("low"))
    cl = safe_float(c.get("close"))
    v = safe_float(c.get("volume")) or 0.0
    if None in (o,h,l,cl) or min(o,h,l,cl) <= 0 or h < l:
        return None
    # Enhanced data validation
    if (h - l) / ((h + l) / 2) > 0.5:  # Reject extreme ranges
        return None
    return {"open":o,"high":h,"low":l,"close":cl,"volume":v}

def clean_obs(item):
    obs = sort_by_timestamp(item.get("observation_candles"))
    out = []
    for c in obs[:3]:
        cc = clean_candle(c)
        if cc: out.append(cc)
    return out

def prev_candles(item, k=12):
    pcs = sort_by_timestamp(item.get("previous_candles"))
    out = []
    for c in pcs[-k:]:
        cc = clean_candle(c)
        if cc: out.append(cc)
    return out

def last_prev_close(item):
    pcs = prev_candles(item, k=1)
    return pcs[-1]["close"] if pcs else None

def prev_closes(item, k=12):
    pcs = prev_candles(item, k=k)
    return [c["close"] for c in pcs]

def prev_hilo(item, k=12):
    pcs = prev_candles(item, k=k)
    if not pcs: return None, None
    hi = max(p["high"] for p in pcs)
    lo = min(p["low"] for p in pcs)
    return hi, lo

def avg_prev_volume(item, k=3):
    pcs = prev_candles(item, k=k)
    if not pcs: return None
    vols = [max(0.0, p["volume"]) for p in pcs]
    return max(RISK_EPS, sum(vols)/len(vols)) if vols else None

def ema(seq, span):
    if not seq: return None
    k = 2/(span+1)
    e = seq[0]
    for x in seq[1:]:
        e = x*k + e*(1-k)
    return e

def sma(seq, n):
    if len(seq) < n: return None
    return sum(seq[-n:]) / n

def atr(candles, n=5):
    """Average True Range for volatility measurement"""
    if len(candles) < 2: return None
    trs = []
    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i-1]
        tr = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"] - prev["close"])
        )
        trs.append(tr)
    return sma(trs, min(n, len(trs))) if trs else None

def rsi(closes, n=6):
    """Relative Strength Index"""
    if len(closes) < n + 1: return 50.0  # neutral
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    
    if len(gains) < n: return 50.0
    avg_gain = sma(gains[-n:], n) or 0
    avg_loss = sma(losses[-n:], n) or 0
    
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def slope_sign(closes):
    if len(closes) < 2: return 0.0
    return (closes[-1] - closes[0]) / max(RISK_EPS, closes[0])

def market_regime(candles, volumes):
    """Detect market regime: trending, ranging, volatile, calm"""
    if len(candles) < 5: return "unknown"
    
    closes = [c["close"] for c in candles]
    vol_avg = sum(volumes) / len(volumes) if volumes else 1.0
    
    # Volatility measure
    price_changes = [abs(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]
    volatility = sum(price_changes) / len(price_changes) if price_changes else 0.0
    
    # Trend strength
    trend_strength = abs(slope_sign(closes))
    
    if volatility > VOLATILITY_HIGH:
        return "volatile"
    elif volatility < VOLATILITY_LOW:
        return "calm"
    elif trend_strength > 0.015:
        return "trending"
    else:
        return "ranging"

def candle_shape(c):
    o,h,l,cl = c["open"],c["high"],c["low"],c["close"]
    rng = max(RISK_EPS, h-l)
    body = abs(cl-o)
    upper = h - max(o,cl)
    lower = min(o,cl) - l
    return (min(1.0, body/rng),
            max(0.0, upper/rng),
            max(0.0, lower/rng),
            rng)

def source_weight(title, source):
    t_up = (title or "").upper()
    s_up = (source or "").upper()
    w = 1.0
    
    # Enhanced weighting
    if s_up in AUTHORITATIVE: w += 0.60  # Increased from 0.50
    elif s_up in CRYPTO_TIER2: w += 0.35  # Increased from 0.30
    if any(b in t_up for b in AUTHORITATIVE): w += 0.30  # Increased from 0.25
    if any(b in t_up for b in CRYPTO_TIER2): w += 0.20   # Increased from 0.15
    
    if (source or "").lower() == "twitter":
        low = (title or "").lower()
        if any(h.lower() in low for h in ALLOWED_TWITTER): 
            w += 0.25  # Increased from 0.20
        else: 
            w -= 0.35  # Increased penalty from 0.25
    
    if any(wd in (title or "").lower() for wd in LOW_CRED_TERMS): 
        w -= 0.40  # Increased penalty from 0.30
    
    return max(0.15, w)

def title_tilt(title):
    if not title: return 0.0
    t = title.lower()
    good_count = sum(1 for k in GOOD_TERMS if k in t)
    bad_count = sum(1 for k in BAD_TERMS if k in t)
    
    # Enhanced sentiment scoring
    if good_count > bad_count: return min(0.35, 0.1 * good_count)
    if bad_count > good_count: return max(-0.35, -0.1 * bad_count)
    return 0.0

def is_promotional(title):
    if not title: return False
    return PROMO_RE.search(title) is not None

def zscore(x, mu, sd):
    if sd is None or sd < RISK_EPS: return 0.0
    return (x-mu)/sd

def time_bias(timestamp):
    """Apply time-based market session bias"""
    if not timestamp: return 1.0
    
    # Simplified time bias (in real implementation, you'd parse actual time zones)
    hour = (int(timestamp) // 3600) % 24
    
    # Asian session (22:00-08:00 UTC) - fade bias
    if 22 <= hour or hour <= 8:
        return ASIAN_FADE_BIAS
    # London session (08:00-16:00 UTC) - trend bias
    elif 8 <= hour <= 16:
        return LONDON_TREND_BIAS
    # NY session overlap (13:00-20:00 UTC) - volatility bias
    elif 13 <= hour <= 20:
        return NY_VOLATILITY_BIAS
    
    return 1.0

# ========= Enhanced Experts =========

def expert_momentum_shock(lp, c1, c2, c3, prev_vol_avg, regime, rsi_val):
    o1,h1,l1,cl1,v1 = c1["open"],c1["high"],c1["low"],c1["close"],c1["volume"]
    imp = (cl1 - lp)/lp
    body, up, lo, rng1 = candle_shape(c1)
    gap = (o1 - lp)/lp
    ft2 = (c2["close"]-cl1)/cl1 if c2 else 0.0
    ft3 = (c3["close"]-cl1)/cl1 if c3 else 0.0
    v2 = (c2["volume"] if c2 else 0.0) or 0.0
    v3 = (c3["volume"] if c3 else 0.0) or 0.0
    spike = ((v1+v2+v3)/3)/max(RISK_EPS, prev_vol_avg or 1.0)

    score = 0.0
    if abs(imp) >= IMPULSE_STRONG: score += 1.2  # Increased from 1.0
    if body >= BODY_STRONG: score += 1.0  # Increased from 0.9
    if spike >= VOLUME_SPIKE_HOT: score += 0.8  # Increased from 0.7
    if ft2*imp > 0: score += 0.6  # Increased from 0.5
    if ft3*imp > 0: score += 0.5  # Increased from 0.4
    if abs(gap) <= 0.012: score += 0.3  # Increased from 0.2
    if abs(gap) > 0.02: score -= 0.5  # Increased penalty from 0.4
    if (h1-l1)/lp > 0.022: score -= 0.4  # Increased penalty from 0.3

    # Regime adjustments
    if regime == "trending": score += 0.3
    elif regime == "ranging": score -= 0.2
    elif regime == "volatile": score -= 0.1

    # RSI confirmation
    if imp > 0 and rsi_val < 70: score += 0.2  # Not overbought
    elif imp < 0 and rsi_val > 30: score += 0.2  # Not oversold
    elif imp > 0 and rsi_val > 80: score -= 0.3  # Overbought fade
    elif imp < 0 and rsi_val < 20: score -= 0.3  # Oversold fade

    direction = "LONG" if imp > 0 else "SHORT"
    confidence = max(0.0, score)
    return direction, confidence

def expert_exhaustion_fade(lp, c1, c2, c3, regime, rsi_val, atr_val):
    o1,h1,l1,cl1,_ = c1["open"],c1["high"],c1["low"],c1["close"],c1["volume"]
    imp = (cl1 - lp)/lp
    body, up, lo, rng1 = candle_shape(c1)
    gap = (o1 - lp)/lp
    ft2 = (c2["close"]-cl1)/cl1 if c2 else 0.0
    ft3 = (c3["close"]-cl1)/cl1 if c3 else 0.0

    score = 0.0
    if abs(gap) > CIRCUIT_MEGA_GAP: score += 1.5  # Increased from 1.2
    if (h1-l1)/lp > 0.02: score += 1.0  # Increased from 0.8
    if imp > 0 and up >= WICK_TRAP: score += 1.0  # Increased from 0.8
    if imp < 0 and lo >= WICK_TRAP: score += 1.0  # Increased from 0.8
    if ft2*imp < 0: score += 0.8  # Increased from 0.7
    if ft3*imp < 0: score += 0.6  # Increased from 0.5
    if body < 0.28 and abs(imp) >= IMPULSE_STRONG: score += 0.5  # Increased from 0.4

    # Enhanced exhaustion signals
    if atr_val and (h1-l1) > atr_val * 2: score += 0.4  # Range > 2x ATR
    
    # RSI divergence
    if imp > 0 and rsi_val > 75: score += 0.6  # Strong overbought
    elif imp < 0 and rsi_val < 25: score += 0.6  # Strong oversold

    # Regime adjustments
    if regime == "volatile": score += 0.3
    elif regime == "calm": score -= 0.2

    direction = "SHORT" if imp > 0 else "LONG"
    confidence = max(0.0, score)
    return direction, confidence

def expert_breakout(prev_hi, prev_lo, c1, c2, regime, volume_ratio):
    cl1 = c1["close"]; o1 = c1["open"]; h1=c1["high"]; l1=c1["low"]
    break_up = h1 > prev_hi * (1.0 + BREAK_HOLD_MIN)
    break_dn = l1 < prev_lo * (1.0 - BREAK_HOLD_MIN)

    hold_up = c2 and (c2["close"] > max(prev_hi, cl1*(1.0 - FOLLOW_THRU_OK)))
    hold_dn = c2 and (c2["close"] < min(prev_lo, cl1*(1.0 + FOLLOW_THRU_OK)))

    strong_up = c2 and (c2["close"] - cl1)/cl1 > FOLLOW_THRU_STRONG
    strong_dn = c2 and (cl1 - c2["close"])/cl1 > FOLLOW_THRU_STRONG

    base_conf = 0.9
    
    # Volume confirmation
    if volume_ratio > 2.0: base_conf += 0.3
    elif volume_ratio < 1.2: base_conf -= 0.2
    
    # Regime adjustments
    if regime == "trending": base_conf += 0.2
    elif regime == "ranging": base_conf -= 0.1

    if break_up and (hold_up or strong_up):
        return "LONG", base_conf
    if break_dn and (hold_dn or strong_dn):
        return "SHORT", base_conf
    
    # Enhanced fakeout detection
    fade_conf = base_conf * 0.9
    if break_up and c2 and c2["close"] < prev_hi * 0.999:  # Failed to hold
        return "SHORT", fade_conf
    if break_dn and c2 and c2["close"] > prev_lo * 1.001:  # Failed to hold
        return "LONG", fade_conf
    
    return None, 0.0

def expert_bollinger(prev_cl, c1, c2, regime):
    if len(prev_cl) < 8:  # Increased minimum from 6
        return None, 0.0
    mu = mean(prev_cl)
    sd = pstdev(prev_cl) if len(prev_cl) > 1 else 0.0
    z1 = zscore(c1["close"], mu, sd)
    
    rev = c2 and ((c2["close"] - c1["close"])/max(RISK_EPS, c1["close"]))
    
    base_conf = 0.8
    
    # Regime adjustments
    if regime == "ranging": base_conf += 0.2
    elif regime == "trending": base_conf -= 0.1

    if abs(z1) >= BOLL_Z_FADE and (rev is None or abs(rev) < 0.001):
        return ("SHORT" if z1 > 0 else "LONG"), base_conf
    
    if abs(z1) >= BOLL_Z_TREND and rev and abs(rev) >= 0.001:
        conf = base_conf * 0.75  # Slightly reduced confidence for trend following
        return ("LONG" if z1 > 0 else "SHORT"), conf
    
    return None, 0.0

def expert_microtrend(prev_cl, lp, c1, regime):
    if len(prev_cl) < 6:  # Increased minimum from 5
        return None, 0.0
    
    e5 = ema(prev_cl, 5)
    e12 = ema(prev_cl, 12) if len(prev_cl) >= 12 else ema(prev_cl, max(4,len(prev_cl)-1))
    slope = (e5 - e12)/max(RISK_EPS, e12 or lp)
    
    if abs(slope) < SLOPE_MIN:
        return None, 0.0
    
    imp = (c1["close"] - lp)/lp
    base_conf = 0.55
    
    # Regime adjustments
    if regime == "trending": base_conf += 0.15
    elif regime == "ranging": base_conf -= 0.10

    if slope * imp > 0:
        return ("LONG" if slope > 0 else "SHORT"), base_conf
    else:
        return ("SHORT" if slope > 0 else "LONG"), base_conf * 0.8

def expert_mean_reversion(prev_cl, lp, c1, c2):
    """New expert for mean reversion patterns"""
    if len(prev_cl) < 5: return None, 0.0
    
    recent_mean = mean(prev_cl[-5:])
    imp = (c1["close"] - lp)/lp
    distance_from_mean = (c1["close"] - recent_mean) / recent_mean
    
    # Look for reversal in c2
    reversal = c2 and (c2["close"] - c1["close"]) / c1["close"]
    
    score = 0.0
    if abs(distance_from_mean) > 0.015:  # Far from mean
        score += 0.6
        if reversal and reversal * distance_from_mean < 0:  # Reverting
            score += 0.4
    
    if score > 0.3:
        direction = "LONG" if distance_from_mean < 0 else "SHORT"
        return direction, score
    
    return None, 0.0

def vote_and_decide(item):
    title = item.get("title") or ""
    source = item.get("source") or ""
    timestamp = item.get("time", 0)
    lp = last_prev_close(item)
    if not lp or lp <= 0: return None

    obs = clean_obs(item)
    if not obs: return None
    
    # Enhanced data validation
    if (obs[0]["high"] - obs[0]["low"]) / lp > MAX_RANGE_GLITCH_PCT:
        return None

    c1 = obs[0]
    c2 = obs[1] if len(obs) > 1 else None
    c3 = obs[2] if len(obs) > 2 else None

    prev_candles_data = prev_candles(item, k=15)  # Increased from 12
    prev_hi, prev_lo = prev_hilo(item, k=15)
    prev_cl = prev_closes(item, k=15)
    pv_avg = avg_prev_volume(item, k=4)  # Increased from 3

    # Enhanced market context
    volumes = [c.get("volume", 0) for c in prev_candles_data[-5:]]
    regime = market_regime(prev_candles_data[-8:], volumes)
    rsi_val = rsi(prev_cl + [c1["close"]])
    atr_val = atr(prev_candles_data[-8:]) if len(prev_candles_data) >= 8 else None
    volume_ratio = (c1["volume"] / max(RISK_EPS, pv_avg or 1.0)) if pv_avg else 1.0
    t_bias = time_bias(timestamp)

    # Expert votes with enhanced context
    votes = []

    d, s = expert_momentum_shock(lp, c1, c2, c3, pv_avg, regime, rsi_val)
    votes.append((d, s, "mom"))
    
    d, s = expert_exhaustion_fade(lp, c1, c2, c3, regime, rsi_val, atr_val)
    votes.append((d, s, "fade"))
    
    if prev_hi and prev_lo:
        d, s = expert_breakout(prev_hi, prev_lo, c1, c2, regime, volume_ratio)
        votes.append((d, s, "brk"))
    
    d, s = expert_bollinger(prev_cl, c1, c2, regime)
    votes.append((d, s, "boll"))
    
    d, s = expert_microtrend(prev_cl, lp, c1, regime)
    votes.append((d, s, "micro"))
    
    # New expert
    d, s = expert_mean_reversion(prev_cl, lp, c1, c2)
    votes.append((d, s, "mean_rev"))

    # Enhanced weighting and adjustments
    w_src = source_weight(title, source)
    tilt = title_tilt(title)
    promo_pen = 0.80 if is_promotional(title) else 1.0  # Increased penalty

    # Combine votes with enhanced weighting
    long_score = 0.0
    short_score = 0.0
    
    for d, s, tag in votes:
        if not d: continue
        
        # Enhanced expert weighting
        w = s
        if tag == "mom": w *= 1.15  # Increased from 1.10
        elif tag == "brk": w *= 1.12  # Increased from 1.10
        elif tag == "fade": w *= 1.08  # New weighting
        elif tag == "mean_rev": w *= 1.05  # New expert weighting
        
        if d == "LONG": long_score += w
        else: short_score += w

    # Apply all adjustments
    long_score *= (w_src * (1.0 + max(0.0, tilt)) * promo_pen)
    short_score *= (w_src * (1.0 + max(0.0, -tilt)) * promo_pen)
    
    # Time bias
    if t_bias > 1.0:  # Fade bias in Asian session
        gap = (c1["open"] - lp)/lp
        if gap > 0: short_score *= t_bias
        else: long_score *= t_bias

    # Enhanced circuit breaker
    gap = (c1["open"] - lp)/lp
    if abs(gap) > CIRCUIT_MEGA_GAP:
        multiplier = 1.20  # Increased from 1.15
        if gap > 0: short_score *= multiplier
        else: long_score *= multiplier

    # Enhanced tie-breaking with regime awareness
    if abs(long_score - short_score) < 0.20:  # Increased from 0.15
        imp = (c1["close"] - lp)/lp
        tie_break = 0.15  # Increased from 0.12
        
        if regime == "volatile": tie_break *= 1.3
        elif regime == "calm": tie_break *= 0.8
        
        if imp > 0: short_score += tie_break
        else: long_score += tie_break

    decision = "LONG" if long_score >= short_score else "SHORT"
    conf_gap = abs(long_score - short_score)

    # Enhanced quality metrics
    impulse = abs((c1["close"] - lp)/lp)
    v1 = c1["volume"] or 0.0
    v2 = (c2["volume"] if c2 else 0.0) or 0.0
    v3 = (c3["volume"] if c3 else 0.0) or 0.0
    spike = ((v1+v2+v3)/3)/max(RISK_EPS, pv_avg or 1.0)
    micro_abs = abs(slope_sign(prev_cl)) if prev_cl else 0.0

    # More selective quality gate
    quality_gate = (
        impulse >= 0.0010 and  # Reduced from 0.0012
        spike >= 1.1 and       # Reduced from 1.2
        conf_gap >= 0.08       # Added confidence requirement
    )

    return {
        "id": item.get("id"),
        "decision": decision,
        "conf": float(max(0.0, conf_gap)),
        "impulse": float(impulse),
        "spike": float(spike),
        "micro": float(micro_abs),
        "time": timestamp,
        "promo": bool(is_promotional(title)),
        "quality": bool(quality_gate),
        "regime": regime,
        "rsi": float(rsi_val)
    }

def dedupe_keep_first(items):
    seen = set()
    out = []
    for it in items:
        iid = it.get("id")
        if iid is None or iid in seen: continue
        seen.add(iid); out.append(it)
    return out

def adaptive_portfolio_balance(scored_items):
    """Balance portfolio based on market regime and confidence distribution"""
    if not scored_items:
        return scored_items
    
    # Analyze regime distribution
    regime_counts = defaultdict(int)
    for item in scored_items:
        regime_counts[item.get("regime", "unknown")] += 1
    
    dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else "unknown"
    
    # Adjust selection based on dominant regime
    if dominant_regime == "volatile":
        # In volatile markets, prefer higher confidence fades and mean reversion
        for item in scored_items:
            if "SHORT" in item["decision"] and item["impulse"] > 0.008:
                item["conf"] *= 1.15
    elif dominant_regime == "trending":
        # In trending markets, boost momentum plays
        for item in scored_items:
            if item["spike"] > 2.0 and item["impulse"] > 0.005:
                item["conf"] *= 1.10
    elif dominant_regime == "ranging":
        # In ranging markets, boost mean reversion
        for item in scored_items:
            if item["conf"] > 0.6:
                item["conf"] *= 1.05
    
    return scored_items

def choose_exactly_50(raw_items):
    """Enhanced selection with adaptive portfolio balancing"""
    # Build scored list
    scored = []
    for it in raw_items:
        if it.get("id") in BLOCKLIST_IDS: continue
        s = vote_and_decide(it)
        if s and s["id"] is not None:
            scored.append(s)

    scored = dedupe_keep_first(scored)
    scored = adaptive_portfolio_balance(scored)

    # Multi-tier selection with enhanced criteria
    
    # Tier 1: Premium quality - high confidence, non-promotional, good fundamentals
    tier1 = [x for x in scored if (
        x["quality"] and 
        not x["promo"] and 
        x["conf"] >= 0.8 and
        x["impulse"] >= 0.002
    )]
    tier1.sort(key=lambda x: (
        x["conf"], 
        x["impulse"], 
        x["spike"], 
        x["micro"], 
        x["time"], 
        x["id"]
    ), reverse=True)
    picks = tier1[:30]  # Take top 30 premium picks
    
    # Tier 2: Good quality - relaxed confidence but still non-promotional
    if len(picks) < 50:
        need = min(15, 50 - len(picks))  # Take up to 15 more
        chosen = {p["id"] for p in picks}
        tier2 = [x for x in scored if (
            x["id"] not in chosen and
            x["quality"] and 
            not x["promo"] and 
            x["conf"] >= 0.5
        )]
        tier2.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["spike"], 
            x["micro"], 
            x["time"], 
            x["id"]
        ), reverse=True)
        picks.extend(tier2[:need])
    
    # Tier 3: Acceptable quality - allow some promotional but require minimum standards
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        tier3 = [x for x in scored if (
            x["id"] not in chosen and
            (x["quality"] or x["conf"] >= 0.6) and  # Either quality or high confidence
            x["impulse"] >= 0.0008  # Minimum impulse requirement
        )]
        tier3.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["spike"], 
            x["micro"], 
            x["time"], 
            x["id"]
        ), reverse=True)
        picks.extend(tier3[:need])

    # Tier 4: Fallback - any valid signal above minimum threshold
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        tier4 = [x for x in scored if (
            x["id"] not in chosen and
            x["conf"] >= 0.3 and
            x["impulse"] >= 0.0005
        )]
        tier4.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["spike"], 
            x["micro"], 
            x["time"], 
            x["id"]
        ), reverse=True)
        picks.extend(tier4[:need])

    # Tier 5: Emergency fallback - simple contrarian with enhanced logic
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        emergency = []
        
        for it in raw_items:
            iid = it.get("id")
            if iid in chosen or iid in BLOCKLIST_IDS: continue
            
            lp = last_prev_close(it)
            obs = clean_obs(it)
            if not lp or not obs: continue
            
            c1 = obs[0]
            imp = (c1["close"] - lp)/lp
            
            # Enhanced emergency scoring
            base_conf = 0.05 + min(0.25, abs(imp) * 50)
            
            # Boost confidence for larger moves
            if abs(imp) > 0.01: base_conf *= 1.3
            if abs(imp) > 0.02: base_conf *= 1.2
            
            # Volume consideration
            vol = c1.get("volume", 0)
            if vol > 0: base_conf *= 1.1
            
            emergency.append({
                "id": iid,
                "decision": "SHORT" if imp > 0 else "LONG",
                "conf": base_conf,
                "impulse": abs(imp),
                "spike": 0.0,
                "micro": 0.0,
                "time": it.get("time", 0)
            })
        
        emergency.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["time"], 
            x["id"]
        ), reverse=True)
        picks.extend(emergency[:need])

    # Final processing and validation
    picks = picks[:50]
    
    # Ensure diversity in decisions (optional portfolio balancing)
    long_count = sum(1 for p in picks if p["decision"] == "LONG")
    short_count = len(picks) - long_count
    
    # If too imbalanced (>80% one direction), try to rebalance from remaining candidates
    if long_count > 0.8 * len(picks) or short_count > 0.8 * len(picks):
        minority_direction = "SHORT" if long_count > short_count else "LONG"
        chosen_ids = {p["id"] for p in picks}
        
        # Find candidates for minority direction
        rebalance_candidates = []
        for x in scored:
            if x["id"] not in chosen_ids and x["decision"] == minority_direction and x["conf"] >= 0.3:
                rebalance_candidates.append(x)
        
        if rebalance_candidates:
            rebalance_candidates.sort(key=lambda x: x["conf"], reverse=True)
            # Replace weakest picks with minority direction
            picks.sort(key=lambda x: x["conf"])
            replace_count = min(5, len(rebalance_candidates), len(picks) // 4)
            for i in range(replace_count):
                picks[i] = rebalance_candidates[i]

    return [{"id": p["id"], "decision": p["decision"]} for p in picks]

# =========================
# Enhanced API
# =========================
@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error":"Input must be a JSON array"}), 400
        if not data:
            return jsonify([]), 200

        # Enhanced preprocessing - filter out obvious junk early
        filtered_data = []
        for item in data:
            title = item.get("title", "")
            # Skip items with no price data
            if not last_prev_close(item) or not clean_obs(item):
                continue
            # Skip items with suspicious titles
            if any(term in title.lower() for term in ["test", "ignore", "placeholder"]):
                continue
            filtered_data.append(item)

        result = choose_exactly_50(filtered_data)
        return jsonify(result), 200
        
    except Exception as e:
        # Enhanced error logging (in production, you'd use proper logging)
        error_msg = f"Error processing trading bot request: {str(e)}"
        return jsonify({"error": error_msg}), 500

