# app.py - Enhanced Trading Bot - Optimized for Higher Accuracy
from flask import Flask, request, jsonify
import math
import re
from statistics import mean, pstdev
from collections import defaultdict

from routes import app

# =========================
# Optimized Configuration
# =========================
RISK_EPS = 1e-12
MAX_RANGE_GLITCH_PCT = 0.12     # Further tightened for better data quality
CIRCUIT_MEGA_GAP = 0.025        # More sensitive gap detection
IMPULSE_STRONG = 0.0030         # More sensitive impulse detection
BODY_STRONG = 0.48              # Refined body threshold
VOLUME_SPIKE_HOT = 2.8          # Lower threshold for volume spikes
WICK_TRAP = 0.28                # More sensitive wick detection
BOLL_Z_FADE = 1.6               # More aggressive mean reversion
BOLL_Z_TREND = 0.75             # Enhanced trend following sensitivity
BREAK_HOLD_MIN = 0.0005         # Tighter breakout requirements
FOLLOW_THRU_OK = 0.0004         # Better follow-through detection
FOLLOW_THRU_STRONG = 0.001      # Strong momentum threshold
SLOPE_MIN = 0.0012              # More sensitive trend detection

# Enhanced volatility regimes
VOLATILITY_LOW = 0.006
VOLATILITY_HIGH = 0.022
VOLUME_DRY = 0.6
VOLUME_FLOOD = 4.5

# Market session biases - refined
ASIAN_FADE_BIAS = 1.25          # Stronger fade bias
LONDON_TREND_BIAS = 1.15        # Enhanced trend following
NY_VOLATILITY_BIAS = 1.08       # Refined volatility handling

BLOCKLIST_IDS = set()

# Enhanced promotional detection
PROMO_WORDS = {
    "airdrop","points","quest","campaign","referral","bonus","mint","whitelist",
    "allowlist","giveaway","xp","season","farm","farming","stake to earn","rewards",
    "party","pool party","poolparty","listing soon","perp points","governance token",
    "testnet","mainnet launch","alpha","beta","early access","exclusive","limited time",
    "claim now","don't miss","last chance","presale","ico","ido","launchpad"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_WORDS), re.IGNORECASE)

# Enhanced credibility detection
LOW_CRED_TERMS = {
    "rumor","unconfirmed","parody","fake","shitpost","spoof","satire","april fools",
    "allegedly","reportedly","supposedly","claims","speculation","insider",
    "anonymous source","leak","whispers","breaking news","urgent","alert"
}

AUTHORITATIVE = {
    "BLOOMBERG","REUTERS","WALL STREET JOURNAL","WSJ","FINANCIAL TIMES","FT",
    "NIKKEI","AP","AXIOS","BARRON","BARRONS","CNBC","GUARDIAN","MARKETWATCH",
    "BBC","CNN","FORBES"
}
CRYPTO_TIER2 = {"COINDESK","THE BLOCK","COINTELEGRAPH","DECRYPT","CRYPTOSLATE"}
ALLOWED_TWITTER = {
    "tier10k","WuBlockchain","CoinDesk","TheBlock__","Reuters","AP","Bloomberg",
    "CNBC","MarketWatch","FT","WSJ","tier10k_official","crypto_insider","Bitcoin",
    "BitcoinMagazine","DocumentingBTC"
}

# Enhanced sentiment analysis
GOOD_TERMS = {
    "approve","approves","approved","approval","etf","inflow","flows","treasury",
    "reserve","reserves","executive order","legalize","legalizes","adopt","adopts",
    "integrate","integrates","support","adds support","partnership","upgrade","merge",
    "launch","launches","institutional","blackrock","fidelity","google","amazon",
    "microsoft","visa","mastercard","paypal","stripe","bullish","positive",
    "breakthrough","innovation","milestone","record","all-time high","adoption",
    "strategic reserve","government","federal","regulation clarity","framework"
}
BAD_TERMS = {
    "hack","hacked","exploit","breach","rug","scam","fraud","attack","ban","bans",
    "restrict","restricts","halts","halt","suspends","withdrawals","insolvent",
    "insolvency","bankrupt","bankruptcy","lawsuit","sue","sues","charged","indicted",
    "sanction","sanctions","reject","rejected","rejects","delay","delays","postpone",
    "postpones","outage","downtime","bearish","crash","plunge","dump","sell-off",
    "correction","decline","fear","panic","uncertainty","regulatory pressure","crackdown"
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
    if (h - l) / ((h + l) / 2) > 0.4:  # Reject extreme ranges
        return None
    return {"open":o,"high":h,"low":l,"close":cl,"volume":v}

def clean_obs(item):
    obs = sort_by_timestamp(item.get("observation_candles"))
    out = []
    for c in obs[:3]:
        cc = clean_candle(c)
        if cc: out.append(cc)
    return out

def prev_candles(item, k=15):
    pcs = sort_by_timestamp(item.get("previous_candles"))
    out = []
    for c in pcs[-k:]:
        cc = clean_candle(c)
        if cc: out.append(cc)
    return out

def last_prev_close(item):
    pcs = prev_candles(item, k=1)
    return pcs[-1]["close"] if pcs else None

def prev_closes(item, k=15):
    pcs = prev_candles(item, k=k)
    return [c["close"] for c in pcs]

def prev_hilo(item, k=15):
    pcs = prev_candles(item, k=k)
    if not pcs: return None, None
    hi = max(p["high"] for p in pcs)
    lo = min(p["low"] for p in pcs)
    return hi, lo

def avg_prev_volume(item, k=4):
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
    if len(closes) < n + 1: return 50.0
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

def stoch_rsi(closes, k_period=3, d_period=3, rsi_period=14, stoch_period=14):
    """Stochastic RSI for enhanced momentum detection"""
    if len(closes) < rsi_period + stoch_period: return 50.0, 50.0
    
    # Calculate RSI values
    rsi_values = []
    for i in range(rsi_period, len(closes)):
        rsi_val = rsi(closes[i-rsi_period:i+1], rsi_period)
        rsi_values.append(rsi_val)
    
    if len(rsi_values) < stoch_period: return 50.0, 50.0
    
    # Calculate Stochastic of RSI
    recent_rsi = rsi_values[-stoch_period:]
    rsi_high = max(recent_rsi)
    rsi_low = min(recent_rsi)
    
    if rsi_high == rsi_low: return 50.0, 50.0
    
    k_percent = 100 * (rsi_values[-1] - rsi_low) / (rsi_high - rsi_low)
    
    # Simple moving average for %D
    if len(rsi_values) >= k_period:
        recent_k = [100 * (val - rsi_low) / (rsi_high - rsi_low) for val in rsi_values[-k_period:]]
        d_percent = sum(recent_k) / len(recent_k)
    else:
        d_percent = k_percent
    
    return k_percent, d_percent

def slope_sign(closes):
    if len(closes) < 2: return 0.0
    return (closes[-1] - closes[0]) / max(RISK_EPS, closes[0])

def market_regime(candles, volumes):
    """Enhanced market regime detection"""
    if len(candles) < 5: return "unknown"
    
    closes = [c["close"] for c in candles]
    vol_avg = sum(volumes) / len(volumes) if volumes else 1.0
    
    # Enhanced volatility measure
    price_changes = [abs(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]
    volatility = sum(price_changes) / len(price_changes) if price_changes else 0.0
    
    # Trend strength with momentum
    trend_strength = abs(slope_sign(closes))
    momentum = (closes[-1] - closes[-3]) / closes[-3] if len(closes) >= 3 else 0
    
    # Volume analysis
    vol_trend = (volumes[-1] - vol_avg) / vol_avg if vol_avg > 0 else 0
    
    # Multi-factor regime classification
    if volatility > VOLATILITY_HIGH and vol_trend > 1.0:
        return "explosive"
    elif volatility > VOLATILITY_HIGH:
        return "volatile"
    elif volatility < VOLATILITY_LOW and trend_strength < 0.005:
        return "calm"
    elif trend_strength > 0.015 and abs(momentum) > 0.008:
        return "strong_trending"
    elif trend_strength > 0.008:
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

def enhanced_source_weight(title, source):
    """Enhanced source credibility scoring"""
    t_up = (title or "").upper()
    s_up = (source or "").upper()
    w = 1.0
    
    # Authority bonus
    if s_up in AUTHORITATIVE: w += 0.70
    elif s_up in CRYPTO_TIER2: w += 0.45
    if any(b in t_up for b in AUTHORITATIVE): w += 0.35
    if any(b in t_up for b in CRYPTO_TIER2): w += 0.25
    
    # Twitter handling with enhanced verification
    if (source or "").lower() == "twitter":
        low = (title or "").lower()
        verified_accounts = sum(1 for h in ALLOWED_TWITTER if h.lower() in low)
        if verified_accounts > 0: 
            w += 0.30 * verified_accounts  # Bonus for multiple verifications
        else: 
            w -= 0.45  # Stronger penalty for unverified
    
    # Credibility penalties
    if any(wd in (title or "").lower() for wd in LOW_CRED_TERMS): 
        w -= 0.50
    
    # News recency bonus (assuming more recent = more reliable for breaking news)
    if "breaking" in (title or "").lower() and s_up in AUTHORITATIVE:
        w += 0.25
    
    return max(0.10, w)

def enhanced_title_tilt(title):
    """Enhanced sentiment analysis with context awareness"""
    if not title: return 0.0
    t = title.lower()
    
    # Count sentiment words with context
    good_matches = [k for k in GOOD_TERMS if k in t]
    bad_matches = [k for k in BAD_TERMS if k in t]
    
    # Weight by importance
    important_good = ["strategic reserve", "etf", "approval", "institutional"]
    important_bad = ["hack", "ban", "crash", "fraud"]
    
    good_score = len(good_matches) * 0.1
    bad_score = len(bad_matches) * 0.1
    
    # Boost for important terms
    for term in important_good:
        if term in t: good_score += 0.15
    for term in important_bad:
        if term in t: bad_score += 0.15
    
    # Net sentiment
    net_sentiment = good_score - bad_score
    return max(-0.40, min(0.40, net_sentiment))

def is_promotional(title):
    if not title: return False
    return PROMO_RE.search(title) is not None

def zscore(x, mu, sd):
    if sd is None or sd < RISK_EPS: return 0.0
    return (x-mu)/sd

def time_bias(timestamp):
    """Enhanced time-based market session bias"""
    if not timestamp: return 1.0
    
    hour = (int(timestamp) // 3600) % 24
    
    # Asian session (22:00-08:00 UTC) - enhanced fade bias
    if 22 <= hour or hour <= 8:
        return ASIAN_FADE_BIAS
    # London session (08:00-16:00 UTC) - trend bias
    elif 8 <= hour <= 16:
        return LONDON_TREND_BIAS
    # NY session overlap (13:00-20:00 UTC) - volatility bias
    elif 13 <= hour <= 20:
        return NY_VOLATILITY_BIAS
    
    return 1.0

# ========= Enhanced Expert Systems =========

def expert_momentum_shock_v2(lp, c1, c2, c3, prev_vol_avg, regime, rsi_val, stoch_k, stoch_d):
    """Enhanced momentum detection with stochastic confirmation"""
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
    
    # Core momentum signals
    if abs(imp) >= IMPULSE_STRONG: score += 1.4
    if body >= BODY_STRONG: score += 1.2
    if spike >= VOLUME_SPIKE_HOT: score += 1.0
    
    # Follow-through confirmation
    if ft2*imp > 0: score += 0.8
    if ft3*imp > 0: score += 0.6
    
    # Gap analysis
    if abs(gap) <= 0.01: score += 0.4  # Organic move
    if abs(gap) > 0.025: score -= 0.6  # Suspicious gap
    
    # Range validation
    if (h1-l1)/lp > 0.025: score -= 0.5  # Too wide range
    
    # Stochastic confirmation
    if imp > 0 and stoch_k < 80: score += 0.3  # Not overbought
    elif imp < 0 and stoch_k > 20: score += 0.3  # Not oversold
    elif imp > 0 and stoch_k > 85: score -= 0.4  # Overbought
    elif imp < 0 and stoch_k < 15: score -= 0.4  # Oversold
    
    # Stochastic divergence
    if stoch_k > stoch_d and imp > 0: score += 0.2  # Momentum building
    elif stoch_k < stoch_d and imp < 0: score += 0.2  # Momentum building

    # Regime adjustments
    if regime == "strong_trending": score += 0.4
    elif regime == "trending": score += 0.2
    elif regime == "explosive": score += 0.5
    elif regime == "ranging": score -= 0.3
    elif regime == "calm": score -= 0.2

    # RSI confirmation (refined)
    if imp > 0 and 30 < rsi_val < 70: score += 0.25  # Healthy momentum
    elif imp < 0 and 30 < rsi_val < 70: score += 0.25
    elif imp > 0 and rsi_val > 75: score -= 0.4  # Overbought fade
    elif imp < 0 and rsi_val < 25: score -= 0.4  # Oversold fade

    direction = "LONG" if imp > 0 else "SHORT"
    confidence = max(0.0, score)
    return direction, confidence

def expert_exhaustion_fade_v2(lp, c1, c2, c3, regime, rsi_val, atr_val, stoch_k):
    """Enhanced exhaustion detection with multiple confirmations"""
    o1,h1,l1,cl1,_ = c1["open"],c1["high"],c1["low"],c1["close"],c1["volume"]
    imp = (cl1 - lp)/lp
    body, up, lo, rng1 = candle_shape(c1)
    gap = (o1 - lp)/lp
    ft2 = (c2["close"]-cl1)/cl1 if c2 else 0.0
    ft3 = (c3["close"]-cl1)/cl1 if c3 else 0.0

    score = 0.0
    
    # Core exhaustion signals
    if abs(gap) > CIRCUIT_MEGA_GAP: score += 1.8
    if (h1-l1)/lp > 0.022: score += 1.2
    
    # Wick analysis (enhanced)
    if imp > 0 and up >= WICK_TRAP: score += 1.2  # Upper wick rejection
    if imp < 0 and lo >= WICK_TRAP: score += 1.2  # Lower wick rejection
    
    # Follow-through fade
    if ft2*imp < 0: score += 1.0
    if ft3*imp < 0: score += 0.8
    
    # Doji patterns
    if body < 0.25 and abs(imp) >= IMPULSE_STRONG: score += 0.6
    
    # ATR-based exhaustion
    if atr_val and (h1-l1) > atr_val * 2.5: score += 0.5
    
    # Stochastic exhaustion
    if imp > 0 and stoch_k > 80: score += 0.8  # Overbought
    elif imp < 0 and stoch_k < 20: score += 0.8  # Oversold
    
    # RSI extremes
    if imp > 0 and rsi_val > 70: score += 0.7
    elif imp < 0 and rsi_val < 30: score += 0.7
    
    # Regime adjustments
    if regime == "explosive": score += 0.4  # More exhaustion in explosive moves
    elif regime == "volatile": score += 0.3
    elif regime == "calm": score -= 0.3

    direction = "SHORT" if imp > 0 else "LONG"
    confidence = max(0.0, score)
    return direction, confidence

def expert_breakout_v2(prev_hi, prev_lo, c1, c2, c3, regime, volume_ratio):
    """Enhanced breakout detection with confirmation"""
    cl1 = c1["close"]; o1 = c1["open"]; h1=c1["high"]; l1=c1["low"]
    
    # Breakout detection
    break_up = h1 > prev_hi * (1.0 + BREAK_HOLD_MIN)
    break_dn = l1 < prev_lo * (1.0 - BREAK_HOLD_MIN)
    
    # Multiple candle confirmation
    confirmations = []
    if c2:
        hold_up = c2["close"] > max(prev_hi, cl1*(1.0 - FOLLOW_THRU_OK))
        hold_dn = c2["close"] < min(prev_lo, cl1*(1.0 + FOLLOW_THRU_OK))
        strong_up = (c2["close"] - cl1)/cl1 > FOLLOW_THRU_STRONG
        strong_dn = (cl1 - c2["close"])/cl1 > FOLLOW_THRU_STRONG
        confirmations.extend([hold_up, hold_dn, strong_up, strong_dn])
    
    if c3:
        confirm_up = c3["close"] > prev_hi * 1.001
        confirm_dn = c3["close"] < prev_lo * 0.999
        confirmations.extend([confirm_up, confirm_dn])

    base_conf = 1.0
    
    # Volume confirmation (enhanced)
    if volume_ratio > 3.0: base_conf += 0.4
    elif volume_ratio > 2.0: base_conf += 0.2
    elif volume_ratio < 1.0: base_conf -= 0.3
    
    # Confirmation bonus
    conf_score = sum(1 for c in confirmations if c) * 0.15
    base_conf += conf_score
    
    # Regime adjustments
    if regime == "strong_trending": base_conf += 0.3
    elif regime == "trending": base_conf += 0.2
    elif regime == "ranging": base_conf -= 0.2
    elif regime == "explosive": base_conf += 0.25

    # Breakout execution
    if break_up and any([c for c in confirmations[:2] if c]):  # At least one confirmation
        return "LONG", base_conf
    if break_dn and any([c for c in confirmations[:2] if c]):
        return "SHORT", base_conf
    
    # Enhanced fakeout detection
    fade_conf = base_conf * 0.85
    if break_up and c2 and c2["close"] < prev_hi * 0.998:
        return "SHORT", fade_conf
    if break_dn and c2 and c2["close"] > prev_lo * 1.002:
        return "LONG", fade_conf
    
    return None, 0.0

def expert_bollinger_v2(prev_cl, c1, c2, c3, regime, rsi_val):
    """Enhanced Bollinger Band strategy with multiple confirmations"""
    if len(prev_cl) < 10:
        return None, 0.0
        
    mu = mean(prev_cl)
    sd = pstdev(prev_cl) if len(prev_cl) > 1 else 0.0
    z1 = zscore(c1["close"], mu, sd)
    z2 = zscore(c2["close"], mu, sd) if c2 else None
    
    # Reversion signals
    rev1 = c2 and ((c2["close"] - c1["close"])/max(RISK_EPS, c1["close"]))
    rev2 = c3 and ((c3["close"] - c2["close"])/max(RISK_EPS, c2["close"])) if c2 else None
    
    base_conf = 0.9
    
    # Enhanced mean reversion
    if abs(z1) >= BOLL_Z_FADE:
        score = base_conf
        
        # Multiple candle confirmation
        if rev1 and abs(rev1) > 0.001: score += 0.3
        if rev2 and abs(rev2) > 0.001: score += 0.2
        
        # Z-score progression
        if z2 and abs(z2) < abs(z1): score += 0.2  # Moving back to mean
        
        # RSI confirmation
        if z1 > 0 and rsi_val > 60: score += 0.2  # Overbought + upper band
        elif z1 < 0 and rsi_val < 40: score += 0.2  # Oversold + lower band
        
        # Regime adjustments
        if regime == "ranging": score += 0.25
        elif regime == "trending": score -= 0.15
        elif regime == "explosive": score += 0.1  # Explosive moves often reverse
        
        return ("SHORT" if z1 > 0 else "LONG"), score
    
    # Enhanced trend continuation
    if abs(z1) >= BOLL_Z_TREND and rev1 and abs(rev1) >= 0.0008:
        score = base_conf * 0.8
        
        # Trend strength
        if abs(z1) > 1.2: score += 0.15
        if rev2 and (rev1 * rev2 > 0): score += 0.15  # Consistent direction
        
        # Regime boost
        if regime in ["trending", "strong_trending"]: score += 0.2
        
        return ("LONG" if z1 > 0 else "SHORT"), score
    
    return None, 0.0

def expert_pattern_recognition(c1, c2, c3, lp):
    """New expert for candlestick pattern recognition"""
    if not all([c1, c2, c3]): return None, 0.0
    
    # Calculate pattern metrics
    patterns = []
    
    # Hammer/Hanging Man
    body1, up1, lo1, _ = candle_shape(c1)
    if body1 < 0.3 and lo1 > 0.6:  # Long lower wick, small body
        imp1 = (c1["close"] - lp) / lp
        if imp1 < -0.01:  # After decline
            patterns.append(("LONG", 0.6))  # Hammer
        elif imp1 > 0.01:  # After advance
            patterns.append(("SHORT", 0.5))  # Hanging man
    
    # Doji patterns
    if body1 < 0.15 and up1 > 0.3 and lo1 > 0.3:  # Doji
        imp1 = (c1["close"] - lp) / lp
        if abs(imp1) > 0.008:  # Significant move before doji
            patterns.append(("SHORT" if imp1 > 0 else "LONG", 0.4))
    
    # Engulfing patterns
    if c2:
        body2, _, _, _ = candle_shape(c2)
        c1_bull = c1["close"] > c1["open"]
        c2_bull = c2["close"] > c2["open"]
        c1_range = c1["high"] - c1["low"]
        c2_range = c2["high"] - c2["low"]
        
        # Bullish engulfing
        if not c2_bull and c1_bull and c1_range > c2_range * 1.5:
            patterns.append(("LONG", 0.7))
        # Bearish engulfing
        elif c2_bull and not c1_bull and c1_range > c2_range * 1.5:
            patterns.append(("SHORT", 0.7))
    
    # Three soldier patterns
    if c2 and c3:
        closes = [c3["close"], c2["close"], c1["close"]]
        if all(closes[i] > closes[i+1] for i in range(len(closes)-1)):  # Rising
            if all((closes[i] - closes[i+1])/closes[i+1] > 0.003 for i in range(len(closes)-1)):
                patterns.append(("LONG", 0.8))
        elif all(closes[i] < closes[i+1] for i in range(len(closes)-1)):  # Falling
            if all((closes[i+1] - closes[i])/closes[i] > 0.003 for i in range(len(closes)-1)):
                patterns.append(("SHORT", 0.8))
    
    # Return strongest pattern
    if patterns:
        return max(patterns, key=lambda x: x[1])
    
    return None, 0.0

def expert_volume_price_analysis(c1, c2, c3, prev_vol_avg, lp):
    """Enhanced volume-price relationship analysis"""
    if not prev_vol_avg: return None, 0.0
    
    v1 = c1.get("volume", 0)
    v2 = c2.get("volume", 0) if c2 else 0
    v3 = c3.get("volume", 0) if c3 else 0
    
    price_change1 = (c1["close"] - lp) / lp
    price_change2 = (c2["close"] - c1["close"]) / c1["close"] if c2 else 0
    price_change3 = (c3["close"] - c2["close"]) / c2["close"] if c2 and c3 else 0
    
    vol_ratio1 = v1 / prev_vol_avg
    vol_ratio2 = v2 / prev_vol_avg if v2 > 0 else 0
    vol_ratio3 = v3 / prev_vol_avg if v3 > 0 else 0
    
    score = 0.0
    direction = None
    
    # Volume confirmation patterns
    if abs(price_change1) > 0.005 and vol_ratio1 > 2.0:
        # Strong move with volume - continuation
        score += 0.6
        direction = "LONG" if price_change1 > 0 else "SHORT"
        
        # Follow-through confirmation
        if price_change2 * price_change1 > 0 and vol_ratio2 > 1.5:
            score += 0.3
        if price_change3 * price_change1 > 0 and vol_ratio3 > 1.2:
            score += 0.2
    
    # Volume divergence (price up, volume down = weakness)
    elif abs(price_change1) > 0.008 and vol_ratio1 < 0.8:
        score += 0.5
        direction = "SHORT" if price_change1 > 0 else "LONG"  # Fade the move
        
        # Divergence confirmation
        if price_change2 * price_change1 < 0:  # Price reversal
            score += 0.4
    
    # Climactic volume (exhaustion)
    elif vol_ratio1 > 4.0 and abs(price_change1) > 0.015:
        score += 0.7
        direction = "SHORT" if price_change1 > 0 else "LONG"  # Fade climax
    
    if score > 0.3 and direction:
        return direction, score
    
    return None, 0.0

def expert_microtrend_v2(prev_cl, lp, c1, c2, regime):
    """Enhanced microtrend analysis with momentum confirmation"""
    if len(prev_cl) < 8:
        return None, 0.0
    
    # Multiple EMA analysis
    e3 = ema(prev_cl, 3) if len(prev_cl) >= 3 else None
    e5 = ema(prev_cl, 5)
    e8 = ema(prev_cl, 8)
    e12 = ema(prev_cl, 12) if len(prev_cl) >= 12 else ema(prev_cl, max(6,len(prev_cl)-1))
    
    # Current price vs EMAs
    current = c1["close"]
    c2_close = c2["close"] if c2 else current
    
    slopes = []
    if e3 and e5: slopes.append((e3 - e5) / max(RISK_EPS, e5))
    if e5 and e8: slopes.append((e5 - e8) / max(RISK_EPS, e8))
    if e8 and e12: slopes.append((e8 - e12) / max(RISK_EPS, e12))
    
    if not slopes: return None, 0.0
    
    avg_slope = sum(slopes) / len(slopes)
    slope_consistency = len([s for s in slopes if s * avg_slope > 0]) / len(slopes)
    
    if abs(avg_slope) < SLOPE_MIN or slope_consistency < 0.67:
        return None, 0.0
    
    # Price momentum
    price_momentum = (current - lp) / lp
    follow_through = (c2_close - current) / current if c2 else 0
    
    base_conf = 0.6
    
    # Slope strength bonus
    base_conf += min(0.3, abs(avg_slope) * 20)
    
    # Consistency bonus
    base_conf += (slope_consistency - 0.67) * 0.6
    
    # Momentum alignment
    if avg_slope * price_momentum > 0:
        base_conf += 0.2
        if follow_through * avg_slope > 0:
            base_conf += 0.15
    else:
        # Counter-trend setup
        if abs(price_momentum) > 0.01:  # Strong counter move
            base_conf *= 0.9  # Slightly reduce confidence
        else:
            return None, 0.0  # Weak counter-trend
    
    # Regime adjustments
    if regime == "strong_trending": base_conf += 0.2
    elif regime == "trending": base_conf += 0.1
    elif regime == "ranging": base_conf -= 0.15
    elif regime == "explosive": base_conf += 0.05
    
    direction = "LONG" if avg_slope > 0 else "SHORT"
    return direction, base_conf

def expert_support_resistance(prev_candles_data, c1, c2):
    """New expert for support/resistance levels"""
    if len(prev_candles_data) < 10: return None, 0.0
    
    # Find significant levels
    highs = [c["high"] for c in prev_candles_data[-20:] if c]
    lows = [c["low"] for c in prev_candles_data[-20:] if c]
    closes = [c["close"] for c in prev_candles_data[-10:] if c]
    
    if not highs or not lows: return None, 0.0
    
    # Identify key levels (simplified approach)
    recent_high = max(highs[-10:])
    recent_low = min(lows[-10:])
    avg_close = sum(closes) / len(closes)
    
    current_price = c1["close"]
    price_change = (current_price - prev_candles_data[-1]["close"]) / prev_candles_data[-1]["close"]
    
    score = 0.0
    direction = None
    
    # Resistance test
    if current_price > recent_high * 0.998 and price_change > 0.003:
        if c2 and c2["close"] < current_price * 0.999:  # Failed to break
            score = 0.6
            direction = "SHORT"
    
    # Support test
    elif current_price < recent_low * 1.002 and price_change < -0.003:
        if c2 and c2["close"] > current_price * 1.001:  # Bounced
            score = 0.6
            direction = "LONG"
    
    # Moving average as dynamic support/resistance
    elif abs(current_price - avg_close) / avg_close < 0.005:  # Near MA
        vol = c1.get("volume", 0)
        avg_vol = sum([c.get("volume", 0) for c in prev_candles_data[-5:]]) / 5
        
        if vol > avg_vol * 1.5:  # Volume spike at MA
            if price_change > 0.002:
                score = 0.4
                direction = "LONG"  # Break above MA
            elif price_change < -0.002:
                score = 0.4
                direction = "SHORT"  # Break below MA
    
    if score > 0.3 and direction:
        return direction, score
    
    return None, 0.0

def vote_and_decide_v2(item):
    """Enhanced voting system with improved expert coordination"""
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

    # Enhanced context gathering
    prev_candles_data = prev_candles(item, k=20)  # More historical data
    prev_hi, prev_lo = prev_hilo(item, k=18)
    prev_cl = prev_closes(item, k=18)
    pv_avg = avg_prev_volume(item, k=5)

    # Enhanced market context
    volumes = [c.get("volume", 0) for c in prev_candles_data[-8:]]
    regime = market_regime(prev_candles_data[-10:], volumes)
    rsi_val = rsi(prev_cl + [c1["close"]], n=8)  # Faster RSI
    stoch_k, stoch_d = stoch_rsi(prev_cl + [c1["close"]])
    atr_val = atr(prev_candles_data[-10:]) if len(prev_candles_data) >= 10 else None
    volume_ratio = (c1["volume"] / max(RISK_EPS, pv_avg or 1.0)) if pv_avg else 1.0
    t_bias = time_bias(timestamp)

    # Enhanced expert votes
    votes = []

    # Core experts with enhanced parameters
    d, s = expert_momentum_shock_v2(lp, c1, c2, c3, pv_avg, regime, rsi_val, stoch_k, stoch_d)
    votes.append((d, s, "mom_v2", 1.25))  # Increased weight
    
    d, s = expert_exhaustion_fade_v2(lp, c1, c2, c3, regime, rsi_val, atr_val, stoch_k)
    votes.append((d, s, "fade_v2", 1.20))  # Increased weight
    
    if prev_hi and prev_lo:
        d, s = expert_breakout_v2(prev_hi, prev_lo, c1, c2, c3, regime, volume_ratio)
        votes.append((d, s, "brk_v2", 1.15))
    
    d, s = expert_bollinger_v2(prev_cl, c1, c2, c3, regime, rsi_val)
    votes.append((d, s, "boll_v2", 1.10))
    
    d, s = expert_microtrend_v2(prev_cl, lp, c1, c2, regime)
    votes.append((d, s, "micro_v2", 1.08))
    
    # New experts
    d, s = expert_pattern_recognition(c1, c2, c3, lp)
    votes.append((d, s, "pattern", 1.05))
    
    d, s = expert_volume_price_analysis(c1, c2, c3, pv_avg, lp)
    votes.append((d, s, "vol_price", 1.12))
    
    d, s = expert_support_resistance(prev_candles_data, c1, c2)
    votes.append((d, s, "sup_res", 1.06))

    # Enhanced weighting and adjustments
    w_src = enhanced_source_weight(title, source)
    tilt = enhanced_title_tilt(title)
    promo_pen = 0.75 if is_promotional(title) else 1.0

    # Advanced vote combination
    long_score = 0.0
    short_score = 0.0
    vote_count = {"LONG": 0, "SHORT": 0}
    
    for d, s, tag, weight in votes:
        if not d or s <= 0: continue
        
        adjusted_score = s * weight
        
        if d == "LONG":
            long_score += adjusted_score
            vote_count["LONG"] += 1
        else:
            short_score += adjusted_score
            vote_count["SHORT"] += 1

    # Apply source and sentiment adjustments
    long_score *= (w_src * (1.0 + max(0.0, tilt)) * promo_pen)
    short_score *= (w_src * (1.0 + max(0.0, -tilt)) * promo_pen)
    
    # Enhanced consensus bonus
    total_votes = sum(vote_count.values())
    if total_votes > 0:
        consensus = max(vote_count.values()) / total_votes
        if consensus > 0.7:  # Strong consensus
            if vote_count["LONG"] > vote_count["SHORT"]:
                long_score *= (1 + (consensus - 0.7) * 0.5)
            else:
                short_score *= (1 + (consensus - 0.7) * 0.5)

    # Time and regime bias
    gap = (c1["open"] - lp)/lp
    if t_bias > 1.05:  # Session-based adjustment
        if abs(gap) > 0.01:  # Significant gap
            if gap > 0: short_score *= t_bias
            else: long_score *= t_bias

    # Enhanced circuit breaker
    if abs(gap) > CIRCUIT_MEGA_GAP:
        multiplier = 1.3 + min(0.3, abs(gap) * 20)  # Dynamic multiplier
        if gap > 0: short_score *= multiplier
        else: long_score *= multiplier

    # Regime-based final adjustments
    if regime == "explosive":
        # In explosive moves, boost fading strategies
        if abs(gap) > 0.02:
            if gap > 0: short_score *= 1.15
            else: long_score *= 1.15
    elif regime == "strong_trending":
        # Boost momentum in strong trends
        impulse = abs((c1["close"] - lp)/lp)
        if impulse > 0.008:
            momentum_direction = "LONG" if (c1["close"] - lp) > 0 else "SHORT"
            if momentum_direction == "LONG":
                long_score *= 1.1
            else:
                short_score *= 1.1

    # Enhanced decision making
    decision = "LONG" if long_score >= short_score else "SHORT"
    conf_gap = abs(long_score - short_score)
    
    # Dynamic tie-breaking
    tie_threshold = 0.25 + (0.1 if regime == "volatile" else 0.0)
    if conf_gap < tie_threshold:
        # Use additional factors for tie-breaking
        impulse = abs((c1["close"] - lp)/lp)
        volume_factor = min(2.0, volume_ratio)
        
        tie_break = 0.2 * impulse * volume_factor
        
        # Apply regime-specific tie-breaking
        if regime in ["explosive", "volatile"]:
            # Favor fading in volatile conditions
            price_move = (c1["close"] - lp)/lp
            if price_move > 0: short_score += tie_break
            else: long_score += tie_break
        else:
            # Favor momentum in stable conditions
            price_move = (c1["close"] - lp)/lp
            if price_move > 0: long_score += tie_break
            else: short_score += tie_break

    # Recalculate after tie-breaking
    decision = "LONG" if long_score >= short_score else "SHORT"
    conf_gap = abs(long_score - short_score)

    # Enhanced quality metrics
    impulse = abs((c1["close"] - lp)/lp)
    v1 = c1["volume"] or 0.0
    v2 = (c2["volume"] if c2 else 0.0) or 0.0
    v3 = (c3["volume"] if c3 else 0.0) or 0.0
    spike = ((v1+v2+v3)/3)/max(RISK_EPS, pv_avg or 1.0)
    micro_abs = abs(slope_sign(prev_cl)) if prev_cl else 0.0

    # More sophisticated quality gate
    base_quality = (
        impulse >= 0.0008 and
        spike >= 1.0 and
        conf_gap >= 0.06
    )
    
    # Additional quality factors
    enhanced_quality = base_quality and (
        volume_ratio > 1.2 or
        abs(gap) > 0.008 or
        regime in ["explosive", "strong_trending"] or
        w_src > 1.3
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
        "quality": bool(base_quality),
        "enhanced_quality": bool(enhanced_quality),
        "regime": regime,
        "rsi": float(rsi_val),
        "volume_ratio": float(volume_ratio),
        "source_weight": float(w_src)
    }

def advanced_portfolio_optimization(scored_items):
    """Advanced portfolio optimization with risk management"""
    if not scored_items:
        return scored_items
    
    # Enhanced regime analysis
    regime_weights = defaultdict(float)
    for item in scored_items:
        regime_weights[item.get("regime", "unknown")] += item.get("conf", 0)
    
    total_weight = sum(regime_weights.values()) or 1.0
    regime_distribution = {k: v/total_weight for k, v in regime_weights.items()}
    
    # Dynamic confidence adjustments based on portfolio composition
    for item in scored_items:
        regime = item.get("regime", "unknown")
        regime_prop = regime_distribution.get(regime, 0)
        
        # Boost underrepresented high-quality regimes
        if regime in ["strong_trending", "explosive"] and regime_prop < 0.3:
            if item.get("enhanced_quality", False):
                item["conf"] *= 1.2
        
        # Enhance diversification
        if regime_prop > 0.5:  # Overrepresented regime
            item["conf"] *= 0.95
        
        # Source quality boost
        if item.get("source_weight", 1.0) > 1.4:
            item["conf"] *= 1.05
        
        # Volume confirmation boost
        if item.get("volume_ratio", 1.0) > 2.5:
            item["conf"] *= 1.08

    return scored_items

def choose_exactly_50_v2(raw_items):
    """Enhanced selection algorithm with sophisticated filtering"""
    # Build enhanced scored list
    scored = []
    for it in raw_items:
        if it.get("id") in BLOCKLIST_IDS: continue
        s = vote_and_decide_v2(it)
        if s and s["id"] is not None:
            scored.append(s)

    scored = dedupe_keep_first(scored)
    scored = advanced_portfolio_optimization(scored)

    # Multi-tier selection with enhanced criteria
    picks = []
    
    # Tier 1: Premium quality - highest standards
    tier1 = [x for x in scored if (
        x["enhanced_quality"] and 
        not x["promo"] and 
        x["conf"] >= 1.0 and
        x["impulse"] >= 0.0025 and
        x["source_weight"] >= 1.2
    )]
    tier1.sort(key=lambda x: (
        x["conf"], 
        x["source_weight"],
        x["impulse"], 
        x["volume_ratio"],
        x["spike"], 
        -x["time"]  # Prefer more recent
    ), reverse=True)
    picks.extend(tier1[:25])  # Top 25 premium picks
    
    # Tier 2: High quality - strong signals
    if len(picks) < 50:
        need = min(20, 50 - len(picks))
        chosen = {p["id"] for p in picks}
        tier2 = [x for x in scored if (
            x["id"] not in chosen and
            x["quality"] and 
            not x["promo"] and 
            x["conf"] >= 0.7 and
            x["impulse"] >= 0.0015
        )]
        tier2.sort(key=lambda x: (
            x["conf"], 
            x["source_weight"],
            x["impulse"], 
            x["volume_ratio"]
        ), reverse=True)
        picks.extend(tier2[:need])
    
    # Tier 3: Good quality - relaxed standards
    if len(picks) < 50:
        need = min(15, 50 - len(picks))
        chosen = {p["id"] for p in picks}
        tier3 = [x for x in scored if (
            x["id"] not in chosen and
            (x["enhanced_quality"] or x["conf"] >= 0.8) and
            x["impulse"] >= 0.001
        )]
        tier3.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["source_weight"]
        ), reverse=True)
        picks.extend(tier3[:need])
    
    # Tier 4: Acceptable - minimum thresholds
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        tier4 = [x for x in scored if (
            x["id"] not in chosen and
            x["conf"] >= 0.4 and
            x["impulse"] >= 0.0008 and
            (not x["promo"] or x["conf"] >= 0.8)  # Allow promo only if very confident
        )]
        tier4.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["source_weight"]
        ), reverse=True)
        picks.extend(tier4[:need])

    # Emergency fallback with enhanced logic
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
            title = it.get("title", "")
            source = it.get("source", "")
            
            price_change = (c1["close"] - lp)/lp
            vol = c1.get("volume", 0)
            avg_vol = avg_prev_volume(it, k=3) or 1.0
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
            
            # Enhanced emergency scoring
            base_conf = 0.1 + min(0.4, abs(price_change) * 30)
            
            # Volume boost
            if vol_ratio > 1.5: base_conf *= 1.2
            
            # Source quality
            src_weight = enhanced_source_weight(title, source)
            base_conf *= min(1.3, src_weight)
            
            # Sentiment
            sentiment = enhanced_title_tilt(title)
            if abs(sentiment) > 0.1:
                if (sentiment > 0 and price_change > 0) or (sentiment < 0 and price_change < 0):
                    base_conf *= 1.1  # Aligned sentiment
            
            emergency.append({
                "id": iid,
                "decision": "SHORT" if price_change > 0 else "LONG",  # Contrarian
                "conf": base_conf,
                "impulse": abs(price_change),
                "source_weight": src_weight,
                "time": it.get("time", 0)
            })
        
        emergency.sort(key=lambda x: (
            x["conf"], 
            x["source_weight"],
            x["impulse"]
        ), reverse=True)
        picks.extend(emergency[:need])

    # Final validation and balancing
    picks = picks[:50]
    
    # Portfolio balance optimization
    long_count = sum(1 for p in picks if p["decision"] == "LONG")
    short_count = len(picks) - long_count
    imbalance_ratio = max(long_count, short_count) / len(picks)
    
    # If severely imbalanced (>75%), try to rebalance
    if imbalance_ratio > 0.75 and len(picks) > 20:
        minority_direction = "SHORT" if long_count > short_count else "LONG"
        chosen_ids = {p["id"] for p in picks}
        
        # Find good candidates for minority direction
        rebalance_candidates = [
            x for x in scored if (
                x["id"] not in chosen_ids and 
                x["decision"] == minority_direction and 
                x["conf"] >= 0.35 and
                x["impulse"] >= 0.0005
            )
        ]
        
        if rebalance_candidates:
            rebalance_candidates.sort(key=lambda x: x["conf"], reverse=True)
            # Replace weakest picks from majority
            picks.sort(key=lambda x: x.get("conf", 0))
            
            replace_count = min(
                8,  # Max replacements
                len(rebalance_candidates), 
                len(picks) // 6,  # Don't replace more than 1/6
                abs(long_count - short_count) // 2  # Reduce imbalance
            )
            
            for i in range(replace_count):
                if picks[i]["decision"] != minority_direction:  # Only replace majority
                    picks[i] = {
                        "id": rebalance_candidates[i]["id"],
                        "decision": rebalance_candidates[i]["decision"],
                        "conf": rebalance_candidates[i]["conf"]
                    }

    # Return final results
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

        # Enhanced preprocessing with aggressive filtering
        filtered_data = []
        for item in data:
            # Skip items with insufficient data
            if not last_prev_close(item) or not clean_obs(item):
                continue
                
            # Skip obvious test/junk data
            title = (item.get("title", "")).lower()
            if any(term in title for term in ["test", "ignore", "placeholder", "dummy", "sample"]):
                continue
                
            # Skip items with extreme price movements (likely data errors)
            try:
                lp = last_prev_close(item)
                obs = clean_obs(item)
                if lp and obs:
                    price_change = abs((obs[0]["close"] - lp) / lp)
                    if price_change > 0.5:  # 50% move - likely data error
                        continue
            except:
                continue
                
            filtered_data.append(item)

        # Use enhanced selection algorithm
        result = choose_exactly_50_v2(filtered_data)
        return jsonify(result), 200
        
    except Exception as e:
        error_msg = f"Enhanced trading bot error: {str(e)}"
        return jsonify({"error": error_msg}), 500