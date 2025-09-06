# app.py - Enhanced Trading Bot with Advanced Signal Processing
from flask import Flask, request, jsonify
import math
import re
from statistics import mean, pstdev
from collections import defaultdict, Counter
import numpy as np

from routes import app

# =========================
# Enhanced Configuration
# =========================
RISK_EPS = 1e-12
MAX_RANGE_GLITCH_PCT = 0.12     # Further tightened from 0.15
CIRCUIT_MEGA_GAP = 0.025        # Further reduced from 0.028
IMPULSE_STRONG = 0.0030         # Further reduced from 0.0035
BODY_STRONG = 0.48              # Further reduced from 0.52
VOLUME_SPIKE_HOT = 2.8          # Further reduced from 3.2
WICK_TRAP = 0.28                # Further reduced from 0.32
BOLL_Z_FADE = 1.6               # Further reduced from 1.8
BOLL_Z_TREND = 0.75             # Further reduced from 0.85
BREAK_HOLD_MIN = 0.0005         # Further reduced from 0.0006
FOLLOW_THRU_OK = 0.0004         # Further reduced from 0.0005
FOLLOW_THRU_STRONG = 0.0010     # Further reduced from 0.0012
SLOPE_MIN = 0.0012              # Further reduced from 0.0015

# Enhanced market regime detection
VOLATILITY_LOW = 0.006
VOLATILITY_HIGH = 0.020
VOLUME_DRY = 0.6
VOLUME_FLOOD = 4.5

# Time-based adjustments
ASIAN_FADE_BIAS = 1.20
LONDON_TREND_BIAS = 1.15
NY_VOLATILITY_BIAS = 1.08

# News sentiment enhancements
SUPER_BULLISH_TERMS = {
    "strategic bitcoin reserve", "bitcoin etf approved", "blackrock bitcoin", 
    "institutional adoption", "government adoption", "legal tender", "treasury bitcoin",
    "fed pivot", "rate cut", "stimulus", "money printing", "currency debasement",
    "halving", "supply shock", "scarcity", "digital gold"
}

SUPER_BEARISH_TERMS = {
    "bitcoin ban", "crypto ban", "exchange hack", "regulatory crackdown",
    "sec lawsuit", "ponzi", "bubble burst", "crash", "liquidation cascade",
    "mt gox", "ftx collapse", "celsius", "terra luna", "stablecoin depeg",
    "quantum computing threat", "mining ban", "energy concerns"
}

BLOCKLIST_IDS = set()

# Enhanced promotional detection
PROMO_WORDS = {
    "airdrop","points","quest","campaign","referral","bonus","mint","whitelist",
    "allowlist","giveaway","xp","season","farm","farming","stake to earn","rewards",
    "party","pool party","poolparty","listing soon","perp points","governance token",
    "testnet","mainnet launch","alpha","beta","early access","exclusive","limited time",
    "claim now","don't miss","last chance","presale","ico","ido","fair launch"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_WORDS), re.IGNORECASE)

# Enhanced credibility terms
LOW_CRED_TERMS = {
    "rumor","unconfirmed","parody","fake","shitpost","spoof","satire","april fools",
    "allegedly","reportedly","supposedly","claims","speculation","insider",
    "anonymous source","leak","whispers","breaking (unconfirmed)",
    "if true","sources say","according to rumors"
}

AUTHORITATIVE = {
    "BLOOMBERG","REUTERS","WALL STREET JOURNAL","WSJ","FINANCIAL TIMES","FT",
    "NIKKEI","AP","AXIOS","BARRON","BARRONS","CNBC","GUARDIAN","MARKETWATCH",
    "BBC","CNN","FORBES","ASSOCIATED PRESS"
}
CRYPTO_TIER2 = {"COINDESK","THE BLOCK","COINTELEGRAPH","DECRYPT","CRYPTOSLATE","BITCOINIST"}
ALLOWED_TWITTER = {
    "tier10k","WuBlockchain","CoinDesk","TheBlock__","Reuters","AP","Bloomberg",
    "CNBC","MarketWatch","FT","WSJ","tier10k_official","crypto_insider",
    "PeterSchiff","elonmusk","sundarpichai","michael_saylor","APompliano"
}

# Enhanced sentiment terms
GOOD_TERMS = {
    "approve","approves","approved","approval","etf","inflow","flows","treasury",
    "reserve","reserves","executive order","legalize","legalizes","adopt","adopts",
    "integrate","integrates","support","adds support","partnership","upgrade","merge",
    "launch","launches","institutional","blackrock","fidelity","google","amazon",
    "microsoft","visa","mastercard","paypal","stripe","bullish","positive",
    "breakthrough","innovation","milestone","record","all-time high","adoption",
    "mainnet","scaling solution","layer 2","defi boom","yield farming"
}
BAD_TERMS = {
    "hack","hacked","exploit","breach","rug","scam","fraud","attack","ban","bans",
    "restrict","restricts","halts","halt","suspends","withdrawals","insolvent",
    "insolvency","bankrupt","bankruptcy","lawsuit","sue","sues","charged","indicted",
    "sanction","sanctions","reject","rejected","rejects","delay","delays","postpone",
    "postpones","outage","downtime","bearish","crash","plunge","dump","sell-off",
    "correction","decline","fear","panic","uncertainty","regulatory pressure",
    "fud","capitulation","death cross","bear market"
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
    if (h - l) / ((h + l) / 2) > 0.4:  # Further tightened from 0.5
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

def macd(closes, fast=8, slow=21, signal=9):
    """MACD indicator for momentum"""
    if len(closes) < slow: return None, None, None
    
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    if not ema_fast or not ema_slow: return None, None, None
    
    macd_line = ema_fast - ema_slow
    # Simplified signal line calculation
    signal_line = macd_line * 0.8  # Approximation
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def stochastic(candles, k=14):
    """Stochastic oscillator"""
    if len(candles) < k: return None, None
    
    highs = [c["high"] for c in candles[-k:]]
    lows = [c["low"] for c in candles[-k:]]
    current_close = candles[-1]["close"]
    
    highest_high = max(highs)
    lowest_low = min(lows)
    
    if highest_high == lowest_low: return 50.0, 50.0
    
    k_percent = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent * 0.8  # Simplified D%
    
    return k_percent, d_percent

def slope_sign(closes):
    if len(closes) < 2: return 0.0
    return (closes[-1] - closes[0]) / max(RISK_EPS, closes[0])

def market_regime(candles, volumes):
    """Enhanced market regime detection"""
    if len(candles) < 5: return "unknown"
    
    closes = [c["close"] for c in candles]
    vol_avg = sum(volumes) / len(volumes) if volumes else 1.0
    
    # Volatility measure
    price_changes = [abs(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]
    volatility = sum(price_changes) / len(price_changes) if price_changes else 0.0
    
    # Trend strength
    trend_strength = abs(slope_sign(closes))
    
    # Range compression
    ranges = [(c["high"] - c["low"])/c["close"] for c in candles]
    avg_range = sum(ranges) / len(ranges) if ranges else 0.0
    
    if volatility > VOLATILITY_HIGH:
        return "volatile"
    elif volatility < VOLATILITY_LOW and avg_range < 0.005:
        return "compressed"  # New regime
    elif volatility < VOLATILITY_LOW:
        return "calm"
    elif trend_strength > 0.02:
        return "strong_trending"  # New regime
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

def enhanced_sentiment_score(title):
    """Enhanced sentiment analysis with super terms"""
    if not title: return 0.0
    t = title.lower()
    
    # Super sentiment terms (stronger signals)
    super_good = sum(1 for term in SUPER_BULLISH_TERMS if term in t)
    super_bad = sum(1 for term in SUPER_BEARISH_TERMS if term in t)
    
    # Regular sentiment terms
    good_count = sum(1 for k in GOOD_TERMS if k in t)
    bad_count = sum(1 for k in BAD_TERMS if k in t)
    
    # Calculate weighted sentiment
    sentiment = super_good * 0.25 + good_count * 0.1 - super_bad * 0.25 - bad_count * 0.1
    return max(-0.5, min(0.5, sentiment))

def source_weight(title, source):
    t_up = (title or "").upper()
    s_up = (source or "").upper()
    w = 1.0
    
    # Enhanced weighting with new sources
    if s_up in AUTHORITATIVE: w += 0.70  # Further increased
    elif s_up in CRYPTO_TIER2: w += 0.40  # Further increased
    if any(b in t_up for b in AUTHORITATIVE): w += 0.35
    if any(b in t_up for b in CRYPTO_TIER2): w += 0.25
    
    if (source or "").lower() == "twitter":
        low = (title or "").lower()
        if any(h.lower() in low for h in ALLOWED_TWITTER): 
            w += 0.30  # Further increased
        else: 
            w -= 0.40  # Maintained penalty
    
    if any(wd in (title or "").lower() for wd in LOW_CRED_TERMS): 
        w -= 0.45  # Further increased penalty
    
    return max(0.12, w)  # Slightly reduced minimum

def title_tilt(title):
    return enhanced_sentiment_score(title)

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
    
    # Asian session (22:00-08:00 UTC) - stronger fade bias
    if 22 <= hour or hour <= 8:
        return ASIAN_FADE_BIAS
    # London session (08:00-16:00 UTC) - stronger trend bias
    elif 8 <= hour <= 16:
        return LONDON_TREND_BIAS
    # NY session overlap (13:00-20:00 UTC) - enhanced volatility bias
    elif 13 <= hour <= 20:
        return NY_VOLATILITY_BIAS
    
    return 1.0

def liquidity_profile(volumes, avg_vol):
    """Analyze liquidity conditions"""
    if not volumes or not avg_vol: return "normal"
    
    recent_vol = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else volumes[-1]
    ratio = recent_vol / avg_vol
    
    if ratio > VOLUME_FLOOD: return "high_liquidity"
    elif ratio < VOLUME_DRY: return "low_liquidity"
    else: return "normal"

# ========= Enhanced Experts =========

def expert_momentum_shock(lp, c1, c2, c3, prev_vol_avg, regime, rsi_val, macd_data, stoch_data):
    """Enhanced momentum expert with additional indicators"""
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
    if abs(imp) >= IMPULSE_STRONG: score += 1.3  # Further increased
    if body >= BODY_STRONG: score += 1.1  # Further increased
    if spike >= VOLUME_SPIKE_HOT: score += 0.9  # Further increased
    if ft2*imp > 0: score += 0.7  # Further increased
    if ft3*imp > 0: score += 0.6  # Further increased
    if abs(gap) <= 0.010: score += 0.4  # Further increased
    if abs(gap) > 0.018: score -= 0.6  # Further increased penalty
    if (h1-l1)/lp > 0.020: score -= 0.5  # Further increased penalty

    # Enhanced regime adjustments
    if regime == "strong_trending": score += 0.4  # New regime bonus
    elif regime == "trending": score += 0.3
    elif regime == "ranging": score -= 0.2
    elif regime == "volatile": score -= 0.1
    elif regime == "compressed": score += 0.2  # New regime bonus

    # Enhanced RSI confirmation
    if imp > 0 and rsi_val < 65: score += 0.3  # Adjusted threshold
    elif imp < 0 and rsi_val > 35: score += 0.3  # Adjusted threshold
    elif imp > 0 and rsi_val > 85: score -= 0.4  # Enhanced overbought penalty
    elif imp < 0 and rsi_val < 15: score -= 0.4  # Enhanced oversold penalty

    # MACD confirmation
    if macd_data and macd_data[0] is not None:
        macd_line, signal_line, histogram = macd_data
        if imp > 0 and macd_line > signal_line: score += 0.2
        elif imp < 0 and macd_line < signal_line: score += 0.2

    # Stochastic confirmation
    if stoch_data and stoch_data[0] is not None:
        k_percent, d_percent = stoch_data
        if imp > 0 and k_percent < 80: score += 0.15
        elif imp < 0 and k_percent > 20: score += 0.15

    direction = "LONG" if imp > 0 else "SHORT"
    confidence = max(0.0, score)
    return direction, confidence

def expert_exhaustion_fade(lp, c1, c2, c3, regime, rsi_val, atr_val, stoch_data):
    """Enhanced exhaustion expert"""
    o1,h1,l1,cl1,_ = c1["open"],c1["high"],c1["low"],c1["close"],c1["volume"]
    imp = (cl1 - lp)/lp
    body, up, lo, rng1 = candle_shape(c1)
    gap = (o1 - lp)/lp
    ft2 = (c2["close"]-cl1)/cl1 if c2 else 0.0
    ft3 = (c3["close"]-cl1)/cl1 if c3 else 0.0

    score = 0.0
    if abs(gap) > CIRCUIT_MEGA_GAP: score += 1.6  # Further increased
    if (h1-l1)/lp > 0.018: score += 1.2  # Further increased
    if imp > 0 and up >= WICK_TRAP: score += 1.2  # Further increased
    if imp < 0 and lo >= WICK_TRAP: score += 1.2  # Further increased
    if ft2*imp < 0: score += 0.9  # Further increased
    if ft3*imp < 0: score += 0.7  # Further increased
    if body < 0.25 and abs(imp) >= IMPULSE_STRONG: score += 0.6  # Further increased

    # Enhanced exhaustion signals
    if atr_val and (h1-l1) > atr_val * 2.2: score += 0.5  # Enhanced ATR multiple
    
    # Enhanced RSI divergence
    if imp > 0 and rsi_val > 78: score += 0.7  # Further increased
    elif imp < 0 and rsi_val < 22: score += 0.7  # Further increased

    # Stochastic exhaustion
    if stoch_data and stoch_data[0] is not None:
        k_percent, d_percent = stoch_data
        if imp > 0 and k_percent > 85: score += 0.3  # Extreme overbought
        elif imp < 0 and k_percent < 15: score += 0.3  # Extreme oversold

    # Enhanced regime adjustments
    if regime == "volatile": score += 0.4  # Further increased
    elif regime == "compressed": score += 0.3  # New regime bonus
    elif regime == "calm": score -= 0.2

    direction = "SHORT" if imp > 0 else "LONG"
    confidence = max(0.0, score)
    return direction, confidence

def expert_breakout(prev_hi, prev_lo, c1, c2, regime, volume_ratio, atr_val):
    """Enhanced breakout expert"""
    cl1 = c1["close"]; o1 = c1["open"]; h1=c1["high"]; l1=c1["low"]
    break_up = h1 > prev_hi * (1.0 + BREAK_HOLD_MIN)
    break_dn = l1 < prev_lo * (1.0 - BREAK_HOLD_MIN)

    hold_up = c2 and (c2["close"] > max(prev_hi, cl1*(1.0 - FOLLOW_THRU_OK)))
    hold_dn = c2 and (c2["close"] < min(prev_lo, cl1*(1.0 + FOLLOW_THRU_OK)))

    strong_up = c2 and (c2["close"] - cl1)/cl1 > FOLLOW_THRU_STRONG
    strong_dn = c2 and (cl1 - c2["close"])/cl1 > FOLLOW_THRU_STRONG

    base_conf = 1.0  # Increased from 0.9
    
    # Enhanced volume confirmation
    if volume_ratio > 2.5: base_conf += 0.4  # Further increased
    elif volume_ratio > 1.8: base_conf += 0.2
    elif volume_ratio < 1.1: base_conf -= 0.3  # Enhanced penalty

    # ATR-based confirmation
    if atr_val:
        breakout_size = max(h1 - prev_hi, prev_lo - l1) if break_up or break_dn else 0
        if breakout_size > atr_val * 0.5: base_conf += 0.2
    
    # Enhanced regime adjustments
    if regime == "strong_trending": base_conf += 0.3  # New regime bonus
    elif regime == "trending": base_conf += 0.2
    elif regime == "compressed": base_conf += 0.4  # Compression breakout bonus
    elif regime == "ranging": base_conf -= 0.1

    if break_up and (hold_up or strong_up):
        return "LONG", base_conf
    if break_dn and (hold_dn or strong_dn):
        return "SHORT", base_conf
    
    # Enhanced fakeout detection
    fade_conf = base_conf * 0.95  # Increased from 0.9
    if break_up and c2 and c2["close"] < prev_hi * 0.9985:  # Tighter fakeout detection
        return "SHORT", fade_conf
    if break_dn and c2 and c2["close"] > prev_lo * 1.0015:  # Tighter fakeout detection
        return "LONG", fade_conf
    
    return None, 0.0

def expert_bollinger(prev_cl, c1, c2, regime, rsi_val):
    """Enhanced Bollinger Bands expert"""
    if len(prev_cl) < 8:
        return None, 0.0
    mu = mean(prev_cl)
    sd = pstdev(prev_cl) if len(prev_cl) > 1 else 0.0
    z1 = zscore(c1["close"], mu, sd)
    
    rev = c2 and ((c2["close"] - c1["close"])/max(RISK_EPS, c1["close"]))
    
    base_conf = 0.85  # Increased from 0.8
    
    # Enhanced regime adjustments
    if regime == "ranging": base_conf += 0.25  # Further increased
    elif regime == "compressed": base_conf += 0.20  # New regime bonus
    elif regime == "trending": base_conf -= 0.1

    # RSI confirmation for mean reversion
    if abs(z1) >= BOLL_Z_FADE:
        rsi_boost = 0.0
        if z1 > 0 and rsi_val > 70: rsi_boost += 0.15  # Overbought confirmation
        elif z1 < 0 and rsi_val < 30: rsi_boost += 0.15  # Oversold confirmation
        
        if rev is None or abs(rev) < 0.0008:  # Tighter reversal threshold
            return ("SHORT" if z1 > 0 else "LONG"), base_conf + rsi_boost
    
    if abs(z1) >= BOLL_Z_TREND and rev and abs(rev) >= 0.0008:  # Tighter trend threshold
        conf = base_conf * 0.8
        return ("LONG" if z1 > 0 else "SHORT"), conf
    
    return None, 0.0

def expert_microtrend(prev_cl, lp, c1, regime, macd_data):
    """Enhanced microtrend expert"""
    if len(prev_cl) < 6:
        return None, 0.0
    
    e5 = ema(prev_cl, 5)
    e12 = ema(prev_cl, 12) if len(prev_cl) >= 12 else ema(prev_cl, max(4,len(prev_cl)-1))
    slope = (e5 - e12)/max(RISK_EPS, e12 or lp)
    
    if abs(slope) < SLOPE_MIN:
        return None, 0.0
    
    imp = (c1["close"] - lp)/lp
    base_conf = 0.60  # Increased from 0.55
    
    # Enhanced regime adjustments
    if regime == "strong_trending": base_conf += 0.20  # New regime bonus
    elif regime == "trending": base_conf += 0.15
    elif regime == "ranging": base_conf -= 0.10

    # MACD trend confirmation
    if macd_data and macd_data[0] is not None:
        macd_line, signal_line, _ = macd_data
        macd_trend = 1 if macd_line > signal_line else -1
        slope_trend = 1 if slope > 0 else -1
        if macd_trend == slope_trend:
            base_conf += 0.1

    if slope * imp > 0:
        return ("LONG" if slope > 0 else "SHORT"), base_conf
    else:
        return ("SHORT" if slope > 0 else "LONG"), base_conf * 0.85  # Increased from 0.8

def expert_mean_reversion(prev_cl, lp, c1, c2, rsi_val):
    """Enhanced mean reversion expert"""
    if len(prev_cl) < 5: return None, 0.0
    
    recent_mean = mean(prev_cl[-5:])
    imp = (c1["close"] - lp)/lp
    distance_from_mean = (c1["close"] - recent_mean) / recent_mean
    
    # Look for reversal in c2
    reversal = c2 and (c2["close"] - c1["close"]) / c1["close"]
    
    score = 0.0
    if abs(distance_from_mean) > 0.012:  # Reduced threshold
        score += 0.7  # Increased from 0.6
        if reversal and reversal * distance_from_mean < 0:  # Reverting
            score += 0.5  # Increased from 0.4
    
    # RSI confirmation for mean reversion
    if distance_from_mean > 0 and rsi_val > 65:
        score += 0.2  # Overbought mean reversion
    elif distance_from_mean < 0 and rsi_val < 35:
        score += 0.2  # Oversold mean reversion
    
    if score > 0.35:  # Reduced threshold from 0.3
        direction = "LONG" if distance_from_mean < 0 else "SHORT"
        return direction, score
    
    return None, 0.0

def expert_gap_analysis(lp, c1, c2, prev_vol_avg):
    """New expert for gap analysis and gap filling"""
    gap = (c1["open"] - lp) / lp
    if abs(gap) < 0.003: return None, 0.0  # Only significant gaps
    
    # Gap fill tendency analysis
    if c2:
        gap_fill = (c2["close"] - c1["open"]) / c1["open"]
        if gap > 0 and gap_fill < -0.001:  # Gap up being filled
            return "SHORT", 0.6
        elif gap < 0 and gap_fill > 0.001:  # Gap down being filled
            return "LONG", 0.6
    
    # Volume confirmation for gap
    volume_ratio = c1.get("volume", 0) / max(prev_vol_avg or 1.0, 0.001)
    base_score = 0.4
    
    if volume_ratio > 2.0:  # High volume gap
        if abs(gap) > 0.015:  # Large gap with high volume - likely continuation
            return ("LONG" if gap > 0 else "SHORT"), base_score + 0.3
        else:  # Medium gap with high volume - likely to fill
            return ("SHORT" if gap > 0 else "LONG"), base_score + 0.2
    
    return None, 0.0

def expert_pattern_recognition(candles, lp):
    """New expert for candlestick pattern recognition"""
    if len(candles) < 3: return None, 0.0
    
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    
    # Doji patterns
    def is_doji(candle):
        body = abs(candle["close"] - candle["open"])
        range_size = candle["high"] - candle["low"]
        return body / max(range_size, 0.0001) < 0.1
    
    # Hammer/Shooting star patterns
    def is_hammer(candle):
        body = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        return lower_wick > body * 2 and upper_wick < body * 0.5
    
    def is_shooting_star(candle):
        body = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        return upper_wick > body * 2 and lower_wick < body * 0.5
    
    # Pattern scoring
    score = 0.0
    direction = None
    
    # Engulfing patterns
    if c2["close"] < c2["open"] and c3["close"] > c3["open"]:  # Bullish engulfing
        if c3["open"] < c2["close"] and c3["close"] > c2["open"]:
            score += 0.5
            direction = "LONG"
    
    if c2["close"] > c2["open"] and c3["close"] < c3["open"]:  # Bearish engulfing
        if c3["open"] > c2["close"] and c3["close"] < c2["open"]:
            score += 0.5
            direction = "SHORT"
    
    # Doji reversal
    if is_doji(c3):
        prev_trend = (c2["close"] - c1["close"]) / c1["close"]
        if prev_trend > 0.005:  # Uptrend, doji suggests reversal
            score += 0.4
            direction = "SHORT"
        elif prev_trend < -0.005:  # Downtrend, doji suggests reversal
            score += 0.4
            direction = "LONG"
    
    # Hammer at support/resistance
    if is_hammer(c3) and c3["close"] > lp * 0.995:  # Hammer near support
        score += 0.6
        direction = "LONG"
    
    # Shooting star at resistance
    if is_shooting_star(c3) and c3["close"] < lp * 1.005:  # Shooting star near resistance
        score += 0.6
        direction = "SHORT"
    
    if score > 0.3 and direction:
        return direction, score
    
    return None, 0.0

def expert_volume_profile(candles, volumes):
    """New expert for volume profile analysis"""
    if len(candles) < 5 or len(volumes) < 5: return None, 0.0
    
    # Volume-weighted average price (VWAP) approximation
    total_volume = sum(volumes)
    if total_volume == 0: return None, 0.0
    
    vwap = sum(candles[i]["close"] * volumes[i] for i in range(len(candles))) / total_volume
    current_price = candles[-1]["close"]
    
    # Distance from VWAP
    vwap_distance = (current_price - vwap) / vwap
    
    # Volume trend
    recent_vol = sum(volumes[-3:]) / 3
    older_vol = sum(volumes[-6:-3]) / 3 if len(volumes) >= 6 else recent_vol
    vol_trend = (recent_vol - older_vol) / max(older_vol, 0.001)
    
    score = 0.0
    
    # Price far from VWAP with volume confirmation
    if abs(vwap_distance) > 0.008 and vol_trend > 0.5:  # High volume, price away from VWAP
        score += 0.5
        direction = "LONG" if vwap_distance < 0 else "SHORT"  # Mean reversion
        return direction, score
    
    # Volume breakout
    if vol_trend > 2.0:  # Significant volume increase
        price_trend = (candles[-1]["close"] - candles[-3]["close"]) / candles[-3]["close"]
        if abs(price_trend) > 0.003:  # Price moving with volume
            score += 0.4
            direction = "LONG" if price_trend > 0 else "SHORT"
            return direction, score
    
    return None, 0.0

def expert_multi_timeframe_confluence(prev_candles, c1):
    """New expert analyzing multiple timeframe confluence"""
    if len(prev_candles) < 10: return None, 0.0
    
    closes = [c["close"] for c in prev_candles]
    current_close = c1["close"]
    
    # Short-term trend (3 periods)
    short_trend = (closes[-1] - closes[-3]) / closes[-3] if len(closes) >= 3 else 0
    
    # Medium-term trend (6 periods)
    med_trend = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
    
    # Long-term trend (10 periods)
    long_trend = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
    
    # Current impulse
    current_impulse = (current_close - closes[-1]) / closes[-1]
    
    score = 0.0
    confluence_count = 0
    
    # Count trend alignment
    trends = [short_trend, med_trend, long_trend]
    positive_trends = sum(1 for t in trends if t > 0.002)
    negative_trends = sum(1 for t in trends if t < -0.002)
    
    # All trends aligned
    if positive_trends >= 2 and current_impulse > 0.001:
        score += 0.6
        confluence_count += 1
        direction = "LONG"
    elif negative_trends >= 2 and current_impulse < -0.001:
        score += 0.6
        confluence_count += 1
        direction = "SHORT"
    # Trend reversal confluence
    elif positive_trends >= 2 and current_impulse < -0.003:  # Strong counter-move
        score += 0.7
        confluence_count += 1
        direction = "SHORT"
    elif negative_trends >= 2 and current_impulse > 0.003:  # Strong counter-move
        score += 0.7
        confluence_count += 1
        direction = "LONG"
    else:
        return None, 0.0
    
    # Boost score for stronger confluence
    if confluence_count > 0:
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

    prev_candles_data = prev_candles(item, k=18)  # Increased from 15
    prev_hi, prev_lo = prev_hilo(item, k=15)
    prev_cl = prev_closes(item, k=18)  # Increased from 15
    pv_avg = avg_prev_volume(item, k=5)  # Increased from 4

    # Enhanced market context with new indicators
    volumes = [c.get("volume", 0) for c in prev_candles_data[-8:]]
    regime = market_regime(prev_candles_data[-10:], volumes[-5:])
    rsi_val = rsi(prev_cl + [c1["close"]])
    atr_val = atr(prev_candles_data[-10:]) if len(prev_candles_data) >= 10 else None
    volume_ratio = (c1["volume"] / max(RISK_EPS, pv_avg or 1.0)) if pv_avg else 1.0
    t_bias = time_bias(timestamp)
    
    # New indicators
    macd_data = macd(prev_cl + [c1["close"]])
    stoch_data = stochastic(prev_candles_data + [c1]) if prev_candles_data else None
    liquidity = liquidity_profile(volumes, pv_avg)

    # Expert votes with enhanced context
    votes = []

    d, s = expert_momentum_shock(lp, c1, c2, c3, pv_avg, regime, rsi_val, macd_data, stoch_data)
    votes.append((d, s, "mom"))
    
    d, s = expert_exhaustion_fade(lp, c1, c2, c3, regime, rsi_val, atr_val, stoch_data)
    votes.append((d, s, "fade"))
    
    if prev_hi and prev_lo:
        d, s = expert_breakout(prev_hi, prev_lo, c1, c2, regime, volume_ratio, atr_val)
        votes.append((d, s, "brk"))
    
    d, s = expert_bollinger(prev_cl, c1, c2, regime, rsi_val)
    votes.append((d, s, "boll"))
    
    d, s = expert_microtrend(prev_cl, lp, c1, regime, macd_data)
    votes.append((d, s, "micro"))
    
    d, s = expert_mean_reversion(prev_cl, lp, c1, c2, rsi_val)
    votes.append((d, s, "mean_rev"))
    
    # New experts
    d, s = expert_gap_analysis(lp, c1, c2, pv_avg)
    votes.append((d, s, "gap"))
    
    d, s = expert_pattern_recognition(prev_candles_data + [c1], lp)
    votes.append((d, s, "pattern"))
    
    d, s = expert_volume_profile(prev_candles_data[-5:] + [c1], volumes[-5:] + [c1.get("volume", 0)])
    votes.append((d, s, "vol_profile"))
    
    d, s = expert_multi_timeframe_confluence(prev_candles_data, c1)
    votes.append((d, s, "mtf"))

    # Enhanced weighting and adjustments
    w_src = source_weight(title, source)
    tilt = title_tilt(title)
    promo_pen = 0.75 if is_promotional(title) else 1.0  # Further increased penalty

    # Combine votes with enhanced weighting
    long_score = 0.0
    short_score = 0.0
    
    for d, s, tag in votes:
        if not d: continue
        
        # Enhanced expert weighting with new experts
        w = s
        if tag == "mom": w *= 1.20  # Further increased
        elif tag == "brk": w *= 1.15  # Further increased
        elif tag == "fade": w *= 1.10  # Further increased
        elif tag == "pattern": w *= 1.08  # New expert weighting
        elif tag == "mtf": w *= 1.12  # New expert weighting (high value)
        elif tag == "vol_profile": w *= 1.05  # New expert weighting
        elif tag == "gap": w *= 1.03  # New expert weighting
        elif tag == "mean_rev": w *= 1.05
        
        # Liquidity adjustments
        if liquidity == "low_liquidity": w *= 0.9  # Reduce confidence in low liquidity
        elif liquidity == "high_liquidity": w *= 1.05  # Boost confidence in high liquidity
        
        if d == "LONG": long_score += w
        else: short_score += w

    # Apply all adjustments
    long_score *= (w_src * (1.0 + max(0.0, tilt)) * promo_pen)
    short_score *= (w_src * (1.0 + max(0.0, -tilt)) * promo_pen)
    
    # Enhanced time bias
    if t_bias > 1.0:  # Fade bias in Asian session
        gap = (c1["open"] - lp)/lp
        if gap > 0: short_score *= t_bias
        else: long_score *= t_bias

    # Enhanced circuit breaker with regime awareness
    gap = (c1["open"] - lp)/lp
    if abs(gap) > CIRCUIT_MEGA_GAP:
        multiplier = 1.25  # Further increased from 1.20
        if regime == "volatile": multiplier *= 1.1  # Extra boost in volatile regime
        elif regime == "compressed": multiplier *= 1.2  # Even more boost after compression
        
        if gap > 0: short_score *= multiplier
        else: long_score *= multiplier

    # Enhanced tie-breaking with multiple factors
    if abs(long_score - short_score) < 0.25:  # Further increased threshold
        imp = (c1["close"] - lp)/lp
        tie_break = 0.18  # Further increased
        
        # Regime-based tie breaking
        if regime == "volatile": tie_break *= 1.4
        elif regime == "compressed": tie_break *= 1.3
        elif regime == "calm": tie_break *= 0.8
        
        # Volume-based tie breaking
        if volume_ratio > 3.0: tie_break *= 1.2
        elif volume_ratio < 1.0: tie_break *= 0.8
        
        # RSI-based tie breaking
        if rsi_val > 75 and imp > 0: tie_break *= 1.3  # Overbought reversal
        elif rsi_val < 25 and imp < 0: tie_break *= 1.3  # Oversold reversal
        
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

    # More sophisticated quality gate
    quality_gate = (
        (impulse >= 0.0008 and spike >= 1.0 and conf_gap >= 0.06) or  # Standard quality
        (impulse >= 0.0015 and conf_gap >= 0.10) or  # High impulse override
        (spike >= 2.5 and conf_gap >= 0.08) or  # High volume override
        (regime in ["compressed", "volatile"] and conf_gap >= 0.12)  # Regime-specific override
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
        "rsi": float(rsi_val),
        "liquidity": liquidity,
        "volume_ratio": float(volume_ratio)
    }

def dedupe_keep_first(items):
    seen = set()
    out = []
    for it in items:
        iid = it.get("id")
        if iid is None or iid in seen: continue
        seen.add(iid); out.append(it)
    return out

def advanced_portfolio_optimization(scored_items):
    """Advanced portfolio balancing with machine learning-inspired scoring"""
    if not scored_items:
        return scored_items
    
    # Analyze multiple dimensions
    regime_counts = defaultdict(int)
    rsi_levels = []
    volume_ratios = []
    impulse_levels = []
    
    for item in scored_items:
        regime_counts[item.get("regime", "unknown")] += 1
        rsi_levels.append(item.get("rsi", 50))
        volume_ratios.append(item.get("volume_ratio", 1.0))
        impulse_levels.append(item.get("impulse", 0))
    
    dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else "unknown"
    avg_rsi = sum(rsi_levels) / len(rsi_levels) if rsi_levels else 50
    avg_volume_ratio = sum(volume_ratios) / len(volume_ratios) if volume_ratios else 1.0
    avg_impulse = sum(impulse_levels) / len(impulse_levels) if impulse_levels else 0
    
    # Dynamic adjustments based on market conditions
    for item in scored_items:
        original_conf = item["conf"]
        
        # Regime-based adjustments
        if dominant_regime == "volatile":
            if "SHORT" in item["decision"] and item["impulse"] > avg_impulse * 1.5:
                item["conf"] *= 1.20  # Boost fades in volatile markets
        elif dominant_regime == "compressed":
            if item["spike"] > avg_volume_ratio * 1.5:
                item["conf"] *= 1.25  # Boost breakouts after compression
        elif dominant_regime == "strong_trending":
            if item["impulse"] > avg_impulse and item["spike"] > 1.5:
                item["conf"] *= 1.15  # Boost momentum in strong trends
        elif dominant_regime == "ranging":
            if item["rsi"] > 70 or item["rsi"] < 30:
                item["conf"] *= 1.10  # Boost extremes in ranging markets
        
        # Market-wide RSI adjustments
        if avg_rsi > 65:  # Overbought market
            if "SHORT" in item["decision"]: item["conf"] *= 1.15
        elif avg_rsi < 35:  # Oversold market
            if "LONG" in item["decision"]: item["conf"] *= 1.15
        
        # Volume environment adjustments
        if avg_volume_ratio > 2.5:  # High volume environment
            if item["volume_ratio"] > avg_volume_ratio: item["conf"] *= 1.10
        elif avg_volume_ratio < 1.2:  # Low volume environment
            if item["quality"]: item["conf"] *= 1.05  # Boost quality in low volume
        
        # Impulse distribution adjustments
        if item["impulse"] > avg_impulse * 2.0:  # Outlier impulse
            item["conf"] *= 1.12
    
    return scored_items

def choose_exactly_50(raw_items):
    """Enhanced selection with advanced portfolio optimization"""
    # Build scored list
    scored = []
    for it in raw_items:
        if it.get("id") in BLOCKLIST_IDS: continue
        s = vote_and_decide(it)
        if s and s["id"] is not None:
            scored.append(s)

    scored = dedupe_keep_first(scored)
    scored = advanced_portfolio_optimization(scored)

    # Enhanced multi-tier selection system
    
    # Tier 0: Ultra-premium - exceptional signals across all metrics
    tier0 = [x for x in scored if (
        x["quality"] and 
        not x["promo"] and 
        x["conf"] >= 1.2 and
        x["impulse"] >= 0.003 and
        x["spike"] >= 2.0
    )]
    tier0.sort(key=lambda x: (
        x["conf"], 
        x["impulse"], 
        x["spike"], 
        x["micro"], 
        -x["rsi"] if x["decision"] == "SHORT" else x["rsi"],  # RSI alignment
        x["time"], 
        x["id"]
    ), reverse=True)
    picks = tier0[:20]  # Take top 20 ultra-premium picks
    
    # Tier 1: Premium quality - high confidence, non-promotional
    if len(picks) < 50:
        need = min(20, 50 - len(picks))
        chosen = {p["id"] for p in picks}
        tier1 = [x for x in scored if (
            x["id"] not in chosen and
            x["quality"] and 
            not x["promo"] and 
            x["conf"] >= 0.9 and
            x["impulse"] >= 0.0018
        )]
        tier1.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["spike"], 
            x["micro"], 
            x["volume_ratio"],
            x["time"], 
            x["id"]
        ), reverse=True)
        picks.extend(tier1[:need])
    
    # Tier 2: Quality with regime bonus
    if len(picks) < 50:
        need = min(10, 50 - len(picks))
        chosen = {p["id"] for p in picks}
        tier2 = [x for x in scored if (
            x["id"] not in chosen and
            (x["quality"] or x["regime"] in ["compressed", "volatile"]) and
            not x["promo"] and 
            x["conf"] >= 0.7
        )]
        tier2.sort(key=lambda x: (
            x["conf"], 
            x["impulse"], 
            x["spike"], 
            1.0 if x["regime"] in ["compressed", "volatile"] else 0.8,  # Regime bonus
            x["time"], 
            x["id"]
        ), reverse=True)
        picks.extend(tier2[:need])
    
    # Continue with existing tiers but with refined thresholds...
    # Tier 3: Acceptable quality
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        tier3 = [x for x in scored if (
            x["id"] not in chosen and
            (x["quality"] or x["conf"] >= 0.65) and
            x["impulse"] >= 0.0010
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

    # Enhanced fallback tiers...
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        tier4 = [x for x in scored if (
            x["id"] not in chosen and
            x["conf"] >= 0.35 and
            x["impulse"] >= 0.0006
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

    # Enhanced emergency fallback
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
            
            # Enhanced emergency scoring with sentiment
            base_conf = 0.08 + min(0.30, abs(imp) * 60)
            
            # Sentiment boost
            sentiment = enhanced_sentiment_score(it.get("title", ""))
            if sentiment != 0:
                base_conf += abs(sentiment) * 0.5
            
            # Volume consideration
            vol = c1.get("volume", 0)
            if vol > 0: base_conf *= 1.15
            
            # Time-based boost
            if time_bias(it.get("time", 0)) > 1.0: base_conf *= 1.1
            
            emergency.append({
                "id": iid,
                "decision": "SHORT" if (imp > 0 and sentiment <= 0) or (imp > 0 and sentiment == 0) else "LONG",
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
    
    # Advanced portfolio balancing
    long_count = sum(1 for p in picks if p["decision"] == "LONG")
    short_count = len(picks) - long_count
    
    # More sophisticated rebalancing
    if long_count > 0.85 * len(picks) or short_count > 0.85 * len(picks):
        minority_direction = "SHORT" if long_count > short_count else "LONG"
        chosen_ids = {p["id"] for p in picks}
        
        # Find high-quality candidates for minority direction
        rebalance_candidates = []
        for x in scored:
            if (x["id"] not in chosen_ids and 
                x["decision"] == minority_direction and 
                x["conf"] >= 0.4 and
                x["impulse"] >= 0.0008):
                rebalance_candidates.append(x)
        
        if rebalance_candidates:
            rebalance_candidates.sort(key=lambda x: x["conf"], reverse=True)
            # Replace weakest picks with minority direction
            picks.sort(key=lambda x: (x["conf"], x["impulse"]))
            replace_count = min(8, len(rebalance_candidates), len(picks) // 3)  # More aggressive rebalancing
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

        # Enhanced preprocessing with more sophisticated filtering
        filtered_data = []
        title_counter = Counter()
        
        for item in data:
            title = item.get("title", "")
            source = item.get("source", "")
            
            # Skip items with no price data
            if not last_prev_close(item) or not clean_obs(item):
                continue
                
            # Skip obvious junk
            if any(term in title.lower() for term in ["test", "ignore", "placeholder", "deleted"]):
                continue
                
            # Skip duplicate titles (keep first occurrence)
            title_key = title.lower().strip()[:100]  # First 100 chars
            title_counter[title_key] += 1
            if title_counter[title_key] > 1:
                continue
            
            # Skip very low credibility sources
            if source.lower() in ["reddit", "4chan", "telegram"] and "tier10k" not in title.lower():
                continue
                
            # Skip items with extreme price anomalies
            obs = clean_obs(item)
            if obs:
                c1 = obs[0]
                lp = last_prev_close(item)
                if lp and abs((c1["close"] - lp) / lp) > 0.50:  # >50% move, likely error
                    continue
            
            filtered_data.append(item)

        result = choose_exactly_50(filtered_data)
        return jsonify(result), 200
        
    except Exception as e:
        # Enhanced error logging
        error_msg = f"Error processing trading bot request: {str(e)}"
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)