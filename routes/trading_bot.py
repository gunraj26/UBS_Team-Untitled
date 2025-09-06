# app.py
from flask import Flask, request, jsonify
import math
import re
from statistics import mean, pstdev

from routes import app 

# =========================
# Tunables
# =========================
RISK_EPS = 1e-12

# data sanity
MAX_RANGE_GLITCH_PCT = 0.18     # if 1m range > 18% of price → bad tick
MAX_WICK_ANOMALY = 8.0          # wick/body > 8x → suspicious
MIN_BODY_TINY = 0.04            # body/range too tiny → indecision

# regime
RET_LOOKBACK = 12               # prev candles lookback
HIGH_VOL_Z = 1.25               # realized vol zscore vs 6-sample mean
STRONG_TREND_SLOPE = 0.0030     # EMA(5)-EMA(12) slope
WEAK_TREND_SLOPE = 0.0012

# impulses & gaps
IMPULSE_STRONG = 0.0048         # 0.48% close-vs-last-prev
IMPULSE_MED = 0.0030
CIRCUIT_MEGA_GAP = 0.035        # >3.5% open gap → reversion bias
GAP_MAGNET = 0.0025             # small gaps tend to fill quickly

# volume & follow-through
VOLUME_SPIKE_HOT = 4.0          # (v1..v3)/avg(prev vol)
FOLLOW_THRU_OK = 0.0006
FOLLOW_THRU_STRONG = 0.0016

# bands
BOLL_Z_FADE = 2.1
BOLL_Z_TREND = 1.0

# wicks, traps
WICK_TRAP = 0.36
WICK_IMBAL = 0.22               # extra wick one side

# ranking & gates
PRIMARY_QUALITY_IMPULSE = 0.0012
PRIMARY_QUALITY_SPIKE = 1.2
CONSENSUS_MIN = 2               # need at least 2 experts aligned unless ultra-strong
ULTRA_STRONG = 1.8              # single expert confidence to allow bypass

# news credibility
AUTHORITATIVE = {
    "BLOOMBERG","REUTERS","WALL STREET JOURNAL","WSJ","FINANCIAL TIMES","FT",
    "NIKKEI","AP","AXIOS","BARRON","BARRONS","CNBC","GUARDIAN","MARKETWATCH"
}
CRYPTO_TIER2 = {"COINDESK","THE BLOCK","COINTELEGRAPH","DECRYPT"}
ALLOWED_TWITTER = {"tier10k","WuBlockchain","CoinDesk","TheBlock__","Reuters","AP","Bloomberg","CNBC","MarketWatch","FT","WSJ"}
LOW_CRED_TERMS = {"rumor","unconfirmed","parody","fake","shitpost","spoof","satire","april fools"}
GOOD_TERMS = {
    "approve","approved","approval","etf","inflow","flows","treasury","reserve","reserves",
    "executive order","legalize","adopt","adopts","integrate","support","adds support",
    "partnership","upgrade","merge","launch","launches","institutional","blackrock",
    "fidelity","google","amazon","microsoft","visa","mastercard","paypal","stripe"
}
BAD_TERMS = {
    "hack","hacked","exploit","breach","rug","scam","fraud","attack","ban","bans",
    "restrict","halts","suspends","withdrawals","insolvent","insolvency","bankrupt",
    "bankruptcy","lawsuit","sue","sues","charged","indicted","sanction","sanctions",
    "reject","rejected","rejects","delay","delays","postpone","postpones","outage","downtime"
}
PROMO_WORDS = {
    "airdrop","points","quest","campaign","referral","bonus","mint","whitelist",
    "allowlist","giveaway","xp","season","farm","farming","stake to earn","rewards",
    "party","pool party","poolparty","listing soon","perp points"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_WORDS), re.IGNORECASE)

BLOCKLIST_IDS = set()  # fill if you learn per-dataset junk ids

# =========================
# Utils
# =========================
def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def sort_by_timestamp(candles):
    def key(c):
        ts = c.get("timestamp")
        return int(ts) if ts is not None else (c.get("datetime") or "")
    return sorted(candles or [], key=key)

def clean_candle(c):
    o = safe_float(c.get("open")); h = safe_float(c.get("high"))
    l = safe_float(c.get("low")); cl = safe_float(c.get("close"))
    v = safe_float(c.get("volume")) or 0.0
    if None in (o,h,l,cl) or min(o,h,l,cl) <= 0 or h < l:
        return None
    return {"open":o,"high":h,"low":l,"close":cl,"volume":max(0.0, v)}

def prev_candles(item, k=RET_LOOKBACK):
    pcs = sort_by_timestamp(item.get("previous_candles"))
    out = []
    for c in pcs[-k:]:
        cc = clean_candle(c)
        if cc: out.append(cc)
    return out

def clean_obs(item):
    obs = sort_by_timestamp(item.get("observation_candles"))
    out = []
    for c in obs[:3]:
        cc = clean_candle(c)
        if cc: out.append(cc)
    return out

def last_prev_close(item):
    pcs = prev_candles(item, k=1)
    return pcs[-1]["close"] if pcs else None

def prev_closes(item, k=RET_LOOKBACK):
    pcs = prev_candles(item, k=k)
    return [c["close"] for c in pcs]

def prev_hilo(item, k=RET_LOOKBACK):
    pcs = prev_candles(item, k=k)
    if not pcs: return None, None
    return max(p["high"] for p in pcs), min(p["low"] for p in pcs)

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

def slope_sign(closes):
    if len(closes) < 2: return 0.0
    return (closes[-1] - closes[0]) / max(RISK_EPS, closes[0])

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
    if s_up in AUTHORITATIVE: w += 0.50
    elif s_up in CRYPTO_TIER2: w += 0.30
    if any(b in t_up for b in AUTHORITATIVE): w += 0.25
    if any(b in t_up for b in CRYPTO_TIER2): w += 0.15
    if (source or "").lower() == "twitter":
        low = (title or "").lower()
        if any(h.lower() in low for h in ALLOWED_TWITTER): w += 0.20
        else: w -= 0.25
    if any(wd in (title or "").lower() for wd in LOW_CRED_TERMS): w -= 0.30
    return max(0.2, w)

def title_tilt(title):
    if not title: return 0.0
    t = title.lower()
    pos = any(k in t for k in GOOD_TERMS)
    neg = any(k in t for k in BAD_TERMS)
    if pos and not neg: return 0.25
    if neg and not pos: return -0.25
    return 0.0

def is_promotional(title):
    if not title: return False
    return PROMO_RE.search(title) is not None

def zscore(x, mu, sd):
    if sd is None or sd < RISK_EPS: return 0.0
    return (x-mu)/sd

def realized_vol_z(prev_cl):
    if len(prev_cl) < 6: return 0.0, 0.0
    rets = []
    for i in range(1,len(prev_cl)):
        a,b = prev_cl[i-1], prev_cl[i]
        if a and b:
            rets.append((b-a)/max(RISK_EPS,a))
    if not rets: return 0.0, 0.0
    rv = (sum(r*r for r in rets)/len(rets))**0.5
    # z vs trailing short mean of abs returns
    win = min(6, len(rets))
    base = sum(abs(x) for x in rets[-win:])/win
    sd = pstdev([abs(x) for x in rets[-win:]]) if win>1 else 0.0
    return rv, zscore(rv, base, sd if sd>0 else None)

# =========================
# Experts
# =========================
def expert_momentum_shock(lp, c1, c2, c3, prev_vol_avg):
    imp = (c1["close"] - lp)/lp
    body, up, lo, rng1 = candle_shape(c1)
    gap = (c1["open"] - lp)/lp
    ft2 = (c2["close"]-c1["close"])/max(RISK_EPS, c1["close"]) if c2 else 0.0
    ft3 = (c3["close"]-c1["close"])/max(RISK_EPS, c1["close"]) if c3 else 0.0
    v1 = c1["volume"] or 0.0
    v2 = (c2["volume"] if c2 else 0.0) or 0.0
    v3 = (c3["volume"] if c3 else 0.0) or 0.0
    spike = ((v1+v2+v3)/3)/max(RISK_EPS, prev_vol_avg or 1.0)

    score = 0.0
    if abs(imp) >= IMPULSE_STRONG: score += 1.05
    elif abs(imp) >= IMPULSE_MED: score += 0.55
    if body >= 0.58: score += 0.9
    if spike >= VOLUME_SPIKE_HOT: score += 0.75
    if ft2*imp > 0: score += 0.5
    if ft3*imp > 0: score += 0.4
    if abs(gap) <= 0.012: score += 0.2
    if abs(gap) > 0.02: score -= 0.4
    if (c1["high"]-c1["low"])/lp > 0.022: score -= 0.3

    direction = "LONG" if imp > 0 else "SHORT"
    return direction, max(0.0, score)

def expert_exhaustion_fade(lp, c1, c2, c3):
    imp = (c1["close"] - lp)/lp
    body, up, lo, rng1 = candle_shape(c1)
    gap = (c1["open"] - lp)/lp
    ft2 = (c2["close"]-c1["close"])/max(RISK_EPS, c1["close"]) if c2 else 0.0
    ft3 = (c3["close"]-c1["close"])/max(RISK_EPS, c1["close"]) if c3 else 0.0

    score = 0.0
    if abs(gap) > CIRCUIT_MEGA_GAP: score += 1.2
    if (c1["high"]-c1["low"])/lp > 0.02: score += 0.8
    if imp > 0 and (c1["high"]-max(c1["open"],c1["close"]))/(max(RISK_EPS, c1["high"]-c1["low"])) >= WICK_TRAP: score += 0.8
    if imp < 0 and (min(c1["open"],c1["close"])-c1["low"])/(max(RISK_EPS, c1["high"]-c1["low"])) >= WICK_TRAP: score += 0.8
    if ft2*imp < 0: score += 0.7
    if ft3*imp < 0: score += 0.5
    if body < 0.28 and abs(imp) >= IMPULSE_STRONG: score += 0.4

    direction = "SHORT" if imp > 0 else "LONG"
    return direction, max(0.0, score)

def expert_breakout(prev_hi, prev_lo, c1, c2):
    cl1 = c1["close"]; o1=c1["open"]; h1=c1["high"]; l1=c1["low"]
    break_up = h1 > prev_hi * (1.0 + 0.0008)
    break_dn = l1 < prev_lo * (1.0 - 0.0008)
    hold_up = c2 and (c2["close"] > max(prev_hi, cl1*(1.0 - FOLLOW_THRU_OK)))
    hold_dn = c2 and (c2["close"] < min(prev_lo, cl1*(1.0 + FOLLOW_THRU_OK)))
    strong_up = c2 and (c2["close"] - cl1)/cl1 > FOLLOW_THRU_STRONG
    strong_dn = c2 and (cl1 - c2["close"])/cl1 > FOLLOW_THRU_STRONG

    if break_up and (hold_up or strong_up):
        return "LONG", 0.95
    if break_dn and (hold_dn or strong_dn):
        return "SHORT", 0.95
    if break_up and c2 and c2["close"] < prev_hi:
        return "SHORT", 0.85
    if break_dn and c2 and c2["close"] > prev_lo:
        return "LONG", 0.85
    return None, 0.0

def expert_bollinger(prev_cl, c1, c2):
    if len(prev_cl) < 6: return (None, 0.0)
    mu = mean(prev_cl)
    sd = pstdev(prev_cl) if len(prev_cl) > 1 else 0.0
    z1 = zscore(c1["close"], mu, sd)
    rev = c2 and ((c2["close"] - c1["close"])/max(RISK_EPS, c1["close"]))
    if abs(z1) >= BOLL_Z_FADE and (rev is None or abs(rev) < 0.001):
        return ("SHORT" if z1 > 0 else "LONG"), 0.85
    if abs(z1) >= BOLL_Z_TREND and rev and abs(rev) >= 0.001:
        return ("LONG" if z1 > 0 else "SHORT"), 0.6
    return (None, 0.0)

def expert_microtrend(prev_cl, lp, c1):
    if len(prev_cl) < 5: return (None, 0.0)
    e5 = ema(prev_cl, 5); e12 = ema(prev_cl, 12) if len(prev_cl)>=12 else ema(prev_cl, max(3,len(prev_cl)-1))
    slope = (e5 - e12)/max(RISK_EPS, e12 or lp)
    if abs(slope) < WEAK_TREND_SLOPE: return (None, 0.0)
    imp = (c1["close"] - lp)/lp
    if abs(slope) >= STRONG_TREND_SLOPE and slope*imp > 0:
        return ("LONG" if slope > 0 else "SHORT"), 0.7
    if slope*imp > 0:
        return ("LONG" if slope > 0 else "SHORT"), 0.55
    else:
        return ("SHORT" if slope > 0 else "LONG"), 0.45

def expert_wick_imbalance(c1):
    body, up, lo, rng = candle_shape(c1)
    if rng <= 0: return (None, 0.0)
    # very long upper wick after up move → short; long lower wick after down move → long
    if up - lo >= WICK_IMBAL and body >= 0.25:
        # direction of wick dominance tends to mean revert over 30m
        if c1["close"] >= c1["open"]:
            return "SHORT", 0.55
        else:
            return "LONG", 0.55
    return (None, 0.0)

def expert_gap_magnet(lp, c1, c2):
    gap = (c1["open"] - lp)/lp
    if abs(gap) < GAP_MAGNET:
        return (None, 0.0)
    # if c1 closes toward filling the gap, continuation to close gap often persists in next 30m
    # measure whether c1 close moved back toward lp
    moved_to_fill = ( (gap > 0 and c1["close"] < c1["open"]) or (gap < 0 and c1["close"] > c1["open"]) )
    if moved_to_fill:
        # continuation toward lp
        return ("SHORT" if gap > 0 else "LONG"), 0.6
    # if c2 also opposes the gap direction → stronger magnet
    if c2:
        c2_dir = 1 if c2["close"] > c2["open"] else -1 if c2["close"] < c2["open"] else 0
        if (gap > 0 and c2_dir < 0) or (gap < 0 and c2_dir > 0):
            return ("SHORT" if gap > 0 else "LONG"), 0.7
    return (None, 0.0)

def expert_microstructure_follow(c1, c2, c3):
    if not c2: return (None, 0.0)
    dir1 = 1 if c1["close"] > c1["open"] else -1 if c1["close"] < c1["open"] else 0
    dir2 = 1 if c2["close"] > c2["open"] else -1 if c2["close"] < c2["open"] else 0
    ft = (c2["close"] - c1["close"])/max(RISK_EPS, c1["close"])
    conf = 0.0
    if dir1 == dir2 and abs(ft) >= FOLLOW_THRU_OK:
        conf += 0.5
        if abs(ft) >= FOLLOW_THRU_STRONG: conf += 0.3
        if c3:
            dir3 = 1 if c3["close"] > c3["open"] else -1 if c3["close"] < c3["open"] else 0
            if dir3 == dir2: conf += 0.2
        return ("LONG" if dir2 > 0 else "SHORT"), conf
    # 2 cancels 1 by >60% of body → fade
    body1 = abs(c1["close"] - c1["open"])
    body2 = abs(c2["close"] - c2["open"])
    if body2 >= 0.6 * body1 and dir1 * dir2 < 0:
        return ("SHORT" if dir1 > 0 else "LONG"), 0.6
    return (None, 0.0)

# =========================
# Consensus + Regime
# =========================
def vote_and_decide(item):
    if item.get("id") in BLOCKLIST_IDS: return None

    title = item.get("title") or ""
    source = item.get("source") or ""
    lp = last_prev_close(item)
    if not lp or lp <= 0: return None

    obs = clean_obs(item)
    if not obs: return None

    # outlier guard #1: insane candle range
    if (obs[0]["high"] - obs[0]["low"]) / lp > MAX_RANGE_GLITCH_PCT:
        return None

    c1 = obs[0]; c2 = obs[1] if len(obs)>1 else None; c3 = obs[2] if len(obs)>2 else None
    prev_hi, prev_lo = prev_hilo(item, k=RET_LOOKBACK)
    prev_cl = prev_closes(item, k=RET_LOOKBACK)
    pv_avg = avg_prev_volume(item, k=3)

    # outlier guard #2: absurd wick/body
    body, up, lo, rng = candle_shape(c1)
    if body == 0 and rng > 0 and max(up,lo)/max(body,1e-9) > MAX_WICK_ANOMALY:
        return None

    # realized vol regime
    rv, rv_z = realized_vol_z(prev_cl)
    trend = slope_sign(prev_cl)
    strong_trend = abs(trend) >= STRONG_TREND_SLOPE
    weak_trend = abs(trend) >= WEAK_TREND_SLOPE
    high_vol = rv_z >= HIGH_VOL_Z

    # experts
    experts = []

    d,s = expert_momentum_shock(lp, c1, c2, c3, pv_avg); experts.append(("mom", d, s))
    d,s = expert_exhaustion_fade(lp, c1, c2, c3);       experts.append(("fade", d, s))
    if prev_hi and prev_lo:
        d,s = expert_breakout(prev_hi, prev_lo, c1, c2);   experts.append(("brk", d, s))
    d,s = expert_bollinger(prev_cl, c1, c2);             experts.append(("boll", d, s))
    d,s = expert_microtrend(prev_cl, lp, c1);            experts.append(("micro", d, s))
    d,s = expert_wick_imbalance(c1);                     experts.append(("wick", d, s))
    d,s = expert_gap_magnet(lp, c1, c2);                 experts.append(("gapmag", d, s))
    d,s = expert_microstructure_follow(c1, c2, c3);      experts.append(("microft", d, s))

    # aggregate raw votes
    long_score = 0.0; short_score = 0.0; long_cnt = 0; short_cnt = 0; max_single = 0.0
    for tag, d, s in experts:
        if not d or s <= 0: continue
        w = s
        # slightly boost high-signal experts
        if tag in ("mom","brk","microft"): w *= 1.08
        if tag in ("fade","boll","wick"):  w *= 1.03
        if d == "LONG":
            long_score += w; long_cnt += 1
            max_single = max(max_single, w)
        else:
            short_score += w; short_cnt += 1
            max_single = max(max_single, w)

    # Regime reweighting
    # - high vol favors fade unless strong trend confirmed
    # - strong trend favors trend-follow
    if high_vol and not strong_trend:
        short_score *= 1.06  # fade bias: if c1 up, short more likely; if down, long more likely (handled by experts)
        long_score  *= 1.06
        # small nudge toward opposite of c1 direction by adding tie-bias later
    if strong_trend:
        if trend > 0: long_score *= 1.08
        else:         short_score *= 1.08
    elif weak_trend:
        if trend > 0: long_score *= 1.03
        else:         short_score *= 1.03

    # credibility & title polarity
    w_src = source_weight(title, source)
    tilt = title_tilt(title)
    promo_pen = 0.85 if is_promotional(title) else 1.0
    long_score *= (w_src * (1.0 + max(0.0, tilt)) * promo_pen)
    short_score *= (w_src * (1.0 + max(0.0, -tilt)) * promo_pen)

    # mega-gap override: reversion bias
    gap = (c1["open"] - lp)/lp
    if abs(gap) > CIRCUIT_MEGA_GAP:
        if gap > 0: short_score *= 1.15
        else:       long_score  *= 1.15

    # consensus requirement (unless ultra-strong single expert)
    aligned = max(long_cnt, short_cnt)
    # If tie-ish, add mean-revert tie-break in high_vol
    if abs(long_score - short_score) < 0.14:
        imp = (c1["close"] - lp)/lp
        if high_vol:
            if imp > 0: short_score += 0.12
            else:       long_score  += 0.12
        else:
            # small-gap trending environment: nudge with microtrend
            if trend > 0: long_score += 0.06
            elif trend < 0: short_score += 0.06

    decision = "LONG" if long_score >= short_score else "SHORT"

    # gate: require consensus or ultra-strong
    strong_ok = (max_single >= ULTRA_STRONG) or (aligned >= CONSENSUS_MIN)

    impulse = abs((c1["close"] - lp)/lp)
    v1 = c1["volume"] or 0.0
    v2 = (c2["volume"] if c2 else 0.0) or 0.0
    v3 = (c3["volume"] if c3 else 0.0) or 0.0
    spike = ((v1+v2+v3)/3)/max(RISK_EPS, pv_avg or 1.0)
    micro_abs = abs(trend)

    quality = (impulse >= PRIMARY_QUALITY_IMPULSE and spike >= PRIMARY_QUALITY_SPIKE and strong_ok)

    return {
        "id": item.get("id"),
        "decision": decision,
        "conf": float(max(0.0, abs(long_score - short_score))),
        "impulse": float(impulse),
        "spike": float(spike),
        "micro": float(micro_abs),
        "time": item.get("time", 0),
        "promo": bool(is_promotional(title)),
        "quality": bool(quality)
    }

def dedupe_keep_first(items):
    seen = set(); out = []
    for it in items:
        iid = it.get("id")
        if iid is None or iid in seen: continue
        seen.add(iid); out.append(it)
    return out

def choose_exactly_50(raw_items):
    scored = []
    for it in raw_items:
        s = vote_and_decide(it)
        if s and s["id"] is not None:
            scored.append(s)

    scored = dedupe_keep_first(scored)

    # Stage A: high-quality, non-promo
    primary = [x for x in scored if x["quality"] and not x["promo"]]
    primary.sort(key=lambda x: (x["conf"], x["impulse"], x["spike"], x["micro"], x["time"], x["id"]), reverse=True)
    picks = primary[:50]

    # Stage B: quality including promo
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        rest = [x for x in scored if x["quality"] and x["id"] not in chosen]
        rest.sort(key=lambda x: (x["conf"], x["impulse"], x["spike"], x["micro"], x["time"], x["id"]), reverse=True)
        picks.extend(rest[:need])

    # Stage C: decent signals (drop quality flag)
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        rest = [x for x in scored if x["id"] not in chosen]
        rest.sort(key=lambda x: (x["conf"], x["impulse"], x["spike"], x["micro"], x["time"], x["id"]), reverse=True)
        picks.extend(rest[:need])

    # Stage D: safety net — impulse fade
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        safes = []
        for it in raw_items:
            iid = it.get("id")
            if iid in chosen or iid in BLOCKLIST_IDS: continue
            lp = last_prev_close(it); obs = clean_obs(it)
            if not lp or not obs: continue
            c1 = obs[0]
            imp = (c1["close"] - lp)/lp
            safes.append({
                "id": iid,
                "decision": "SHORT" if imp > 0 else "LONG",
                "conf": 0.05 + min(0.3, abs(imp)*60),
                "impulse": abs(imp),
                "spike": 0.0,
                "micro": 0.0,
                "time": it.get("time", 0)
            })
        safes.sort(key=lambda x:(x["conf"],x["impulse"],x["time"],x["id"]), reverse=True)
        picks.extend(safes[:need])

    picks = picks[:50]
    return [{"id": p["id"], "decision": p["decision"]} for p in picks]

# =========================
# API
# =========================
@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error":"Input must be a JSON array"}), 400
        if not data:
            return jsonify([]), 200

        result = choose_exactly_50(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # pip install flask
    # python app.py
    app.run(host="0.0.0.0", port=8000)
