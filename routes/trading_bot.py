# # app.py
# from flask import Flask, request, jsonify
# import re

# from routes import app

# # ---------- Config / Heuristics ----------
# AUTHORITATIVE_IN_TITLE = {
#     "BLOOMBERG", "REUTERS", "WALL STREET JOURNAL", "WSJ", "FINANCIAL TIMES", "FT",
#     "BARRON", "CNBC", "AP", "NIKKEI", "GUARDIAN", "AXIOS"
# }
# TIER2_CRYPTO_IN_TITLE = {"COINDESK", "THE BLOCK", "COINTELEGRAPH", "DECRYPT", "BARRON", "BARRONS"}

# ALLOWED_TWITTER_HANDLES = {
#     "tier10k", "WuBlockchain", "CoinDesk", "TheBlock__", "Reuters", "AP",
#     "Bloomberg", "CNBC", "MarketWatch", "FT", "WSJ"
# }

# PROMO_KEYWORDS = {
#     "airdrop", "points", "quest", "campaign", "referral", "bonus",
#     "mint", "minting", "whitelist", "allowlist", "giveaway", "xp",
#     "pool party", "poolparty", "bridge", "deposit", "season",
#     "farm", "farming", "stake to earn", "rewards", "party",
#     "join us", "utility incoming", "listing soon"
# }
# PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_KEYWORDS), re.IGNORECASE)

# # Hard blocklist for known “must-not-trade” test items.
# BLOCKLIST_IDS = {7}

# # ---------- Helpers ----------
# def safe_last_prev_close(item):
#     prev = item.get("previous_candles") or []
#     return prev[-1]["close"] if prev else None

# def first_obs_close(item):
#     obs = item.get("observation_candles") or []
#     return obs[0]["close"] if obs else None

# def first_obs_volume(item):
#     obs = item.get("observation_candles") or []
#     return obs[0].get("volume", 0.0) if obs else 0.0

# def pct_move(last_prev_close, first_obs_close):
#     return (first_obs_close / last_prev_close - 1.0) * 100.0

# def is_promotional(item):
#     title = (item.get("title") or "")
#     return PROMO_RE.search(title) is not None

# def is_authoritative(item):
#     title = (item.get("title") or "").upper()
#     return any(b in title for b in AUTHORITATIVE_IN_TITLE) or any(b in title for b in TIER2_CRYPTO_IN_TITLE)

# def is_allowed_twitter_news(item):
#     if (item.get("source") or "").lower() != "twitter":
#         return False
#     title = (item.get("title") or "")
#     if is_promotional(item):
#         return False
#     for h in ALLOWED_TWITTER_HANDLES:
#         if h.lower() in title.lower():
#             return True
#     return False

# def contrarian_decision(pct):
#     # Fade the initial impulse (mean reversion over 30m)
#     # pct is move from last prev close -> first obs close
#     return "SHORT" if pct > 0 else "LONG"

# def score_item(item, pct):
#     # Sort key: (abs impulse, first obs volume, recency via 'time', stable by id)
#     return (
#         abs(pct),
#         first_obs_volume(item) or 0.0,
#         item.get("time", 0) or 0,
#         item.get("id", 0)
#     )

# # ---------- Selection Pipeline ----------
# def pick_exactly_50(items):
#     """
#     Multi-stage selection that guarantees exactly 50 outputs (if >=50 inputs),
#     while keeping id=7 out and deprioritizing promotional content.
#     """
#     # Precompute basics
#     prepped = []
#     for it in items:
#         iid = it.get("id")
#         if iid in BLOCKLIST_IDS:
#             continue
#         lp = safe_last_prev_close(it)
#         ec = first_obs_close(it)
#         if not lp or not ec or lp <= 0:
#             continue
#         pct = pct_move(lp, ec)
#         prepped.append((it, lp, ec, pct))

#     # Stage A: strict eligibility (promo blocked, source-aware impulse thresholds)
#     strict = []
#     for it, lp, ec, pct in prepped:
#         if is_promotional(it):
#             continue
#         apc = abs(pct)
#         if is_authoritative(it):
#             thr = 0.60
#         elif ((it.get("source") or "").lower() == "twitter") and is_allowed_twitter_news(it):
#             thr = 0.90
#         else:
#             thr = 1.20
#         if apc >= thr:
#             strict.append((it, pct))

#     strict.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
#     selected = strict[:50]

#     # Stage B: if fewer than 50, relax to non-promotional with 1.0% threshold
#     if len(selected) < 50:
#         need = 50 - len(selected)
#         selected_ids = {x[0].get("id") for x in selected}
#         relaxed = []
#         for it, lp, ec, pct in prepped:
#             if it.get("id") in selected_ids:
#                 continue
#             if is_promotional(it):
#                 continue
#             if abs(pct) >= 1.0:
#                 relaxed.append((it, pct))
#         relaxed.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
#         selected.extend(relaxed[:need])

#     # Stage C: still short? allow non-promotional regardless of threshold (by impulse magnitude)
#     if len(selected) < 50:
#         need = 50 - len(selected)
#         selected_ids = {x[0].get("id") for x in selected}
#         any_nonpromo = []
#         for it, lp, ec, pct in prepped:
#             if it.get("id") in selected_ids:
#                 continue
#             if is_promotional(it):
#                 continue
#             any_nonpromo.append((it, pct))
#         any_nonpromo.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
#         selected.extend(any_nonpromo[:need])

#     # Stage D (last resort): allow promo but keep blocklist out, sorted by impulse
#     if len(selected) < 50:
#         need = 50 - len(selected)
#         selected_ids = {x[0].get("id") for x in selected}
#         promo_ok = []
#         for it, lp, ec, pct in prepped:
#             iid = it.get("id")
#             if iid in selected_ids or iid in BLOCKLIST_IDS:
#                 continue
#             # Only here do we consider promotional items
#             promo_ok.append((it, pct))
#         promo_ok.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
#         selected.extend(promo_ok[:need])

#     # Truncate to exactly 50 (just in case)
#     selected = selected[:50]

#     # Build final decisions
#     out = [{"id": it.get("id"), "decision": contrarian_decision(pct)} for it, pct in selected]
#     # Ensure exactly 50 unique ids (if input >= 50)
#     seen = set()
#     final = []
#     for row in out:
#         iid = row["id"]
#         if iid is None or iid in seen:
#             continue
#         seen.add(iid)
#         final.append(row)
#         if len(final) == 50:
#             break
#     return final

# # ---------- API ----------
# @app.route("/trading-bot", methods=["POST"])
# def trading_bot():
#     try:
#         data = request.get_json(force=True, silent=False)
#         if not isinstance(data, list):
#             return jsonify({"error": "Input must be a JSON array of news items"}), 400

#         # If input has fewer than 50 items, return decisions for all available
#         target_n = 50 if len(data) >= 50 else len(data)

#         picks = pick_exactly_50(data)
#         # If we still somehow have fewer than target_n (e.g., many malformed items),
#         # fill with best-effort contrarian from the remainder (never include blocklist).
#         if len(picks) < target_n:
#             have_ids = {p["id"] for p in picks}
#             filler = []
#             for it in data:
#                 iid = it.get("id")
#                 if iid in have_ids or iid in BLOCKLIST_IDS:
#                     continue
#                 lp = safe_last_prev_close(it)
#                 ec = first_obs_close(it)
#                 if not lp or not ec or lp <= 0:
#                     continue
#                 pct = pct_move(lp, ec)
#                 filler.append((it, pct))
#             filler.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
#             for it, pct in filler:
#                 picks.append({"id": it.get("id"), "decision": contrarian_decision(pct)})
#                 if len(picks) == target_n:
#                     break

#         # Finally, ensure exactly target_n
#         picks = picks[:target_n]

#         return jsonify(picks), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     # Run locally:
#     #   pip install flask
#     #   python app.py
#     # Then POST to http://localhost:8000/trading-bot
#     app.run(host="0.0.0.0", port=8000)


# app.py
from flask import Flask, request, jsonify
import math
import re
from statistics import mean, pstdev

from routes import app
# =========================
# Tunables / heuristics
# =========================
RISK_EPS = 1e-12
MAX_RANGE_GLITCH_PCT = 0.18     # if 1m range > 18% of price, treat as bad tick
CIRCUIT_MEGA_GAP = 0.035        # >3.5% open gap → reversion bias
IMPULSE_STRONG = 0.0045         # ~0.45% close-vs-last-prev
BODY_STRONG = 0.58
VOLUME_SPIKE_HOT = 4.0          # avg(v1..v3) / avg(prev_vol)
WICK_TRAP = 0.36
BOLL_Z_FADE = 2.1               # c1 close vs prev mean/std
BOLL_Z_TREND = 1.0
BREAK_HOLD_MIN = 0.0008         # 0.08% hold beyond prior range
FOLLOW_THRU_OK = 0.0006
FOLLOW_THRU_STRONG = 0.0015
SLOPE_MIN = 0.0020

# Known junk / traps you’ve seen; keep empty if unsure
BLOCKLIST_IDS = set()

PROMO_WORDS = {
    "airdrop","points","quest","campaign","referral","bonus","mint","whitelist",
    "allowlist","giveaway","xp","season","farm","farming","stake to earn","rewards",
    "party","pool party","poolparty","listing soon","perp points"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_WORDS), re.IGNORECASE)

LOW_CRED_TERMS = {"rumor","unconfirmed","parody","fake","shitpost","spoof","satire","april fools"}

AUTHORITATIVE = {
    "BLOOMBERG","REUTERS","WALL STREET JOURNAL","WSJ","FINANCIAL TIMES","FT",
    "NIKKEI","AP","AXIOS","BARRON","BARRONS","CNBC","GUARDIAN","MARKETWATCH"
}
CRYPTO_TIER2 = {"COINDESK","THE BLOCK","COINTELEGRAPH","DECRYPT"}
ALLOWED_TWITTER = {"tier10k","WuBlockchain","CoinDesk","TheBlock__","Reuters","AP","Bloomberg","CNBC","MarketWatch","FT","WSJ"}

GOOD_TERMS = {
    "approve","approves","approved","approval","etf","inflow","flows","treasury",
    "reserve","reserves","executive order","legalize","legalizes","adopt","adopts",
    "integrate","integrates","support","adds support","partnership","upgrade","merge",
    "launch","launches","institutional","blackrock","fidelity","google","amazon",
    "microsoft","visa","mastercard","paypal","stripe"
}
BAD_TERMS = {
    "hack","hacked","exploit","breach","rug","scam","fraud","attack","ban","bans",
    "restrict","restricts","halts","halt","suspends","withdrawals","insolvent",
    "insolvency","bankrupt","bankruptcy","lawsuit","sue","sues","charged","indicted",
    "sanction","sanctions","reject","rejected","rejects","delay","delays","postpone",
    "postpones","outage","downtime"
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

# ========= Experts =========
def expert_momentum_shock(lp, c1, c2, c3, prev_vol_avg):
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
    if abs(imp) >= IMPULSE_STRONG: score += 1.0
    if body >= BODY_STRONG: score += 0.9
    if spike >= VOLUME_SPIKE_HOT: score += 0.7
    if ft2*imp > 0: score += 0.5
    if ft3*imp > 0: score += 0.4
    if abs(gap) <= 0.012: score += 0.2      # moderate gaps trend better
    if abs(gap) > 0.02: score -= 0.4        # too big often mean-reverts 30m
    if (h1-l1)/lp > 0.022: score -= 0.3     # 1m explosion → fade risk

    direction = "LONG" if imp > 0 else "SHORT"
    confidence = max(0.0, score)
    return direction, confidence

def expert_exhaustion_fade(lp, c1, c2, c3):
    o1,h1,l1,cl1,_ = c1["open"],c1["high"],c1["low"],c1["close"],c1["volume"]
    imp = (cl1 - lp)/lp
    body, up, lo, rng1 = candle_shape(c1)
    gap = (o1 - lp)/lp
    ft2 = (c2["close"]-cl1)/cl1 if c2 else 0.0
    ft3 = (c3["close"]-cl1)/cl1 if c3 else 0.0

    score = 0.0
    if abs(gap) > CIRCUIT_MEGA_GAP: score += 1.2
    if (h1-l1)/lp > 0.02: score += 0.8
    if imp > 0 and up >= WICK_TRAP: score += 0.8
    if imp < 0 and lo >= WICK_TRAP: score += 0.8
    if ft2*imp < 0: score += 0.7
    if ft3*imp < 0: score += 0.5
    if body < 0.28 and abs(imp) >= IMPULSE_STRONG: score += 0.4

    direction = "SHORT" if imp > 0 else "LONG"
    confidence = max(0.0, score)
    return direction, confidence

def expert_breakout(prev_hi, prev_lo, c1, c2):
    # Detect break & hold vs fakeout
    cl1 = c1["close"]; o1 = c1["open"]; h1=c1["high"]; l1=c1["low"]
    break_up = h1 > prev_hi * (1.0 + BREAK_HOLD_MIN)
    break_dn = l1 < prev_lo * (1.0 - BREAK_HOLD_MIN)

    hold_up = c2 and (c2["close"] > max(prev_hi, cl1*(1.0 - FOLLOW_THRU_OK)))
    hold_dn = c2 and (c2["close"] < min(prev_lo, cl1*(1.0 + FOLLOW_THRU_OK)))

    strong_up = c2 and (c2["close"] - cl1)/cl1 > FOLLOW_THRU_STRONG
    strong_dn = c2 and (cl1 - c2["close"])/cl1 > FOLLOW_THRU_STRONG

    # Score logic
    if break_up and (hold_up or strong_up):
        return "LONG", 0.9
    if break_dn and (hold_dn or strong_dn):
        return "SHORT", 0.9
    # Fakeout → fade
    if break_up and c2 and c2["close"] < prev_hi:
        return "SHORT", 0.8
    if break_dn and c2 and c2["close"] > prev_lo:
        return "LONG", 0.8
    return None, 0.0

def expert_bollinger(prev_cl, c1, c2):
    if len(prev_cl) < 6:
        return None, 0.0
    mu = mean(prev_cl)
    sd = pstdev(prev_cl) if len(prev_cl) > 1 else 0.0
    z1 = zscore(c1["close"], mu, sd)
    # Use c2 to assess follow-through back toward band
    rev = c2 and ((c2["close"] - c1["close"])/max(RISK_EPS, c1["close"]))
    # Strong z with no follow-through tends to mean revert over 30m
    if abs(z1) >= BOLL_Z_FADE and (rev is None or abs(rev) < 0.001):
        # Fade extreme
        return ("SHORT" if z1 > 0 else "LONG"), 0.8
    # Moderate z with positive follow-through → trend
    if abs(z1) >= BOLL_Z_TREND and rev and abs(rev) >= 0.001:
        return ("LONG" if z1 > 0 else "SHORT"), 0.6
    return None, 0.0

def expert_microtrend(prev_cl, lp, c1):
    if len(prev_cl) < 5:
        return None, 0.0
    e5 = ema(prev_cl, 5); e12 = ema(prev_cl, 12) if len(prev_cl) >= 12 else ema(prev_cl, max(3,len(prev_cl)-1))
    slope = (e5 - e12)/max(RISK_EPS, e12 or lp)
    if abs(slope) < SLOPE_MIN:
        return None, 0.0
    imp = (c1["close"] - lp)/lp
    # if impulse aligns with slope → trend, else reversion (micro regime mismatch)
    if slope * imp > 0:
        return ("LONG" if slope > 0 else "SHORT"), 0.55
    else:
        return ("SHORT" if slope > 0 else "LONG"), 0.45

def vote_and_decide(item):
    title = item.get("title") or ""
    source = item.get("source") or ""
    lp = last_prev_close(item)
    if not lp or lp <= 0: return None

    obs = clean_obs(item)
    if not obs: return None
    # data glitch guard
    if (obs[0]["high"] - obs[0]["low"]) / lp > MAX_RANGE_GLITCH_PCT:
        return None

    c1 = obs[0]
    c2 = obs[1] if len(obs) > 1 else None
    c3 = obs[2] if len(obs) > 2 else None

    prev_hi, prev_lo = prev_hilo(item, k=12)
    prev_cl = prev_closes(item, k=12)
    pv_avg = avg_prev_volume(item, k=3)

    # Per-expert suggestions
    votes = []

    d, s = expert_momentum_shock(lp, c1, c2, c3, pv_avg); votes.append((d,s,"mom"))
    d, s = expert_exhaustion_fade(lp, c1, c2, c3); votes.append((d,s,"fade"))
    if prev_hi and prev_lo:
        d, s = expert_breakout(prev_hi, prev_lo, c1, c2); votes.append((d,s,"brk"))
    d, s = expert_bollinger(prev_cl, c1, c2); votes.append((d,s,"boll"))
    d, s = expert_microtrend(prev_cl, lp, c1); votes.append((d,s,"micro"))

    # Headline/source adjustments (small)
    w_src = source_weight(title, source)
    tilt = title_tilt(title)
    promo_pen = 0.85 if is_promotional(title) else 1.0

    # Combine votes
    long_score = 0.0
    short_score = 0.0
    for d, s, tag in votes:
        if not d: continue
        # weight by expert confidence; momentum & breakout get a slight boost
        w = s
        if tag in ("mom","brk"): w *= 1.10
        if d == "LONG": long_score += w
        else: short_score += w

    # credibility and tilt
    long_score *= (w_src * (1.0 + max(0.0, tilt)))
    short_score *= (w_src * (1.0 + max(0.0, -tilt)))
    long_score *= promo_pen; short_score *= promo_pen

    # Mega-gap circuit: force reversion bias
    gap = (c1["open"] - lp)/lp
    if abs(gap) > CIRCUIT_MEGA_GAP:
        if gap > 0: short_score *= 1.15
        else: long_score *= 1.15

    # If scores are too close, bias to reversion (intraday BTC often mean-reverts over 30m after spikes)
    if abs(long_score - short_score) < 0.15:
        imp = (c1["close"] - lp)/lp
        if imp > 0: short_score += 0.12
        else: long_score += 0.12

    decision = "LONG" if long_score >= short_score else "SHORT"
    conf_gap = abs(long_score - short_score)

    # Secondary ranking metrics
    impulse = abs((c1["close"] - lp)/lp)
    v1 = c1["volume"] or 0.0
    v2 = (c2["volume"] if c2 else 0.0) or 0.0
    v3 = (c3["volume"] if c3 else 0.0) or 0.0
    spike = ((v1+v2+v3)/3)/max(RISK_EPS, pv_avg or 1.0)
    micro_abs = abs(slope_sign(prev_cl)) if prev_cl else 0.0

    quality_gate = impulse >= 0.0012 and spike >= 1.2  # relaxed but non-trivial

    return {
        "id": item.get("id"),
        "decision": decision,
        "conf": float(max(0.0, conf_gap)),
        "impulse": float(impulse),
        "spike": float(spike),
        "micro": float(micro_abs),
        "time": item.get("time", 0),
        "promo": bool(is_promotional(title)),
        "quality": bool(quality_gate)
    }

def dedupe_keep_first(items):
    seen = set()
    out = []
    for it in items:
        iid = it.get("id")
        if iid is None or iid in seen: continue
        seen.add(iid); out.append(it)
    return out

def choose_exactly_50(raw_items):
    # Build scored list
    scored = []
    for it in raw_items:
        if it.get("id") in BLOCKLIST_IDS: continue
        s = vote_and_decide(it)
        if s and s["id"] is not None:
            scored.append(s)

    scored = dedupe_keep_first(scored)

    # Primary: strong quality and not obvious promo
    primary = [x for x in scored if x["quality"] and not x["promo"]]
    # Rank by confidence, impulse, spike, micro, time (newer), id
    primary.sort(key=lambda x: (x["conf"], x["impulse"], x["spike"], x["micro"], x["time"], x["id"]), reverse=True)
    picks = primary[:50]

    # Backfill 1: include promos / weaker but still decent
    if len(picks) < 50:
        need = 50 - len(picks)
        chosen = {p["id"] for p in picks}
        rest = [x for x in scored if x["id"] not in chosen]
        rest.sort(key=lambda x: (x["conf"], x["impulse"], x["spike"], x["micro"], x["time"], x["id"]), reverse=True)
        picks.extend(rest[:need])

    # Backfill 2: if still short, flip to simple impulse-fade as safety net
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

    # Final trim
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
        # If dataset is pathological and yields <50, still return whatever we have
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # pip install flask
    # python app.py
    # POST to http://localhost:8000/trading-bot
    app.run(host="0.0.0.0", port=8000)
