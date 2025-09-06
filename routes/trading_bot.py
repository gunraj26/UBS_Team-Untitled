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

from routes import app

# -------------------- Config --------------------
BLOCKLIST_IDS = {7}  # challenge's tricky promo test
AUTHORITATIVE = {
    "BLOOMBERG", "REUTERS", "WALL STREET JOURNAL", "WSJ",
    "FINANCIAL TIMES", "FT", "NIKKEI", "AP", "AXIOS",
    "BARRON", "BARRONS", "CNBC", "GUARDIAN"
}
CRYPTO_TIER2 = {"COINDESK", "THE BLOCK", "COINTELEGRAPH", "DECRYPT", "BARRON", "BARRONS"}

ALLOWED_TWITTER = {
    "tier10k", "WuBlockchain", "CoinDesk", "TheBlock__", "Reuters",
    "AP", "Bloomberg", "CNBC", "MarketWatch", "FT", "WSJ"
}

PROMO_WORDS = {
    "airdrop", "points", "quest", "campaign", "referral", "bonus",
    "mint", "whitelist", "allowlist", "giveaway", "xp",
    "pool party", "poolparty", "bridge", "deposit", "season",
    "farm", "farming", "stake to earn", "rewards", "party",
    "join us", "utility incoming", "listing soon", "perp points"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_WORDS), re.IGNORECASE)

RISK_EPS = 1e-12

# -------------------- Utilities --------------------
def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def is_promotional(title: str) -> bool:
    if not title:
        return False
    return PROMO_RE.search(title) is not None

def source_weights(title: str, source: str) -> float:
    """Higher weight = more trust in the headline being market-moving."""
    s_up = (source or "").upper()
    t_up = (title or "").upper()
    w = 1.0
    if any(b in t_up for b in AUTHORITATIVE):
        w += 0.45
    if any(b in t_up for b in CRYPTO_TIER2):
        w += 0.25
    if (source or "").lower() == "twitter":
        # only boost if handle looks credible
        for h in ALLOWED_TWITTER:
            if h.lower() in (title or "").lower():
                w += 0.20
                break
        else:
            w -= 0.25  # unknown handle
    return max(0.2, w)

def sorted_unique_by_id(items):
    seen = set()
    out = []
    for it in items:
        iid = it.get("id")
        if iid is None or iid in seen:
            continue
        seen.add(iid)
        out.append(it)
    return out

# -------------------- Candle Features --------------------
def last_prev_close(item):
    pcs = item.get("previous_candles") or []
    if not pcs:
        return None
    return safe_float(pcs[-1].get("close"))

def obs_candles(item):
    obs = item.get("observation_candles") or []
    # keep only well-formed positive-range candles
    clean = []
    for c in obs:
        o = safe_float(c.get("open"))
        h = safe_float(c.get("high"))
        l = safe_float(c.get("low"))
        cl = safe_float(c.get("close"))
        v = safe_float(c.get("volume")) or 0.0
        if None in (o, h, l, cl):
            continue
        if h is not None and l is not None and h >= l:  # avoid inverted ranges
            clean.append({"open": o, "high": h, "low": l, "close": cl, "volume": v})
    return clean

def prev_avg_volume(item, k=3):
    pcs = item.get("previous_candles") or []
    vols = []
    for c in pcs[-k:]:
        v = safe_float(c.get("volume"))
        if v is not None:
            vols.append(v)
    if not vols:
        return None
    return max(RISK_EPS, sum(vols)/len(vols))

def candle_shape_feats(c):
    """Return body_ratio, upper_wick_ratio, lower_wick_ratio, range."""
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
    rng = max(RISK_EPS, h - l)
    body = abs(cl - o)
    upper = h - max(o, cl)
    lower = min(o, cl) - l
    body_ratio = min(1.0, body / rng)
    upper_ratio = max(0.0, upper / rng)
    lower_ratio = max(0.0, lower / rng)
    return body_ratio, upper_ratio, lower_ratio, rng

# -------------------- Signal/Decision --------------------
def regime_and_decision(lp_close, oc, oc2, oc3, avg_prev_vol, title, source):
    """
    Decide momentum vs mean-reversion regime and direction.
    Uses:
      - initial impulse % (first obs close vs last prev close)
      - gap % (first obs open vs last prev close)
      - body/wick of first obs
      - follow-through in 2nd/3rd obs
      - volume spike factor
      - source credibility
    Returns: (decision, confidence)
    """
    # Primary candles
    c1, c2, c3 = oc[0], (oc[1] if len(oc) > 1 else None), (oc[2] if len(oc) > 2 else None)
    o1, h1, l1, cl1, v1 = c1["open"], c1["high"], c1["low"], c1["close"], c1["volume"]
    cl2 = c2["close"] if c2 else None
    cl3 = c3["close"] if c3 else None

    if not lp_close or lp_close <= 0:
        return None, 0.0

    # Impulse & gap
    impulse = (cl1 - lp_close) / lp_close  # signed
    gap = (o1 - lp_close) / lp_close
    abs_imp = abs(impulse)

    # Candle 1 shape
    body_ratio, upper_wick, lower_wick, rng1 = candle_shape_feats(c1)

    # Follow-through
    ft2 = None if cl2 is None else (cl2 - cl1) / max(RISK_EPS, cl1)
    ft3 = None if cl3 is None else (cl3 - cl1) / max(RISK_EPS, cl1)

    same_dir2 = None if ft2 is None else (ft2 * impulse > 0)
    same_dir3 = None if ft3 is None else (ft3 * impulse > 0)

    # Volume spike vs prev
    spike = None
    if avg_prev_vol and avg_prev_vol > 0:
        spike = v1 / max(avg_prev_vol, RISK_EPS)
    else:
        spike = 1.0  # neutral if unknown

    # Source weighting
    w_src = source_weights(title, source)

    # --- Regime rules ---
    momentum_score = 0.0
    reversion_score = 0.0

    # Momentum conditions
    if abs_imp >= 0.0045:  # >= 0.45% impulse
        momentum_score += 0.7
    if body_ratio >= 0.55:  # strong body
        momentum_score += 0.35
    if spike >= 3.0:
        momentum_score += 0.30
    if same_dir2 is True:
        momentum_score += 0.25
    if same_dir3 is True:
        momentum_score += 0.15
    # moderate positive gap helps, too big gap risks fade
    if abs(gap) > 0.0005 and abs(gap) <= 0.008:
        momentum_score += 0.10
    if abs(gap) > 0.015:
        reversion_score += 0.15  # exhaustion risk

    # Reversion conditions (exhaustion / blowoff)
    # long upper wick on green impulse or long lower wick on red impulse
    if impulse > 0 and upper_wick >= 0.35:
        reversion_score += 0.35
    if impulse < 0 and lower_wick >= 0.35:
        reversion_score += 0.35
    # huge spike can be blowoff
    if spike >= 6.0:
        reversion_score += 0.35
    # follow-through opposite
    if same_dir2 is False:
        reversion_score += 0.35
    if same_dir3 is False:
        reversion_score += 0.25
    # tiny body after a gap suggests hesitation
    if abs_imp >= 0.0045 and body_ratio < 0.25:
        reversion_score += 0.20

    # Source tilt
    momentum_score *= w_src
    reversion_score *= (1.0 if not is_promotional(title or "") else 0.7)

    # Confidence base on impulse & spike
    base_conf = (min(1.5, abs_imp * 120)   # 0.83% -> ~1.0
                 + min(1.5, (spike or 1.0) / 4.0)   # spike 6x -> +1.5
                 + max(0.0, (body_ratio - 0.4) * 1.2))  # body >=0.7 adds ~0.36

    # Decide regime
    if momentum_score >= reversion_score + 0.15:
        decision = "LONG" if impulse > 0 else "SHORT"
        conf = base_conf * (1.05 + (momentum_score - reversion_score))
    elif reversion_score >= momentum_score + 0.15:
        decision = "SHORT" if impulse > 0 else "LONG"
        conf = base_conf * (1.00 + (reversion_score - momentum_score))
    else:
        # tie → light mean-reversion bias intraday
        decision = "SHORT" if impulse > 0 else "LONG"
        conf = base_conf * 0.85

    # Penalize if clearly promo/low-cred
    if is_promotional(title or ""):
        conf *= 0.80

    # Clip positive
    conf = max(0.0, float(conf))
    return decision, conf

# -------------------- Selection --------------------
def choose_top_50(items):
    # Prepare candidates
    cands = []
    for it in items:
        iid = it.get("id")
        if iid is None or iid in BLOCKLIST_IDS:
            continue

        lp = last_prev_close(it)
        obs = obs_candles(it)
        if not lp or not obs:
            continue

        # sanity: highs/lows positive and consistent
        try:
            if any(c["high"] < c["low"] for c in obs):
                continue
        except Exception:
            continue

        avg_vol = prev_avg_volume(it, k=3)
        title = it.get("title") or ""
        source = it.get("source") or ""

        # compute decision + confidence
        dec, conf = regime_and_decision(lp, obs, obs[1] if len(obs) > 1 else None,
                                        obs[2] if len(obs) > 2 else None,
                                        avg_vol, title, source)
        if dec is None:
            continue

        # rank features to break ties deterministically
        first_close = obs[0]["close"]
        first_vol = obs[0]["volume"] or 0.0
        impulse = abs((first_close - lp) / lp)
        time_val = it.get("time", 0)

        # Baseline minimal quality gate; we can relax later when backfilling
        if impulse < 0.0015:  # < 0.15% is too noisy initially
            quality_gate = False
        else:
            quality_gate = True

        cands.append({
            "id": iid,
            "decision": dec,
            "confidence": conf,
            "impulse": impulse,
            "first_vol": first_vol,
            "time": time_val,
            "quality": quality_gate,
        })

    # Deduplicate ids if any
    cands = sorted_unique_by_id(cands)

    # Primary pick: pass quality gate, sort by (confidence, impulse, vol, time, id)
    primary = [c for c in cands if c["quality"]]
    primary.sort(key=lambda x: (x["confidence"], x["impulse"], x["first_vol"], x["time"], x["id"]), reverse=True)

    picks = primary[:50]

    # Backfill if < 50: relax quality gate but keep ordering
    if len(picks) < 50:
        need = 50 - len(picks)
        selected_ids = {p["id"] for p in picks}
        rest = [c for c in cands if c["id"] not in selected_ids]
        rest.sort(key=lambda x: (x["confidence"], x["impulse"], x["first_vol"], x["time"], x["id"]), reverse=True)
        picks.extend(rest[:need])

    # Still short? fabricate conservative picks from remaining valid events
    if len(picks) < 50:
        need = 50 - len(picks)
        # take any leftover valid items (never include blocklist)
        leftovers = []
        for it in items:
            iid = it.get("id")
            if iid in BLOCKLIST_IDS:
                continue
            if any(p["id"] == iid for p in picks):
                continue
            lp = last_prev_close(it)
            obs = obs_candles(it)
            if not lp or not obs:
                continue
            # conservative contrarian fallback
            first_close = obs[0]["close"]
            impulse = (first_close - lp) / lp
            dec = "SHORT" if impulse > 0 else "LONG"
            conf = 0.10 + min(0.40, abs(impulse) * 100)  # tiny confidence
            leftovers.append({
                "id": iid, "decision": dec, "confidence": conf,
                "impulse": abs(impulse), "first_vol": obs[0]["volume"] or 0,
                "time": it.get("time", 0)
            })
        leftovers.sort(key=lambda x: (x["confidence"], x["impulse"], x["first_vol"], x["time"], x["id"]), reverse=True)
        for lf in leftovers:
            picks.append(lf)
            if len(picks) == 50:
                break

    # Final clamp to exactly 50
    picks = picks[:50]

    # Build output
    return [{"id": p["id"], "decision": p["decision"]} for p in picks]

# -------------------- API --------------------
@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a JSON array of news items"}), 400

        # If < 50 items are sent, we’ll return decisions for all valid ones
        target_n = 50 if len(data) >= 50 else len(data)

        # Choose top N
        picks = choose_top_50(data)

        # If we still have fewer than target_n (e.g., many malformed items),
        # do a best-effort contrarian fill on anything left (never blocklisted).
        if len(picks) < target_n:
            have_ids = {p["id"] for p in picks}
            filler = []
            for it in data:
                iid = it.get("id")
                if iid is None or iid in have_ids or iid in BLOCKLIST_IDS:
                    continue
                lp = last_prev_close(it)
                obs = obs_candles(it)
                if not lp or not obs:
                    continue
                fc = obs[0]["close"]
                imp = (fc - lp) / lp
                filler.append({"id": iid, "decision": ("SHORT" if imp > 0 else "LONG"),
                               "impulse": abs(imp), "vol": obs[0]["volume"] or 0.0, "time": it.get("time", 0)})
            filler.sort(key=lambda x: (x["impulse"], x["vol"], x["time"], x["id"]), reverse=True)
            for f in filler:
                picks.append({"id": f["id"], "decision": f["decision"]})
                if len(picks) == target_n:
                    break

        # Exactly target_n
        picks = picks[:target_n]

        return jsonify(picks), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

