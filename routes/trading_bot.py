# app.py
from flask import Flask, request, jsonify
import math
import re

from routes import app
# -------- Heuristics / Lists --------
AUTHORITATIVE_SOURCES = {
    # News brands you trust for market-moving items
    "BLOOMBERG", "REUTERS", "WSJ", "FINANCIAL TIMES", "FT", "BARRONS",
    "CNBC", "AP", "NIKKEI", "GUARDIAN", "AXIOS"
}

# Crypto trade press that is usually okay but a notch below the above list
TIER2_BRANDS_IN_TITLE = {
    "COINDESK", "THE BLOCK", "COINTELEGRAPH", "DECRYPT", "BARRONS", "BLOOMBERG"
}

# Twitter/X accounts considered market-moving (not promotional). Case-insensitive substring match on the `title`
ALLOWED_TWITTER_HANDLES = {
    "tier10k", "WuBlockchain", "CoinDesk", "TheBlock__", "Reuters", "AP",
    "Bloomberg", "CNBC", "MarketWatch", "FT", "WSJ"
}

# Promotional keywords that imply referral/airdrop/points/quests/etc → no trade
PROMO_KEYWORDS = {
    "airdrop", "points", "quest", "campaign", "referral", "bonus",
    "mint", "minting", "whitelist", "allowlist", "giveaway", "xp",
    "pool party", "poolparty", "bridge", "deposit", "season", "quest",
    "farm", "farming", "stake to earn", "rewards", "party", "drops",
    "join us", "utility incoming", "listing soon"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_KEYWORDS), re.IGNORECASE)

def is_authoritative(item):
    """Decide if an item is from an authoritative outlet (or strong tier2)."""
    source = (item.get("source") or "").strip()
    title = (item.get("title") or "").upper()

    # If the source string itself contains an authoritative brand
    for brand in AUTHORITATIVE_SOURCES:
        if brand in title:
            return True

    # Treat high-quality crypto press headlines as acceptable
    for brand in TIER2_BRANDS_IN_TITLE:
        if brand in title:
            return True

    # Otherwise, not authoritative by default
    return False

def is_allowed_twitter_news(item):
    """Allow only certain Twitter accounts unless the tweet looks promotional."""
    source = (item.get("source") or "").strip()
    title = (item.get("title") or "")
    if source.lower() != "twitter":
        return False

    if PROMO_RE.search(title):
        return False

    # allow if any of the known handles appear in the title
    for handle in ALLOWED_TWITTER_HANDLES:
        if handle.lower() in title.lower():
            return True
    return False

def is_promotional(item):
    """Block obvious promotional content, especially on Twitter."""
    title = (item.get("title") or "")
    return PROMO_RE.search(title) is not None

def safe_last_prev_close(item):
    prev = item.get("previous_candles") or []
    return prev[-1]["close"] if prev else None

def first_obs_close(item):
    obs = item.get("observation_candles") or []
    return obs[0]["close"] if obs else None

def first_obs_volume(item):
    obs = item.get("observation_candles") or []
    return obs[0]["volume"] if obs else None

def abs_pct_change(entry, base):
    return abs((entry / base - 1.0) * 100.0)

def decide_contrarian(pct_change):
    # Mean-reversion take: fade the initial impulse at entry
    return "SHORT" if pct_change > 0 else "LONG"

def eligible(item):
    """Eligibility filter + thresholding. Ensures id=7 is filtered out."""
    if not item.get("previous_candles") or not item.get("observation_candles"):
        return False, None, None, None

    lp = safe_last_prev_close(item)
    ec = first_obs_close(item)
    if lp is None or ec is None or lp <= 0:
        return False, None, None, None

    pct = (ec / lp - 1.0) * 100.0
    apc = abs(pct)

    title = (item.get("title") or "")
    source = (item.get("source") or "").strip()

    # Promotional → reject outright (this blocks the id=7 "Pool Points Party" case)
    if is_promotional(item):
        return False, lp, ec, pct

    # Authority tiers set different impulse thresholds
    if is_authoritative(item):
        threshold = 0.60   # lower threshold for strong outlets
    elif source.lower() == "twitter" and is_allowed_twitter_news(item):
        threshold = 0.90   # slightly stricter for Twitter newswire-style accounts
    else:
        # generic blogs/unknown → require larger move
        threshold = 1.20

    # Require sufficient impulse on the **first observation close** vs last prev close
    if apc < threshold:
        return False, lp, ec, pct

    return True, lp, ec, pct

@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a JSON array of news items"}), 400

        scored = []
        for item in data:
            try:
                ok, lp, ec, pct = eligible(item)
                if not ok:
                    continue
                # score by absolute impulse; break ties by higher first obs volume when available
                vol = first_obs_volume(item) or 0.0
                score = (abs(pct), vol)
                decision = decide_contrarian(pct)
                scored.append({
                    "id": item.get("id"),
                    "decision": decision,
                    "score": score
                })
            except Exception:
                # Skip malformed entries
                continue

        # Sort by impulse magnitude desc, then volume desc; take top 50
        scored.sort(key=lambda x: (x["score"][0], x["score"][1]), reverse=True)
        top50 = scored[:50]

        # Fallback: if fewer than 50 made the cut, relax to grab highest impulses from the rest
        if len(top50) < 50:
            # try to pull additional non-promotional Twitter/press with slightly relaxed threshold (1.0%)
            extras = []
            for item in data:
                if any(s["id"] == item.get("id") for s in top50):
                    continue
                if is_promotional(item):
                    continue
                lp = safe_last_prev_close(item)
                ec = first_obs_close(item)
                if not lp or not ec:
                    continue
                pct = (ec / lp - 1.0) * 100.0
                apc = abs(pct)
                if apc >= 1.0:
                    vol = first_obs_volume(item) or 0.0
                    extras.append({
                        "id": item.get("id"),
                        "decision": decide_contrarian(pct),
                        "score": (apc, vol)
                    })
            extras.sort(key=lambda x: (x["score"][0], x["score"][1]), reverse=True)
            need = 50 - len(top50)
            top50.extend(extras[:need])

        # final payload (id/decision only)
        out = [{"id": e["id"], "decision": e["decision"]} for e in top50 if e.get("id") is not None]

        return jsonify(out), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500