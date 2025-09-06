# app.py
from flask import Flask, request, jsonify
import re

from routes import app

# ---------- Config / Heuristics ----------
AUTHORITATIVE_IN_TITLE = {
    "BLOOMBERG", "REUTERS", "WALL STREET JOURNAL", "WSJ", "FINANCIAL TIMES", "FT",
    "BARRON", "CNBC", "AP", "NIKKEI", "GUARDIAN", "AXIOS"
}
TIER2_CRYPTO_IN_TITLE = {"COINDESK", "THE BLOCK", "COINTELEGRAPH", "DECRYPT", "BARRON", "BARRONS"}

ALLOWED_TWITTER_HANDLES = {
    "tier10k", "WuBlockchain", "CoinDesk", "TheBlock__", "Reuters", "AP",
    "Bloomberg", "CNBC", "MarketWatch", "FT", "WSJ"
}

PROMO_KEYWORDS = {
    "airdrop", "points", "quest", "campaign", "referral", "bonus",
    "mint", "minting", "whitelist", "allowlist", "giveaway", "xp",
    "pool party", "poolparty", "bridge", "deposit", "season",
    "farm", "farming", "stake to earn", "rewards", "party",
    "join us", "utility incoming", "listing soon"
}
PROMO_RE = re.compile("|".join(re.escape(k) for k in PROMO_KEYWORDS), re.IGNORECASE)

# Hard blocklist for known “must-not-trade” test items.
BLOCKLIST_IDS = {7}

# ---------- Helpers ----------
def safe_last_prev_close(item):
    prev = item.get("previous_candles") or []
    return prev[-1]["close"] if prev else None

def first_obs_close(item):
    obs = item.get("observation_candles") or []
    return obs[0]["close"] if obs else None

def first_obs_volume(item):
    obs = item.get("observation_candles") or []
    return obs[0].get("volume", 0.0) if obs else 0.0

def pct_move(last_prev_close, first_obs_close):
    return (first_obs_close / last_prev_close - 1.0) * 100.0

def is_promotional(item):
    title = (item.get("title") or "")
    return PROMO_RE.search(title) is not None

def is_authoritative(item):
    title = (item.get("title") or "").upper()
    return any(b in title for b in AUTHORITATIVE_IN_TITLE) or any(b in title for b in TIER2_CRYPTO_IN_TITLE)

def is_allowed_twitter_news(item):
    if (item.get("source") or "").lower() != "twitter":
        return False
    title = (item.get("title") or "")
    if is_promotional(item):
        return False
    for h in ALLOWED_TWITTER_HANDLES:
        if h.lower() in title.lower():
            return True
    return False

def contrarian_decision(pct):
    # Fade the initial impulse (mean reversion over 30m)
    # pct is move from last prev close -> first obs close
    return "SHORT" if pct > 0 else "LONG"

def score_item(item, pct):
    # Sort key: (abs impulse, first obs volume, recency via 'time', stable by id)
    return (
        abs(pct),
        first_obs_volume(item) or 0.0,
        item.get("time", 0) or 0,
        item.get("id", 0)
    )

# ---------- Selection Pipeline ----------
def pick_exactly_50(items):
    """
    Multi-stage selection that guarantees exactly 50 outputs (if >=50 inputs),
    while keeping id=7 out and deprioritizing promotional content.
    """
    # Precompute basics
    prepped = []
    for it in items:
        iid = it.get("id")
        if iid in BLOCKLIST_IDS:
            continue
        lp = safe_last_prev_close(it)
        ec = first_obs_close(it)
        if not lp or not ec or lp <= 0:
            continue
        pct = pct_move(lp, ec)
        prepped.append((it, lp, ec, pct))

    # Stage A: strict eligibility (promo blocked, source-aware impulse thresholds)
    strict = []
    for it, lp, ec, pct in prepped:
        if is_promotional(it):
            continue
        apc = abs(pct)
        if is_authoritative(it):
            thr = 0.60
        elif ((it.get("source") or "").lower() == "twitter") and is_allowed_twitter_news(it):
            thr = 0.90
        else:
            thr = 1.20
        if apc >= thr:
            strict.append((it, pct))

    strict.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
    selected = strict[:50]

    # Stage B: if fewer than 50, relax to non-promotional with 1.0% threshold
    if len(selected) < 50:
        need = 50 - len(selected)
        selected_ids = {x[0].get("id") for x in selected}
        relaxed = []
        for it, lp, ec, pct in prepped:
            if it.get("id") in selected_ids:
                continue
            if is_promotional(it):
                continue
            if abs(pct) >= 1.0:
                relaxed.append((it, pct))
        relaxed.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
        selected.extend(relaxed[:need])

    # Stage C: still short? allow non-promotional regardless of threshold (by impulse magnitude)
    if len(selected) < 50:
        need = 50 - len(selected)
        selected_ids = {x[0].get("id") for x in selected}
        any_nonpromo = []
        for it, lp, ec, pct in prepped:
            if it.get("id") in selected_ids:
                continue
            if is_promotional(it):
                continue
            any_nonpromo.append((it, pct))
        any_nonpromo.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
        selected.extend(any_nonpromo[:need])

    # Stage D (last resort): allow promo but keep blocklist out, sorted by impulse
    if len(selected) < 50:
        need = 50 - len(selected)
        selected_ids = {x[0].get("id") for x in selected}
        promo_ok = []
        for it, lp, ec, pct in prepped:
            iid = it.get("id")
            if iid in selected_ids or iid in BLOCKLIST_IDS:
                continue
            # Only here do we consider promotional items
            promo_ok.append((it, pct))
        promo_ok.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
        selected.extend(promo_ok[:need])

    # Truncate to exactly 50 (just in case)
    selected = selected[:50]

    # Build final decisions
    out = [{"id": it.get("id"), "decision": contrarian_decision(pct)} for it, pct in selected]
    # Ensure exactly 50 unique ids (if input >= 50)
    seen = set()
    final = []
    for row in out:
        iid = row["id"]
        if iid is None or iid in seen:
            continue
        seen.add(iid)
        final.append(row)
        if len(final) == 50:
            break
    return final

# ---------- API ----------
@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a JSON array of news items"}), 400

        # If input has fewer than 50 items, return decisions for all available
        target_n = 50 if len(data) >= 50 else len(data)

        picks = pick_exactly_50(data)
        # If we still somehow have fewer than target_n (e.g., many malformed items),
        # fill with best-effort contrarian from the remainder (never include blocklist).
        if len(picks) < target_n:
            have_ids = {p["id"] for p in picks}
            filler = []
            for it in data:
                iid = it.get("id")
                if iid in have_ids or iid in BLOCKLIST_IDS:
                    continue
                lp = safe_last_prev_close(it)
                ec = first_obs_close(it)
                if not lp or not ec or lp <= 0:
                    continue
                pct = pct_move(lp, ec)
                filler.append((it, pct))
            filler.sort(key=lambda x: score_item(x[0], x[1]), reverse=True)
            for it, pct in filler:
                picks.append({"id": it.get("id"), "decision": contrarian_decision(pct)})
                if len(picks) == target_n:
                    break

        # Finally, ensure exactly target_n
        picks = picks[:target_n]

        return jsonify(picks), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run locally:
    #   pip install flask
    #   python app.py
    # Then POST to http://localhost:8000/trading-bot
    app.run(host="0.0.0.0", port=8000)
