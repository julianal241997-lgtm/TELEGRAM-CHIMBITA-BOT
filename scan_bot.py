# -*- coding: utf-8 -*-
# CHIMBITA-RAILWAY-PROXY
import os, time, math, json, traceback
from datetime import datetime, timezone
import requests, numpy as np, pandas as pd, ccxt

TELEGRAM_TOKEN = "8331474411:AAG5wW_4m7KuBUdQ8nNi8L1gntx0NN2RUwU"
CHAT_ID = "1982879600"

TIMEFRAMES = ["1h", "2h", "4h", "1d"]
CANDLE_LIMIT = 300
REQUEST_TIMEOUT = 10
ENDPOINTS = [
    "https://api4.binance.com",
    "https://api-gcp.binance.com",
    "https://fapi.binance.com",
    "https://api.binance.vision"
]

def tg_send(text: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=10)
    except Exception:
        pass

def tg_send_document(caption: str, filename: str, content: bytes):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {"document": (filename, content)}
        data = {"chat_id": CHAT_ID, "caption": caption}
        requests.post(url, data=data, files=files, timeout=15)
    except Exception:
        pass

def test_endpoint(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/fapi/v1/exchangeInfo", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            return "symbols" in j
        return False
    except Exception:
        return False

def pick_working_endpoint() -> str:
    for base in ENDPOINTS:
        if test_endpoint(base):
            return base
    return ""

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    out = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(out.values, index=series.index)

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, period=20, mult=2):
    basis = sma(series, period)
    dev = series.rolling(period).std(ddof=0)
    upper = basis + mult * dev
    lower = basis - mult * dev
    width = (upper - lower) / (basis + 1e-9)
    return upper, basis, lower, width

def check_golden_cross(df):
    if len(df) < 205: 
        return False
    close = df["close"]
    ma50 = sma(close, 50)
    ma200 = sma(close, 200)
    cross = (ma50.iloc[-2] < ma200.iloc[-2]) and (ma50.iloc[-1] > ma200.iloc[-1])
    return bool(cross)

def check_bullish_divergence(df):
    if len(df) < 60:
        return False
    close = df["close"]
    r = rsi(close, 14).fillna(0)
    macd_line, signal_line, hist = macd(close)
    window = 30
    p1 = close.iloc[-window:-window//2].min()
    p2 = close.iloc[-window//2:].min()
    r1 = r.iloc[-window:-window//2].min()
    r2 = r.iloc[-window//2:].min()
    m1 = macd_line.iloc[-window:-window//2].min()
    m2 = macd_line.iloc[-window//2:].min()
    price_lower_low = p2 < p1 * 0.999
    rsi_higher_low = r2 > r1
    macd_higher_low = m2 > m1
    return bool(price_lower_low and (rsi_higher_low or macd_higher_low))

def check_bb_squeeze_breakout(df):
    if len(df) < 60:
        return False
    close = df["close"]
    upper, basis, lower, width = bollinger_bands(close, 20, 2)
    w = width.fillna(0)
    thresh = w.rolling(100).quantile(0.2).iloc[-1] if len(w) >= 100 else w.quantile(0.2)
    is_squeeze = w.iloc[-5:].mean() < thresh
    breakout = close.iloc[-1] > upper.iloc[-1]
    return bool(is_squeeze and breakout)

def check_ema_ribbon_flip(df):
    if len(df) < 60:
        return False
    close = df["close"]
    e8 = ema(close, 8)
    e13 = ema(close, 13)
    e21 = ema(close, 21)
    e34 = ema(close, 34)
    e55 = ema(close, 55)
    prev_bear = (e8.iloc[-3] < e13.iloc[-3] < e21.iloc[-3] < e34.iloc[-3] < e55.iloc[-3])
    now_bull = (e8.iloc[-1] > e13.iloc[-1] > e21.iloc[-1] > e34.iloc[-1] > e55.iloc[-1])
    return bool(prev_bear and now_bull)

def check_volume_spike_breakout(df):
    if len(df) < 50:
        return False
    vol = df["volume"]
    close = df["close"]
    high = df["high"]
    avg20 = vol.rolling(20).mean()
    spike = vol.iloc[-1] > 1.8 * (avg20.iloc[-1] + 1e-9)
    breakout = close.iloc[-1] > high.iloc[-20:-1].max()
    return bool(spike and breakout)

def evaluate_criteria(df):
    hits = 0
    if check_golden_cross(df): hits += 1
    if check_bullish_divergence(df): hits += 1
    if check_bb_squeeze_breakout(df): hits += 1
    if check_ema_ribbon_flip(df): hits += 1
    if check_volume_spike_breakout(df): hits += 1
    return hits

def prob_label(n):
    if n >= 5: return "HIGH"
    if n == 4: return "MEDIUM"
    if n == 3: return "LOW"
    return "NONE"

def fetch_ohlcv_safe(ex, symbol, timeframe):
    try:
        return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
    except Exception:
        time.sleep(0.25)
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
        except Exception:
            return []

def get_usdt_perp_symbols(ex):
    markets = ex.load_markets()
    symbols = []
    for s, m in markets.items():
        try:
            if m.get('swap', False) and m.get('linear', True) and m.get('quote', '') == 'USDT':
                info = m.get('info', {})
                if str(info.get('contractType', '')).upper() == 'PERPETUAL':
                    symbols.append(s)
        except Exception:
            continue
    return sorted(list(set(symbols)))

def scan_once():
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    tg_send(f"🔄 Bot CHIMBITA-RAILWAY-PROXY started on Railway\n🕒 {started}")

    base = pick_working_endpoint()
    if not base:
        tg_send("❌ Critical bot error: No working Binance endpoint (all blocked).")
        return

    ex = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
        "timeout": 15000,
    })
    try:
        ex.urls["api"]["fapi"] = base
    except Exception:
        pass

    alerts = []
    near = []

    try:
        symbols = get_usdt_perp_symbols(ex)
    except Exception as e:
        try:
            r = requests.get(f"{base}/fapi/v1/exchangeInfo", timeout=REQUEST_TIMEOUT)
            txt = f"[CRITICAL] exchangeInfo GET {base}/fapi/v1/exchangeInfo {r.status_code}\n{json.dumps(r.json(), indent=2) if r.text else ''}"
            tg_send(txt[:3900])
            if r.text:
                tg_send_document("exchangeInfo.json", "exchangeInfo.json", r.content)
        except Exception as ee:
            tg_send(f"[CRITICAL] Could not fetch exchangeInfo at {base} : {ee}")
        return

    for sym in symbols:
        for tf in TIMEFRAMES:
            ohlcv = fetch_ohlcv_safe(ex, sym, tf)
            if not ohlcv or len(ohlcv) < 60:
                continue
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            score = evaluate_criteria(df)
            if score >= 3:
                lbl = prob_label(score)
                alerts.append((score, f"• {sym} [{tf.upper()}] — {lbl} ({score}/5)"))
            elif score == 2:
                near.append((score, f"{sym} [{tf.upper()}] ({score}/5)"))

    if alerts:
        alerts.sort(key=lambda x: (-x[0], x[1]))
        lines = [f"✅ Signals ({len(alerts)}) — 3+ criteria met"]
        lines.extend([a[1] for a in alerts[:60]])
        tg_send("\n".join(lines)[:3900])
    else:
        near.sort(key=lambda x: (-x[0], x[1]))
        top5 = ", ".join([n[1] for n in near[:5]]) if near else "—"
        msg = "ℹ️ No signals found (3+ criteria).\n" + f"🔎 Closest (2/5): {top5}"
        tg_send(msg)

def main():
    try:
        print("[INFO] Starting scan (Railway cron)")
        scan_once()
    except Exception as e:
        import traceback
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        tg_send(f"❌ Unhandled error:\n{err[:3900]}")

if __name__ == "__main__":
    main()
