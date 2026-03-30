import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

CHECK_INTERVAL   = 60
ET               = pytz.timezone("America/New_York")
TWELVE_BASE      = "https://api.twelvedata.com"

# Tickers
STOCK_WATCHLIST  = ["QQQ", "SPY", "NVDA", "TSLA", "META", "AMZN"]

# Thresholds
VIX_MAX           = 30.0
VOLUME_SPIKE_MULT = 1.5
RSI_OVERBOUGHT    = 70
RSI_OVERSOLD      = 30
IV_RANK_HIGH      = 50
IV_RANK_LOW       = 20
TREND_BARS        = 6
EMA_FAST          = 9
EMA_SLOW          = 21
ALERT_COOLDOWN    = 900  # 15 min

# Scan window
MORNING_OPEN  = (9, 30)
MORNING_CLOSE = (11, 0)

# ── State ─────────────────────────────────────────────────────────────────────
alert_history   = {}
prev_closes     = {}
ema_cross_state = {}

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[Telegram error] {e}")

# ── Time helpers ──────────────────────────────────────────────────────────────
def now_et():
    return datetime.now(ET)

def in_morning_window():
    now = now_et()
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=MORNING_OPEN[0],  minute=MORNING_OPEN[1],  second=0, microsecond=0)
    close_t = now.replace(hour=MORNING_CLOSE[0], minute=MORNING_CLOSE[1], second=0, microsecond=0)
    return open_t <= now <= close_t

# ── Cooldown ──────────────────────────────────────────────────────────────────
def cooldown_ok(ticker, signal):
    now = time.time()
    if ticker not in alert_history:
        alert_history[ticker] = {}
    return (now - alert_history[ticker].get(signal, 0)) >= ALERT_COOLDOWN

def update_cooldown(ticker, signal):
    if ticker not in alert_history:
        alert_history[ticker] = {}
    alert_history[ticker][signal] = time.time()

# ── Twelve Data API ───────────────────────────────────────────────────────────
def twelve_get(endpoint, params={}):
    p = {**params, "apikey": TWELVE_API_KEY}
    r = requests.get(f"{TWELVE_BASE}/{endpoint}", params=p, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_5min(ticker):
    """Fetch 5-minute bars for a ticker."""
    try:
        data = twelve_get("time_series", {
            "symbol":     ticker,
            "interval":   "5min",
            "outputsize": "78",   # ~1 trading day of 5min bars
            "format":     "JSON",
        })
        if "values" not in data:
            print(f"[{ticker}] No values: {data.get('message', data.get('status', ''))}")
            return None
        rows = []
        for v in reversed(data["values"]):  # oldest first
            rows.append({
                "Open":   float(v["open"]),
                "High":   float(v["high"]),
                "Low":    float(v["low"]),
                "Close":  float(v["close"]),
                "Volume": float(v["volume"]),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[5min fetch error] {ticker}: {e}")
        return None

def fetch_daily(ticker):
    """Fetch daily bars for IV rank calculation."""
    try:
        data = twelve_get("time_series", {
            "symbol":     ticker,
            "interval":   "1day",
            "outputsize": "252",
            "format":     "JSON",
        })
        if "values" not in data:
            return None
        rows = []
        for v in reversed(data["values"]):
            rows.append({
                "Open":   float(v["open"]),
                "High":   float(v["high"]),
                "Low":    float(v["low"]),
                "Close":  float(v["close"]),
                "Volume": float(v["volume"]),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[Daily fetch error] {ticker}: {e}")
        return None

def get_vix():
    """Fetch latest VIX value."""
    try:
        data = twelve_get("quote", {"symbol": "VIX", "format": "JSON"})
        if "close" in data:
            return round(float(data["close"]), 2)
    except Exception as e:
        print(f"[VIX error] {e}")
    return None

def load_prev_closes():
    for ticker in STOCK_WATCHLIST:
        try:
            df = fetch_daily(ticker)
            if df is not None and len(df) >= 2:
                prev_closes[ticker] = round(float(df["Close"].iloc[-2]), 2)
                print(f"[Prev close] {ticker}: ${prev_closes[ticker]}")
            time.sleep(1)  # stay within rate limits
        except Exception as e:
            print(f"[Prev close error] {ticker}: {e}")

# ── Technical indicators ──────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calc_vwap(df):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def detect_trend(df):
    bars  = df.tail(TREND_BARS)
    highs = bars["High"].tolist()
    lows  = bars["Low"].tolist()
    hh = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
    hl = all(lows[i]  > lows[i-1]  for i in range(1, len(lows)))
    lh = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
    ll = all(lows[i]  < lows[i-1]  for i in range(1, len(lows)))
    if hh and hl: return "CALLS"
    if lh and ll: return "PUTS"
    return None

def detect_volume_spike(df):
    if len(df) < 21: return False, 0.0
    avg  = df["Volume"].iloc[-21:-1].mean()
    last = df["Volume"].iloc[-1]
    if avg == 0: return False, 0.0
    ratio = last / avg
    return ratio >= VOLUME_SPIKE_MULT, round(float(ratio), 2)

def detect_ema_cross(ticker, df):
    closes    = df["Close"]
    ema_f     = calc_ema(closes, EMA_FAST)
    ema_s     = calc_ema(closes, EMA_SLOW)
    prev_state = ema_cross_state.get(ticker)
    new_cross  = None
    if ema_f.iloc[-2] <= ema_s.iloc[-2] and ema_f.iloc[-1] > ema_s.iloc[-1]:
        new_cross = "bullish"
    elif ema_f.iloc[-2] >= ema_s.iloc[-2] and ema_f.iloc[-1] < ema_s.iloc[-1]:
        new_cross = "bearish"
    ema_cross_state[ticker] = "bullish" if ema_f.iloc[-1] > ema_s.iloc[-1] else "bearish"
    return new_cross if new_cross != prev_state else None

def calc_iv_rank(ticker):
    try:
        df = fetch_daily(ticker)
        if df is None or len(df) < 52: return None
        rv      = df["Close"].pct_change().dropna().rolling(21).std() * np.sqrt(252) * 100
        rv      = rv.dropna()
        curr_rv = rv.iloc[-1]
        iv_rank = ((curr_rv - rv.min()) / (rv.max() - rv.min())) * 100
        return round(float(iv_rank), 1)
    except: return None

def near_key_level(ticker, price):
    threshold = price * 0.003
    levels    = []
    base      = round(price)
    for offset in range(-3, 4):
        lvl = float(base + offset)
        if abs(lvl - price) <= threshold:
            levels.append(f"${lvl:.0f}")
    pc = prev_closes.get(ticker)
    if pc and abs(pc - price) <= threshold:
        levels.append(f"prev close ${pc:.2f}")
    return levels

# ── Economic events ───────────────────────────────────────────────────────────
ECON_KEYWORDS = [
    "fomc", "federal reserve", "interest rate", "cpi", "consumer price",
    "ppi", "producer price", "gdp", "jobs report", "nonfarm", "payroll",
    "retail sales", "unemployment", "inflation", "fed minutes", "powell",
]
_econ_cache      = []
_econ_cache_date = ""

def check_economic_events():
    import datetime as _dt
    global _econ_cache, _econ_cache_date
    today     = now_et().date()
    today_str = str(today)
    if _econ_cache_date == today_str:
        return _econ_cache
    try:
        url    = "https://financialmodelingprep.com/api/v3/economic_calendar"
        params = {"from": today_str, "to": str(today + _dt.timedelta(days=2))}
        r      = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        events = []
        for item in r.json():
            name    = item.get("event", "").lower()
            country = item.get("country", "").upper()
            impact  = item.get("impact", "").lower()
            date_s  = item.get("date", "")[:10]
            if country != "US": continue
            if impact not in ("high", "medium"): continue
            if not any(kw in name for kw in ECON_KEYWORDS): continue
            try:
                event_date = _dt.date.fromisoformat(date_s)
                days_away  = (event_date - today).days
                if 0 <= days_away <= 1:
                    label = "TODAY" if days_away == 0 else "TOMORROW"
                    events.append(f"{item.get('event', 'Event')} — {label}")
            except: pass
        _econ_cache      = events
        _econ_cache_date = today_str
        return events
    except Exception as e:
        print(f"[Econ events error] {e}")
        _econ_cache      = []
        _econ_cache_date = today_str
        return []

# ── Scanner ───────────────────────────────────────────────────────────────────
def scan_ticker(ticker, vix):
    df = fetch_5min(ticker)
    if df is None or len(df) < 30:
        return

    price  = round(float(df["Close"].iloc[-1]), 2)
    closes = df["Close"]

    rsi_series       = calc_rsi(closes)
    rsi              = round(float(rsi_series.iloc[-1]), 1) if not rsi_series.isna().iloc[-1] else None
    vwap             = round(float(calc_vwap(df).iloc[-1]), 2)
    trend            = detect_trend(df)
    vol_spike, vol_ratio = detect_volume_spike(df)
    ema_cross        = detect_ema_cross(ticker, df)
    key_levels       = near_key_level(ticker, price)
    iv_rank          = calc_iv_rank(ticker)

    rsi_ok  = rsi is not None and (
        (trend == "CALLS" and rsi < RSI_OVERBOUGHT) or
        (trend == "PUTS"  and rsi > RSI_OVERSOLD)
    )
    vwap_ok = (trend == "CALLS" and price > vwap) or (trend == "PUTS" and price < vwap)

    conditions = [trend is not None, vol_spike, vix <= VIX_MAX, rsi_ok, vwap_ok]
    score      = sum(conditions)

    print(f"  {ticker}: ${price} | Trend:{trend} | Vol:{vol_spike}({vol_ratio}x) "
          f"| RSI:{rsi} | VWAP:{'✅' if vwap_ok else '❌'} | Score:{score}/5")

    if trend and score >= 4 and cooldown_ok(ticker, trend):
        emoji     = "🟢" if trend == "CALLS" else "🔴"
        direction = "CALLS ↑" if trend == "CALLS" else "PUTS ↓"
        vwap_rel  = "above ✅" if price > vwap else "below ⚠️"
        iv_label  = ""
        if iv_rank is not None:
            iv_tag   = "🔥 Expensive" if iv_rank >= IV_RANK_HIGH else ("💎 Cheap" if iv_rank <= IV_RANK_LOW else "Moderate")
            iv_label = f"📉 IV Rank: <b>{iv_rank}%</b> — {iv_tag}\n"

        msg = (
            f"{emoji} <b>{ticker} SIGNAL — {direction}</b> [🌅 Morning]\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price: <b>${price:.2f}</b>\n"
            f"😌 VIX: <b>{vix}</b> ({'✅' if vix <= VIX_MAX else '⚠️'})\n"
            f"📈 Volume: <b>{vol_ratio}x avg</b>\n"
            f"📊 VWAP: <b>${vwap:.2f}</b> — price {vwap_rel}\n"
            f"💹 RSI: <b>{rsi}</b>\n"
            f"{iv_label}"
            + (f"📍 Key levels: <b>{', '.join(key_levels)}</b>\n" if key_levels else "")
            + f"━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {now_et().strftime('%I:%M %p ET')}\n"
            f"⚡️ Confirm on chart before entering"
        )
        send_telegram(msg)
        update_cooldown(ticker, trend)
        print(f"  [ALERT SENT] {ticker} {trend}")

    # EMA cross standalone alert
    if ema_cross and cooldown_ok(ticker, f"EMA_{ema_cross}"):
        emoji = "⚡️🟢" if ema_cross == "bullish" else "⚡️🔴"
        send_telegram(
            f"{emoji} <b>EMA CROSS — {ticker}</b>\n"
            f"9 EMA crossed <b>{'above' if ema_cross == 'bullish' else 'below'}</b> 21 EMA\n"
            f"Price: ${price:.2f} | RSI: {rsi}\n"
            f"Potential {'CALLS' if ema_cross == 'bullish' else 'PUTS'} setup forming\n"
            f"⏰ {now_et().strftime('%I:%M %p ET')}"
        )
        update_cooldown(ticker, f"EMA_{ema_cross}")

    time.sleep(1)  # rate limit between tickers

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    send_telegram(
        "✅ <b>Options Scanner v4 is LIVE</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "📊 Tickers: QQQ | SPY | NVDA | TSLA | META | AMZN\n"
        "🕐 Morning session: 9:30–11:00 AM ET\n"
        "📡 Data: Twelve Data (real-time)\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "Signals: Trend + VWAP + RSI + EMA + Volume\n"
        "Checking every 60 seconds 👀"
    )
    print("[v4 Scanner started]")
    load_prev_closes()

    econ_alerted_today = False
    last_date          = ""

    while True:
        now  = now_et()
        date = now.strftime("%Y-%m-%d")

        # Refresh prev closes at market open
        if now.hour == 9 and now.minute == 30:
            load_prev_closes()

        # Reset daily flags
        if date != last_date:
            econ_alerted_today = False
            last_date          = date

        # Economic event warning at 8am
        if not econ_alerted_today and now.hour >= 8 and now.weekday() < 5:
            events = check_economic_events()
            if events:
                lines = ["📅 <b>ECONOMIC EVENT WARNING</b>\n━━━━━━━━━━━━━━━━━━━"]
                lines += [f"⚠️ {e}" for e in events]
                lines.append("Options premiums may be elevated. Size carefully.")
                send_telegram("\n".join(lines))
            econ_alerted_today = True

        if not in_morning_window():
            print(f"[{now.strftime('%H:%M ET')}] Outside scan window, sleeping...")
            time.sleep(CHECK_INTERVAL)
            continue

        vix = get_vix()
        if vix is None:
            print("[VIX unavailable] Skipping cycle")
            time.sleep(CHECK_INTERVAL)
            continue

        print(f"[{now.strftime('%H:%M ET')}] Scanning | VIX: {vix}")

        for ticker in STOCK_WATCHLIST:
            scan_ticker(ticker, vix)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
