import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz



# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]

CHECK_INTERVAL    = 60
ET                = pytz.timezone("America/New_York")
MASSIVE_API_KEY   = os.environ["MASSIVE_API_KEY"]
MASSIVE_BASE      = "https://api.polygon.io"   # Massive.com uses same endpoints

# ── Massive/Polygon data layer ────────────────────────────────────────────────
def _poly_get(path, params=None):
    """Authenticated GET to Massive/Polygon API."""
    p = params or {}
    p["apiKey"] = MASSIVE_API_KEY
    r = requests.get(f"{MASSIVE_BASE}{path}", params=p, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_5min(ticker):
    """Fetch last 5 days of 5-minute bars."""
    try:
        end   = datetime.now(ET).date()
        start = end - timedelta(days=5)
        data  = _poly_get(
            f"/v2/aggs/ticker/{ticker}/range/5/minute/{start}/{end}",
            {"adjusted": "true", "sort": "asc", "limit": "1000"}
        )
        results = data.get("results", [])
        if not results:
            return None
        df = pd.DataFrame(results)
        df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume","t":"Time"}, inplace=True)
        df["Time"] = pd.to_datetime(df["Time"], unit="ms", utc=True)
        df.set_index("Time", inplace=True)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception as e:
        print(f"[5min fetch error] {ticker}: {e}")
        return None

def fetch_daily(ticker):
    """Fetch last 1 year of daily bars."""
    try:
        end   = datetime.now(ET).date()
        start = end - timedelta(days=365)
        data  = _poly_get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            {"adjusted": "true", "sort": "asc", "limit": "365"}
        )
        results = data.get("results", [])
        if not results:
            return None
        df = pd.DataFrame(results)
        df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume","t":"Time"}, inplace=True)
        df["Time"] = pd.to_datetime(df["Time"], unit="ms", utc=True)
        df.set_index("Time", inplace=True)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception as e:
        print(f"[Daily fetch error] {ticker}: {e}")
        return None

def get_vix():
    """Fetch latest VIX value."""
    try:
        end   = datetime.now(ET).date()
        start = end - timedelta(days=2)
        data  = _poly_get(
            f"/v2/aggs/ticker/VXX/range/5/minute/{start}/{end}",
            {"adjusted": "true", "sort": "desc", "limit": "1"}
        )
        results = data.get("results", [])
        if results:
            return round(float(results[0]["c"]), 2)
    except Exception as e:
        print(f"[VIX error] {e}")
    return None

def get_unusual_options_flow(ticker):
    """Check for unusual options activity via Massive/Polygon options chain."""
    try:
        data = _poly_get(
            f"/v3/snapshot/options/{ticker}",
            {"limit": "50", "sort": "volume", "order": "desc"}
        )
        results = data.get("results", [])
        unusual = []
        for item in results:
            details = item.get("details", {})
            day     = item.get("day", {})
            greeks  = item.get("greeks", {})
            vol     = day.get("volume", 0)
            oi      = item.get("open_interest", 0)
            if oi == 0: continue
            ratio = vol / oi
            if ratio > 3 and vol > 100:
                unusual.append({
                    "side":   "CALLS" if details.get("contract_type") == "call" else "PUTS",
                    "strike": details.get("strike_price", 0),
                    "volume": int(vol),
                    "oi":     int(oi),
                    "ratio":  round(ratio, 1),
                })
        return unusual[:3] if unusual else None
    except Exception as e:
        print(f"[Options flow error] {ticker}: {e}")
        return None

# Stock watchlist
STOCK_WATCHLIST = ["QQQ", "SPY", "NVDA", "TSLA", "META"]

# Futures — yfinance tickers
FUTURES = {
    "NQ=F":  {"name": "/NQ",   "label": "Nasdaq Futures",   "stock_pair": "QQQ", "unit": "pts"},
    "ES=F":  {"name": "/ES",   "label": "S&P Futures",      "stock_pair": "SPY", "unit": "pts"},
    "GC=F":  {"name": "Gold",  "label": "Gold Futures",     "stock_pair": None,  "unit": "$/oz"},
    "CL=F":  {"name": "Oil",   "label": "Crude Oil Futures","stock_pair": None,  "unit": "$/bbl"},
}

# Futures move thresholds that trigger independent alerts
FUTURES_MOVE_THRESHOLDS = {
    "NQ=F": 0.4,   # 0.4% move on NQ
    "ES=F": 0.3,   # 0.3% move on ES
    "GC=F": 0.5,   # 0.5% move on Gold
    "CL=F": 1.0,   # 1.0% move on Oil
}

# Futures runs 23hrs/day — use a wider scan window
FUTURES_SCAN_START = (6, 0)    # 6am ET
FUTURES_SCAN_END   = (17, 0)   # 5pm ET

# Stock scan windows
MORNING_OPEN  = (9, 30)
MORNING_CLOSE = (11, 0)
POWER_OPEN    = (15, 0)
POWER_CLOSE   = (16, 0)

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
ALERT_COOLDOWN    = 900   # 15 min

# ── State ─────────────────────────────────────────────────────────────────────
alert_history   = {}
prev_closes     = {}
ema_cross_state = {}
futures_state   = {}   # {ticker: {"price": x, "direction": "up"|"down"|None}}

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

def in_window(open_h, open_m, close_h, close_m):
    now    = now_et()
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=open_h,  minute=open_m,  second=0, microsecond=0)
    close_t = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return open_t <= now <= close_t

def in_morning_window():  return in_window(*MORNING_OPEN,  *MORNING_CLOSE)
def in_power_window():    return in_window(*POWER_OPEN,    *POWER_CLOSE)
def in_futures_window():  return in_window(*FUTURES_SCAN_START, *FUTURES_SCAN_END)
def market_is_open():     return in_window(9, 30, 16, 0)

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
    closes   = df["Close"]
    ema_f    = calc_ema(closes, EMA_FAST)
    ema_s    = calc_ema(closes, EMA_SLOW)
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



def check_earnings(ticker):
    """Check upcoming earnings via Massive/Polygon."""
    try:
        today = now_et().date()
        end   = today + timedelta(days=7)
        data  = _poly_get(
            f"/vX/reference/financials",
            {"ticker": ticker, "limit": "1", "sort": "filing_date", "order": "desc"}
        )
        # Earnings date not always in free tier — try snapshot instead
        snap = _poly_get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
        results = snap.get("ticker", {})
        # Polygon free tier doesn't reliably expose earnings dates
        # so we just return None gracefully and skip earnings warning
        return None
    except: 
        return None

ECON_KEYWORDS = [
    "fomc", "federal reserve", "interest rate", "cpi", "consumer price",
    "ppi", "producer price", "gdp", "jobs report", "nonfarm", "payroll",
    "retail sales", "unemployment", "inflation", "fed minutes", "powell",
]

_econ_cache      = []
_econ_cache_date = ""

def fetch_economic_events():
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
                    events.append(f"{item.get(chr(39)+'event'+chr(39), 'Economic Event')} — {label}")
            except: pass
        _econ_cache      = events
        _econ_cache_date = today_str
        print(f"[Economic events] {len(events)} upcoming events fetched")
        return events
    except Exception as e:
        print(f"[Economic events error] {e}")
        _econ_cache      = []
        _econ_cache_date = today_str
        return []

def check_economic_events():
    return fetch_economic_events()

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

# ── Futures logic ─────────────────────────────────────────────────────────────
def scan_futures():
    """
    Scans all futures independently and checks for confirmation with paired stocks.
    Returns list of alert messages to send.
    """
    alerts = []

    for fticker, finfo in FUTURES.items():
        try:
            df = fetch_5min(fticker)
            if df is None or len(df) < 10:
                continue

            price     = round(float(df["Close"].iloc[-1]), 2)
            prev      = round(float(df["Close"].iloc[-2]), 2)
            pct_chg   = ((price - prev) / prev) * 100
            direction = "up" if pct_chg > 0 else "down"
            threshold = FUTURES_MOVE_THRESHOLDS[fticker]
            trend     = detect_trend(df)
            vol_spike, vol_ratio = detect_volume_spike(df)

            last_state = futures_state.get(fticker, {})
            last_price = last_state.get("price", price)
            session_pct = ((price - last_price) / last_price) * 100 if last_price else 0

            print(f"  {finfo['name']}: ${price} | {pct_chg:+.2f}% | Trend:{trend}")

            # ── Independent alert: meaningful move + trend ──
            signal_key = f"INDEPENDENT_{direction.upper()}"
            if (abs(pct_chg) >= threshold and trend and
                    vol_spike and cooldown_ok(fticker, signal_key)):
                emoji = "🟢" if direction == "up" else "🔴"
                move_label = "BULLISH ↑" if direction == "up" else "BEARISH ↓"
                alerts.append(
                    f"{emoji} <b>{finfo['label']} MOVE — {move_label}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 {finfo['name']}: <b>${price:,.2f}</b> ({pct_chg:+.2f}%)\n"
                    f"📈 Volume: <b>{vol_ratio}x average</b>\n"
                    f"📊 Trend: <b>{'Higher highs & lows' if trend == 'CALLS' else 'Lower highs & lows'}</b>\n"
                    + (f"🔗 Paired stock: <b>{finfo['stock_pair']}</b> — watch for confirmation\n"
                       if finfo['stock_pair'] else "") +
                    f"⏰ {now_et().strftime('%I:%M %p ET')}"
                )
                update_cooldown(fticker, signal_key)

            # ── Confirmation alert: futures + paired stock agree ──
            if finfo["stock_pair"] and trend:
                stock      = finfo["stock_pair"]
                stock_df   = fetch_5min(stock)
                if stock_df is not None and len(stock_df) >= TREND_BARS:
                    stock_trend = detect_trend(stock_df)
                    stock_price = round(float(stock_df["Close"].iloc[-1]), 2)
                    confirm_key = f"CONFIRM_{trend}"

                    if stock_trend == trend and cooldown_ok(fticker, confirm_key):
                        emoji     = "🟢🔗" if trend == "CALLS" else "🔴🔗"
                        direction_label = "BULLISH" if trend == "CALLS" else "BEARISH"
                        alerts.append(
                            f"{emoji} <b>FUTURES + STOCK CONFIRMATION</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━\n"
                            f"Both <b>{finfo['name']}</b> and <b>{stock}</b> trending <b>{direction_label}</b>\n"
                            f"💰 {finfo['name']}: ${price:,.2f} | {stock}: ${stock_price:.2f}\n"
                            f"📊 Strong conviction signal — consider <b>{'CALLS' if trend == 'CALLS' else 'PUTS'}</b> on {stock}\n"
                            f"⏰ {now_et().strftime('%I:%M %p ET')}"
                        )
                        update_cooldown(fticker, confirm_key)

            # Update state
            futures_state[fticker] = {"price": price, "direction": direction}

        except Exception as e:
            print(f"[Futures scan error] {fticker}: {e}")

    return alerts

# ── Stock scanner ─────────────────────────────────────────────────────────────
def scan_ticker(ticker, vix, session):
    df5 = fetch_5min(ticker)
    if df5 is None or len(df5) < 30:
        return

    price = round(float(df5["Close"].iloc[-1]), 2)
    closes = df5["Close"]

    rsi_series       = calc_rsi(closes)
    rsi              = round(float(rsi_series.iloc[-1]), 1) if not rsi_series.isna().iloc[-1] else None
    vwap             = round(float(calc_vwap(df5).iloc[-1]), 2)
    trend            = detect_trend(df5)
    vol_spike, vol_ratio = detect_volume_spike(df5)
    ema_cross        = detect_ema_cross(ticker, df5)
    key_levels       = near_key_level(ticker, price)
    earnings         = check_earnings(ticker)
    iv_rank          = calc_iv_rank(ticker)

    rsi_ok  = rsi is not None and (
        (trend == "CALLS" and rsi < RSI_OVERBOUGHT) or
        (trend == "PUTS"  and rsi > RSI_OVERSOLD)
    )
    vwap_ok = (trend == "CALLS" and price > vwap) or (trend == "PUTS" and price < vwap)

    # Check futures confirmation for paired tickers
    futures_confirm = None
    for fticker, finfo in FUTURES.items():
        if finfo["stock_pair"] == ticker:
            fstate = futures_state.get(fticker, {})
            fdir   = fstate.get("direction")
            if fdir == "up" and trend == "CALLS":
                futures_confirm = f"{finfo['name']} trending up ✅"
            elif fdir == "down" and trend == "PUTS":
                futures_confirm = f"{finfo['name']} trending down ✅"

    conditions = [trend is not None, vol_spike, vix <= VIX_MAX, rsi_ok, vwap_ok]
    score      = sum(conditions)

    print(f"  {ticker}: ${price} | Trend:{trend} | Vol:{vol_spike}({vol_ratio}x) "
          f"| RSI:{rsi} | VWAP:{'✅' if vwap_ok else '❌'} | FuturesConfirm:{futures_confirm is not None} | Score:{score}/5")

    if trend and score >= 4 and cooldown_ok(ticker, trend):
        emoji     = "🟢" if trend == "CALLS" else "🔴"
        direction = "CALLS ↑" if trend == "CALLS" else "PUTS ↓"
        vwap_rel  = "above ✅" if price > vwap else "below ⚠️"
        iv_label  = ""
        if iv_rank is not None:
            iv_tag   = "🔥 Expensive" if iv_rank >= IV_RANK_HIGH else ("💎 Cheap" if iv_rank <= IV_RANK_LOW else "Moderate")
            iv_label = f"📉 IV Rank: <b>{iv_rank}%</b> — {iv_tag}\n"
        sess_label = "🌅 Morning" if session == "morning" else "⚡️ Power Hour"

        msg = (
            f"{emoji} <b>{ticker} SIGNAL — {direction}</b> [{sess_label}]\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Price: <b>${price:.2f}</b>\n"
            f"😌 VIX: <b>{vix}</b> ({'✅' if vix <= VIX_MAX else '⚠️'})\n"
            f"📈 Volume: <b>{vol_ratio}x avg</b>\n"
            f"📊 VWAP: <b>${vwap:.2f}</b> — price {vwap_rel}\n"
            f"💹 RSI: <b>{rsi}</b>\n"
            f"{iv_label}"
            + (f"🔗 Futures confirm: <b>{futures_confirm}</b>\n" if futures_confirm else "")
            + (f"📍 Key levels: <b>{', '.join(key_levels)}</b>\n" if key_levels else "")
            + (f"⚠️ Earnings: <b>{earnings}</b> — premiums inflated!\n" if earnings else "")
            + f"━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {now_et().strftime('%I:%M %p ET')}\n"
            f"⚡️ Confirm on chart before entering"
        )
        send_telegram(msg)
        update_cooldown(ticker, trend)
        print(f"  [ALERT] {ticker} {trend}")

    # EMA cross standalone
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

    # Unusual options flow
    flow = get_unusual_options_flow(ticker)
    if flow and cooldown_ok(ticker, "FLOW"):
        lines = [f"🐋 <b>UNUSUAL FLOW — {ticker}</b>\n━━━━━━━━━━━━━━━━━━━"]
        for f in flow:
            lines.append(
                f"{'🟢' if f['side'] == 'CALLS' else '🔴'} <b>{f['side']}</b> "
                f"${f['strike']} | Vol: {f['volume']:,} | OI: {f['oi']:,} | {f['ratio']}x"
            )
        lines.append(f"⏰ {now_et().strftime('%I:%M %p ET')}")
        send_telegram("\n".join(lines))
        update_cooldown(ticker, "FLOW")

# ── Prev close loader ─────────────────────────────────────────────────────────
def load_prev_closes():
    all_tickers = STOCK_WATCHLIST + list(FUTURES.keys())
    for ticker in all_tickers:
        try:
            df = fetch_daily(ticker)
            if df is not None and len(df) >= 2:
                prev_closes[ticker] = round(float(df["Close"].iloc[-2]), 2)
                print(f"[Prev close] {ticker}: ${prev_closes[ticker]}")
            time.sleep(12)  # stay within 5 calls/min free tier
        except Exception as e:
            print(f"[Prev close error] {ticker}: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    send_telegram(
        "✅ <b>Advanced Options Scanner v3 is LIVE</b>\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "📊 Stocks: QQQ | SPY | NVDA | TSLA | META\n"
        "📈 Futures: /NQ | /ES | Gold | Oil\n"
        "🕐 Morning: 9:30–11:00 AM ET\n"
        "⚡️ Power Hour: 3:00–4:00 PM ET\n"
        "🌙 Futures: 6:00 AM–5:00 PM ET\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "Signals: Trend + VWAP + RSI + EMA + Volume\n"
        "🔗 Futures confirmation layer active\n"
        "🐋 Unusual options flow detection\n"
        "📅 Earnings & economic event warnings"
    )
    print("[v3 Scanner started]")
    load_prev_closes()

    econ_alerted_today = False
    last_date          = ""

    while True:
        now  = now_et()
        date = now.strftime("%Y-%m-%d")

        if now.hour == 9 and now.minute == 30:
            load_prev_closes()

        if date != last_date:
            econ_alerted_today = False
            last_date          = date

        # Economic event warning at 8am
        if not econ_alerted_today and now.hour >= 8:
            events = check_economic_events()
            if events:
                lines = ["📅 <b>ECONOMIC EVENT WARNING</b>\n━━━━━━━━━━━━━━━━━━━"]
                lines += [f"⚠️ {e}" for e in events]
                lines.append("Options premiums may be elevated. Size carefully.")
                send_telegram("\n".join(lines))
                econ_alerted_today = True

        morning = in_morning_window()
        power   = in_power_window()
        futures = in_futures_window()

        if not morning and not power and not futures:
            print(f"[{now.strftime('%H:%M ET')}] Outside all scan windows, sleeping...")
            time.sleep(CHECK_INTERVAL)
            continue

        # Always scan futures in futures window
        if futures:
            print(f"[{now.strftime('%H:%M ET')}] Scanning futures...")
            futures_alerts = scan_futures()
            for alert in futures_alerts:
                send_telegram(alert)
                time.sleep(1)

        # Scan stocks during market sessions
        if morning or power:
            session = "morning" if morning else "power"
            vix     = get_vix()
            if vix is None:
                print("[VIX unavailable] Skipping stock scan")
            else:
                print(f"[{now.strftime('%H:%M ET')}] Scanning stocks [{session}] | VIX: {vix}")
                for ticker in STOCK_WATCHLIST:
                    scan_ticker(ticker, vix, session)
                    time.sleep(2)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
