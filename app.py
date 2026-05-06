import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="VN30F Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background: #080c18; color: #dde4f0; }
.stApp { background: #080c18; }
section[data-testid="stSidebar"] { background: #0c1020; border-right: 1px solid #192138; }
section[data-testid="stSidebar"] * { color: #c0ccdf !important; }
section[data-testid="stSidebar"] .stCheckbox label p { font-size: 12px !important; font-family: 'JetBrains Mono', monospace !important; }

.metric-box { background: #0f1626; border: 1px solid #192138; border-radius: 8px; padding: 12px 14px; text-align: center; }
.metric-label { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 17px; font-weight: 700; }
.green { color: #00e676; } .red { color: #ff5252; } .yellow { color: #ffd600; }
.white { color: #f1f5f9; } .blue { color: #38bdf8; } .purple { color: #a78bfa; }

.signal-card { border-radius: 10px; padding: 14px 18px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 13px; letter-spacing: 0.5px; text-align: center; }
.uptrend   { background: linear-gradient(135deg,#0a2218,#0d311f); border: 1.5px solid #00e676; color: #00e676; }
.downtrend { background: linear-gradient(135deg,#220a0a,#310d0d); border: 1.5px solid #ff5252; color: #ff5252; }
.sideway   { background: linear-gradient(135deg,#18180a,#26240a); border: 1.5px solid #ffd600; color: #ffd600; }

.sec-hdr { font-size: 10px; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: #334155; border-bottom: 1px solid #192138; padding-bottom: 5px; margin-bottom: 10px; font-family: 'JetBrains Mono', monospace; }

.sig-row-buy  { background: #0a1f12; border: 1px solid #00e67633; border-left: 3px solid #00e676; border-radius: 6px; padding: 8px 12px; margin-bottom: 5px; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.sig-row-sell { background: #1f0a0a; border: 1px solid #ff525233; border-left: 3px solid #ff5252; border-radius: 6px; padding: 8px 12px; margin-bottom: 5px; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.sig-row-watch{ background: #1a180a; border: 1px solid #ffd60033; border-left: 3px solid #ffd600; border-radius: 6px; padding: 8px 12px; margin-bottom: 5px; font-family: 'JetBrains Mono', monospace; font-size: 11px; }

.stButton > button { background: linear-gradient(135deg,#1e3a8a,#1d4ed8); color: #fff; border: none; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.stButton > button:hover { background: linear-gradient(135deg,#1d4ed8,#3b82f6); }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1px; color: #475569; }
.stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; }
.stSelectbox > div > div { background: #0f1626; border-color: #192138; color: #dde4f0; }
.stNumberInput > div > div > input { background: #0f1626; border-color: #192138; color: #dde4f0; }
.stTextInput > div > div > input { background: #0f1626; border-color: #192138; color: #dde4f0; }
.toggle-title { font-size: 10px; color: #334155; letter-spacing: 1.5px; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin: 6px 0 4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
defaults = {
    "trade_history":  [],
    "signal_history": [],
    "last_refresh":   datetime.now(),
    "seed":           random.randint(0, 9999),
    "prev_sig_keys":  set(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════
# DATA – Nguồn không phải công ty chứng khoán:
# Ưu tiên 1: xnoapi (thư viện độc lập, lấy thẳng từ HNX)
# Ưu tiên 2: Wbgapis proxy (dữ liệu công khai HNX)
# Fallback: Mô phỏng
# Chart chính: TradingView Widget nhúng trực tiếp (không cần API)
# ══════════════════════════════════════════════════════
import requests as _requests
import math as _math

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}


def _ts_range(tf_minutes: int, n_bars: int):
    bars_per_day = max(1, 285 // tf_minutes)
    days_needed  = max(7, _math.ceil(n_bars / bars_per_day) + 5)
    to_ts   = int(datetime.now().timestamp())
    from_ts = int((datetime.now() - timedelta(days=days_needed)).timestamp())
    return from_ts, to_ts


def _parse_tv_udf(raw: dict, n_bars: int) -> pd.DataFrame:
    status = raw.get("s", "")
    if status != "ok":
        raise ValueError(f"API status='{status}', errmsg={raw.get('errmsg','?')}")
    if not raw.get("t"):
        raise ValueError("Không có dữ liệu (mảng 't' rỗng)")
    df = pd.DataFrame({
        "time":   pd.to_datetime(raw["t"], unit="s", utc=True).tz_convert("Asia/Ho_Chi_Minh").tz_localize(None),
        "open":   pd.to_numeric(raw["o"], errors="coerce"),
        "high":   pd.to_numeric(raw["h"], errors="coerce"),
        "low":    pd.to_numeric(raw["l"], errors="coerce"),
        "close":  pd.to_numeric(raw["c"], errors="coerce"),
        "volume": pd.to_numeric(raw["v"], errors="coerce"),
    })
    df = df.set_index("time").dropna().sort_index()
    return df.tail(n_bars)


# ── Nguồn 1: xnoapi – thư viện độc lập, không phải công ty CK ───────────────
def _fetch_xnoapi(tf_minutes: int, n_bars: int) -> pd.DataFrame:
    try:
        from xnoapi.vn.data import get_derivatives_hist
        res_map = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        res = res_map.get(tf_minutes, "1m")
        df = get_derivatives_hist("VN30F1M", res)
        if df is None or len(df) < 10:
            raise ValueError("Không đủ dữ liệu từ xnoapi")
        # Chuẩn hóa tên cột
        df.columns = [c.lower() for c in df.columns]
        col_map = {"date": "time", "datetime": "time", "open": "open",
                   "high": "high", "low": "low", "close": "close", "volume": "volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "time" not in df.index.names and "time" in df.columns:
            df = df.set_index("time")
        elif df.index.name != "time":
            df.index.name = "time"
        df.index = pd.to_datetime(df.index)
        return df[["open","high","low","close","volume"]].dropna().sort_index().tail(n_bars)
    except ImportError:
        raise ValueError("xnoapi chưa cài: pip install xnoapi")


# ── Nguồn 2: HNX Datafeed trực tiếp (sàn giao dịch nhà nước, không phải CK) ──
def _fetch_hnx_direct(tf_minutes: int, n_bars: int) -> pd.DataFrame:
    from_ts, to_ts = _ts_range(tf_minutes, n_bars)
    res = {1: "1", 5: "5", 15: "15", 30: "30", 60: "60"}.get(tf_minutes, "1")
    # HNX cung cấp data qua nhiều mirror, thử lần lượt
    endpoints = [
        ("https://hnx.vn/api/datafeed/history", {
            "symbol": "VN30F1M", "resolution": res,
            "from": from_ts, "to": to_ts,
        }, {"Referer": "https://hnx.vn/"}),
        ("https://api.hnx.vn/datafeed/history", {
            "symbol": "VN30F1M", "resolution": res,
            "from": from_ts, "to": to_ts,
        }, {"Referer": "https://hnx.vn/"}),
        # Investing.com public feed (nguồn quốc tế)
        ("https://tvc4.investing.com/14a2ae7e3d49f20a5d35ef01c91e8c21/1/14/14/14/history", {
            "symbol": "VN30F1M", "resolution": res,
            "from": from_ts, "to": to_ts,
        }, {"Referer": "https://vn.investing.com/", "X-Requested-With": "XMLHttpRequest"}),
    ]
    for url, params, extra_hdrs in endpoints:
        try:
            resp = _requests.get(url, params=params,
                                 headers={**_HEADERS, **extra_hdrs}, timeout=8)
            resp.raise_for_status()
            return _parse_tv_udf(resp.json(), n_bars)
        except Exception:
            continue
    raise ValueError("HNX direct: tất cả endpoint thất bại")


# ── Nguồn 3: Wbgapis / public proxy aggregate ────────────────────────────────
def _fetch_public_proxy(tf_minutes: int, n_bars: int) -> pd.DataFrame:
    """Thử các public proxy/mirror không phải công ty chứng khoán."""
    from_ts, to_ts = _ts_range(tf_minutes, n_bars)
    res = {1: "1", 5: "5", 15: "15", 30: "30", 60: "60"}.get(tf_minutes, "1")
    candidates = [
        # Market open API (nguồn dữ liệu phi công ty CK)
        ("https://api.simplize.vn/api/market/derivative/history", {
            "symbol": "VN30F1M", "resolution": res,
            "from": from_ts, "to": to_ts,
        }, {"Referer": "https://simplize.vn/"}),
        # Fmarket (quỹ đầu tư, không phải CK)
        ("https://fmarket.vn/fund/api/fund-derivative/history", {
            "symbol": "VN30F1M", "resolution": res,
            "from": from_ts, "to": to_ts,
        }, {"Referer": "https://fmarket.vn/"}),
    ]
    for url, params, extra_hdrs in candidates:
        try:
            resp = _requests.get(url, params=params,
                                 headers={**_HEADERS, **extra_hdrs}, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("t"):
                return _parse_tv_udf({**data, "s": "ok"}, n_bars)
            elif isinstance(data, dict) and data.get("s") == "ok":
                return _parse_tv_udf(data, n_bars)
        except Exception:
            continue
    raise ValueError("Public proxy: tất cả thất bại")


# ── Fallback mô phỏng ─────────────────────────────────────────────────────────
def _simulate_ohlcv(tf_minutes, n_bars=300, seed=42):
    np.random.seed(seed + tf_minutes * 7)
    now = datetime.now().replace(second=0, microsecond=0)
    now -= timedelta(minutes=now.minute % tf_minutes)
    times = [now - timedelta(minutes=tf_minutes * i) for i in range(n_bars)][::-1]
    prices = [1280.0]
    for i in range(1, n_bars):
        phase = (i // 40) % 3
        drift = 0.18 if phase == 0 else (-0.14 if phase == 2 else 0.0)
        vol   = 0.32 if phase == 1 else 0.58
        prices.append(max(prices[-1] + drift + np.random.normal(0, vol), 100))
    df = pd.DataFrame({"time": times, "close": prices})
    noise = np.abs(np.random.normal(0, 0.28, n_bars)) + 0.08
    df["open"]   = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"]   = df[["open","close"]].max(axis=1) + noise
    df["low"]    = df[["open","close"]].min(axis=1) - noise
    df["volume"] = np.random.randint(150, 3000, n_bars)
    return df.set_index("time")


@st.cache_data(ttl=60)
def fetch_ohlcv(symbol: str, tf_minutes: int, n_bars: int = 300):
    """
    Lấy dữ liệu OHLCV VN30F1M – KHÔNG dùng công ty chứng khoán.
    Thứ tự: xnoapi → HNX Direct → Public Proxy → Mô phỏng.
    """
    sources = [
        ("xnoapi (độc lập)",    _fetch_xnoapi),
        ("HNX Direct",          _fetch_hnx_direct),
        ("Public Proxy",        _fetch_public_proxy),
    ]
    errors = []
    for name, fn in sources:
        try:
            df = fn(tf_minutes, n_bars)
            if df is not None and len(df) >= 10:
                st.session_state["data_source"] = f"✅ {name} · {len(df)} bars"
                st.session_state["data_errors"] = []
                return df
            errors.append(f"{name}: chỉ {len(df) if df is not None else 0} bars")
        except Exception as e:
            errors.append(f"{name}: {str(e)[:150]}")

    st.session_state["data_source"] = "⚠️ Mô phỏng (tất cả API ngoài giờ/bị chặn)"
    st.session_state["data_errors"] = errors
    return _simulate_ohlcv(tf_minutes, n_bars, seed=random.randint(0, 9999))


# ══════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════
def add_indicators(df):
    c, h, l, n = df["close"].values, df["high"].values, df["low"].values, len(df)

    def ema(arr, p):
        r = np.full(len(arr), np.nan)
        k = 2 / (p + 1)
        r[p-1] = arr[:p].mean()
        for i in range(p, len(arr)):
            r[i] = arr[i] * k + r[i-1] * (1 - k)
        return r

    df["ema9"]  = ema(c, 9)
    df["ema21"] = ema(c, 21)
    df["ema50"] = ema(c, 50)

    rm = pd.Series(c).rolling(20).mean().values
    rs = pd.Series(c).rolling(20).std().values
    df["bb_mid"]   = rm
    df["bb_upper"] = rm + 2 * rs
    df["bb_lower"] = rm - 2 * rs
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / pd.Series(rm).replace(0, np.nan).values

    d = pd.Series(c).diff()
    g_  = d.clip(lower=0).rolling(14).mean()
    lo_ = (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + g_ / lo_)

    e12, e26 = ema(c, 12), ema(c, 26)
    ml = e12 - e26
    df["macd"]        = ml
    df["macd_signal"] = ema(np.nan_to_num(ml), 9)
    df["macd_hist"]   = ml - df["macd_signal"].values

    tr = np.zeros(n); dmp = np.zeros(n); dmm = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up_, dn_ = h[i]-h[i-1], l[i-1]-l[i]
        dmp[i] = max(up_, 0) if up_ > dn_ else 0
        dmm[i] = max(dn_, 0) if dn_ > up_ else 0

    def wilder(a, p):
        r = np.full(len(a), np.nan)
        r[p] = sum(a[1:p+1])
        for i in range(p+1, len(a)):
            r[i] = r[i-1] - r[i-1]/p + a[i]
        return r

    atr14 = wilder(tr, 14); dmp14 = wilder(dmp, 14); dmm14 = wilder(dmm, 14)
    safe  = lambda x: np.where(x == 0, 1, x)
    dp    = 100 * dmp14 / safe(atr14)
    dm    = 100 * dmm14 / safe(atr14)
    dx    = 100 * np.abs(dp - dm) / safe(dp + dm)
    df["adx"]    = wilder(dx, 14)
    df["di_pos"] = dp
    df["di_neg"] = dm
    df["atr"]    = atr14
    df["vol_ma"] = pd.Series(df["volume"].values).rolling(20).mean().values

    lo14 = pd.Series(l).rolling(14).min()
    hi14 = pd.Series(h).rolling(14).max()
    k_raw = (pd.Series(c) - lo14) / (hi14 - lo14 + 1e-9) * 100
    df["stoch_k"] = k_raw.rolling(3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["pivot"] = (pd.Series(h).shift(1) + pd.Series(l).shift(1) + pd.Series(c).shift(1)) / 3
    df["res1"]  = 2 * df["pivot"] - pd.Series(l).shift(1).values
    df["sup1"]  = 2 * df["pivot"] - pd.Series(h).shift(1).values

    return df


# ══════════════════════════════════════════════════════
# SIGNAL DEFINITIONS
# ══════════════════════════════════════════════════════
SIGNAL_DEFS = {
    "EMA_CROSS_UP":    {"label":"EMA 9×21 Cắt Lên",         "action":"BUY",  "icon":"🟢","color":"#00e676","sym":"triangle-up",  "pos":"below"},
    "EMA_CROSS_DOWN":  {"label":"EMA 9×21 Cắt Xuống",       "action":"SELL", "icon":"🔴","color":"#ff5252","sym":"triangle-down","pos":"above"},
    "EMA_ALIGN_BULL":  {"label":"EMA Xếp chuẩn BULL",       "action":"BUY",  "icon":"🟢","color":"#00e676","sym":"triangle-up",  "pos":"below"},
    "EMA_ALIGN_BEAR":  {"label":"EMA Xếp chuẩn BEAR",       "action":"SELL", "icon":"🔴","color":"#ff5252","sym":"triangle-down","pos":"above"},
    "RSI_OVERSOLD":    {"label":"RSI Quá Bán (<30)",         "action":"BUY",  "icon":"💎","color":"#00e676","sym":"triangle-up",  "pos":"below"},
    "RSI_OVERBOUGHT":  {"label":"RSI Quá Mua (>70)",         "action":"SELL", "icon":"🔥","color":"#ff5252","sym":"triangle-down","pos":"above"},
    "MACD_CROSS_UP":   {"label":"MACD Cắt Lên",             "action":"BUY",  "icon":"📈","color":"#38bdf8","sym":"triangle-up",  "pos":"below"},
    "MACD_CROSS_DOWN": {"label":"MACD Cắt Xuống",           "action":"SELL", "icon":"📉","color":"#f97316","sym":"triangle-down","pos":"above"},
    "BB_BREAK_UP":     {"label":"Phá BB Trên – Breakout",   "action":"BUY",  "icon":"🚀","color":"#38bdf8","sym":"star",         "pos":"below"},
    "BB_BREAK_DOWN":   {"label":"Phá BB Dưới – Breakdown",  "action":"SELL", "icon":"💥","color":"#f97316","sym":"star",         "pos":"above"},
    "BB_SQUEEZE":      {"label":"BB Squeeze – Sắp bứt phá", "action":"WATCH","icon":"⚡","color":"#ffd600","sym":"diamond",      "pos":"below"},
    "BB_BOUNCE_UP":    {"label":"Nảy BB Dưới (Hỗ trợ)",    "action":"BUY",  "icon":"🟢","color":"#00e676","sym":"triangle-up",  "pos":"below"},
    "BB_BOUNCE_DOWN":  {"label":"Từ chối BB Trên (KCự)",   "action":"SELL", "icon":"🔴","color":"#ff5252","sym":"triangle-down","pos":"above"},
    "STOCH_CROSS_UP":  {"label":"Stoch K×D Cắt Lên",       "action":"BUY",  "icon":"🟢","color":"#a78bfa","sym":"triangle-up",  "pos":"below"},
    "STOCH_CROSS_DOWN":{"label":"Stoch K×D Cắt Xuống",     "action":"SELL", "icon":"🔴","color":"#a78bfa","sym":"triangle-down","pos":"above"},
    "VOL_SPIKE":       {"label":"Volume Đột Biến (>2×MA)",  "action":"WATCH","icon":"📊","color":"#a78bfa","sym":"diamond",      "pos":"below"},
}


# ══════════════════════════════════════════════════════
# DETECT SIGNALS (latest bar)
# ══════════════════════════════════════════════════════
def g(row, col, default=0):
    v = row.get(col, default)
    return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

def detect_signals(df):
    if len(df) < 60:
        return {"regime":"SIDEWAY","strength":"YẾU","adx":0,"di_pos":0,"di_neg":0,
                "rsi":50,"bb_width":0.02,"ema9":0,"ema21":0,"ema50":0,
                "close":float(df["close"].iloc[-1]),"macd_hist":0,"stoch_k":50,"stoch_d":50,"signals":[]}

    cur  = df.iloc[-1]
    prev = df.iloc[-2]

    adx    = g(cur,"adx",20); di_pos = g(cur,"di_pos",20); di_neg = g(cur,"di_neg",20)
    rsi    = g(cur,"rsi",50); bb_w   = g(cur,"bb_width",0.03)
    ema9   = g(cur,"ema9"); ema21 = g(cur,"ema21"); ema50 = g(cur,"ema50")
    p_ema9 = g(prev,"ema9"); p_ema21 = g(prev,"ema21")
    bb_up  = g(cur,"bb_upper"); bb_lo = g(cur,"bb_lower")
    close  = float(cur["close"]); prev_close = float(prev["close"])
    mh     = g(cur,"macd_hist"); pmh = g(prev,"macd_hist")
    sk     = g(cur,"stoch_k",50); sd  = g(cur,"stoch_d",50)
    psk    = g(prev,"stoch_k",50); psd = g(prev,"stoch_d",50)
    vol    = float(cur["volume"]); vol_ma = g(cur,"vol_ma",vol)

    regime   = ("SIDEWAY" if (adx < 22 or bb_w < 0.015) else ("UPTREND" if di_pos > di_neg else "DOWNTREND"))
    strength = ("YẾU" if adx < 18 else ("MẠNH" if adx > 35 else "VỪA"))

    fired = []
    if p_ema9 <= p_ema21 and ema9 > ema21:                              fired.append("EMA_CROSS_UP")
    if p_ema9 >= p_ema21 and ema9 < ema21:                              fired.append("EMA_CROSS_DOWN")
    if ema9 > ema21 > ema50 and regime == "UPTREND":                    fired.append("EMA_ALIGN_BULL")
    if ema9 < ema21 < ema50 and regime == "DOWNTREND":                  fired.append("EMA_ALIGN_BEAR")
    if rsi < 30:                                                         fired.append("RSI_OVERSOLD")
    if rsi > 70:                                                         fired.append("RSI_OVERBOUGHT")
    if pmh <= 0 < mh:                                                    fired.append("MACD_CROSS_UP")
    if pmh >= 0 > mh:                                                    fired.append("MACD_CROSS_DOWN")
    if close > bb_up:                                                    fired.append("BB_BREAK_UP")
    if close < bb_lo:                                                    fired.append("BB_BREAK_DOWN")
    if prev_close <= g(prev,"bb_lower") and close > bb_lo:              fired.append("BB_BOUNCE_UP")
    if prev_close >= g(prev,"bb_upper") and close < bb_up:              fired.append("BB_BOUNCE_DOWN")
    hist_bbw = df["bb_width"].dropna().tail(60)
    if len(hist_bbw) > 15 and bb_w < hist_bbw.quantile(0.15):           fired.append("BB_SQUEEZE")
    if psk <= psd and sk > sd and sk < 80:                               fired.append("STOCH_CROSS_UP")
    if psk >= psd and sk < sd and sk > 20:                               fired.append("STOCH_CROSS_DOWN")
    if vol_ma > 0 and vol / vol_ma > 2.0:                                fired.append("VOL_SPIKE")

    return {"regime":regime,"strength":strength,"adx":adx,"di_pos":di_pos,"di_neg":di_neg,
            "rsi":rsi,"bb_width":bb_w,"ema9":ema9,"ema21":ema21,"ema50":ema50,
            "close":close,"macd_hist":mh,"stoch_k":sk,"stoch_d":sd,"signals":fired}


# ══════════════════════════════════════════════════════
# SCAN ALL HISTORICAL SIGNAL MARKERS FOR CHART
# ══════════════════════════════════════════════════════
def scan_chart_signals(df, lookback=100):
    results = []
    df_s = df.tail(lookback).copy()
    for i in range(2, len(df_s)):
        sub  = df_s.iloc[:i+1]
        cur  = sub.iloc[-1]; prev = sub.iloc[-2]
        ema9  = g(cur,"ema9"); ema21 = g(cur,"ema21")
        p9    = g(prev,"ema9"); p21   = g(prev,"ema21")
        rsi   = g(cur,"rsi",50)
        mh    = g(cur,"macd_hist"); pmh = g(prev,"macd_hist")
        bb_up = g(cur,"bb_upper"); bb_lo = g(cur,"bb_lower")
        close = float(cur["close"])
        sk = g(cur,"stoch_k",50); sd = g(cur,"stoch_d",50)
        psk=g(prev,"stoch_k",50); psd=g(prev,"stoch_d",50)
        atr  = max(g(cur,"atr",1), 0.3)
        t    = sub.index[-1]

        def add(key):
            d   = SIGNAL_DEFS[key]
            off = atr * 0.7
            y   = (close - off) if d["pos"] == "below" else (close + off)
            results.append({"time":t,"price":close,"y":y,"key":key,
                             "action":d["action"],"sym":d["sym"],
                             "color":d["color"],"pos":d["pos"],"label":d["label"]})

        if p9 <= p21 and ema9 > ema21:    add("EMA_CROSS_UP")
        if p9 >= p21 and ema9 < ema21:    add("EMA_CROSS_DOWN")
        if rsi < 30:                       add("RSI_OVERSOLD")
        if rsi > 70:                       add("RSI_OVERBOUGHT")
        if pmh <= 0 < mh:                  add("MACD_CROSS_UP")
        if pmh >= 0 > mh:                  add("MACD_CROSS_DOWN")
        if close > bb_up:                  add("BB_BREAK_UP")
        if close < bb_lo:                  add("BB_BREAK_DOWN")
        if psk <= psd and sk > sd and sk < 80: add("STOCH_CROSS_UP")
        if psk >= psd and sk < sd and sk > 20: add("STOCH_CROSS_DOWN")

    return results


# ══════════════════════════════════════════════════════
# PUSH SIGNALS TO HISTORY
# ══════════════════════════════════════════════════════
def push_signals(regime_info, tf, price):
    now_str  = datetime.now().strftime("%H:%M:%S")
    date_str = datetime.now().strftime("%d/%m")
    for key in regime_info["signals"]:
        uid = f"{date_str}_{now_str[:-3]}_{key}_{tf}"
        if uid not in st.session_state.prev_sig_keys:
            st.session_state.prev_sig_keys.add(uid)
            d = SIGNAL_DEFS.get(key, {})
            st.session_state.signal_history.insert(0, {
                "time":   now_str, "date": date_str, "tf": tf, "key": key,
                "label":  d.get("label", key), "action": d.get("action","WATCH"),
                "icon":   d.get("icon","•"),    "color":  d.get("color","#64748b"),
                "price":  price, "regime": regime_info["regime"],
                "rsi":    regime_info["rsi"],   "adx": regime_info["adx"],
            })
    st.session_state.signal_history = st.session_state.signal_history[:200]


# ══════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════
BG = "#080c18"; GRID = "#192138"

def build_chart(df, title, show, chart_sigs):
    df  = df.dropna(subset=["ema21"]).iloc[-100:]

    sub_map = [("vol",   show.get("volume",True),   0.12),
               ("macd",  show.get("macd",True),     0.15),
               ("rsi",   show.get("rsi",True),      0.13),
               ("stoch", show.get("stoch",False),   0.13),
               ("adx",   show.get("adx_panel",False),0.13)]
    active  = [(n, h) for n, on, h in sub_map if on]

    heights = [0.50] + [h for _, h in active]
    total   = sum(heights); heights = [h/total for h in heights]
    n_rows  = len(heights)

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=heights, vertical_spacing=0.008,
                        specs=[[{"secondary_y":False}]]*n_rows)

    # ── Candles ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff5252",
        increasing_fillcolor="#00e676",  decreasing_fillcolor="#ff5252",
        line_width=1, name="OHLC",
    ), row=1, col=1)

    # ── Bollinger Bands ──
    if show.get("bb", True):
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"],
            line=dict(color="#475569",width=1,dash="dot"), name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"],
            line=dict(color="#475569",width=1,dash="dot"), fill="tonexty",
            fillcolor="rgba(71,85,105,0.06)", name="BB Lower", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_mid"],
            line=dict(color="#2d3f5e",width=0.8), name="BB Mid", showlegend=False), row=1, col=1)

    # ── EMAs ──
    ema_cfg = [("ema9","#f59e0b","EMA9"),("ema21","#38bdf8","EMA21"),("ema50","#a78bfa","EMA50")]
    for col_name, color, lbl in ema_cfg:
        if show.get(col_name, True):
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name],
                line=dict(color=color, width=1.5), name=lbl), row=1, col=1)

    # ── Pivot S/R ──
    if show.get("pivot", False):
        fig.add_trace(go.Scatter(x=df.index, y=df["pivot"],
            line=dict(color="#64748b",width=1,dash="dash"), name="Pivot"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["res1"],
            line=dict(color="#ff526655",width=0.9,dash="dot"), name="R1"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["sup1"],
            line=dict(color="#00e67655",width=0.9,dash="dot"), name="S1"), row=1, col=1)

    # ── TP / SL lines for open trades ──
    if show.get("tp_sl", True):
        for t in st.session_state.trade_history:
            if t["status"] == "OPEN":
                dc = "#00e676" if t["direction"] == "LONG" else "#ff5252"
                for val, lc, lbl in [(t["tp"],"#00e676",f"TP#{t['id']}"),(t["sl"],"#ff5252",f"SL#{t['id']}")]:
                    fig.add_hline(y=val, line_color=lc, line_width=1, line_dash="dot",
                                  annotation_text=lbl, annotation_font_color=lc,
                                  annotation_font_size=9, row=1, col=1)

    # ── Signal markers on chart ──
    if show.get("signals_on_chart", True) and chart_sigs:
        # Group by action for legend clarity
        for action_filter, sym_default in [("BUY","triangle-up"),("SELL","triangle-down"),("WATCH","diamond")]:
            grp = [s for s in chart_sigs if s["action"] == action_filter]
            if not grp: continue
            fig.add_trace(go.Scatter(
                x    = [s["time"]  for s in grp],
                y    = [s["y"]     for s in grp],
                mode = "markers",
                marker=dict(
                    symbol = [s["sym"]  for s in grp],
                    size   = 11,
                    color  = [s["color"] for s in grp],
                    line   = dict(color=BG, width=1),
                ),
                hovertext = [f"{SIGNAL_DEFS[s['key']]['icon']} {s['label']}<br>@ {s['price']:.2f}" for s in grp],
                hoverinfo = "text",
                name      = f"▲ Tín hiệu {action_filter}" if action_filter=="BUY" else (f"▼ Tín hiệu {action_filter}" if action_filter=="SELL" else f"◆ WATCH"),
            ), row=1, col=1)

    # ── Sub-panels ──
    for ri, (sub_name, _) in enumerate(active, start=2):
        if sub_name == "vol":
            vc = ["rgba(0,230,118,0.6)" if c>=o else "rgba(255,82,82,0.6)"
                  for c, o in zip(df["close"], df["open"])]
            fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vc, name="Vol", showlegend=False), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["vol_ma"],
                line=dict(color="#ffd600",width=1.2), name="Vol MA", showlegend=False), row=ri, col=1)

        elif sub_name == "macd":
            hc = ["#00e676" if v >= 0 else "#ff5252" for v in df["macd_hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], marker_color=hc, name="Hist", showlegend=False), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
                line=dict(color="#38bdf8",width=1.2), name="MACD"), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
                line=dict(color="#ffd600",width=1.2), name="Signal"), row=ri, col=1)
            fig.add_hline(y=0, line_color=GRID, line_width=0.8, row=ri, col=1)

        elif sub_name == "rsi":
            fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
                line=dict(color="#38bdf8",width=1.5), name="RSI",
                fill="tozeroy", fillcolor="rgba(56,189,248,0.04)"), row=ri, col=1)
            fig.add_hline(y=70, line_color="#ff5252", line_width=0.8, line_dash="dot", row=ri, col=1)
            fig.add_hline(y=30, line_color="#00e676", line_width=0.8, line_dash="dot", row=ri, col=1)
            fig.add_hline(y=50, line_color=GRID,      line_width=0.6, row=ri, col=1)
            fig.update_yaxes(row=ri, col=1, range=[0,100])

        elif sub_name == "stoch":
            fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"],
                line=dict(color="#a78bfa",width=1.3), name="%K"), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"],
                line=dict(color="#ffd600",width=1.1), name="%D"), row=ri, col=1)
            fig.add_hline(y=80, line_color="#ff5252", line_width=0.7, line_dash="dot", row=ri, col=1)
            fig.add_hline(y=20, line_color="#00e676", line_width=0.7, line_dash="dot", row=ri, col=1)
            fig.update_yaxes(row=ri, col=1, range=[0,100])

        elif sub_name == "adx":
            fig.add_trace(go.Scatter(x=df.index, y=df["adx"],
                line=dict(color="#f59e0b",width=1.4), name="ADX"), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["di_pos"],
                line=dict(color="#00e676",width=1.0), name="DI+"), row=ri, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["di_neg"],
                line=dict(color="#ff5252",width=1.0), name="DI-"), row=ri, col=1)
            fig.add_hline(y=25, line_color="#475569", line_width=0.8, line_dash="dot", row=ri, col=1)

    h_total = min(520 + 90 * len(active), 800)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=0,r=0,t=28,b=0), height=h_total,
        title=dict(text=title, font=dict(family="JetBrains Mono",size=11,color="#475569"), x=0.01),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    font=dict(size=9,color="#64748b"), bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    for i in range(1, n_rows+1):
        fig.update_xaxes(row=i, col=1, gridcolor=GRID, showgrid=True, zeroline=False,
                         tickfont=dict(size=8,color="#334155"))
        fig.update_yaxes(row=i, col=1, gridcolor=GRID, showgrid=True, zeroline=False,
                         tickfont=dict(size=8,color="#475569"))
    return fig


# ══════════════════════════════════════════════════════
# TRADE HELPERS
# ══════════════════════════════════════════════════════
def add_trade(direction, entry, tp, sl, size, note=""):
    st.session_state.trade_history.insert(0, {
        "id": len(st.session_state.trade_history)+1,
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%d/%m"),
        "direction": direction, "entry": entry, "tp": tp, "sl": sl,
        "size": size, "status": "OPEN", "pnl": 0.0, "note": note,
    })

def close_trade(idx, price):
    t = st.session_state.trade_history[idx]
    if t["status"] != "OPEN": return
    mult = 1 if t["direction"] == "LONG" else -1
    t.update({"status":"CLOSED","exit_price":price,
              "pnl":(price-t["entry"])*mult*t["size"]*100_000,
              "close_time":datetime.now().strftime("%H:%M:%S")})

def render_trades(cp):
    if not st.session_state.trade_history:
        st.markdown('<div style="color:#334155;font-family:JetBrains Mono,monospace;font-size:11px;padding:8px">Chưa có lệnh</div>', unsafe_allow_html=True)
        return
    op = sum((cp-t["entry"])*(1 if t["direction"]=="LONG" else -1)*t["size"]
             for t in st.session_state.trade_history if t["status"]=="OPEN")
    cl = sum(t.get("pnl",0) for t in st.session_state.trade_history if t["status"]=="CLOSED")
    c1,c2 = st.columns(2)
    c1.markdown(f'<div class="metric-box"><div class="metric-label">P&L Đóng</div><div class="metric-value {"green" if cl>=0 else "red"}" style="font-size:13px">{cl:+,.0f}đ</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-label">P&L Mở</div><div class="metric-value {"green" if op>=0 else "red"}" style="font-size:13px">{op:+.2f}pt</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    for i, t in enumerate(st.session_state.trade_history):
        dc = "#00e676" if t["direction"]=="LONG" else "#ff5252"
        sc = "#ffd600" if t["status"]=="OPEN" else "#334155"
        lp_str = ""
        if t["status"] == "OPEN":
            lp = (cp-t["entry"])*(1 if t["direction"]=="LONG" else -1)*t["size"]
            lp_str = f'<span class="{"green" if lp>=0 else "red"}">{lp:+.2f}pt</span>'
        st.markdown(f"""
        <div style="background:#0f1626;border:1px solid #192138;border-left:2px solid {dc};border-radius:5px;padding:8px 10px;margin-bottom:5px;font-family:'JetBrains Mono',monospace;font-size:10px">
          <div style="display:flex;justify-content:space-between"><b style="color:{dc};font-size:12px">#{t['id']} {t['direction']}</b><span style="color:{sc}">{t['status']}</span></div>
          <div style="color:#475569">{t['date']} {t['time']} · {t['size']}HĐ</div>
          <div style="display:flex;justify-content:space-between;margin-top:2px">
            <span>In <b style="color:#f1f5f9">{t['entry']:.2f}</b> TP <b style="color:#00e676">{t['tp']:.2f}</b> SL <b style="color:#ff5252">{t['sl']:.2f}</b></span>
            <span>{lp_str}</span>
          </div>
        </div>""", unsafe_allow_html=True)
        if t["status"] == "OPEN":
            if st.button(f"Đóng #{t['id']}", key=f"cl_{i}"): close_trade(i, cp); st.rerun()


# ══════════════════════════════════════════════════════
# SIGNAL HISTORY PANEL
# ══════════════════════════════════════════════════════
def render_signal_history():
    hist = st.session_state.signal_history
    if not hist:
        st.markdown('<div style="color:#334155;font-family:JetBrains Mono,monospace;font-size:11px;padding:8px">Chưa có tín hiệu</div>', unsafe_allow_html=True)
        return

    f1, f2, f3 = st.columns(3)
    fa = f1.selectbox("Hành động", ["Tất cả","BUY","SELL","WATCH"], key="sf_a")
    ft = f2.selectbox("Khung",     ["Tất cả","1P","5P"],            key="sf_t")
    fk = f3.selectbox("Loại tín hiệu",
                      ["Tất cả"] + sorted({s["key"] for s in hist}), key="sf_k")

    filt = [s for s in hist
            if (fa=="Tất cả" or s["action"]==fa)
            and (ft=="Tất cả" or s["tf"]==ft)
            and (fk=="Tất cả" or s["key"]==fk)]

    # Stats row
    n_buy  = sum(1 for s in filt if s["action"]=="BUY")
    n_sell = sum(1 for s in filt if s["action"]=="SELL")
    n_watch= sum(1 for s in filt if s["action"]=="WATCH")
    st.markdown(f"""
    <div style="display:flex;gap:8px;margin-bottom:10px;font-family:'JetBrains Mono',monospace;font-size:11px">
      <div style="background:#0a1f12;border:1px solid #00e67633;border-radius:5px;padding:5px 10px;color:#00e676">🟢 BUY: {n_buy}</div>
      <div style="background:#1f0a0a;border:1px solid #ff525233;border-radius:5px;padding:5px 10px;color:#ff5252">🔴 SELL: {n_sell}</div>
      <div style="background:#1a180a;border:1px solid #ffd60033;border-radius:5px;padding:5px 10px;color:#ffd600">⚡ WATCH: {n_watch}</div>
      <div style="background:#0f1626;border:1px solid #192138;border-radius:5px;padding:5px 10px;color:#64748b">TỔNG: {len(filt)}</div>
    </div>""", unsafe_allow_html=True)

    css_map = {"BUY":"sig-row-buy","SELL":"sig-row-sell","WATCH":"sig-row-watch"}
    col_map = {"BUY":"#00e676","SELL":"#ff5252","WATCH":"#ffd600"}
    rc_map  = {"UPTREND":"#00e676","DOWNTREND":"#ff5252","SIDEWAY":"#ffd600"}

    for s in filt[:80]:
        col = col_map.get(s["action"],"#64748b")
        css = css_map.get(s["action"],"sig-row-watch")
        rc  = rc_map.get(s.get("regime",""),"#64748b")
        st.markdown(f"""
        <div class="{css}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span>{s['icon']} <b style="color:{col}">{s['action']}</b>
              <span style="color:#475569"> [{s['tf']}] {s['date']} {s['time']}</span></span>
            <b style="color:#f1f5f9;font-size:12px">{s['price']:.2f}</b>
          </div>
          <div style="color:#94a3b8;margin-top:2px;font-size:10px">{s['label']}</div>
          <div style="display:flex;gap:10px;margin-top:2px;font-size:10px">
            <span style="color:{rc}">{s.get('regime','')}</span>
            <span style="color:#475569">RSI {s.get('rsi',0):.1f}</span>
            <span style="color:#475569">ADX {s.get('adx',0):.1f}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    if not filt:
        st.markdown('<div style="color:#334155;font-size:11px;font-family:JetBrains Mono,monospace;padding:8px">Không có tín hiệu khớp bộ lọc</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TRADE EXECUTION TABLE
# ══════════════════════════════════════════════════════
def render_trade_table(current_price):
    trades = st.session_state.trade_history
    closed = [t for t in trades if t["status"] == "CLOSED"]
    open_  = [t for t in trades if t["status"] == "OPEN"]

    if not trades:
        st.markdown('<div style="color:#334155;font-family:JetBrains Mono,monospace;font-size:11px;padding:8px">Chưa có lệnh nào được thực hiện.</div>', unsafe_allow_html=True)
        return

    # ── Summary stats ──
    total_pnl_pts = sum(
        (t["exit_price"] - t["entry"]) * (1 if t["direction"] == "LONG" else -1)
        for t in closed
    )
    wins  = [t for t in closed if (t["exit_price"] - t["entry"]) * (1 if t["direction"] == "LONG" else -1) > 0]
    losses= [t for t in closed if (t["exit_price"] - t["entry"]) * (1 if t["direction"] == "LONG" else -1) <= 0]
    win_rate = len(wins)/len(closed)*100 if closed else 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.markdown(f'<div class="metric-box"><div class="metric-label">Tổng Lệnh</div><div class="metric-value white" style="font-size:15px">{len(trades)}</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">Mở: {len(open_)} · Đóng: {len(closed)}</div></div>', unsafe_allow_html=True)
    sc2.markdown(f'<div class="metric-box"><div class="metric-label">Tổng P&L (điểm)</div><div class="metric-value {"green" if total_pnl_pts>=0 else "red"}" style="font-size:15px">{total_pnl_pts:+.2f}</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">từ {len(closed)} lệnh đóng</div></div>', unsafe_allow_html=True)
    sc3.markdown(f'<div class="metric-box"><div class="metric-label">Win Rate</div><div class="metric-value {"green" if win_rate>=50 else "red"}" style="font-size:15px">{win_rate:.0f}%</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">✅{len(wins)} ❌{len(losses)}</div></div>', unsafe_allow_html=True)
    avg_win  = sum((t["exit_price"]-t["entry"])*(1 if t["direction"]=="LONG" else -1) for t in wins)/len(wins)   if wins   else 0
    avg_loss = sum((t["exit_price"]-t["entry"])*(1 if t["direction"]=="LONG" else -1) for t in losses)/len(losses) if losses else 0
    sc4.markdown(f'<div class="metric-box"><div class="metric-label">Avg Win / Loss</div><div class="metric-value yellow" style="font-size:13px">{avg_win:+.2f} / {avg_loss:+.2f}</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">điểm mỗi lệnh</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Table header ──
    st.markdown("""
    <div style="display:grid;grid-template-columns:40px 60px 70px 70px 70px 70px 80px 90px 80px;
                gap:2px;padding:6px 10px;background:#0a0e1a;border:1px solid #192138;
                border-radius:6px 6px 0 0;font-family:'JetBrains Mono',monospace;
                font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px;margin-top:6px">
      <div>#</div><div>Giờ</div><div>L/S</div><div>Điểm vào</div>
      <div>Điểm ra</div><div>TP / SL</div><div>Kết quả</div>
      <div>±Điểm</div><div>Trạng thái</div>
    </div>""", unsafe_allow_html=True)

    # ── Table rows ──
    for t in trades:
        dc   = "#00e676" if t["direction"] == "LONG" else "#ff5252"
        is_open = t["status"] == "OPEN"

        if is_open:
            exit_pt   = current_price
            pts       = (current_price - t["entry"]) * (1 if t["direction"] == "LONG" else -1)
            exit_str  = f'{current_price:.2f} <span style="color:#ffd600">(LV)</span>'
            result_lbl= "MỞ"
            result_col= "#ffd600"
        else:
            exit_pt   = t.get("exit_price", t["entry"])
            pts       = (exit_pt - t["entry"]) * (1 if t["direction"] == "LONG" else -1)
            exit_str  = f'{exit_pt:.2f}'
            if pts > 0:
                result_lbl = "CHỐT LỜI"
                result_col = "#00e676"
            else:
                result_lbl = "CẮT LỖ"
                result_col = "#ff5252"

        pts_col = "#00e676" if pts >= 0 else "#ff5252"
        tp_str  = f'{t["tp"]:.2f}'
        sl_str  = f'{t["sl"]:.2f}'

        # Check if exit was at TP or SL
        if not is_open:
            tp_reached = abs(exit_pt - t["tp"]) < 0.15
            sl_reached = abs(exit_pt - t["sl"]) < 0.15
            tpsl_str = f'<span style="color:#00e676">{tp_str}</span> / <span style="color:#ff5252">{sl_str}</span>'
            if tp_reached: tpsl_str += ' <span style="color:#00e676">✓TP</span>'
            if sl_reached: tpsl_str += ' <span style="color:#ff5252">✓SL</span>'
        else:
            tpsl_str = f'<span style="color:#00e676">{tp_str}</span> / <span style="color:#ff5252">{sl_str}</span>'

        row_bg = "#0a1f12" if t["direction"] == "LONG" else "#1f0a0a"

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:40px 60px 70px 70px 70px 70px 80px 90px 80px;
                    gap:2px;padding:7px 10px;background:{row_bg};
                    border:1px solid #19213844;border-top:none;
                    font-family:'JetBrains Mono',monospace;font-size:10px;align-items:center">
          <div style="color:#64748b">#{t['id']}</div>
          <div style="color:#475569">{t['time']}</div>
          <div><b style="color:{dc}">{t['direction']}</b></div>
          <div style="color:#f1f5f9"><b>{t['entry']:.2f}</b></div>
          <div style="color:#94a3b8">{exit_str}</div>
          <div style="font-size:9px">{tpsl_str}</div>
          <div style="color:{result_col};font-weight:700;font-size:9px">{result_lbl}</div>
          <div style="color:{pts_col};font-weight:700;font-size:12px">{pts:+.2f} đ</div>
          <div style="color:{'#ffd600' if is_open else '#334155'};font-size:9px">{'● ĐANG MỞ' if is_open else f'✕ {t.get("close_time","--:--")}'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='border:1px solid #192138;border-top:none;border-radius:0 0 6px 6px;height:4px;background:#080c18'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ██ SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:#38bdf8;padding:6px 0 14px">⚡ VN30F TERMINAL</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">⚙️ CÀI ĐẶT</div>', unsafe_allow_html=True)
    symbol       = "VN30F1M"
    st.markdown('<div style="background:#0f1626;border:1px solid #192138;border-radius:6px;padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:12px;color:#38bdf8;font-weight:700">📌 VN30F1M</div>' , unsafe_allow_html=True)
    auto_refresh = st.toggle("🔄 Tự động cập nhật", value=True)
    refresh_sec  = st.slider("Chu kỳ (giây)", 10, 120, 30) if auto_refresh else 30

    st.markdown('<div class="sec-hdr" style="margin-top:14px">📐 QUẢN LÝ RỦI RO</div>', unsafe_allow_html=True)
    lot_size  = st.number_input("Số hợp đồng", min_value=1, max_value=50, value=1)
    tp_points = st.number_input("TP (điểm)", min_value=1.0, max_value=50.0, value=8.0, step=0.5)
    sl_points = st.number_input("SL (điểm)", min_value=1.0, max_value=30.0, value=4.0, step=0.5)
    st.markdown(f'<div style="color:#ffd600;font-family:JetBrains Mono,monospace;font-size:11px">R:R = 1 : {tp_points/sl_points:.1f}</div>', unsafe_allow_html=True)

    # ────────── INDICATOR TOGGLES ──────────
    st.markdown('<div class="sec-hdr" style="margin-top:16px">📊 BẬT / TẮT CHỈ BÁO</div>', unsafe_allow_html=True)

    st.markdown('<div class="toggle-title">OVERLAY (nằm trên chart)</div>', unsafe_allow_html=True)
    show_ema9  = st.checkbox("🟡 EMA 9",             value=True,  key="t_ema9")
    show_ema21 = st.checkbox("🔵 EMA 21",            value=True,  key="t_ema21")
    show_ema50 = st.checkbox("🟣 EMA 50",            value=True,  key="t_ema50")
    show_bb    = st.checkbox("⬜ Bollinger Bands",   value=True,  key="t_bb")
    show_pivot = st.checkbox("📏 Pivot / S&R",       value=False, key="t_pivot")
    show_tpsl  = st.checkbox("🎯 Đường TP / SL",     value=True,  key="t_tpsl")
    st.markdown('<div class="toggle-title" style="margin-top:8px">MARKERS TÍN HIỆU</div>', unsafe_allow_html=True)
    show_sigs  = st.checkbox("⬆⬇ Hiện markers chart",value=True,  key="t_sigs")

    st.markdown('<div class="toggle-title" style="margin-top:8px">PANELS PHỤ (bên dưới)</div>', unsafe_allow_html=True)
    show_vol   = st.checkbox("📊 Volume",            value=True,  key="t_vol")
    show_macd  = st.checkbox("📈 MACD",              value=True,  key="t_macd")
    show_rsi   = st.checkbox("📉 RSI (14)",          value=True,  key="t_rsi")
    show_stoch = st.checkbox("〰 Stochastic",        value=False, key="t_stoch")
    show_adxp  = st.checkbox("📐 ADX / DI+DI-",     value=False, key="t_adxp")

    SHOW = {
        "ema9":show_ema9, "ema21":show_ema21, "ema50":show_ema50,
        "bb":show_bb, "pivot":show_pivot, "tp_sl":show_tpsl,
        "signals_on_chart":show_sigs,
        "volume":show_vol, "macd":show_macd, "rsi":show_rsi,
        "stoch":show_stoch, "adx_panel":show_adxp,
    }

    st.markdown("---")
    col_clr1, col_clr2 = st.columns(2)
    if col_clr1.button("🗑️ Xóa tín hiệu", use_container_width=True):
        st.session_state.signal_history = []; st.session_state.prev_sig_keys = set(); st.rerun()
    if col_clr2.button("🗑️ Xóa lệnh", use_container_width=True):
        st.session_state.trade_history = []; st.rerun()
    st.markdown('<div style="font-size:10px;color:#334155;font-family:JetBrains Mono,monospace;margin-top:6px">📡 Chart: TradingView/HNX (thật)<br>Data: xnoapi → HNX Direct → Proxy<br>Fallback mô phỏng nếu ngoài giờ</div>', unsafe_allow_html=True)

    # ── DEBUG: Trạng thái nguồn dữ liệu ──
    src = st.session_state.get("data_source", "...")
    src_color = "#00e676" if src.startswith("✅") else "#ff5252" if src.startswith("⚠️") else "#ffd600"
    st.markdown(f'<div style="margin-top:8px;padding:6px 8px;background:#0a1218;border:1px solid #192138;border-radius:6px;font-family:JetBrains Mono,monospace;font-size:10px"><div style="color:#334155;margin-bottom:2px">NGUỒN DỮ LIỆU</div><div style="color:{src_color}">{src}</div></div>', unsafe_allow_html=True)

    errs = st.session_state.get("data_errors", [])
    if errs:
        with st.expander("🔍 Chi tiết lỗi", expanded=False):
            for i, e in enumerate(errs, 1):
                st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#ff7043;margin-bottom:4px">{i}. {e}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# AUTO-REFRESH
# ══════════════════════════════════════════════════════
if auto_refresh:
    if (datetime.now() - st.session_state.last_refresh).seconds >= refresh_sec:
        st.session_state.seed = random.randint(0, 9999)
        st.session_state.last_refresh = datetime.now()


# ══════════════════════════════════════════════════════
# DATA + SIGNALS
# ══════════════════════════════════════════════════════
df1 = add_indicators(fetch_ohlcv(symbol, 1, 300))
df5 = add_indicators(fetch_ohlcv(symbol, 5, 200))

current_price = float(df1["close"].iloc[-1])
prev_close    = float(df1["close"].iloc[-2])
regime1       = detect_signals(df1)
regime5       = detect_signals(df5)

push_signals(regime1, "1P", current_price)
push_signals(regime5, "5P", current_price)

chart_sigs1 = scan_chart_signals(df1, lookback=100) if show_sigs else []
chart_sigs5 = scan_chart_signals(df5, lookback=100) if show_sigs else []


# ══════════════════════════════════════════════════════
# ██ HEADER
# ══════════════════════════════════════════════════════
price_chg = current_price - prev_close; pct_chg = price_chg / prev_close * 100
up = price_chg >= 0

h1,h2,h3,h4,h5,h6 = st.columns([2.2,1.5,1.3,1.3,1.3,1.3])

h1.markdown(f"""
<div class="metric-box">
  <div class="metric-label">{symbol}</div>
  <div class="metric-value white" style="font-size:24px">{current_price:.2f}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{'#00e676' if up else '#ff5252'}">
    {'▲' if up else '▼'} {price_chg:+.2f} ({pct_chg:+.2f}%)</div>
</div>""", unsafe_allow_html=True)

rc5 = {"UPTREND":"#00e676","DOWNTREND":"#ff5252","SIDEWAY":"#ffd600"}.get(regime5["regime"],"#64748b")
h2.markdown(f'<div class="metric-box"><div class="metric-label">Xu hướng 5P</div><div class="metric-value" style="color:{rc5};font-size:13px">{regime5["regime"]}</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">ADX {regime5["adx"]:.1f} · {regime5["strength"]}</div></div>', unsafe_allow_html=True)

rsi_col = "green" if regime1["rsi"]<40 else ("red" if regime1["rsi"]>60 else "yellow")
h3.markdown(f'<div class="metric-box"><div class="metric-label">RSI 14</div><div class="metric-value {rsi_col}">{regime1["rsi"]:.1f}</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">{"Quá bán" if regime1["rsi"]<30 else "Quá mua" if regime1["rsi"]>70 else "Trung tính"}</div></div>', unsafe_allow_html=True)

bull = regime1["ema9"] > regime1["ema21"]
h4.markdown(f'<div class="metric-box"><div class="metric-label">EMA 9/21</div><div class="metric-value {"green" if bull else "red"}" style="font-size:13px">{"BULL ▲" if bull else "BEAR ▼"}</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">{regime1["ema9"]:.1f}/{regime1["ema21"]:.1f}</div></div>', unsafe_allow_html=True)

h5.markdown(f'<div class="metric-box"><div class="metric-label">DI+ / DI−</div><div class="metric-value" style="font-size:13px"><span class="green">{regime1["di_pos"]:.1f}</span>/<span class="red">{regime1["di_neg"]:.1f}</span></div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">ADX {regime1["adx"]:.1f}</div></div>', unsafe_allow_html=True)

vr = float(df1["volume"].iloc[-1]) / max(float(df1["vol_ma"].iloc[-1]),1)
h6.markdown(f'<div class="metric-box"><div class="metric-label">Volume</div><div class="metric-value {"green" if vr>1.5 else "yellow"}" style="font-size:14px">{vr:.1f}× MA</div><div style="font-size:10px;color:#475569;font-family:JetBrains Mono,monospace">{int(df1["volume"].iloc[-1]):,}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Regime banners ──
def regime_banner(r, label):
    regime = r["regime"]
    css  = {"UPTREND":"uptrend","DOWNTREND":"downtrend","SIDEWAY":"sideway"}.get(regime,"sideway")
    icon = {"UPTREND":"🚀","DOWNTREND":"💥","SIDEWAY":"🔄"}.get(regime,"")
    desc = {"UPTREND":"Ưu tiên BUY – pullback về EMA21, ride trend","DOWNTREND":"Ưu tiên SELL – hồi kháng cự rồi SHORT","SIDEWAY":"Range trade – canh BB biên, chờ Breakout xác nhận"}.get(regime,"")
    return f'<div class="signal-card {css}">{icon} [{label}] {regime} — {r["strength"]}<br><span style="font-size:10px;font-weight:400;letter-spacing:0;opacity:0.8">{desc}</span></div>'

cr1, cr5 = st.columns(2)
with cr1: st.markdown(regime_banner(regime1,"KHUNG 1 PHÚT"), unsafe_allow_html=True)
with cr5: st.markdown(regime_banner(regime5,"KHUNG 5 PHÚT"), unsafe_allow_html=True)

# ── Live signal cards ──
all_sigs = [(k,"1P") for k in regime1["signals"]] + [(k,"5P") for k in regime5["signals"]]
if all_sigs:
    st.markdown('<div class="sec-hdr" style="margin-top:8px">🎯 TÍN HIỆU ĐANG KÍCH HOẠT</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(all_sigs),5))
    for idx,(key,tf) in enumerate(all_sigs[:5]):
        d = SIGNAL_DEFS.get(key,{})
        color = d.get("color","#64748b")
        cols[idx].markdown(f"""
        <div style="background:#0f1626;border:1px solid {color}44;border-top:2px solid {color};border-radius:6px;padding:8px 10px;font-family:'JetBrains Mono',monospace;font-size:10px">
          <div style="color:{color};font-weight:700;font-size:12px">{d.get('icon','')} {d.get('action','')}</div>
          <div style="color:#64748b;margin-top:2px">[{tf}] {d.get('label','')}</div>
          <div style="color:#334155;margin-top:3px">{current_price:.2f}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ██ MAIN: CHART + TRADE PANEL
# ══════════════════════════════════════════════════════
chart_col, right_col = st.columns([3.1, 1.1])

with chart_col:
    tab1m, tab5m = st.tabs(["📊 Biểu đồ 1 Phút", "📊 Biểu đồ 5 Phút"])

    def build_lwc(df_src, label, tf_min):
        """Render Lightweight Charts (open-source, không cần login) với data Python."""
        rows = df_src.dropna(subset=["open","high","low","close"]).iloc[-200:]
        candles = []
        for ts, row in rows.iterrows():
            t = int(ts.timestamp())
            candles.append({"time": t, "open": round(float(row["open"]),2),
                            "high": round(float(row["high"]),2),
                            "low": round(float(row["low"]),2),
                            "close": round(float(row["close"]),2)})
        vols = []
        for ts, row in rows.iterrows():
            t = int(ts.timestamp())
            color = "rgba(0,230,118,0.5)" if float(row["close"]) >= float(row["open"]) else "rgba(255,82,82,0.5)"
            vols.append({"time": t, "value": int(row["volume"]), "color": color})
        ema9_data  = [{"time": int(ts.timestamp()), "value": round(float(v),2)}
                      for ts, v in rows["ema9"].dropna().items()]
        ema21_data = [{"time": int(ts.timestamp()), "value": round(float(v),2)}
                      for ts, v in rows["ema21"].dropna().items()]
        ema50_data = [{"time": int(ts.timestamp()), "value": round(float(v),2)}
                      for ts, v in rows["ema50"].dropna().items()]
        import json as _json
        c_js = _json.dumps(candles)
        v_js = _json.dumps(vols)
        e9_js = _json.dumps(ema9_data)
        e21_js = _json.dumps(ema21_data)
        e50_js = _json.dumps(ema50_data)
        last_p = candles[-1]["close"] if candles else 0
        last_t = label

        html = f"""
        <div id="lw_chart_{tf_min}" style="position:relative;width:100%;height:480px;background:#080c18;border-radius:8px;overflow:hidden;">
          <div style="position:absolute;top:8px;left:12px;z-index:10;font-family:JetBrains Mono,monospace;font-size:11px;color:#475569">{last_t} · <span id="lw_price_{tf_min}" style="color:#f1f5f9;font-weight:700">{last_p:.2f}</span></div>
        </div>
        <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
        <script>
        (function() {{
          var container = document.getElementById('lw_chart_{tf_min}');
          var chart = LightweightCharts.createChart(container, {{
            width: container.offsetWidth, height: 480,
            layout: {{ background: {{ color: '#080c18' }}, textColor: '#64748b' }},
            grid: {{ vertLines: {{ color: '#192138' }}, horzLines: {{ color: '#192138' }} }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{ borderColor: '#192138' }},
            timeScale: {{ borderColor: '#192138', timeVisible: true, secondsVisible: false }},
          }});
          var candleSeries = chart.addCandlestickSeries({{
            upColor: '#00e676', downColor: '#ff5252',
            borderUpColor: '#00e676', borderDownColor: '#ff5252',
            wickUpColor: '#00e676', wickDownColor: '#ff5252',
          }});
          candleSeries.setData({c_js});
          var ema9 = chart.addLineSeries({{ color: '#f59e0b', lineWidth: 1.5, title: 'EMA9' }});
          ema9.setData({e9_js});
          var ema21 = chart.addLineSeries({{ color: '#38bdf8', lineWidth: 1.5, title: 'EMA21' }});
          ema21.setData({e21_js});
          var ema50 = chart.addLineSeries({{ color: '#a78bfa', lineWidth: 1.5, title: 'EMA50' }});
          ema50.setData({e50_js});
          chart.timeScale().fitContent();
          // Update price label on crosshair
          chart.subscribeCrosshairMove(function(p) {{
            if (p.seriesData && p.seriesData.get(candleSeries)) {{
              var d = p.seriesData.get(candleSeries);
              var el = document.getElementById('lw_price_{tf_min}');
              if (el) el.innerText = d.close.toFixed(2);
            }}
          }});
          // Resize
          new ResizeObserver(function() {{ chart.applyOptions({{ width: container.offsetWidth }}); }}).observe(container);
        }})();
        </script>"""
        st.components.v1.html(html, height=490, scrolling=False)

    with tab1m:
        build_lwc(df1, f"VN30F1M · 1 Phút · {datetime.now().strftime('%H:%M:%S')}", 1)
        st.plotly_chart(build_chart(df1, f"VN30F1M · Indicators · 1P", SHOW, chart_sigs1),
                        use_container_width=True, config={"displayModeBar":False})
    with tab5m:
        build_lwc(df5, f"VN30F1M · 5 Phút · {datetime.now().strftime('%H:%M:%S')}", 5)
        st.plotly_chart(build_chart(df5, f"VN30F1M · Indicators · 5P", SHOW, chart_sigs5),
                        use_container_width=True, config={"displayModeBar":False})

with right_col:
    st.markdown('<div class="sec-hdr">🔫 VÀO LỆNH NHANH</div>', unsafe_allow_html=True)
    entry_price = st.number_input("Giá vào", value=float(f"{current_price:.2f}"), step=0.1, format="%.2f")
    rr = tp_points / sl_points
    st.markdown(f"""
    <div style="background:#0f1626;border:1px solid #192138;border-radius:6px;padding:9px;font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:8px">
      <div style="color:#334155;margin-bottom:4px">Preview LONG:</div>
      <div>🟢 TP <b style="color:#00e676">{entry_price+tp_points:.2f}</b> &nbsp;🔴 SL <b style="color:#ff5252">{entry_price-sl_points:.2f}</b></div>
      <div style="color:#ffd600;margin-top:3px">R:R = 1:{rr:.1f} · {lot_size}HĐ</div>
    </div>""", unsafe_allow_html=True)
    note = st.text_input("Ghi chú", placeholder="EMA cross, BB bounce...", label_visibility="collapsed")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🟢 LONG",  use_container_width=True):
            add_trade("LONG",  entry_price, round(entry_price+tp_points,2), round(entry_price-sl_points,2), lot_size, note); st.success("✅ Vào LONG!"); st.rerun()
    with c2:
        if st.button("🔴 SHORT", use_container_width=True):
            add_trade("SHORT", entry_price, round(entry_price-tp_points,2), round(entry_price+sl_points,2), lot_size, note); st.error("✅ Vào SHORT!"); st.rerun()

    st.markdown('<div class="sec-hdr" style="margin-top:12px">📋 LỊCH SỬ LỆNH</div>', unsafe_allow_html=True)
    render_trades(current_price)


# ══════════════════════════════════════════════════════
# ██ SIGNAL HISTORY (full width)
# ══════════════════════════════════════════════════════
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
total_sigs = len(st.session_state.signal_history)
buy_count  = sum(1 for s in st.session_state.signal_history if s["action"]=="BUY")
sell_count = sum(1 for s in st.session_state.signal_history if s["action"]=="SELL")

with st.expander(
    f"📡 LỊCH SỬ TÍN HIỆU  ·  {total_sigs} tín hiệu  ·  🟢 {buy_count} BUY  ·  🔴 {sell_count} SELL",
    expanded=True
):
    render_signal_history()


# ══════════════════════════════════════════════════════
# ██ TRADE EXECUTION TABLE (full width)
# ══════════════════════════════════════════════════════
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
total_trades  = len(st.session_state.trade_history)
closed_trades = sum(1 for t in st.session_state.trade_history if t["status"] == "CLOSED")
open_trades   = total_trades - closed_trades

with st.expander(
    f"📋 BẢNG LỆNH ĐÃ THỰC HIỆN  ·  {total_trades} lệnh  ·  🟡 {open_trades} đang mở  ·  ✅ {closed_trades} đã đóng",
    expanded=True
):
    render_trade_table(current_price)


# ══════════════════════════════════════════════════════
# ██ REFERENCE GUIDE
# ══════════════════════════════════════════════════════
with st.expander("📘 HƯỚNG DẪN ĐỌC TÍN HIỆU & CHIẾN LƯỢC"):
    st.markdown("""
| Tín hiệu | Điều kiện | Hành động |
|-----------|-----------|-----------|
| 🟢 **EMA 9×21 Cắt Lên** | EMA9 vừa cắt lên EMA21 | BUY |
| 🔴 **EMA 9×21 Cắt Xuống** | EMA9 vừa cắt xuống EMA21 | SELL |
| 🟢 **EMA Xếp BULL** | EMA9 > EMA21 > EMA50 + UPTREND | BUY |
| 💎 **RSI Quá Bán** | RSI < 30 | BUY |
| 🔥 **RSI Quá Mua** | RSI > 70 | SELL |
| 📈 **MACD Cắt Lên** | Histogram âm → dương | BUY |
| 🚀 **BB Break Up** | Giá > BB Upper | BUY momentum |
| 🟢 **BB Bounce Up** | Giá hồi từ BB Lower | BUY reversion |
| ⚡ **BB Squeeze** | BB Width < p15 lịch sử | WATCH – chờ breakout |
| 📊 **Volume Spike** | Vol > 2× Vol MA | WATCH – xác nhận tín hiệu |

**Marker trên chart:** ▲ = BUY signal · ▼ = SELL signal · ◆ = WATCH

**Chiến lược theo regime:**
- 🔄 **SIDEWAY** (ADX<22): Canh BB Lower mua, BB Upper bán. SL 2-3 điểm.
- 🚀 **UPTREND** (DI+>DI-): Chỉ LONG, pullback về EMA21. Ride trend.
- 💥 **DOWNTREND** (DI->DI+): Chỉ SHORT, hồi về EMA21. Ride trend.
- ⚡ **BB Squeeze**: Chờ Vol Spike xác nhận hướng → vào lệnh.
    """)


# ══════════════════════════════════════════════════════
# ██ FOOTER
# ══════════════════════════════════════════════════════
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
fl, fr = st.columns([4,1])
fl.markdown(f'<div style="font-size:10px;color:#192138;font-family:JetBrains Mono,monospace">VN30F Terminal v3 · DNSE/Vietcap/TCBS · {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</div>', unsafe_allow_html=True)
if auto_refresh:
    rem = max(0, refresh_sec-(datetime.now()-st.session_state.last_refresh).seconds)
    fr.markdown(f'<div style="font-size:10px;color:#38bdf8;font-family:JetBrains Mono,monospace;text-align:right">🔄 {rem}s</div>', unsafe_allow_html=True)
    time.sleep(1)
    st.rerun()
