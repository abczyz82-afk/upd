import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from collections import defaultdict

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="AI Trading Dashboard v3 📊",
    page_icon="📊"
)

st.markdown("""
<style>
div[data-testid="metric-container"] > div {
    background: rgba(255,255,255,0.04);
    border-radius: 10px; padding: 8px;
    border: 1px solid rgba(255,255,255,0.09);
}
.sig-card {
    border-radius: 0 10px 10px 0;
    padding: 8px 14px; margin: 4px 0;
    background: rgba(255,255,255,0.04);
}
.regime-banner {
    border-radius: 12px; padding: 14px 20px;
    margin-bottom: 16px; display: flex;
    align-items: center; gap: 16px;
}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
VN30_TICKERS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","NVL","PDR","PLX","PNJ","POW","SAB","SSI",
    "STB","TCB","TPB","VCB","VHM","VIC","VJC","VNM","VPB","VRE",
]

_FULL_STOCK_UNIVERSE = sorted(set([
    "VIX","VND","SHB","GEX","HPG","SSI","MSB","MBB","CII","NVL",
    "PC1","BSR","DXG","FPT","ACB","VCI","HCM","PDR","CTG","HSG",
    "VCB","VPB","HDB","VRE","TCB","VHM","POW","BID","DIG","MWG",
    "VNM","EIB","TPB","KDH","STB","ORS","TCH","VIC","PLX","VSC",
    "PVD","EVF","VCG","VCK","IJC","HHV","GVR","GEL","PVT","TCX",
    "DXS","DPM","MSN","HDG","VIB","KBC","VPI","NLG","PAN","GEE",
    "VPX","DCM","DGC","HHS","HAG","HDC","CTS","PET","VJC","OCB",
    "PNJ","SAB","VTP","DGW","GMD","CTR","REE","CTD","DSE","SHS",
    "CEO","MBS","PVS","PVC","AGR","LPB","EVG","VHC","FRT","DHC",
    "BSI","KSB","AMD","DBC","HAH","VOS","GIL","TLG","TDM","SBT",
    "MPC","PHR","HCT","TLH","VGC","HVN","SKG","BWE","NT2","PGD",
]))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Cấu hình Hệ thống")
ai_provider = st.sidebar.radio("Chọn nhà cung cấp AI:", ["🟠 Claude (Anthropic)", "🔵 Gemini (Google)"])
api_key = st.sidebar.text_input("Nhập API Key:", type="password")
if not api_key:
    st.sidebar.warning("⚠️ Nhập API Key để bật Trợ lý AI.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**📊 v3.0 — 16 Phương Pháp:**\n"
    "- 🔬 **ICT** — OB · FVG · BOS · CHoCH · OTE\n"
    "- 📊 **VSA** — Stopping · Climax · No Demand/Supply\n"
    "- 🕯️ **PA** — S/R · Inside Bar · Pin Bar · Trend\n"
    "- 💧 **Overflow** — Vol · BB · Breakout · Gap\n"
    "- 🐋 **Smart Money** — OBV · CMF · MFI · A/D\n"
    "- 🌀 **Wyckoff** — SC · Spring · SOS · LPSY\n"
    "- 🔀 **Divergence** — Regular + Hidden RSI\n"
    "- 🏔️ **Volume Profile** — POC · VAH · VAL · HVN\n"
    "- 🎯 **Supply & Demand** — Base+Rally/Drop Zones\n"
    "- 💹 **VWAP** — Rolling 20 · ±1σ ±2σ Bands\n"
    "- ⚡ **TTM Squeeze** — BB vs Keltner Compression\n"
    "- 🌊 **Heikin-Ashi** — Trend Filter\n"
    "- 🔴🟢 **Elder Impulse** — EMA13 + MACD Go/No-Go\n"
    "- 🛑 **Parabolic SAR** — Trend + Trailing Stop\n"
    "- 📉 **StochRSI** — Stochastic RSI K/D\n"
    "- 📏 **ATR Setup** — Auto SL/TP/R:R Calculator\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER & TABS
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 AI Trading Dashboard v3 — 16 Phương Pháp Phân Tích")
st.markdown(
    "Tích hợp **ICT · VSA · PA · Overflow · Smart Money · Wyckoff · Divergence · "
    "Volume Profile · VWAP · TTM Squeeze · Heikin-Ashi · Elder Impulse · PSAR · S&D · StochRSI · ATR**"
    " — Phân tích bởi **Claude** hoặc **Gemini**."
)
_tab1, _tab2 = st.tabs(["📊 Phân Tích Đơn Lẻ", "🔍 Quét Toàn Thị Trường"])

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_clean_stock_data(symbol: str) -> pd.DataFrame | None:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=220)

    if symbol.startswith("VN30F"):
        try:
            url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/derivative"
                   f"?from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
                   f"&symbol={symbol}&resolution=1D")
            resp = requests.get(url, timeout=10).json()
            if "t" in resp and len(resp["t"]) > 0:
                df = pd.DataFrame({"time": pd.to_datetime(resp["t"], unit="s"),
                                   "open": resp["o"], "high": resp["h"],
                                   "low": resp["l"], "close": resp["c"], "volume": resp["v"]})
                df["time"] = df["time"].dt.strftime("%Y-%m-%d")
                return df
        except Exception:
            pass

    try:
        url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
               f"?from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
               f"&symbol={symbol}&resolution=1D")
        resp = requests.get(url, timeout=10).json()
        if "t" in resp and len(resp.get("t", [])) > 0:
            df = pd.DataFrame({"time": pd.to_datetime(resp["t"], unit="s"),
                               "open": resp["o"], "high": resp["h"],
                               "low": resp["l"], "close": resp["c"], "volume": resp["v"]})
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass

    try:
        url = (f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/{symbol}/"
               f"bars-long-term?resolution=D&type=stock"
               f"&to={int(end_date.timestamp())}&countBack=200")
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=12).json()
        bars = resp.get("data", [])
        if bars:
            df = pd.DataFrame(bars)
            df = df.rename(columns={"tradingDate": "time", "open": "open", "high": "high",
                                    "low": "low", "close": "close", "volume": "volume"})
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
            return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        pass

    try:
        str_start = start_date.strftime("%Y-%m-%d")
        str_end   = end_date.strftime("%Y-%m-%d")
        url = (f"https://iboard-query.ssi.com.vn/v2/stock/historical-price"
               f"?symbol={symbol}&fromDate={str_start}&toDate={str_end}&offset=0&limit=200")
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10).json()
        data = resp.get("data", {}).get("items", [])
        if data:
            df = pd.DataFrame(data)
            df = df.rename(columns={"tradingDate": "time", "openPrice": "open",
                                    "highPrice": "high", "lowPrice": "low",
                                    "closePrice": "close", "totalMatchVolume": "volume"})
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
            return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        pass

    st.error(f"Không thể lấy dữ liệu cho mã **{symbol}**.")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def build_scan_universe() -> tuple:
    universe = set(_FULL_STOCK_UNIVERSE)
    source_note = f"Danh sách cố định ({len(universe)} mã)"
    try:
        resp = requests.get("https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/all",
                            timeout=15, headers={"User-Agent": "Mozilla/5.0"}).json()
        items = resp if isinstance(resp, list) else resp.get("listStock", resp.get("data", []))
        added = 0
        for item in (items or []):
            sym = str(item.get("ticker", item.get("symbol", item.get("code", "")))).upper().strip()
            if sym and 2 <= len(sym) <= 4 and sym.isalpha():
                universe.add(sym); added += 1
        if added > 0:
            source_note = f"TCBS + cố định ({len(universe)} mã)"
    except Exception:
        pass
    try:
        resp = requests.get(
            "https://services.entrade.com.vn/market-data/securities?type=stock&exchange=HOSE&size=3000",
            timeout=12, headers={"User-Agent": "Mozilla/5.0"}).json()
        items = resp if isinstance(resp, list) else resp.get("data", [])
        for item in (items or []):
            sym = str(item.get("symbol", item.get("ticker", ""))).upper().strip()
            if sym and 2 <= len(sym) <= 4 and sym.isalpha():
                universe.add(sym)
    except Exception:
        pass
    for exchange in ["HOSE", "HNX"]:
        try:
            url = f"https://iboard-query.ssi.com.vn/v2/stock/board-data/exchange?exchange={exchange}&size=3000"
            resp = requests.get(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}, timeout=15).json()
            items = resp if isinstance(resp, list) else resp.get("data", resp.get("items", []))
            for item in (items or []):
                if not isinstance(item, dict): continue
                sym = str(item.get("symbol", item.get("code", ""))).upper().strip()
                if sym and 2 <= len(sym) <= 4 and sym.isalpha():
                    universe.add(sym)
        except Exception:
            pass
    return sorted(universe), f"{source_note} — HOSE · HNX · UPCOM"


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIC INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def calc_ichimoku(df):
    hi, lo = df["high"], df["low"]
    tenkan = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
    kijun  = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou


def calc_fibonacci(df, lookback=60):
    window  = df.tail(lookback)
    high_val = window["high"].max(); low_val = window["low"].min()
    hi_pos   = int(window["high"].values.argmax()); lo_pos = int(window["low"].values.argmin())
    is_down  = (hi_pos > lo_pos)
    diff = max(high_val - low_val, high_val * 0.01)
    if is_down:
        levels = {
            "0.0% (Đỉnh)": high_val, "23.6%": high_val - 0.236*diff,
            "38.2%": high_val - 0.382*diff, "50.0%": high_val - 0.500*diff,
            "61.8% ✨": high_val - 0.618*diff, "65.0% 🏅": high_val - 0.650*diff,
            "78.6%": high_val - 0.786*diff, "100.0% (Đáy)": low_val,
            "127.2% 📉": low_val - 0.272*diff, "161.8% 📉": low_val - 0.618*diff,
        }
    else:
        levels = {
            "0.0% (Đáy)": low_val, "23.6%": low_val + 0.236*diff,
            "38.2%": low_val + 0.382*diff, "50.0%": low_val + 0.500*diff,
            "61.8% ✨": low_val + 0.618*diff, "65.0% 🏅": low_val + 0.650*diff,
            "78.6%": low_val + 0.786*diff, "100.0% (Đỉnh)": high_val,
            "127.2% 📈": high_val + 0.272*diff, "161.8% 📈": high_val + 0.618*diff,
        }
    levels["_high"] = high_val; levels["_low"] = low_val
    levels["_is_downtrend"] = is_down; levels["_diff"] = diff
    return levels


def calc_mcdx(df):
    macd_hist = MACD(close=df["close"]).macd_diff()
    rsi       = RSIIndicator(close=df["close"], window=14).rsi()
    def _n(s):
        mn, mx = s.min(), s.max()
        return pd.Series(0, index=s.index) if mx == mn else (s - mn)/(mx - mn)*2 - 1
    return ((_n(macd_hist) + _n(rsi - 50)) / 2).rename("MCDX")


def calc_adx(df, window=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr   = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    dm_p = high.diff().clip(lower=0); dm_m = (-low.diff()).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_m, 0); dm_m = dm_m.where(dm_m > dm_p, 0)
    atr  = tr.ewm(span=window, min_periods=window, adjust=False).mean()
    dip  = 100 * dm_p.ewm(span=window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    dim  = 100 * dm_m.ewm(span=window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    df["ADX"]      = dx.ewm(span=window, min_periods=window, adjust=False).mean()
    df["DI_Plus"]  = dip; df["DI_Minus"] = dim
    return df


def calc_ema_cross(df):
    df["EMA5"]  = df["close"].ewm(span=5,  adjust=False).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    return df


def calc_smart_money(df):
    close, high, low, vol, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    obv = [0]
    for i in range(1, len(df)):
        obv.append(obv[-1] + vol.iloc[i] if close.iloc[i] > close.iloc[i-1]
                   else (obv[-1] - vol.iloc[i] if close.iloc[i] < close.iloc[i-1] else obv[-1]))
    df["OBV"]  = pd.array(obv, dtype=float)
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    df["CMF"]  = (mfm * vol).rolling(20).sum() / vol.rolling(20).sum()
    tp = (high + low + close) / 3
    pos_mf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    df["MFI"]  = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))
    df["Up_Vol"]   = np.where(close >= open_, vol, 0).astype(float)
    df["Down_Vol"] = np.where(close <  open_, vol, 0).astype(float)
    buy10 = pd.Series(df["Up_Vol"]).rolling(10).sum()
    sell10 = pd.Series(df["Down_Vol"]).rolling(10).sum()
    df["Buy_Ratio"] = buy10 / (buy10 + sell10).replace(0, np.nan)
    df["Vol_MA20"]  = vol.rolling(20).mean()
    df["Vol_Ratio"] = vol / df["Vol_MA20"]
    df["Body_Ratio"] = (abs(close - open_) / (high - low).replace(0, np.nan)).fillna(0)
    ad_mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    df["AD_Line"] = (ad_mfm * vol).cumsum()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# V3.0 — NEW INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def calc_vwap(df):
    """Rolling 20-day VWAP with ±1σ and ±2σ bands."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["VWAP"]    = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    roll_tpv      = (tp * df["volume"]).rolling(20).sum()
    roll_v        = df["volume"].rolling(20).sum()
    df["VWAP20"]  = roll_tpv / roll_v.replace(0, np.nan)
    deviation     = (tp - df["VWAP20"]).abs().rolling(20).std()
    df["VWAP20_U1"] = df["VWAP20"] + deviation
    df["VWAP20_U2"] = df["VWAP20"] + 2 * deviation
    df["VWAP20_L1"] = df["VWAP20"] - deviation
    df["VWAP20_L2"] = df["VWAP20"] - 2 * deviation
    return df


def calc_ttm_squeeze(df):
    """TTM Squeeze: BB inside Keltner Channel = coiling."""
    if len(df) < 21:
        df["SQ_On"] = 0; df["SQ_Mom"] = 0.0; return df
    try:
        bb  = BollingerBands(close=df["close"], window=20, window_dev=2)
        bb_upper = bb.bollinger_hband(); bb_lower = bb.bollinger_lband()
        # Keltner Channel (EMA20 ± 1.5×ATR)
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        atr14 = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
        kc_upper = ema20 + 1.5 * atr14; kc_lower = ema20 - 1.5 * atr14
        sq = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        df["SQ_On"] = sq.astype(int)
        highest = df["high"].rolling(20).max(); lowest = df["low"].rolling(20).min()
        delta   = df["close"] - (highest + lowest) / 2 - df["close"].rolling(20).mean() / 2
        df["SQ_Mom"] = delta.ewm(span=5, adjust=False).mean()
    except Exception:
        df["SQ_On"] = 0; df["SQ_Mom"] = 0.0
    return df


def calc_volume_profile(df, bins=25):
    """Volume Profile: POC, VAH (70%), VAL, HVN, LVN."""
    if len(df) < 10:
        cp = df["close"].iloc[-1]
        return {"poc": cp, "vah": cp * 1.02, "val": cp * 0.98,
                "profile": [], "hvn_levels": [], "lvn_levels": []}
    p_min, p_max = df["low"].min(), df["high"].max()
    rng = p_max - p_min
    if rng <= 0:
        return {"poc": p_min, "vah": p_max, "val": p_min,
                "profile": [], "hvn_levels": [], "lvn_levels": []}
    edges   = np.linspace(p_min, p_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vols    = np.zeros(bins)
    for _, row in df.iterrows():
        lo, hi, v = row["low"], row["high"], row["volume"]
        cr = hi - lo if hi > lo else rng / bins
        for b in range(bins):
            ov = min(hi, edges[b + 1]) - max(lo, edges[b])
            if ov > 0:
                vols[b] += v * ov / cr
    poc_idx = int(np.argmax(vols)); poc = centers[poc_idx]
    total   = vols.sum(); target = total * 0.70
    sorted_b = np.argsort(vols)[::-1]
    va_vol, va_idx = 0, []
    for idx in sorted_b:
        va_idx.append(idx); va_vol += vols[idx]
        if va_vol >= target: break
    vah      = centers[max(va_idx)] if va_idx else p_max
    val      = centers[min(va_idx)] if va_idx else p_min
    vol_mean = vols.mean()
    hvn      = [centers[i] for i in range(bins) if vols[i] > vol_mean * 1.5]
    lvn      = [centers[i] for i in range(bins) if vols[i] < vol_mean * 0.4]
    profile  = [{"price": centers[i], "volume": vols[i], "pct": vols[i] / max(total, 1) * 100}
                for i in range(bins)]
    return {"poc": poc, "vah": vah, "val": val, "profile": profile,
            "hvn_levels": hvn[-3:], "lvn_levels": lvn[:3], "total_vol": total}


def calc_elder_impulse(df):
    """Elder Impulse: Green=bull, Red=bear, Blue=neutral (EMA13 + MACD hist slopes)."""
    if len(df) < 20:
        df["Elder"] = "neutral"; return df
    ema13      = df["close"].ewm(span=13, adjust=False).mean()
    ema_slope  = ema13.diff()
    hist       = MACD(close=df["close"]).macd_diff()
    hist_slope = hist.diff()
    colors = []
    for e, m in zip(ema_slope, hist_slope):
        if pd.isna(e) or pd.isna(m): colors.append("neutral")
        elif e > 0 and m > 0:        colors.append("bull")
        elif e < 0 and m < 0:        colors.append("bear")
        else:                        colors.append("neutral")
    df["Elder"] = colors
    return df


def calc_stoch_rsi(df):
    """StochRSI K and D lines (0–100)."""
    try:
        from ta.momentum import StochRSIIndicator
        srsi = StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
        df["StochRSI_K"] = srsi.stochrsi_k() * 100
        df["StochRSI_D"] = srsi.stochrsi_d() * 100
    except Exception:
        rsi_raw = RSIIndicator(close=df["close"], window=14).rsi()
        rsi_min = rsi_raw.rolling(14).min(); rsi_max = rsi_raw.rolling(14).max()
        k_raw = (rsi_raw - rsi_min) / (rsi_max - rsi_min + 1e-9) * 100
        df["StochRSI_K"] = k_raw.rolling(3).mean()
        df["StochRSI_D"] = df["StochRSI_K"].rolling(3).mean()
    return df


def calc_psar(df):
    """Parabolic SAR."""
    try:
        from ta.trend import PSARIndicator
        psar = PSARIndicator(high=df["high"], low=df["low"], close=df["close"],
                             step=0.02, max_step=0.2)
        df["PSAR"]      = psar.psar()
        df["PSAR_Up"]   = psar.psar_up()
        df["PSAR_Down"] = psar.psar_down()
    except Exception:
        df["PSAR"] = np.nan; df["PSAR_Up"] = np.nan; df["PSAR_Down"] = np.nan
    return df


def calc_atr_indicator(df, window=14):
    """ATR for risk management setup."""
    try:
        df["ATR14"] = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=window
        ).average_true_range()
    except Exception:
        df["ATR14"] = (df["high"] - df["low"]).rolling(window).mean()
    return df


def calc_market_regime(df):
    """Market Regime: Strong/Moderate Trend, Squeeze, Ranging."""
    if len(df) < 30:
        return {"regime": "UNKNOWN", "color": "#aaa", "icon": "❓",
                "score": 50, "desc": "Không đủ dữ liệu"}
    latest  = df.iloc[-1]
    adx_val = latest.get("ADX", 0) or 0
    di_p    = latest.get("DI_Plus", 0) or 0
    di_m    = latest.get("DI_Minus", 0) or 0
    try:
        bb    = BollingerBands(close=df["close"], window=20, window_dev=2)
        bw    = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
        bw_now = float(bw.iloc[-1]) if pd.notna(bw.iloc[-1]) else 5.0
        bw_avg = float(bw.rolling(50).mean().iloc[-1]) if len(df) >= 50 else bw_now
    except Exception:
        bw_now = bw_avg = 5.0

    if   adx_val > 35 and di_p > di_m:
        return {"regime": "📈 STRONG UPTREND",  "color": "#00C853", "icon": "🚀", "score": 88,
                "desc": f"ADX={adx_val:.1f}>35 · DI+>{di_m:.0f} — Trend tăng rất mạnh"}
    elif adx_val > 35:
        return {"regime": "📉 STRONG DOWNTREND","color": "#FF1744", "icon": "💥", "score": 12,
                "desc": f"ADX={adx_val:.1f}>35 · DI->DI+ — Trend giảm rất mạnh"}
    elif adx_val > 20 and di_p > di_m:
        return {"regime": "📈 UPTREND",          "color": "#69F0AE", "icon": "📈", "score": 68,
                "desc": f"ADX={adx_val:.1f}>20 · DI+>{di_m:.0f} — Xu hướng tăng"}
    elif adx_val > 20:
        return {"regime": "📉 DOWNTREND",        "color": "#FF9100", "icon": "📉", "score": 32,
                "desc": f"ADX={adx_val:.1f}>20 · DI->DI+ — Xu hướng giảm"}
    elif pd.notna(bw_now) and pd.notna(bw_avg) and bw_now < bw_avg * 0.65:
        return {"regime": "⚡ SQUEEZE — Sắp bứt phá","color": "#FFD740","icon": "⚡","score": 50,
                "desc": f"BB Width {bw_now:.1f}% (avg {bw_avg:.1f}%) — Thị trường đang nén, chuẩn bị bứt phá"}
    else:
        return {"regime": "↔️ RANGING / SIDEWAYS","color": "#90A4AE","icon": "↔️","score": 50,
                "desc": "Thị trường đi ngang — Ưu tiên mean-reversion, tránh trend-following"}


def calc_heikin_ashi(df):
    """Heikin-Ashi candles and consecutive trend count."""
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    df["HA_Close"] = ha_close; df["HA_Open"] = ha_open
    df["HA_High"]  = pd.concat([ha_open, ha_close, df["high"]], axis=1).max(axis=1)
    df["HA_Low"]   = pd.concat([ha_open, ha_close, df["low"]],  axis=1).min(axis=1)
    df["HA_Bull"]  = (ha_close >= ha_open).astype(int)
    direction   = 1 if df["HA_Bull"].iloc[-1] == 1 else -1
    consecutive = 0
    for i in range(len(df) - 1, -1, -1):
        if (df["HA_Bull"].iloc[i] == 1 and direction == 1) or \
           (df["HA_Bull"].iloc[i] == 0 and direction == -1):
            consecutive += 1
        else:
            break
    return df, consecutive, direction


def calc_wyckoff_events(df):
    """Detect Wyckoff events: SC, BC, Spring, AR, ST, SOS, LPSY, NoSupply."""
    if len(df) < 40:
        return {"events": [], "phase": "UNKNOWN",
                "phase_desc": "Không đủ dữ liệu", "score": 0}
    c   = df["close"].values; h = df["high"].values
    lo  = df["low"].values;   o = df["open"].values
    v   = df["volume"].values; tms = df["time"].values; n = len(df)
    vm20 = pd.Series(v).rolling(20).mean().values
    events = []; start = max(2, n - 55)

    for i in range(start, n):
        vm = float(vm20[i]) if not np.isnan(vm20[i]) and vm20[i] > 0 else float(v[i])
        if vm == 0: continue
        vr = v[i] / vm
        sp = (h[i] - lo[i]) / max(c[i], 1)
        is_red = c[i] < o[i]; is_grn = c[i] > o[i]

        if is_red and vr > 2.5 and sp > 0.025:
            events.append({"type": "SC", "emoji": "💥", "label": "Selling Climax",
                           "time": tms[i], "price": c[i], "bullish": True,
                           "desc": f"Vol x{vr:.1f} · Nến đỏ rộng → Đáy climax bán tiềm năng"})

        if is_grn and vr > 2.5 and sp > 0.025 and c[i] >= np.max(c[max(0, i - 20):i + 1]):
            events.append({"type": "BC", "emoji": "🔔", "label": "Buying Climax",
                           "time": tms[i], "price": c[i], "bullish": False,
                           "desc": f"Vol x{vr:.1f} · Đỉnh cao → Climax mua, cảnh báo đỉnh"})

        if i > start + 5:
            rl = float(np.min(lo[max(start, i - 10):i]))
            if lo[i] < rl * 0.996 and c[i] > rl and vr < 1.8 and sp > 0.008:
                events.append({"type": "Spring", "emoji": "🌱", "label": "Spring/Shakeout",
                               "time": tms[i], "price": c[i], "bullish": True,
                               "desc": f"Xuyên đáy {rl:,.0f} · Vol x{vr:.1f} thấp → Bẫy gấu, gom hàng"})

        if i > start + 5:
            rh = float(np.max(h[max(start, i - 20):i]))
            if is_grn and c[i] > rh * 0.995 and vr > 1.5 and sp > 0.012:
                events.append({"type": "SOS", "emoji": "🚀", "label": "Sign of Strength",
                               "time": tms[i], "price": c[i], "bullish": True,
                               "desc": f"Phá đỉnh {rh:,.0f} · Vol x{vr:.1f} → Bứt phá tích lũy Wyckoff"})

        if i > start + 5 and is_red and vr > 1.2:
            rh85 = float(np.percentile(h[max(start, i - 20):i], 85))
            if h[i] > rh85:
                events.append({"type": "LPSY", "emoji": "⚠️", "label": "LPSY (Cung Cuối)",
                               "time": tms[i], "price": c[i], "bullish": False,
                               "desc": f"Rally yếu vùng cao · Vol x{vr:.1f} → Phân phối cuối"})

        if is_red and vr < 0.55 and sp < 0.01:
            events.append({"type": "NoSup", "emoji": "🔬", "label": "No Supply",
                           "time": tms[i], "price": c[i], "bullish": True,
                           "desc": f"Giảm nhẹ Vol x{vr:.1f} thấp · Spread hẹp → Cung cạn kiệt"})

    seen = {}
    for e in events:
        seen[e["type"]] = e
    events = list(seen.values())[-10:]
    recent_types = [e["type"] for e in events[-4:]]
    bull_c = sum(1 for e in events[-4:] if e.get("bullish", False))
    score  = bull_c - (len(events[-4:]) - bull_c)

    if "SOS" in recent_types or "Spring" in recent_types:
        phase = "MARKUP 🚀 (Xong tích lũy)"
        phase_desc = "SOS/Spring phát hiện — Smart Money gom xong, chuẩn bị markup giá"
    elif "SC" in recent_types or "NoSup" in recent_types:
        phase = "ACCUMULATION 🏦 (Đang tích lũy)"
        phase_desc = "SC/No Supply — Tiền lớn bí mật gom hàng ở vùng giá thấp"
    elif "BC" in recent_types or "LPSY" in recent_types:
        phase = "DISTRIBUTION 📤 (Phân phối)"
        phase_desc = "BC/LPSY — Tiền lớn đang phân phối, xả hàng cho retail"
    else:
        phase = "UNDEFINED ❓"
        phase_desc = "Chưa xác định pha Wyckoff từ dữ liệu gần đây"
    return {"events": events, "phase": phase, "phase_desc": phase_desc, "score": score}


def calc_hidden_divergence(df, window=40):
    """Regular + Hidden Divergence detection on RSI.

    Fixes:
    - window tăng lên 40 để bắt được nhiều swing hơn
    - find_swings dùng range(3, n-3) tránh index out-of-range khi n nhỏ
    - nearest max_dist tăng lên 10 để map được swing giá ↔ RSI chính xác hơn
    - Tách riêng reg_bear và hid_bear (không dùng cùng elif) để phát hiện đồng thời
    - Thêm kiểm tra p1i != p2i để tránh cùng một swing
    """
    default = {"reg_bull": False, "reg_bear": False,
               "hid_bull": False, "hid_bear": False,
               "signals": [], "score": 0}
    if len(df) < 20 or "RSI" not in df.columns:
        return default

    # Dùng min(window, len(df)) để không bị lỗi khi dữ liệu ít
    use_window = min(window, len(df))
    recent = df.tail(use_window).reset_index(drop=True)
    prices = recent["close"].values
    rsi    = recent["RSI"].fillna(50).values
    n      = len(prices)

    if n < 7:
        return default

    def find_swings(arr, is_high, strength=1):
        """Tìm swing high/low với strength bước xác nhận (mặc định 1 nến 2 phía)."""
        out = []
        s = max(1, strength)
        for i in range(s * 2, n - s * 2):
            if is_high:
                if all(arr[i] > arr[i - k] for k in range(1, s + 1)) and \
                   all(arr[i] > arr[i + k] for k in range(1, s + 1)):
                    out.append((i, float(arr[i])))
            else:
                if all(arr[i] < arr[i - k] for k in range(1, s + 1)) and \
                   all(arr[i] < arr[i + k] for k in range(1, s + 1)):
                    out.append((i, float(arr[i])))
        return out

    # Thử strength=2 trước (ít nhiễu hơn), fallback strength=1 nếu không đủ swing
    ph = find_swings(prices, True,  2) or find_swings(prices, True,  1)
    pl = find_swings(prices, False, 2) or find_swings(prices, False, 1)
    rh = find_swings(rsi,    True,  1)
    rl = find_swings(rsi,    False, 1)

    def nearest_rsi_swing(p_idx, rsi_swings, max_dist=10):
        """Tìm RSI swing gần nhất với giá swing tại p_idx."""
        candidates = [(i, v) for i, v in rsi_swings if abs(i - p_idx) <= max_dist]
        if not candidates:
            return None
        # Ưu tiên swing gần nhất
        return min(candidates, key=lambda x: abs(x[0] - p_idx))

    signals = []
    reg_bull = reg_bear = hid_bull = hid_bear = False
    score = 0

    # ── Bearish divergences (từ swing HIGHS) ────────────────────────────────
    if len(ph) >= 2:
        for idx in range(len(ph) - 1, 0, -1):
            p2i, p2v = ph[idx]
            p1i, p1v = ph[idx - 1]
            if p1i == p2i:
                continue
            r1 = nearest_rsi_swing(p1i, rh)
            r2 = nearest_rsi_swing(p2i, rh)
            if r1 is None or r2 is None:
                continue
            # Regular Bearish: Giá HH nhưng RSI LH
            if p2v > p1v and r2[1] < r1[1] and not reg_bear:
                reg_bear = True
                score -= 2
                signals.append(("📉 Regular Bearish Div",
                                 f"Giá HH ({p2v:,.0f}>{p1v:,.0f}) nhưng RSI LH ({r2[1]:.1f}<{r1[1]:.1f}) → Đỉnh phân kỳ, cảnh báo đảo chiều giảm",
                                 "bearish"))
            # Hidden Bearish: Giá LH nhưng RSI HH
            if p2v < p1v and r2[1] > r1[1] and not hid_bear:
                hid_bear = True
                score -= 3
                signals.append(("🔒 Hidden Bearish Div",
                                 f"Giá LH ({p2v:,.0f}<{p1v:,.0f}) RSI HH ({r2[1]:.1f}>{r1[1]:.1f}) → Tiếp tục downtrend",
                                 "bearish"))
            if reg_bear and hid_bear:
                break

    # ── Bullish divergences (từ swing LOWS) ─────────────────────────────────
    if len(pl) >= 2:
        for idx in range(len(pl) - 1, 0, -1):
            p2i, p2v = pl[idx]
            p1i, p1v = pl[idx - 1]
            if p1i == p2i:
                continue
            r1 = nearest_rsi_swing(p1i, rl)
            r2 = nearest_rsi_swing(p2i, rl)
            if r1 is None or r2 is None:
                continue
            # Regular Bullish: Giá LL nhưng RSI HL
            if p2v < p1v and r2[1] > r1[1] and not reg_bull:
                reg_bull = True
                score += 2
                signals.append(("📈 Regular Bullish Div",
                                 f"Giá LL ({p2v:,.0f}<{p1v:,.0f}) nhưng RSI HL ({r2[1]:.1f}>{r1[1]:.1f}) → Đáy phân kỳ, tiềm năng đảo chiều tăng",
                                 "bullish"))
            # Hidden Bullish: Giá HL nhưng RSI LL
            if p2v > p1v and r2[1] < r1[1] and not hid_bull:
                hid_bull = True
                score += 3
                signals.append(("🔒 Hidden Bullish Div",
                                 f"Giá HL ({p2v:,.0f}>{p1v:,.0f}) RSI LL ({r2[1]:.1f}<{r1[1]:.1f}) → Tiếp tục uptrend mạnh",
                                 "bullish"))
            if reg_bull and hid_bull:
                break

    return {"reg_bull": reg_bull, "reg_bear": reg_bear,
            "hid_bull": hid_bull, "hid_bear": hid_bear,
            "signals": signals, "score": score}


def calc_supply_demand_zones(df, lookback=60):
    """Supply & Demand Zones: Base+Rally=Demand, Base+Drop=Supply."""
    if len(df) < 20:
        return [], []
    c = df["close"].values; h = df["high"].values
    lo = df["low"].values;  o = df["open"].values
    tms = df["time"].values; n = len(df)
    body_sz = np.abs(c - o)
    body_ma = pd.Series(body_sz).rolling(20).mean().values
    demand_zones, supply_zones = [], []
    start = max(3, n - lookback)
    for i in range(start, n - 3):
        bm = body_ma[i] if not np.isnan(body_ma[i]) and body_ma[i] > 0 else 1.0
        if body_sz[i] / bm >= 0.55:
            continue
        move_pct = abs(c[min(i + 3, n - 1)] - c[i]) / max(c[i], 1) * 100
        if move_pct < 1.5:
            continue
        z_top = max(h[max(0, i - 1)], h[i]); z_bot = min(lo[max(0, i - 1)], lo[i])
        if z_top <= z_bot:
            continue
        move = c[min(i + 3, n - 1)] - c[i]
        if move > 0:
            demand_zones.append({"top": z_top, "bottom": z_bot, "time": tms[i],
                                  "strength": move_pct, "fresh": lo[-1] > z_bot})
        else:
            supply_zones.append({"top": z_top, "bottom": z_bot, "time": tms[i],
                                  "strength": move_pct, "fresh": h[-1] < z_top})
    demand_zones = sorted(demand_zones, key=lambda x: -x["strength"])[:4]
    supply_zones = sorted(supply_zones, key=lambda x: -x["strength"])[:4]
    return demand_zones, supply_zones


# ─────────────────────────────────────────────────────────────────────────────
# ICT — Inner Circle Trader
# ─────────────────────────────────────────────────────────────────────────────
def calc_ict(df: pd.DataFrame) -> dict:
    empty = {"signals": [], "order_blocks": [], "fvg_list": [], "structure": "N/A",
             "ote_zone": None, "liq_highs": [], "liq_lows": [], "score": 0,
             "summary": "Không đủ dữ liệu", "bos_bullish": False,
             "bos_bearish": False, "choch": False, "swing_highs": [], "swing_lows": []}
    if len(df) < 20: return empty
    close = df["close"].values; high = df["high"].values
    low   = df["low"].values;   open_ = df["open"].values
    n = len(df); times = df["time"].values; signals = []

    swing_highs = []; swing_lows = []
    for i in range(2, n - 2):
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            swing_highs.append((i, high[i]))
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            swing_lows.append((i, low[i]))

    structure = "SIDEWAYS"; bos_bullish = bos_bearish = choch = False
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        lhh, phh = swing_highs[-1][1], swing_highs[-2][1]
        lll, pll = swing_lows[-1][1],  swing_lows[-2][1]
        if lhh > phh and lll > pll:
            structure = "UPTREND"
            if close[-1] > lhh:
                bos_bullish = True
                signals.append(("🟢 BOS Bullish", f"Phá đỉnh {lhh:,.0f} → Cấu trúc tăng xác nhận", "bullish"))
        elif lhh < phh and lll < pll:
            structure = "DOWNTREND"
            if close[-1] < lll:
                bos_bearish = True
                signals.append(("🔴 BOS Bearish", f"Phá đáy {lll:,.0f} → Cấu trúc giảm xác nhận", "bearish"))
        if structure == "UPTREND" and close[-1] < lll:
            choch = True
            signals.append(("⚡ CHoCH Bearish", f"Phá đáy swing {lll:,.0f} trong uptrend → Đảo chiều", "bearish"))
        elif structure == "DOWNTREND" and close[-1] > lhh:
            choch = True
            signals.append(("⚡ CHoCH Bullish", f"Phá đỉnh swing {lhh:,.0f} trong downtrend → Đảo chiều", "bullish"))

    order_blocks = []
    for i in range(max(0, n - 30), n - 3):
        if (close[i] < open_[i] and close[i+1] > open_[i+1] and close[i+2] > open_[i+2]
                and (high[i+2] - low[i]) / max(close[i], 1) > 0.015):
            order_blocks.append({"type": "bullish", "top": open_[i], "bottom": close[i],
                                  "time": times[i], "idx": i,
                                  "desc": f"Bullish OB {close[i]:,.0f}–{open_[i]:,.0f}"})
        if (close[i] > open_[i] and close[i+1] < open_[i+1] and close[i+2] < open_[i+2]
                and (high[i] - low[i+2]) / max(close[i], 1) > 0.015):
            order_blocks.append({"type": "bearish", "top": close[i], "bottom": open_[i],
                                  "time": times[i], "idx": i,
                                  "desc": f"Bearish OB {open_[i]:,.0f}–{close[i]:,.0f}"})
    order_blocks = order_blocks[-5:]
    cp = close[-1]
    for ob in order_blocks:
        tol = (ob["top"] - ob["bottom"]) * 0.5
        if ob["bottom"] - tol <= cp <= ob["top"] + tol:
            t_ = ob["type"]
            signals.append((f"🏦 {'Bullish' if t_=='bullish' else 'Bearish'} OB",
                            f"Giá {cp:,.0f} tại vùng {ob['bottom']:,.0f}–{ob['top']:,.0f}", t_))

    fvg_list = []
    for i in range(1, n - 1):
        if low[i+1] > high[i-1]:
            gap = (low[i+1] - high[i-1]) / close[i] * 100
            if gap > 0.3:
                fvg_list.append({"type": "bullish", "top": low[i+1], "bottom": high[i-1],
                                  "time": times[i], "pct": gap})
        if high[i+1] < low[i-1]:
            gap = (low[i-1] - high[i+1]) / close[i] * 100
            if gap > 0.3:
                fvg_list.append({"type": "bearish", "top": low[i-1], "bottom": high[i+1],
                                  "time": times[i], "pct": gap})
    fvg_list = fvg_list[-6:]
    for fvg in fvg_list[-3:]:
        if fvg["bottom"] <= cp <= fvg["top"]:
            t_ = fvg["type"]
            signals.append((f"{'📊' if t_=='bullish' else '📉'} FVG {t_.capitalize()}",
                            f"Lấp FVG {fvg['bottom']:,.0f}–{fvg['top']:,.0f} ({fvg['pct']:.1f}%)", t_))

    liq_highs = []; liq_lows = []; tol_pct = 0.002
    for i in range(n - 20, n - 1):
        if i < 0: continue
        for j in range(i + 1, min(i + 8, n - 1)):
            if abs(high[i] - high[j]) / max(high[i], 1) < tol_pct:
                liq_highs.append(round((high[i] + high[j]) / 2))
            if abs(low[i] - low[j]) / max(low[i], 1) < tol_pct:
                liq_lows.append(round((low[i] + low[j]) / 2))
    liq_highs = sorted(set(liq_highs), reverse=True)[:3]
    liq_lows  = sorted(set(liq_lows))[:3]
    for lh in liq_highs:
        if abs(cp - lh) / max(lh, 1) < 0.015:
            signals.append(("⚠️ Liquidity Cao", f"Tiếp cận Equal High {lh:,.0f} — Stop hunt zone", "neutral"))
    for ll in liq_lows:
        if abs(cp - ll) / max(ll, 1) < 0.015:
            signals.append(("⚠️ Liquidity Thấp", f"Tiếp cận Equal Low {ll:,.0f} — Stop hunt zone", "neutral"))

    ote_zone = None
    if swing_highs and swing_lows:
        sh_i, sh_v = swing_highs[-1]; sl_i, sl_v = swing_lows[-1]
        rng = sh_v - sl_v
        if sh_i > sl_i:
            ote_low, ote_high = sh_v - 0.79*rng, sh_v - 0.618*rng
            ote_zone = {"type": "bullish", "low": ote_low, "high": ote_high,
                        "desc": f"OTE Bullish 61.8–79%: {ote_low:,.0f}–{ote_high:,.0f}"}
            if ote_low <= cp <= ote_high:
                signals.append(("🎯 OTE Bullish Zone", f"Vùng vào mua lý tưởng ICT {ote_low:,.0f}–{ote_high:,.0f}", "bullish"))
        else:
            ote_low, ote_high = sl_v + 0.618*rng, sl_v + 0.79*rng
            ote_zone = {"type": "bearish", "low": ote_low, "high": ote_high,
                        "desc": f"OTE Bearish: {ote_low:,.0f}–{ote_high:,.0f}"}
            if ote_low <= cp <= ote_high:
                signals.append(("🎯 OTE Bearish Zone", f"Vùng vào bán ICT {ote_low:,.0f}–{ote_high:,.0f}", "bearish"))

    score = sum(1 if s[2] == "bullish" else -1 if s[2] == "bearish" else 0 for s in signals)
    if structure == "UPTREND": score += 1
    elif structure == "DOWNTREND": score -= 1
    if bos_bullish: score += 2
    if bos_bearish: score -= 2
    bull_s = [s for s in signals if s[2] == "bullish"]
    bear_s = [s for s in signals if s[2] == "bearish"]
    return {"signals": signals, "order_blocks": order_blocks, "fvg_list": fvg_list,
            "structure": structure, "ote_zone": ote_zone, "liq_highs": liq_highs,
            "liq_lows": liq_lows, "score": score, "bos_bullish": bos_bullish,
            "bos_bearish": bos_bearish, "choch": choch,
            "swing_highs": swing_highs[-3:], "swing_lows": swing_lows[-3:],
            "summary": f"Cấu trúc: {structure} · OB: {len(order_blocks)} · FVG: {len(fvg_list)} · {len(bull_s)}🟢/{len(bear_s)}🔴"}


# ─────────────────────────────────────────────────────────────────────────────
# VSA — Volume Spread Analysis
# ─────────────────────────────────────────────────────────────────────────────
def calc_vsa(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {"signals": [], "score": 0, "patterns": [], "summary": "Không đủ dữ liệu"}
    close = df["close"]; high = df["high"]; low = df["low"]
    open_ = df["open"]; vol = df["volume"]
    vol_ma20  = vol.rolling(20).mean()
    spread    = (high - low) / close.replace(0, np.nan)
    spread_ma = spread.rolling(20).mean()
    patterns = []; signals = []
    for i in range(max(1, len(df) - 15), len(df)):
        v   = vol.iloc[i]; vm = vol_ma20.iloc[i] if pd.notna(vol_ma20.iloc[i]) else v
        sp  = spread.iloc[i] if pd.notna(spread.iloc[i]) else 0
        spm = spread_ma.iloc[i] if pd.notna(spread_ma.iloc[i]) else sp
        c   = close.iloc[i]; op = open_.iloc[i]
        h   = high.iloc[i];  lv = low.iloc[i]
        t   = df["time"].iloc[i] if "time" in df.columns else str(i)
        is_up = c >= op; body = abs(c - op); rng = h - lv if h != lv else 1
        upper_w = h - max(c, op); lower_w = min(c, op) - lv
        if vm == 0: continue
        vr = v / vm
        if vr > 1.8 and sp > spm * 0.8 and lower_w > body * 0.5 and c > (lv + rng * 0.4):
            patterns.append(("🛑 Stopping Volume", t, c, f"Vol x{vr:.1f} · Đuôi dưới dài → Lực mua hấp thụ"))
            signals.append(("🛑 Stopping Volume", f"Phiên {t}: Vol x{vr:.1f} + đuôi dưới → buyer hấp thụ seller", "bullish"))
        elif vr < 0.7 and is_up and sp < spm * 0.7:
            patterns.append(("❌ No Demand", t, c, f"Vol thấp x{vr:.1f} · Tăng yếu → Thiếu lực mua"))
            signals.append(("❌ No Demand", f"Phiên {t}: Vol x{vr:.1f} thấp, giá tăng → thiếu lực mua", "bearish"))
        elif vr < 0.7 and not is_up and sp < spm * 0.7:
            patterns.append(("✅ No Supply", t, c, f"Vol thấp x{vr:.1f} · Giảm yếu → Seller kiệt sức"))
            signals.append(("✅ No Supply", f"Phiên {t}: Vol x{vr:.1f} thấp, giá giảm → seller kiệt", "bullish"))
        elif vr > 3.5 and sp > spm * 1.4:
            if is_up:
                patterns.append(("💥 Buying Climax", t, c, f"Vol x{vr:.1f} KHỔNG LỒ nến xanh rộng → Đỉnh tiềm năng"))
                signals.append(("💥 Buying Climax", f"Phiên {t}: Vol x{vr:.1f} cực lớn nến xanh → Đỉnh phân phối", "bearish"))
            else:
                patterns.append(("💥 Selling Climax", t, c, f"Vol x{vr:.1f} KHỔNG LỒ nến đỏ rộng → Đáy tiềm năng"))
                signals.append(("💥 Selling Climax", f"Phiên {t}: Vol x{vr:.1f} cực lớn nến đỏ → Đáy tích lũy", "bullish"))
        elif vr > 1.5 and not is_up and upper_w > body * 1.5 and sp > spm:
            patterns.append(("🪝 Upthrust", t, c, f"Vol x{vr:.1f} · Đuôi trên dài · Đóng thấp → Bẫy bull"))
            signals.append(("🪝 Upthrust", f"Phiên {t}: Giá xuyên lên đóng thấp Vol x{vr:.1f} → trap bull", "bearish"))
        elif vr < 0.6 and sp < spm * 0.6:
            patterns.append(("🔬 Test", t, c, f"Vol thấp x{vr:.1f} spread hẹp → Xác nhận hỗ trợ"))
            signals.append(("🔬 Test", f"Phiên {t}: Vol x{vr:.1f} spread hẹp → hỗ trợ được xác nhận", "bullish"))
    patterns = patterns[-8:]; signals = signals[-8:]
    score = sum(1 if s[2] == "bullish" else -1 for s in signals)
    bull_p = [p for p in patterns if any(k in p[0] for k in ["Stopping", "No Supply", "Selling Climax", "Test", "✅"])]
    bear_p = [p for p in patterns if any(k in p[0] for k in ["No Demand", "Buying Climax", "Upthrust", "❌"])]
    return {"signals": signals, "patterns": patterns, "score": score,
            "summary": f"VSA: {len(bull_p)}🟢 · {len(bear_p)}🔴 · Score: {score:+d}"}


# ─────────────────────────────────────────────────────────────────────────────
# PRICE ACTION
# ─────────────────────────────────────────────────────────────────────────────
def calc_price_action(df: pd.DataFrame) -> dict:
    if len(df) < 10:
        return {"signals": [], "score": 0, "key_levels": [], "trend_structure": "N/A",
                "inside_bars": 0, "is_consolidating": False, "summary": "Không đủ dữ liệu"}
    close = df["close"].values; high = df["high"].values
    low   = df["low"].values;   open_ = df["open"].values
    n = len(df); signals = []; score = 0; key_levels = []

    ib_count = 0; consec_ibs = 0
    for i in range(1, n):
        if high[i] < high[i-1] and low[i] > low[i-1]: ib_count += 1; consec_ibs += 1
        else: consec_ibs = 0
    if consec_ibs >= 2:
        signals.append(("📦 NÉN MẠNH", f"{consec_ibs} Inside Bar liên tiếp → Setup breakout mạnh", "neutral"))
    elif ib_count >= 1:
        signals.append(("📦 Inside Bar", f"{ib_count} Inside Bar → Tích lũy, chờ breakout", "neutral"))

    if n >= 2 and high[-1] > high[-2] and low[-1] < low[-2]:
        d_ = "tăng" if close[-1] > open_[-1] else "giảm"
        ct = "bullish" if d_ == "tăng" else "bearish"
        signals.append(("🔥 Outside Bar", f"Nuốt nến trước → Đảo chiều {d_} mạnh", ct))
        score += (2 if ct == "bullish" else -2)

    sh = []; sl = []
    for i in range(2, n - 2):
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            sh.append(high[i])
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            sl.append(low[i])
    trend_structure = "SIDEWAYS"
    if len(sh) >= 2 and len(sl) >= 2:
        hh = sh[-1] > sh[-2]; hl = sl[-1] > sl[-2]
        lh = sh[-1] < sh[-2]; ll = sl[-1] < sl[-2]
        if hh and hl:
            trend_structure = "UPTREND (HH+HL)"; score += 2
            signals.append(("📈 Uptrend PA", "HH+HL → Xu hướng tăng rõ ràng", "bullish"))
        elif lh and ll:
            trend_structure = "DOWNTREND (LH+LL)"; score -= 2
            signals.append(("📉 Downtrend PA", "LH+LL → Xu hướng giảm rõ ràng", "bearish"))

    for sv in (sh[-3:] if sh else []):
        key_levels.append({"price": sv, "type": "resistance", "label": f"KC: {sv:,.0f}"})
    for sv in (sl[-3:] if sl else []):
        key_levels.append({"price": sv, "type": "support", "label": f"HT: {sv:,.0f}"})
    cp = close[-1]
    for kl in key_levels:
        pct = abs(cp - kl["price"]) / max(kl["price"], 1) * 100
        if pct < 1.5:
            if kl["type"] == "support":
                signals.append(("🟢 Tại Hỗ Trợ PA", f"Tiếp cận {kl['price']:,.0f} (cách {pct:.1f}%)", "bullish"))
                score += 1
            else:
                signals.append(("🔴 Tại Kháng Cự PA", f"Tiếp cận {kl['price']:,.0f} (cách {pct:.1f}%)", "bearish"))
                score -= 1

    r10 = df.tail(10)
    pr_range = (r10["high"].max() - r10["low"].min()) / r10["close"].mean() * 100
    is_cons  = pr_range < 5.0
    if is_cons:
        signals.append(("⏳ Tích Lũy (Consolidation)", f"Biên độ 10P: {pr_range:.1f}% → Sắp breakout", "neutral"))

    lb = abs(close[-1] - open_[-1]); lr = high[-1] - low[-1] if high[-1] != low[-1] else 1
    ll_w = min(close[-1], open_[-1]) - low[-1]; lu_w = high[-1] - max(close[-1], open_[-1])
    if ll_w > 3 * lb and ll_w > lu_w * 2:
        signals.append(("📌 Bullish Pin Bar", "Đuôi dài dưới → Từ chối vùng thấp mạnh", "bullish")); score += 2
    elif lu_w > 3 * lb and lu_w > ll_w * 2:
        signals.append(("📌 Bearish Pin Bar", "Đuôi dài trên → Từ chối vùng cao mạnh", "bearish")); score -= 2

    return {"signals": signals, "score": score, "key_levels": key_levels,
            "trend_structure": trend_structure, "inside_bars": ib_count,
            "is_consolidating": is_cons,
            "summary": f"PA: {trend_structure} · IB: {ib_count} · KL: {len(key_levels)} · Score: {score:+d}"}


# ─────────────────────────────────────────────────────────────────────────────
# OVERFLOW
# ─────────────────────────────────────────────────────────────────────────────
def calc_overflow(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {"signals": [], "score": 0, "overflow_events": [],
                "summary": "Không đủ dữ liệu", "gaps": [], "consecutive_run": 0,
                "bb_overflow_up": False, "bb_overflow_down": False}
    close = df["close"]; high = df["high"]; low = df["low"]
    open_ = df["open"]; vol = df["volume"]
    vol_ma20 = vol.rolling(20).mean(); tms = df["time"].values
    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband(); bb_low = bb.bollinger_lband()
    signals = []; overflow_events = []; gaps = []; score = 0

    for i in range(max(0, len(df) - 10), len(df)):
        vm = vol_ma20.iloc[i] if pd.notna(vol_ma20.iloc[i]) else 0
        if vm == 0: continue
        vr = vol.iloc[i] / vm; t = tms[i]
        color = "bullish" if close.iloc[i] >= open_.iloc[i] else "bearish"
        if vr >= 4.0:
            overflow_events.append({"type": "EXTREME", "time": t, "ratio": vr,
                                     "close": close.iloc[i], "direction": color,
                                     "desc": f"Vol x{vr:.1f} — CỰC ĐẠI"})
            signals.append((f"🌊 Vol Overflow x{vr:.1f} CỰC ĐẠI",
                            f"Phiên {t}: Tổ chức giao dịch ồ ạt", color))
            score += (3 if color == "bullish" else -3)
        elif vr >= 2.5:
            overflow_events.append({"type": "HIGH", "time": t, "ratio": vr,
                                     "close": close.iloc[i], "direction": color,
                                     "desc": f"Vol x{vr:.1f} — Cao bất thường"})
            signals.append((f"🌊 Vol Overflow x{vr:.1f}",
                            f"Phiên {t}: Volume cao bất thường", color))
            score += (1 if color == "bullish" else -1)

    lc = close.iloc[-1]; lb_h = bb_high.iloc[-1]; lb_l = bb_low.iloc[-1]
    if pd.notna(lb_h) and lc > lb_h * 1.01:
        pct = (lc / lb_h - 1) * 100
        signals.append(("⚡ Price Overflow — Quá Mua",
                        f"Vượt BB Upper {lb_h:,.0f} ({pct:.1f}%) → Hồi về mean", "bearish"))
        score -= 2
        overflow_events.append({"type": "Price UP", "time": tms[-1], "ratio": pct,
                                  "close": lc, "direction": "overbought",
                                  "desc": f"{pct:.1f}% trên BB Upper"})
    elif pd.notna(lb_l) and lc < lb_l * 0.99:
        pct = (1 - lc / lb_l) * 100
        signals.append(("💧 Price Overflow — Quá Bán",
                        f"Dưới BB Lower {lb_l:,.0f} ({pct:.1f}%) → Hồi phục", "bullish"))
        score += 2
        overflow_events.append({"type": "Price DOWN", "time": tms[-1], "ratio": pct,
                                  "close": lc, "direction": "oversold",
                                  "desc": f"{pct:.1f}% dưới BB Lower"})

    for i in range(max(1, len(df) - 10), len(df)):
        pc = close.iloc[i - 1]; co = open_.iloc[i]
        gp = (co - pc) / pc * 100
        if abs(gp) > 0.5:
            gaps.append({"time": tms[i], "pct": gp,
                          "filled": bool((gp > 0 and low.iloc[i] <= pc) or (gp < 0 and high.iloc[i] >= pc))})
            if abs(gp) > 1.0:
                color = "bullish" if gp > 0 else "bearish"
                # Bug fix: "GAP" not "GẬP"
                signals.append((f"📐 GAP {'TĂNG' if gp > 0 else 'GIẢM'} {gp:+.1f}%",
                                f"Phiên {tms[i]}: Giá mở cửa lệch {gp:+.1f}%", color))

    consec = 0
    if len(df) >= 2:
        d_last = 1 if close.iloc[-1] > close.iloc[-2] else -1
        for i in range(len(df) - 1, max(0, len(df) - 10) - 1, -1):
            if i == 0: break
            d = 1 if close.iloc[i] > close.iloc[i - 1] else -1
            if d == d_last: consec += 1
            else: break
    if consec >= 6:
        color = "bearish" if d_last == 1 else "bullish"
        dl = "tăng" if d_last == 1 else "giảm"
        signals.append((f"😮 Kiệt sức {dl.upper()} ({consec}P)",
                        f"{consec} phiên {dl} liên tiếp → Đảo chiều/điều chỉnh", color))
        score += (1 if color == "bullish" else -1)

    if len(df) >= 3:
        p20h = high.iloc[-21:-1].max() if len(df) >= 22 else high.iloc[:-1].max()
        p20l = low.iloc[-21:-1].min()  if len(df) >= 22 else low.iloc[:-1].min()
        lvr  = vol.iloc[-1] / vol_ma20.iloc[-1] if pd.notna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] > 0 else 1
        if lc > p20h and lvr > 1.5:
            signals.append(("🚀 BREAKOUT xác nhận",
                            f"Phá đỉnh 20P ({p20h:,.0f}) · Vol x{lvr:.1f}", "bullish"))
            score += 3
        elif lc < p20l and lvr > 1.5:
            signals.append(("💥 BREAKDOWN xác nhận",
                            f"Phá đáy 20P ({p20l:,.0f}) · Vol x{lvr:.1f}", "bearish"))
            score -= 3

    return {"signals": signals, "score": score, "overflow_events": overflow_events,
            "gaps": gaps, "consecutive_run": consec,
            "bb_overflow_up": (pd.notna(lb_h) and lc > lb_h * 1.01),
            "bb_overflow_down": (pd.notna(lb_l) and lc < lb_l * 0.99),
            "summary": f"Overflow: {len(overflow_events)} sự kiện · Gaps: {len(gaps)} · Run: {consec}P · Score: {score:+d}"}


# ─────────────────────────────────────────────────────────────────────────────
# SMART MONEY PHASE
# ─────────────────────────────────────────────────────────────────────────────
def detect_smart_money_phase(df: pd.DataFrame) -> dict:
    win  = df.tail(15); last = df.iloc[-1]
    obv_vals = win["OBV"].dropna().values; obv_slope = 0
    if len(obv_vals) > 3:
        x = np.arange(len(obv_vals)); s = np.polyfit(x, obv_vals, 1)[0]
        obv_slope = s / (abs(obv_vals).mean() + 1e-9) * 10
    prices = win["close"].dropna().values; price_slope = 0
    if len(prices) > 3:
        x = np.arange(len(prices)); s = np.polyfit(x, prices, 1)[0]
        price_slope = s / (prices.mean() + 1e-9) * 100
    cmf_val   = float(last.get("CMF",  0))  if pd.notna(last.get("CMF",  np.nan)) else 0.0
    mfi_val   = float(last.get("MFI", 50))  if pd.notna(last.get("MFI",  np.nan)) else 50.0
    buy_ratio = float(last.get("Buy_Ratio", 0.5)) if pd.notna(last.get("Buy_Ratio", np.nan)) else 0.5
    vol_ratio = float(last.get("Vol_Ratio", 1.0)) if pd.notna(last.get("Vol_Ratio", np.nan)) else 1.0
    score = 0; signals = []
    if obv_slope > 0.3:    score += 2; signals.append(("OBV tăng",  "Tiền lớn tích lũy", "🟢"))
    elif obv_slope < -0.3: score -= 2; signals.append(("OBV giảm",  "Tiền lớn xả",       "🔴"))
    else:                              signals.append(("OBV phẳng", "Chưa rõ",           "⚪"))
    if cmf_val > 0.12:    score += 2; signals.append(("CMF cao",    f"{cmf_val:.3f} Mua mạnh", "🟢"))
    elif cmf_val > 0:     score += 1; signals.append(("CMF dương",  f"{cmf_val:.3f}",         "🟡"))
    elif cmf_val < -0.12: score -= 2; signals.append(("CMF âm",     f"{cmf_val:.3f} Bán mạnh","🔴"))
    else:                 score -= 1; signals.append(("CMF âm nhẹ", f"{cmf_val:.3f}",          "🟠"))
    if mfi_val < 25:   score += 2; signals.append(("MFI quá bán", f"{mfi_val:.1f}", "🟢"))
    elif mfi_val > 80: score -= 2; signals.append(("MFI quá mua", f"{mfi_val:.1f}", "🔴"))
    else:                           signals.append(("MFI trung tính", f"{mfi_val:.1f}", "⚪"))
    if buy_ratio > 0.65:   score += 1.5; signals.append(("KL mua trội", f"{buy_ratio*100:.0f}%", "🟢"))
    elif buy_ratio < 0.35: score -= 1.5; signals.append(("KL bán trội", f"{(1-buy_ratio)*100:.0f}%", "🔴"))
    else:                               signals.append(("Cân bằng",  f"{buy_ratio*100:.0f}%", "⚪"))
    if price_slope > 0.5 and obv_slope < -0.2:
        score -= 1.5; signals.append(("Phân kỳ âm", "Giá↑ OBV↓", "🔴"))
    elif price_slope < -0.5 and obv_slope > 0.2:
        score += 1.5; signals.append(("Phân kỳ dương", "Giá↓ OBV↑", "🟢"))
    max_score = 10.0; pct = score / max_score
    if pct >= 0.4:    phase,icon,color,bg,border = "GOM HÀNG","🏦","#00C853","rgba(0,200,83,0.10)","#00C853"; desc = "Tiền lớn đang **bí mật tích lũy**."
    elif pct >= 0.15: phase,icon,color,bg,border = "ĐẨY GIÁ (MARKUP)","🚀","#40C4FF","rgba(64,196,255,0.10)","#40C4FF"; desc = "Dòng tiền **đang đẩy giá lên mạnh**."
    elif pct >= -0.15:phase,icon,color,bg,border = "TRUNG TÍNH","🔍","#FFD740","rgba(255,215,64,0.08)","#FFD740"; desc = "Dòng tiền **chưa lộ rõ ý định**."
    elif pct >= -0.40:phase,icon,color,bg,border = "XẢ HÀNG","📤","#FF6D00","rgba(255,109,0,0.10)","#FF6D00"; desc = "Tiền lớn **đang lặng lẽ phân phối**."
    else:             phase,icon,color,bg,border = "ĐÈ GIÁ","📉","#FF1744","rgba(255,23,68,0.10)","#FF1744"; desc = "Tiền lớn **đang đè giá và bán tháo**."
    return {"phase": phase, "icon": icon, "color": color, "bg": bg, "border": border,
            "phase_desc": desc, "score": score, "max_score": max_score, "signals": signals,
            "obv_slope": obv_slope, "cmf": cmf_val, "mfi": mfi_val,
            "buy_ratio": buy_ratio, "vol_ratio": vol_ratio, "price_slope": price_slope}


def detect_candlestick_patterns(df):
    if len(df) < 5: return []
    patterns = []
    last = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
    o, h, l, c   = last["open"], last["high"], last["low"], last["close"]
    po, ph, pl, pc= prev["open"], prev["high"], prev["low"], prev["close"]
    p2o, p2c      = prev2["open"], prev2["close"]
    body   = abs(c - o); rng = (h - l) if (h - l) > 0 else 0.001
    up_w   = h - max(o, c); low_w = min(o, c) - l
    if body / rng < 0.1:                              patterns.append(("⚖️ Doji",          "Thị trường do dự — cân bằng lực mua/bán", "neutral"))
    elif low_w > 2*body and up_w < body and c >= o:   patterns.append(("🔨 Hammer",        "Đảo chiều tăng — đuôi dưới dài",          "bullish"))
    elif up_w > 2*body and low_w < body and c < o:    patterns.append(("⭐ Shooting Star", "Đảo chiều giảm — đuôi trên dài",          "bearish"))
    if pc < po and c > o and c >= po and o <= pc:     patterns.append(("🟢 Bullish Engulfing", "Nuốt nến đỏ → Đảo chiều tăng mạnh",  "bullish"))
    elif pc > po and c < o and c <= po and o >= pc:   patterns.append(("🔴 Bearish Engulfing", "Nuốt nến xanh → Đảo chiều giảm mạnh","bearish"))
    if p2c < p2o and abs(pc - po) < 0.3*abs(p2c-p2o) and c > o and c > (p2o+p2c)/2:
        patterns.append(("🌅 Morning Star", "3 nến đảo chiều tăng", "bullish"))
    elif p2c > p2o and abs(pc - po) < 0.3*abs(p2c-p2o) and c < o and c < (p2o+p2c)/2:
        patterns.append(("🌇 Evening Star", "3 nến đảo chiều giảm", "bearish"))
    if low_w > 3 * body:   patterns.append(("📌 Pin Bar Tăng", "Đuôi dài dưới — từ chối vùng thấp", "bullish"))
    elif up_w > 3 * body:  patterns.append(("📌 Pin Bar Giảm", "Đuôi dài trên — từ chối vùng cao",  "bearish"))
    if c > o and body / rng > 0.85: patterns.append(("💚 Marubozu Xanh", "Lực mua áp đảo", "bullish"))
    elif c < o and body / rng > 0.85: patterns.append(("❤️ Marubozu Đỏ", "Lực bán áp đảo", "bearish"))
    # Three White Soldiers / Three Black Crows
    try:
        if all(df["close"].iloc[-i] > df["open"].iloc[-i] and
               df["close"].iloc[-i] > df["close"].iloc[-i - 1] for i in range(1, 4)):
            patterns.append(("🕯️✨ Three White Soldiers", "3 nến xanh liên tiếp → Bullish Continuation mạnh", "bullish"))
        if all(df["close"].iloc[-i] < df["open"].iloc[-i] and
               df["close"].iloc[-i] < df["close"].iloc[-i - 1] for i in range(1, 4)):
            patterns.append(("🕯️⛈️ Three Black Crows", "3 nến đỏ liên tiếp → Bearish Continuation mạnh", "bearish"))
    except Exception:
        pass
    seen = []; result = []
    for p in patterns:
        if p[0] not in seen: seen.append(p[0]); result.append(p)
    return result[:6]


def calc_trend_strength(df, sm_result):
    score = 50.0; signals = []; latest = df.iloc[-1]; c = latest["close"]
    adx_val = latest.get("ADX", np.nan); di_p = latest.get("DI_Plus", 0) or 0; di_m = latest.get("DI_Minus", 0) or 0
    if pd.notna(adx_val):
        if adx_val > 35:   adj = 12 if di_p > di_m else -12; score += adj; signals.append(f"ADX={adx_val:.1f}>35 ({'TĂNG' if di_p>di_m else 'GIẢM'} mạnh)")
        elif adx_val > 22: adj = 6  if di_p > di_m else -6;  score += adj; signals.append(f"ADX={adx_val:.1f}>22 ({'tăng' if di_p>di_m else 'giảm'})")
        else:              signals.append(f"ADX={adx_val:.1f}<22 → Sideway")
    e5 = latest.get("EMA5", np.nan); e20 = latest.get("EMA20", np.nan); e50 = latest.get("EMA50", np.nan)
    if pd.notna(e5) and pd.notna(e20) and pd.notna(e50):
        if e5 > e20 > e50 and c > e5:   score += 16; signals.append("EMA5>EMA20>EMA50: Bull Alignment ✅")
        elif e5 > e20 and c > e20:       score += 8;  signals.append("EMA5>EMA20: Tăng ngắn-trung hạn")
        elif c > e50:                    score += 4;  signals.append("Giá>EMA50: Dài hạn tăng")
        elif e5 < e20 < e50 and c < e5: score -= 16; signals.append("EMA5<EMA20<EMA50: Bear Alignment ❌")
        elif e5 < e20 and c < e20:       score -= 8;  signals.append("EMA5<EMA20: Giảm ngắn-trung hạn")
        elif c < e50:                    score -= 4;  signals.append("Giá<EMA50: Dài hạn yếu")
    spa = latest.get("SpanA", np.nan); spb = latest.get("SpanB", np.nan)
    if pd.notna(spa) and pd.notna(spb):
        ct = max(spa, spb); cb = min(spa, spb)
        if c > ct:   score += 10; signals.append("Trên Kumo ☁️ Bullish")
        elif c < cb: score -= 10; signals.append("Dưới Kumo ☁️ Bearish")
    mv = latest.get("MACD", np.nan); sv = latest.get("MACD_Signal", np.nan)
    if pd.notna(mv) and pd.notna(sv):
        if mv > sv: score += 5; signals.append("MACD>Signal: Momentum tăng")
        else:       score -= 5; signals.append("MACD<Signal: Momentum giảm")
    # TTM Squeeze bonus
    sq_on = latest.get("SQ_On", 0) or 0; sq_mom = latest.get("SQ_Mom", 0) or 0
    if sq_on: signals.append("⚡ SQUEEZE đang nén → Chờ bứt phá mạnh")
    elif sq_mom > 0:  score += 5; signals.append(f"⚡ Squeeze FIRED tăng ({sq_mom:.2f})")
    elif sq_mom < 0:  score -= 5; signals.append(f"⚡ Squeeze FIRED giảm ({sq_mom:.2f})")
    # VWAP
    vwap20 = latest.get("VWAP20", np.nan)
    if pd.notna(vwap20):
        if c > vwap20: score += 4; signals.append(f"Giá>VWAP20 ({vwap20:,.0f}): Bullish bias")
        else:          score -= 4; signals.append(f"Giá<VWAP20 ({vwap20:,.0f}): Bearish bias")
    # StochRSI
    stoch_k = latest.get("StochRSI_K", 50) or 50
    if stoch_k < 20:   score += 3; signals.append(f"StochRSI K={stoch_k:.0f}: Quá bán → Tiềm năng tăng")
    elif stoch_k > 80: score -= 3; signals.append(f"StochRSI K={stoch_k:.0f}: Quá mua → Tiềm năng giảm")
    # Smart Money
    sm_pct = sm_result["score"] / sm_result["max_score"]; score += sm_pct * 8
    signals.append(f"Smart Money: {sm_result['score']:+.1f} ({sm_result['phase']})")
    score = max(0.0, min(100.0, score))
    if score >= 78:   label = "RẤT MẠNH TĂNG 🚀"; color = "#00C853"; bg = "rgba(0,200,83,0.12)"
    elif score >= 62: label = "ĐANG TĂNG 📈";      color = "#69F0AE"; bg = "rgba(105,240,174,0.10)"
    elif score >= 42: label = "TRUNG TÍNH ⚖️";     color = "#FFD740"; bg = "rgba(255,215,64,0.08)"
    elif score >= 28: label = "ĐANG GIẢM 📉";      color = "#FF9100"; bg = "rgba(255,145,0,0.10)"
    else:             label = "RẤT MẠNH GIẢM 🔻"; color = "#FF1744"; bg = "rgba(255,23,68,0.10)"
    return {"score": score, "label": label, "color": color, "bg": bg, "signals": signals}


# ─────────────────────────────────────────────────────────────────────────────
# AI CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(api_key, prompt):
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    body    = {"model": "claude-sonnet-4-6", "max_tokens": 3000, "messages": [{"role": "user", "content": prompt}]}
    resp    = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=90)
    if not resp.ok: st.error(f"Lỗi Claude: {resp.status_code} — {resp.text[:200]}")
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_gemini(api_key, prompt):
    from google import genai
    import time
    MODELS = ["gemini-2.5-flash-preview-05-20", "gemini-2.0-flash", "gemini-1.5-flash-latest"]
    client = genai.Client(api_key=api_key)
    for model in MODELS:
        for attempt in range(2):
            try:
                return client.models.generate_content(model=model, contents=prompt).text
            except Exception as e:
                err = str(e)
                if "RESOURCE_EXHAUSTED" in err or "429" in err:
                    if attempt == 0: time.sleep(5); continue
                    else: break
                elif "NOT_FOUND" in err or "404" in err: break
                else: raise
    raise RuntimeError("⚠️ Hết quota Gemini. Chờ 1 phút hoặc dùng Claude.")


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER HELPER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def scan_single_stock(symbol: str) -> dict | None:
    try:
        df = get_clean_stock_data(symbol)
        if df is None or df.empty or len(df) < 30: return None
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df) < 30: return None

        df["RSI"]        = RSIIndicator(close=df["close"], window=14).rsi()
        bb               = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["BB_High"]    = bb.bollinger_hband(); df["BB_Low"] = bb.bollinger_lband()
        macd_o           = MACD(close=df["close"])
        df["MACD"]       = macd_o.macd(); df["MACD_Signal"] = macd_o.macd_signal()
        df["MCDX"]       = calc_mcdx(df)
        (df["Tenkan"], df["Kijun"], df["SpanA"], df["SpanB"], df["Chikou"]) = calc_ichimoku(df)
        df = calc_adx(df); df = calc_ema_cross(df); df = calc_smart_money(df)
        df = calc_vwap(df); df = calc_ttm_squeeze(df); df = calc_atr_indicator(df)

        fib_levels = calc_fibonacci(df); sm_result = detect_smart_money_phase(df)
        ict_r = calc_ict(df); vsa_r = calc_vsa(df)
        pa_r  = calc_price_action(df); ovf_r = calc_overflow(df)
        wy_r  = calc_wyckoff_events(df); div_r = calc_hidden_divergence(df)

        latest = df.iloc[-1]; prev = df.iloc[-2]
        pct_chg = (latest["close"] - prev["close"]) / prev["close"] * 100
        rsi_now = latest["RSI"] if pd.notna(latest["RSI"]) else 50

        score = 0
        if rsi_now < 35: score += 2
        elif rsi_now < 50: score += 1
        elif rsi_now > 70: score -= 2
        if pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_Signal")):
            score += (2 if latest["MACD"] > latest["MACD_Signal"] else -1)
        mcdx_now = latest.get("MCDX", 0) or 0
        if mcdx_now > 0.2: score += 2
        elif mcdx_now > 0: score += 1
        else: score -= 1
        adx_val = latest.get("ADX", 0) or 0
        if adx_val > 25:
            score += (1 if (latest.get("DI_Plus", 0) or 0) > (latest.get("DI_Minus", 0) or 0) else -1)
        e5 = latest.get("EMA5", np.nan); e20 = latest.get("EMA20", np.nan); e50 = latest.get("EMA50", np.nan)
        if pd.notna(e5) and pd.notna(e20) and pd.notna(e50):
            if e5 > e20 > e50: score += 2
            elif e5 < e20 < e50: score -= 2
        # VWAP
        vwap20 = latest.get("VWAP20", np.nan)
        if pd.notna(vwap20):
            score += (1 if latest["close"] > vwap20 else -1)
        # TTM Squeeze
        sq_on = latest.get("SQ_On", 0) or 0; sq_mom = latest.get("SQ_Mom", 0) or 0
        if not sq_on: score += (1 if sq_mom > 0 else -1)
        score += max(-3, min(3, ict_r["score"]))
        score += max(-3, min(3, vsa_r["score"]))
        score += max(-3, min(3, pa_r["score"]))
        score += max(-2, min(2, ovf_r["score"]))
        score += max(-2, min(2, wy_r["score"]))
        score += max(-2, min(2, div_r["score"]))

        max_score = 26; pct_s = score / max_score
        if pct_s >= 0.40:   recommendation = "✅ MUA"
        elif pct_s >= 0.12: recommendation = "⏳ THEO DÕI"
        else:               recommendation = "🚫 TRÁNH/BÁN"

        _is_dn = fib_levels.get("_is_downtrend", True)
        _fh = fib_levels.get("_high", 0); _fl = fib_levels.get("_low", 0)
        _diff = fib_levels.get("_diff", _fh - _fl); cn = latest["close"]
        if _is_dn:
            _f618 = _fh - 0.618*_diff; _f650 = _fh - 0.650*_diff
            _f786 = _fh - 0.786*_diff; _f382 = _fh - 0.382*_diff
            if _f650 <= cn <= _f618:    fib_zone = "⭐ Golden Pocket"
            elif _f786 <= cn < _f650:   fib_zone = "⚠️ Hỗ trợ sâu (65-78.6%)"
            elif cn < _f786:            fib_zone = "⛔ Dưới hỗ trợ"
            elif _f618 < cn <= _f382:   fib_zone = "📍 Hỗ trợ (38-61.8%)"
            else:                       fib_zone = "🔺 Trên 38.2%"
        else:
            _f236 = _fl + 0.236*_diff; _f382 = _fl + 0.382*_diff
            if cn <= _f236:   fib_zone = "✅ Gần đáy (tích lũy)"
            elif cn <= _f382: fib_zone = "📍 Phục hồi sớm"
            else:             fib_zone = "🔺 Đã phục hồi nhiều"

        ict_struct = ict_r.get("structure", "N/A")
        ict_bos    = "BOS✅" if ict_r.get("bos_bullish") or ict_r.get("bos_bearish") else ""
        ict_choch  = "CHoCH⚡" if ict_r.get("choch") else ""
        ict_label  = f"{ict_struct} {ict_bos}{ict_choch}".strip()
        top_vsa    = vsa_r["patterns"][-1][0] if vsa_r["patterns"] else "—"

        div_label = ""
        if div_r["hid_bull"]:   div_label = "🔒HidBull"
        elif div_r["reg_bull"]: div_label = "📈RegBull"
        elif div_r["hid_bear"]: div_label = "🔒HidBear"
        elif div_r["reg_bear"]: div_label = "📉RegBear"
        else:                   div_label = "—"

        wyckoff_short = wy_r["phase"].split(" ")[0] if wy_r["phase"] != "UNDEFINED ❓" else "?"

        return {
            "symbol": symbol, "close": latest["close"], "pct_chg": pct_chg,
            "rsi": rsi_now, "mcdx": mcdx_now, "adx": float(adx_val),
            "macd_bull": bool(pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_Signal"))
                              and latest["MACD"] > latest["MACD_Signal"]),
            "ema_bull": bool(pd.notna(e5) and pd.notna(e20) and e5 > e20),
            "sm_phase": sm_result["phase"], "sm_icon": sm_result["icon"],
            "sm_color": sm_result["color"], "sm_score": sm_result["score"],
            "fib_zone": fib_zone, "score": score, "max_score": max_score,
            "recommendation": recommendation,
            "obv_slope": sm_result.get("obv_slope", 0), "cmf": sm_result.get("cmf", 0),
            "mfi": sm_result.get("mfi", 50), "buy_ratio": sm_result.get("buy_ratio", 0.5),
            "vol_ratio": sm_result.get("vol_ratio", 1.0),
            "avg_vol": float(latest.get("Vol_MA20", 0) or 0),
            "ict_score": ict_r["score"], "ict_label": ict_label,
            "vsa_score": vsa_r["score"], "vsa_top": top_vsa,
            "pa_score": pa_r["score"], "pa_trend": pa_r.get("trend_structure", "N/A"),
            "ovf_score": ovf_r["score"], "ovf_events": len(ovf_r["overflow_events"]),
            "is_consolidating": pa_r.get("is_consolidating", False),
            "consecutive_run": ovf_r.get("consecutive_run", 0),
            "wyckoff_phase": wyckoff_short, "wyckoff_score": wy_r["score"],
            "div_label": div_label, "div_score": div_r["score"],
            "sq_on": int(sq_on), "sq_mom": float(sq_mom),
            "vwap_bull": bool(pd.notna(vwap20) and latest["close"] > vwap20),
        }
    except Exception:
        return None


def _phase_order(phase):
    return {"GOM HÀNG": 0, "ĐẨY GIÁ (MARKUP)": 1, "TRUNG TÍNH": 2, "XẢ HÀNG": 3, "ĐÈ GIÁ": 4}.get(phase, 5)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PHÂN TÍCH ĐƠN LẺ
# ═════════════════════════════════════════════════════════════════════════════
with _tab1:
    ticker = st.text_input(
        "🔍 Nhập mã chứng khoán (VD: HPG, VCB, VN30F1M):", "VN30F1M"
    ).upper().strip()

    if st.button("🚀 Lấy Dữ Liệu & Phân Tích", type="primary", key="analyze_btn"):
        with st.spinner(f"Đang trích xuất dữ liệu **{ticker}**..."):
            df = get_clean_stock_data(ticker)

        if df is None or df.empty:
            st.error("❌ Không thể lấy dữ liệu."); st.stop()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df) < 30: st.error("❌ Không đủ dữ liệu (cần ≥30 phiên)."); st.stop()

        # ── Tính toán tất cả chỉ báo ─────────────────────────────────────
        df["RSI"]        = RSIIndicator(close=df["close"], window=14).rsi()
        bb               = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["BB_High"]    = bb.bollinger_hband(); df["BB_Low"] = bb.bollinger_lband(); df["BB_Mid"] = bb.bollinger_mavg()
        macd_o           = MACD(close=df["close"])
        df["MACD"]       = macd_o.macd(); df["MACD_Signal"] = macd_o.macd_signal(); df["MACD_Diff"] = macd_o.macd_diff()
        df["MCDX"]       = calc_mcdx(df)
        (df["Tenkan"], df["Kijun"], df["SpanA"], df["SpanB"], df["Chikou"]) = calc_ichimoku(df)
        df = calc_adx(df); df = calc_ema_cross(df); df = calc_smart_money(df)
        df = calc_vwap(df); df = calc_ttm_squeeze(df); df = calc_elder_impulse(df)
        df = calc_stoch_rsi(df); df = calc_psar(df); df = calc_atr_indicator(df)

        fib_levels  = calc_fibonacci(df)
        sm_result   = detect_smart_money_phase(df)
        patterns    = detect_candlestick_patterns(df)
        trend_str   = calc_trend_strength(df, sm_result)
        ict_result  = calc_ict(df); vsa_result = calc_vsa(df)
        pa_result   = calc_price_action(df); ovf_result = calc_overflow(df)
        wy_result   = calc_wyckoff_events(df); div_result = calc_hidden_divergence(df)
        vol_profile = calc_volume_profile(df)
        regime      = calc_market_regime(df)
        df, ha_consec, ha_dir = calc_heikin_ashi(df)
        demand_zones, supply_zones = calc_supply_demand_zones(df)

        latest  = df.iloc[-1]; prev = df.iloc[-2]
        pct_chg = (latest["close"] - prev["close"]) / prev["close"] * 100
        rsi_val = latest["RSI"]; adx_val = latest.get("ADX", 0) or 0
        di_p    = latest.get("DI_Plus", 0) or 0; di_m = latest.get("DI_Minus", 0) or 0
        e5_v    = latest.get("EMA5", 0) or 0; e20_v = latest.get("EMA20", 0) or 0; e50_v = latest.get("EMA50", 0) or 0
        mcdx_val = latest["MCDX"]
        atr14    = float(latest.get("ATR14", 0) or 0)
        vwap20   = latest.get("VWAP20", np.nan)
        sq_on    = int(latest.get("SQ_On", 0) or 0)
        sq_mom   = float(latest.get("SQ_Mom", 0) or 0)
        stoch_k  = float(latest.get("StochRSI_K", 50) or 50)
        stoch_d  = float(latest.get("StochRSI_D", 50) or 50)
        psar_v   = latest.get("PSAR", np.nan)
        psar_up  = pd.notna(latest.get("PSAR_Up", np.nan))

        # ── Market Regime Banner ──────────────────────────────────────────
        rg = regime
        st.markdown(
            f"""<div style="background:rgba(0,0,0,0.35);border:1.5px solid {rg['color']}44;
                border-left:5px solid {rg['color']};border-radius:12px;
                padding:12px 20px;margin-bottom:16px;display:flex;align-items:center;gap:16px;">
  <div style="font-size:2.2rem;">{rg['icon']}</div>
  <div>
    <div style="font-size:0.75rem;color:#aaa;letter-spacing:1.2px;">🌐 MARKET REGIME</div>
    <div style="font-size:1.35rem;font-weight:800;color:{rg['color']};">{rg['regime']}</div>
    <div style="font-size:0.80rem;color:#bbb;margin-top:2px;">{rg['desc']}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Trend Strength Meter ──────────────────────────────────────────
        ts = trend_str
        st.markdown(
            f"""<div style="background:{ts['bg']};border-radius:14px;padding:16px 24px;
                border:1.5px solid {ts['color']}44;margin-bottom:18px;
                display:flex;align-items:center;gap:20px;">
  <div style="flex:1;">
    <div style="font-size:0.78rem;color:#aaa;letter-spacing:1px;">📡 XU HƯỚNG TỔNG HỢP — 16 PHƯƠNG PHÁP</div>
    <div style="font-size:1.55rem;font-weight:800;color:{ts['color']};margin:4px 0;">{ts['label']}</div>
    <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:10px;width:100%;margin-top:6px;">
      <div style="height:10px;width:{ts['score']:.0f}%;border-radius:6px;
                  background:linear-gradient(90deg,#FF1744,#FFD740,#00C853);"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.70rem;color:#666;margin-top:3px;">
      <span>Rất giảm (0)</span><span>Trung tính (50)</span><span>Rất tăng (100)</span>
    </div>
  </div>
  <div style="font-size:2.8rem;font-weight:900;color:{ts['color']};min-width:75px;text-align:right;">
    {ts['score']:.0f}<span style="font-size:1rem;">/100</span>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Method Scores (8 columns) ─────────────────────────────────────
        st.subheader(f"🔬 Điểm Phương Pháp — {ticker}")
        mc1, mc2, mc3, mc4, mc5, mc6, mc7, mc8 = st.columns(8)

        def _sc(s):
            return "#00C853" if s > 0 else ("#FF5252" if s < 0 else "#FFD740")

        methods = [
            (mc1, "🔬 ICT",     ict_result["score"],  ict_result["structure"][:10],  "rgba(64,196,255,0.10)"),
            (mc2, "📊 VSA",     vsa_result["score"],  f"{len(vsa_result['patterns'])} ptns", "rgba(255,167,38,0.10)"),
            (mc3, "🕯️ PA",      pa_result["score"],   pa_result.get("trend_structure","")[:10], "rgba(224,64,251,0.10)"),
            (mc4, "💧 OVF",     ovf_result["score"],  f"{len(ovf_result['overflow_events'])} evt", "rgba(255,23,68,0.10)"),
            (mc5, "🐋 SM",      sm_result["score"],   sm_result["phase"][:8],         sm_result["bg"]),
            (mc6, "🌀 Wyckoff", wy_result["score"],   wy_result["phase"].split(" ")[0][:10], "rgba(100,120,255,0.10)"),
            (mc7, "🔀 Div",     div_result["score"],  "Reg+Hidden",                  "rgba(255,80,120,0.10)"),
        ]
        for col_, lbl, sc, sub, bc in methods:
            col_.markdown(
                f"<div style='text-align:center;padding:10px 4px;border-radius:10px;"
                f"background:{bc};border:1px solid rgba(255,255,255,0.10);'>"
                f"<div style='font-size:0.70rem;color:#aaa;'>{lbl}</div>"
                f"<div style='font-size:1.55rem;font-weight:800;color:{_sc(sc)};'>"
                f"{sc:+.1f}</div>"
                f"<div style='font-size:0.65rem;color:#bbb;overflow:hidden;white-space:nowrap;'>{sub}</div>"
                f"</div>", unsafe_allow_html=True)
        sq_color  = "#FFD740" if sq_on else ("#00C853" if sq_mom > 0 else "#FF5252")
        sq_label2 = "🔴 SQUEEZE ON" if sq_on else ("🟢 Fired ↑" if sq_mom > 0 else "🔴 Fired ↓")
        mc8.markdown(
            f"<div style='text-align:center;padding:10px 4px;border-radius:10px;"
            f"background:rgba(255,215,64,0.10);border:1px solid rgba(255,255,255,0.10);'>"
            f"<div style='font-size:0.70rem;color:#aaa;'>⚡ Squeeze</div>"
            f"<div style='font-size:1.0rem;font-weight:800;color:{sq_color};margin:4px 0;'>{sq_label2}</div>"
            f"<div style='font-size:0.65rem;color:#bbb;'>Mom:{sq_mom:.2f}</div>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Classic Metrics ────────────────────────────────────────────────
        cm1, cm2, cm3, cm4, cm5, cm6, cm7 = st.columns(7)
        cm1.metric("Giá Đóng Cửa", f"{latest['close']:,.0f} đ", f"{pct_chg:+.2f}%")
        rsi_note = "⚠️ Quá mua" if rsi_val > 70 else ("⚠️ Quá bán" if rsi_val < 30 else "Trung tính")
        cm2.metric("RSI (14)", f"{rsi_val:.1f}", rsi_note)
        cm3.metric("ADX (14)", f"{adx_val:.1f}", f"DI+{di_p:.0f}/DI-{di_m:.0f}")
        cm4.metric("StochRSI K/D", f"{stoch_k:.0f}/{stoch_d:.0f}",
                   "⚠️ OB" if stoch_k > 80 else ("⚠️ OS" if stoch_k < 20 else "Bình thường"))
        vwap_disp  = f"{vwap20:,.0f}" if pd.notna(vwap20) else "N/A"
        vwap_delta = f"{'↑ Trên' if pd.notna(vwap20) and latest['close'] > vwap20 else '↓ Dưới'} VWAP20"
        cm5.metric("VWAP20", vwap_disp, vwap_delta)
        psar_disp  = f"{psar_v:,.0f}" if pd.notna(psar_v) else "N/A"
        cm6.metric("Parabolic SAR", psar_disp, "↑ Bull" if psar_up else "↓ Bear")
        cm7.metric("ATR(14)", f"{atr14:,.0f}", f"SL: {latest['close']-1.5*atr14:,.0f}")

        # ── HA + Elder Info ────────────────────────────────────────────────
        ha_label = "🟢 HA Tăng" if ha_dir == 1 else "🔴 HA Giảm"
        elder_now = df["Elder"].iloc[-1] if "Elder" in df.columns else "neutral"
        elder_icon = "🟢" if elder_now == "bull" else ("🔴" if elder_now == "bear" else "🔵")
        elder_text = "Bull Impulse (Mua)" if elder_now == "bull" else ("Bear Impulse (Bán)" if elder_now == "bear" else "Neutral (Chờ)")
        ic1, ic2, ic3 = st.columns(3)
        ic1.info(f"🌊 **Heikin-Ashi:** {ha_label} · {ha_consec} phiên liên tiếp")
        ic2.info(f"{elder_icon} **Elder Impulse:** {elder_text}")
        ic3.info(f"🌀 **Wyckoff:** {wy_result['phase'][:40]}")

        # ── Candlestick Patterns ───────────────────────────────────────────
        if patterns:
            badges = ""
            for pname, pdesc, ptype in patterns:
                pc = {"bullish": "#00E676", "bearish": "#FF5252", "neutral": "#FFD740"}.get(ptype, "#aaa")
                badges += (f"<span style='background:{pc}22;border:1px solid {pc};border-radius:20px;"
                           f"padding:4px 12px;margin:3px;font-size:0.80rem;color:{pc};"
                           f"display:inline-block;' title='{pdesc}'>{pname}</span>")
            st.markdown(
                f"<div style='margin:8px 0 16px 0;'>"
                f"<b style='color:#aaa;font-size:0.78rem;'>🕯️ CANDLESTICK PATTERNS:</b><br>{badges}</div>",
                unsafe_allow_html=True)

        # ── ATR Trade Setup ────────────────────────────────────────────────
        if atr14 > 0:
            entry   = latest["close"]
            sl      = entry - 1.5 * atr14
            t1      = entry + 2.0 * atr14
            t2      = entry + 4.0 * atr14
            rr1     = (t1 - entry) / (entry - sl)
            rr2     = (t2 - entry) / (entry - sl)
            sl_pct  = (entry - sl) / entry * 100
            t1_pct  = (t1 - entry) / entry * 100
            t2_pct  = (t2 - entry) / entry * 100
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.12);
                    border-radius:12px;padding:14px 20px;margin-bottom:16px;">
  <div style="font-size:0.8rem;color:#aaa;margin-bottom:10px;">📏 ATR TRADE SETUP (ATR14={atr14:,.0f}) — Dựa trên giá hiện tại</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;text-align:center;">
    <div style="background:rgba(64,196,255,0.1);border-radius:8px;padding:10px;">
      <div style="font-size:0.72rem;color:#aaa;">📥 Entry</div>
      <div style="font-size:1.1rem;font-weight:700;color:#40C4FF;">{entry:,.0f}</div>
      <div style="font-size:0.70rem;color:#888;">Giá hiện tại</div>
    </div>
    <div style="background:rgba(255,23,68,0.1);border-radius:8px;padding:10px;">
      <div style="font-size:0.72rem;color:#aaa;">🛑 Stop Loss</div>
      <div style="font-size:1.1rem;font-weight:700;color:#FF1744;">{sl:,.0f}</div>
      <div style="font-size:0.70rem;color:#888;">-{sl_pct:.1f}% (1.5×ATR)</div>
    </div>
    <div style="background:rgba(0,200,83,0.1);border-radius:8px;padding:10px;">
      <div style="font-size:0.72rem;color:#aaa;">🎯 Target 1</div>
      <div style="font-size:1.1rem;font-weight:700;color:#00C853;">{t1:,.0f}</div>
      <div style="font-size:0.70rem;color:#888;">+{t1_pct:.1f}% · R:R 1:{rr1:.1f}</div>
    </div>
    <div style="background:rgba(0,200,83,0.15);border-radius:8px;padding:10px;">
      <div style="font-size:0.72rem;color:#aaa;">🎯 Target 2</div>
      <div style="font-size:1.1rem;font-weight:700;color:#69F0AE;">{t2:,.0f}</div>
      <div style="font-size:0.70rem;color:#888;">+{t2_pct:.1f}% · R:R 1:{rr2:.1f}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Main Chart (5 rows) ────────────────────────────────────────────
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            row_heights=[0.44, 0.12, 0.15, 0.14, 0.15],
            vertical_spacing=0.022,
            subplot_titles=(
                f"Giá · BB · EMA · Ichimoku · VWAP · S&D · Fibonacci — {ticker}",
                "Khối Lượng (Elder Impulse Colors)",
                "RSI(14) · StochRSI K/D",
                "MACD · MCDX",
                "TTM Squeeze Momentum"
            )
        )

        # Row 1: Candlestick
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Nến",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
            row=1, col=1)

        # BB
        for col_, color_, nm in [("BB_High", "rgba(255,80,80,0.5)", "BB Upper"),
                                   ("BB_Mid",  "rgba(180,180,180,0.4)", "BB Mid"),
                                   ("BB_Low",  "rgba(80,220,80,0.5)",  "BB Lower")]:
            fig.add_trace(go.Scatter(x=df["time"], y=df[col_],
                                     line=dict(color=color_, width=1), name=nm), row=1, col=1)

        # EMA
        for ec, eco, en in [("EMA5", "#FFD740", "EMA5"), ("EMA20", "#40C4FF", "EMA20"), ("EMA50", "#E040FB", "EMA50")]:
            fig.add_trace(go.Scatter(x=df["time"], y=df[ec],
                                     line=dict(color=eco, width=1.4, dash="dot"),
                                     name=en, opacity=0.85), row=1, col=1)

        # VWAP
        if "VWAP20" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["VWAP20"],
                                     line=dict(color="#FF6F00", width=2.0), name="VWAP20"), row=1, col=1)
            for col_, nm_, op_ in [("VWAP20_U1","VWAP+1σ",0.4),("VWAP20_U2","VWAP+2σ",0.3),
                                    ("VWAP20_L1","VWAP-1σ",0.4),("VWAP20_L2","VWAP-2σ",0.3)]:
                if col_ in df.columns:
                    fig.add_trace(go.Scatter(x=df["time"], y=df[col_],
                                             line=dict(color="#FF8F00", width=0.8, dash="dot"),
                                             name=nm_, opacity=op_, showlegend=False), row=1, col=1)

        # PSAR dots
        if "PSAR_Up" in df.columns and "PSAR_Down" in df.columns:
            psar_up_vals   = df["PSAR_Up"].where(df["PSAR_Up"].notna())
            psar_down_vals = df["PSAR_Down"].where(df["PSAR_Down"].notna())
            fig.add_trace(go.Scatter(x=df["time"], y=psar_up_vals,
                                     mode="markers", marker=dict(color="#00E676", size=4, symbol="circle"),
                                     name="PSAR Bull"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["time"], y=psar_down_vals,
                                     mode="markers", marker=dict(color="#FF5252", size=4, symbol="circle"),
                                     name="PSAR Bear"), row=1, col=1)

        # Ichimoku
        fig.add_trace(go.Scatter(x=df["time"], y=df["Tenkan"],
                                  line=dict(color="#00E5FF", width=1.6), name="Tenkan"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["Kijun"],
                                  line=dict(color="#FF6B6B", width=1.6), name="Kijun"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["Chikou"],
                                  line=dict(color="#B39DDB", width=1.0, dash="dot"),
                                  name="Chikou", opacity=0.6), row=1, col=1)
        spa = df["SpanA"].values; spb = df["SpanB"].values; tms = df["time"].values
        bull_a = np.where(spa >= spb, spa, np.nan); bull_b = np.where(spa >= spb, spb, np.nan)
        bear_a = np.where(spa <  spb, spa, np.nan); bear_b = np.where(spa <  spb, spb, np.nan)
        fig.add_trace(go.Scatter(x=tms, y=bull_a, line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=tms, y=bull_b, fill="tonexty", fillcolor="rgba(38,166,154,0.15)",
                                  line=dict(color="#26a69a", width=0.6), name="Kumo Bull"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tms, y=bear_b, line=dict(color="rgba(0,0,0,0)", width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=tms, y=bear_a, fill="tonexty", fillcolor="rgba(239,83,80,0.15)",
                                  line=dict(color="#ef5350", width=0.6), name="Kumo Bear"), row=1, col=1)

        # Supply & Demand Zones
        for dz in demand_zones:
            fig.add_shape(type="rect", xref="paper", yref="y",
                          x0=0, x1=1, y0=dz["bottom"], y1=dz["top"],
                          fillcolor="rgba(0,200,83,0.08)",
                          line=dict(color="#00C853", width=0.8, dash="dot"), row=1, col=1)
        for sz in supply_zones:
            fig.add_shape(type="rect", xref="paper", yref="y",
                          x0=0, x1=1, y0=sz["bottom"], y1=sz["top"],
                          fillcolor="rgba(255,23,68,0.07)",
                          line=dict(color="#FF1744", width=0.8, dash="dot"), row=1, col=1)

        # Volume Profile key levels on chart
        vp = vol_profile
        for lvl, color_, nm_ in [
            (vp["poc"], "#FFD740", f"VOL POC {vp['poc']:,.0f}"),
            (vp["vah"], "#FF9100", f"VAH {vp['vah']:,.0f}"),
            (vp["val"], "#40C4FF", f"VAL {vp['val']:,.0f}"),
        ]:
            fig.add_shape(type="line", xref="paper", yref="y",
                          x0=0, x1=1, y0=lvl, y1=lvl,
                          line=dict(color=color_, width=1.2, dash="dashdot"), row=1, col=1)
            fig.add_annotation(xref="paper", yref="y", x=1.01, y=lvl,
                               text=nm_, showarrow=False,
                               font=dict(color=color_, size=8), xanchor="left", row=1, col=1)

        # ICT Order Blocks
        for ob in ict_result["order_blocks"]:
            fill_c = "rgba(0,200,83,0.12)" if ob["type"] == "bullish" else "rgba(255,23,68,0.12)"
            line_c = "#00C853" if ob["type"] == "bullish" else "#FF1744"
            x0_ = df["time"].iloc[max(0, ob["idx"] - 1)] if ob["idx"] < len(df) else df["time"].iloc[-5]
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=x0_, x1=df["time"].iloc[-1], y0=ob["bottom"], y1=ob["top"],
                          fillcolor=fill_c, line=dict(color=line_c, width=1.0, dash="dot"), row=1, col=1)
            fig.add_annotation(xref="x", yref="y", x=df["time"].iloc[-1], y=(ob["top"] + ob["bottom"]) / 2,
                               text=f" OB {'🟢' if ob['type'] == 'bullish' else '🔴'}",
                               showarrow=False, font=dict(color=line_c, size=8), xanchor="left", row=1, col=1)

        # FVG
        for fvg in ict_result["fvg_list"][-3:]:
            fill_c = "rgba(0,229,118,0.07)" if fvg["type"] == "bullish" else "rgba(255,82,82,0.07)"
            line_c = "#00E676" if fvg["type"] == "bullish" else "#FF5252"
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=fvg["time"], x1=df["time"].iloc[-1],
                          y0=fvg["bottom"], y1=fvg["top"],
                          fillcolor=fill_c, line=dict(color=line_c, width=0.7, dash="dash"), row=1, col=1)

        # OTE zone
        if ict_result["ote_zone"]:
            ote = ict_result["ote_zone"]
            ote_c = "rgba(255,167,38,0.13)" if ote["type"] == "bullish" else "rgba(224,64,251,0.10)"
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1,
                          y0=ote["low"], y1=ote["high"],
                          fillcolor=ote_c, line=dict(color="#FFA726", width=1.0, dash="dot"), row=1, col=1)
            fig.add_annotation(xref="paper", yref="y", x=1.01, y=(ote["low"] + ote["high"]) / 2,
                               text=" OTE", showarrow=False,
                               font=dict(color="#FFA726", size=8), xanchor="left", row=1, col=1)

        # Fibonacci
        fib_pal = {
            "0.0% (Đỉnh)": "#9E9E9E", "0.0% (Đáy)": "#9E9E9E",
            "23.6%": "#7986CB", "38.2%": "#29B6F6", "50.0%": "#EF5350",
            "61.8% ✨": "#FFA726", "65.0% 🏅": "#FF7043", "78.6%": "#AB47BC",
            "100.0% (Đỉnh)": "#9E9E9E", "100.0% (Đáy)": "#9E9E9E",
            "127.2% 📉": "#546E7A", "127.2% 📈": "#546E7A",
            "161.8% 📉": "#37474F", "161.8% 📈": "#37474F"
        }
        x_range = [df["time"].iloc[0], df["time"].iloc[-1]]
        x_label = df["time"].iloc[-1]
        for label, price in fib_levels.items():
            if label.startswith("_"): continue
            is_key = label in ("38.2%", "50.0%", "61.8% ✨")
            fc = fib_pal.get(label, "#9E9E9E")
            fig.add_trace(go.Scatter(x=x_range, y=[price, price], mode="lines",
                                      line=dict(color=fc, width=2.2 if is_key else 0.9,
                                                dash="dash" if is_key else "dot"),
                                      name=f"Fib {label}", text=f"Fib {label} {price:,.0f}",
                                      hoverinfo="text"), row=1, col=1)
            fig.add_annotation(xref="x", yref="y", x=x_label, y=price,
                               text=f"  {label}·{price:,.0f}",
                               showarrow=False, font=dict(color=fc, size=8 if is_key else 7),
                               xanchor="left", row=1, col=1)

        # Row 2: Volume (Elder Impulse colored)
        elder_colors_vol = []
        for e in df["Elder"]:
            if e == "bull":    elder_colors_vol.append("#26a69a")
            elif e == "bear":  elder_colors_vol.append("#ef5350")
            else:              elder_colors_vol.append("#607D8B")
        fig.add_trace(go.Bar(x=df["time"], y=df["volume"],
                              marker_color=elder_colors_vol, name="Volume (Elder)", showlegend=True), row=2, col=1)

        # Row 3: RSI + StochRSI
        fig.add_trace(go.Scatter(x=df["time"], y=df["RSI"],
                                  line=dict(color="#FF9800", width=1.8), name="RSI"), row=3, col=1)
        if "StochRSI_K" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_K"],
                                      line=dict(color="#40C4FF", width=1.2, dash="dot"), name="StochRSI K"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df["time"], y=df["StochRSI_D"],
                                      line=dict(color="#FF7043", width=1.2, dash="dot"), name="StochRSI D"), row=3, col=1)
        for lv, lc in [(80, "rgba(255,80,80,0.5)"), (70, "rgba(255,80,80,0.3)"),
                        (30, "rgba(80,200,80,0.3)"), (20, "rgba(80,200,80,0.5)")]:
            fig.add_hline(y=lv, line_color=lc, line_width=1, line_dash="dash",
                          row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["ADX"],
                                  line=dict(color="#E040FB", width=1.3, dash="dot"), name="ADX"), row=3, col=1)

        # Row 4: MACD + MCDX
        hc = np.where(df["MACD_Diff"] >= 0, "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(x=df["time"], y=df["MACD_Diff"], marker_color=hc, name="MACD Hist"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["MACD"],
                                  line=dict(color="#2196F3", width=1.5), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["MACD_Signal"],
                                  line=dict(color="#FF5722", width=1.5), name="Signal"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["MCDX"],
                                  line=dict(color="#E040FB", width=1.8, dash="dot"), name="MCDX"), row=4, col=1)

        # Row 5: TTM Squeeze Momentum
        if "SQ_Mom" in df.columns:
            sq_colors = []
            prev_mom  = df["SQ_Mom"].shift(1)
            for i in range(len(df)):
                on  = df["SQ_On"].iloc[i]
                mom = df["SQ_Mom"].iloc[i]
                prv = prev_mom.iloc[i] if pd.notna(prev_mom.iloc[i]) else 0
                if on:
                    sq_colors.append("#607D8B")  # Squeeze on = grey
                elif mom > 0:
                    sq_colors.append("#00C853" if mom > prv else "#69F0AE")
                else:
                    sq_colors.append("#FF1744" if mom < prv else "#FF8A80")
            fig.add_trace(go.Bar(x=df["time"], y=df["SQ_Mom"], marker_color=sq_colors, name="TTM Squeeze Mom"), row=5, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=5, col=1)
            # Squeeze dots
            sq_dot_y = [0.01 if df["SQ_On"].iloc[i] else np.nan for i in range(len(df))]
            fig.add_trace(go.Scatter(x=df["time"], y=sq_dot_y,
                                      mode="markers", marker=dict(color="#FF1744", size=5, symbol="circle"),
                                      name="Squeeze ON"), row=5, col=1)

        fig.update_layout(
            template="plotly_dark", height=1200, xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        font=dict(size=8), bgcolor="rgba(0,0,0,0.3)", borderwidth=1),
            margin=dict(l=10, r=160, t=60, b=10))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Volume Profile Panel ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏔️ Volume Profile — Phân Phối Khối Lượng Theo Giá")
        st.caption("POC = Point of Control (khối lượng cao nhất) · VAH/VAL = Value Area High/Low (70% volume) · HVN/LVN = High/Low Volume Node")
        vp = vol_profile
        vp_col1, vp_col2 = st.columns([1, 2])
        with vp_col1:
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px 20px;
                    border:1px solid rgba(255,255,255,0.10);">
  <div style="margin-bottom:12px;">
    <div style="font-size:0.75rem;color:#aaa;">📍 Point of Control (POC)</div>
    <div style="font-size:1.4rem;font-weight:800;color:#FFD740;">{vp['poc']:,.0f} đ</div>
    <div style="font-size:0.72rem;color:#888;">Mức giá được giao dịch nhiều nhất</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div style="background:rgba(255,145,0,0.1);border-radius:8px;padding:10px;">
      <div style="font-size:0.70rem;color:#aaa;">⬆️ VAH (70%)</div>
      <div style="font-size:1.05rem;font-weight:700;color:#FF9100;">{vp['vah']:,.0f}</div>
    </div>
    <div style="background:rgba(64,196,255,0.1);border-radius:8px;padding:10px;">
      <div style="font-size:0.70rem;color:#aaa;">⬇️ VAL (70%)</div>
      <div style="font-size:1.05rem;font-weight:700;color:#40C4FF;">{vp['val']:,.0f}</div>
    </div>
  </div>
  <div style="margin-top:12px;font-size:0.78rem;color:#bbb;">
    {'🟢 Giá TRÊN POC — Bullish bias' if latest['close'] > vp['poc'] else '🔴 Giá DƯỚI POC — Bearish bias'}
  </div>
</div>""", unsafe_allow_html=True)
        with vp_col2:
            if vp["profile"]:
                max_vol = max(p["volume"] for p in vp["profile"]) or 1
                cp_ = latest["close"]
                vp_fig = go.Figure()
                bar_colors = []
                for p in vp["profile"]:
                    if abs(p["price"] - vp["poc"]) < (vp["vah"] - vp["val"]) / 25:
                        bar_colors.append("#FFD740")
                    elif vp["val"] <= p["price"] <= vp["vah"]:
                        bar_colors.append("#FF9100")
                    else:
                        bar_colors.append("#546E7A")
                vp_fig.add_trace(go.Bar(
                    x=[p["volume"] for p in vp["profile"]],
                    y=[p["price"] for p in vp["profile"]],
                    orientation="h", marker_color=bar_colors, name="Volume Profile",
                    hovertemplate="%{y:,.0f} đ<br>Vol: %{x:,.0f}<extra></extra>"))
                for lvl, color_, nm_ in [(vp["poc"], "#FFD740", "POC"),
                                          (vp["vah"], "#FF9100", "VAH"),
                                          (vp["val"], "#40C4FF", "VAL"),
                                          (cp_,       "#00E676", "Giá hiện tại")]:
                    vp_fig.add_hline(y=lvl, line_color=color_, line_dash="dash",
                                      line_width=1.5, annotation_text=nm_,
                                      annotation_font_color=color_)
                vp_fig.update_layout(template="plotly_dark", height=300,
                                      margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=False,
                                      xaxis_title="Volume", yaxis_title="Giá")
                st.plotly_chart(vp_fig, use_container_width=True)

        # ── ICT Section ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Phân Tích ICT — Inner Circle Trader")
        st.caption("Market Structure · Order Blocks · Fair Value Gaps · Liquidity · OTE")
        ict_c = "#00E676" if ict_result["score"] > 0 else ("#FF5252" if ict_result["score"] < 0 else "#FFD740")
        st.markdown(
            f"""<div style="background:rgba(64,196,255,0.07);border:1px solid #40C4FF44;
                border-left:4px solid #40C4FF;border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.72rem;color:#aaa;">🏗️ Market Structure</div>
         <div style="font-size:1.1rem;font-weight:700;color:#40C4FF;">{ict_result['structure']}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">📦 Order Blocks</div>
         <div style="font-size:1.1rem;font-weight:700;color:#FFA726;">{len(ict_result['order_blocks'])} vùng</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">📊 FVG</div>
         <div style="font-size:1.1rem;font-weight:700;color:#E040FB;">{len(ict_result['fvg_list'])} gap</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">🔀 BOS/CHoCH</div>
         <div style="font-size:1.1rem;font-weight:700;color:#00E676;">
           {'BOS ✅' if ict_result.get('bos_bullish') or ict_result.get('bos_bearish') else '—'}
           {'⚡' if ict_result.get('choch') else ''}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">🎯 ICT Score</div>
         <div style="font-size:1.1rem;font-weight:700;color:{ict_c};">{ict_result['score']:+d}</div></div>
  </div>
</div>""", unsafe_allow_html=True)
        if ict_result["signals"]:
            ict_s1, ict_s2 = st.columns(2)
            ict_col_map = {"bullish": "#00E676", "bearish": "#FF5252", "neutral": "#FFD740"}
            for i, (sname, sdesc, stype) in enumerate(ict_result["signals"]):
                sc = ict_col_map.get(stype, "#aaa")
                with (ict_s1 if i % 2 == 0 else ict_s2):
                    st.markdown(
                        f"<div style='border-left:3px solid {sc};padding:8px 12px;margin:4px 0;"
                        f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                        f"<b style='color:{sc};'>{sname}</b><br>"
                        f"<span style='color:#bbb;font-size:0.82rem;'>{sdesc}</span></div>",
                        unsafe_allow_html=True)
        with st.expander("📋 Chi tiết Order Blocks, FVG & Liquidity"):
            ob_c1, ob_c2 = st.columns(2)
            with ob_c1:
                st.markdown("**📦 Order Blocks:**")
                for ob in ict_result["order_blocks"]:
                    c_ = "🟢" if ob["type"] == "bullish" else "🔴"
                    st.markdown(f"- {c_} {ob['desc']}")
                if not ict_result["order_blocks"]: st.info("Không phát hiện OB trong 30 phiên")
            with ob_c2:
                st.markdown("**📊 Fair Value Gaps:**")
                for fvg in ict_result["fvg_list"][-5:]:
                    c_ = "🟢" if fvg["type"] == "bullish" else "🔴"
                    st.markdown(f"- {c_} FVG {fvg['type']} tại {fvg['bottom']:,.0f}–{fvg['top']:,.0f} ({fvg['pct']:.1f}%)")
                if not ict_result["fvg_list"]: st.info("Không phát hiện FVG đáng kể")
            if ict_result["liq_highs"] or ict_result["liq_lows"]:
                st.markdown(
                    f"**⚠️ Liquidity Zones:** "
                    f"Equal Highs: `{', '.join([f'{h:,.0f}' for h in ict_result['liq_highs']])}` | "
                    f"Equal Lows: `{', '.join([f'{l:,.0f}' for l in ict_result['liq_lows']])}`")
        if ict_result["ote_zone"]:
            ote = ict_result["ote_zone"]
            st.info(f"🎯 **{ote['desc']}** — {'Vùng vào mua lý tưởng ICT' if ote['type'] == 'bullish' else 'Vùng vào bán lý tưởng ICT'}")

        # ── VSA Section ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Phân Tích VSA — Volume Spread Analysis")
        st.caption("Stopping Volume · No Demand · No Supply · Climax · Upthrust · Test")
        vsa_c = "#00E676" if vsa_result["score"] > 0 else ("#FF5252" if vsa_result["score"] < 0 else "#FFD740")
        bull_vsa = [p for p in vsa_result["patterns"] if any(k in p[0] for k in ["Stopping", "No Supply", "Selling Climax", "Test", "✅"])]
        bear_vsa = [p for p in vsa_result["patterns"] if any(k in p[0] for k in ["No Demand", "Buying Climax", "Upthrust", "❌"])]
        st.markdown(
            f"""<div style="background:rgba(255,167,38,0.07);border:1px solid #FFA72644;
                border-left:4px solid #FFA726;border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.72rem;color:#aaa;">🟢 Tín hiệu Tăng</div>
         <div style="font-size:1.4rem;font-weight:800;color:#00E676;">{len(bull_vsa)}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">🔴 Tín hiệu Giảm</div>
         <div style="font-size:1.4rem;font-weight:800;color:#FF5252;">{len(bear_vsa)}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">📋 Tổng Patterns</div>
         <div style="font-size:1.4rem;font-weight:800;color:#FFA726;">{len(vsa_result['patterns'])}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">⚖️ VSA Score</div>
         <div style="font-size:1.4rem;font-weight:800;color:{vsa_c};">{vsa_result['score']:+d}</div></div>
  </div>
</div>""", unsafe_allow_html=True)
        if vsa_result["patterns"]:
            for pname, pt, pc_val, pdesc in vsa_result["patterns"]:
                is_bull = any(k in pname for k in ["Stopping", "No Supply", "Selling Climax", "Test", "✅"])
                is_bear = any(k in pname for k in ["No Demand", "Buying Climax", "Upthrust", "❌"])
                color_v = "#00E676" if is_bull else ("#FF5252" if is_bear else "#FFD740")
                st.markdown(
                    f"<div style='border-left:3px solid {color_v};padding:8px 14px;margin:4px 0;"
                    f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                    f"<b style='color:{color_v};'>{pname}</b> "
                    f"<span style='color:#888;font-size:0.78rem;'>phiên {pt} · {pc_val:,.0f} đ</span><br>"
                    f"<span style='color:#bbb;font-size:0.82rem;'>{pdesc}</span></div>",
                    unsafe_allow_html=True)
        else:
            st.info("Không phát hiện VSA pattern đáng kể trong 15 phiên gần nhất")

        # ── Price Action Section ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("🕯️ Phân Tích Price Action")
        st.caption("Market Structure · Inside/Outside Bar · Key S/R · Pin Bar · Consolidation")
        pa_c = "#00E676" if pa_result["score"] > 0 else ("#FF5252" if pa_result["score"] < 0 else "#FFD740")
        st.markdown(
            f"""<div style="background:rgba(224,64,251,0.07);border:1px solid #E040FB44;
                border-left:4px solid #E040FB;border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.72rem;color:#aaa;">📐 Cấu trúc PA</div>
         <div style="font-size:0.95rem;font-weight:700;color:#E040FB;">{pa_result.get('trend_structure','N/A')}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">📦 Inside Bars</div>
         <div style="font-size:1.4rem;font-weight:800;color:#FFA726;">{pa_result.get('inside_bars',0)}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">🔑 Key Levels</div>
         <div style="font-size:1.4rem;font-weight:800;color:#40C4FF;">{len(pa_result.get('key_levels',[]))}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">⚖️ PA Score</div>
         <div style="font-size:1.4rem;font-weight:800;color:{pa_c};">{pa_result['score']:+d}</div></div>
  </div>
  {('<div style="margin-top:10px;font-size:0.82rem;color:#FFD740;">⏳ ĐANG TÍCH LŨY — Sắp breakout</div>' if pa_result.get("is_consolidating") else '')}
</div>""", unsafe_allow_html=True)
        if pa_result["signals"]:
            pa_c1, pa_c2 = st.columns(2)
            for i, (sn, sd, st_) in enumerate(pa_result["signals"]):
                sc = {"bullish": "#00E676", "bearish": "#FF5252", "neutral": "#FFD740"}.get(st_, "#aaa")
                with (pa_c1 if i % 2 == 0 else pa_c2):
                    st.markdown(
                        f"<div style='border-left:3px solid {sc};padding:8px 12px;margin:4px 0;"
                        f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                        f"<b style='color:{sc};'>{sn}</b><br>"
                        f"<span style='color:#bbb;font-size:0.82rem;'>{sd}</span></div>",
                        unsafe_allow_html=True)

        # ── Overflow Section ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💧 Phân Tích Overflow — Volume · Price · Breakout · Gap")
        ovf_c = "#00E676" if ovf_result["score"] > 0 else ("#FF5252" if ovf_result["score"] < 0 else "#FFD740")
        st.markdown(
            f"""<div style="background:rgba(255,23,68,0.07);border:1px solid #FF174444;
                border-left:4px solid #FF1744;border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.72rem;color:#aaa;">🌊 Volume Events</div>
         <div style="font-size:1.4rem;font-weight:800;color:#FF9100;">{len(ovf_result['overflow_events'])}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">📐 Gaps</div>
         <div style="font-size:1.4rem;font-weight:800;color:#40C4FF;">{len(ovf_result['gaps'])}</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">🔁 Consecutive Run</div>
         <div style="font-size:1.4rem;font-weight:800;color:#FFD740;">{ovf_result.get('consecutive_run',0)}P</div></div>
    <div><div style="font-size:0.72rem;color:#aaa;">⚖️ Overflow Score</div>
         <div style="font-size:1.4rem;font-weight:800;color:{ovf_c};">{ovf_result['score']:+d}</div></div>
  </div>
  {('<div style="margin-top:8px;color:#FF5252;font-size:0.82rem;">⚡ Giá vượt BB Upper — Quá mua</div>' if ovf_result.get("bb_overflow_up") else '')}
  {('<div style="margin-top:8px;color:#00E676;font-size:0.82rem;">💧 Giá dưới BB Lower — Quá bán</div>' if ovf_result.get("bb_overflow_down") else '')}
</div>""", unsafe_allow_html=True)
        if ovf_result["signals"]:
            ovf_c1, ovf_c2 = st.columns(2)
            for i, (sn, sd, stype_) in enumerate(ovf_result["signals"]):
                sc = {"bullish": "#00E676", "bearish": "#FF5252", "neutral": "#FFD740"}.get(stype_, "#aaa")
                with (ovf_c1 if i % 2 == 0 else ovf_c2):
                    st.markdown(
                        f"<div style='border-left:3px solid {sc};padding:8px 12px;margin:4px 0;"
                        f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                        f"<b style='color:{sc};'>{sn}</b><br>"
                        f"<span style='color:#bbb;font-size:0.82rem;'>{sd}</span></div>",
                        unsafe_allow_html=True)

        # ── Wyckoff Events Section ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("🌀 Wyckoff Events & Phase Detection")
        st.caption("SC · BC · Spring · AR · ST · SOS · LPSY · No Supply — Phân tích pha Wyckoff")
        wy = wy_result
        wy_c = "#00C853" if "MARKUP" in wy["phase"] or "ACCUMULATION" in wy["phase"] else \
               ("#FF1744" if "DISTRIBUTION" in wy["phase"] else "#FFD740")
        st.markdown(
            f"""<div style="background:rgba(100,120,255,0.08);border:1px solid rgba(100,120,255,0.3);
                border-left:4px solid #7986CB;border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="font-size:1.1rem;font-weight:700;color:{wy_c};margin-bottom:6px;">🌀 {wy['phase']}</div>
  <div style="font-size:0.85rem;color:#bbb;margin-bottom:10px;">{wy['phase_desc']}</div>
  <div style="font-size:0.75rem;color:#aaa;">Sự kiện phát hiện: {len(wy['events'])} · Score: {wy['score']:+d}</div>
</div>""", unsafe_allow_html=True)
        if wy["events"]:
            for ev in reversed(wy["events"]):
                bull_color = "#00C853" if ev.get("bullish") else "#FF5252"
                st.markdown(
                    f"<div style='border-left:3px solid {bull_color};padding:7px 12px;margin:3px 0;"
                    f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                    f"<b style='color:{bull_color};'>{ev['emoji']} {ev['label']}</b>"
                    f"<span style='color:#888;font-size:0.78rem;'> · phiên {ev['time']} · {ev['price']:,.0f} đ</span><br>"
                    f"<span style='color:#bbb;font-size:0.82rem;'>{ev['desc']}</span></div>",
                    unsafe_allow_html=True)
        else:
            st.info("Chưa phát hiện sự kiện Wyckoff rõ ràng trong dữ liệu gần đây")

        # ── Hidden Divergence Section ──────────────────────────────────────
        st.markdown("---")
        st.subheader("🔀 Phân Kỳ — Regular & Hidden Divergence (RSI)")
        st.caption("Regular Divergence → Đảo chiều · Hidden Divergence → Tiếp tục xu hướng (thường mạnh hơn)")
        div = div_result
        div_items = [
            ("📈 Regular Bullish", "Giá LL · RSI HL → Đảo chiều tăng tiềm năng", div["reg_bull"], "#00C853"),
            ("📉 Regular Bearish", "Giá HH · RSI LH → Đảo chiều giảm tiềm năng", div["reg_bear"], "#FF5252"),
            ("🔒 Hidden Bullish",  "Giá HL · RSI LL → Tiếp tục tăng (mạnh hơn Regular)", div["hid_bull"], "#40C4FF"),
            ("🔒 Hidden Bearish",  "Giá LH · RSI HH → Tiếp tục giảm (mạnh hơn Regular)", div["hid_bear"], "#FF9100"),
        ]
        dc1, dc2, dc3, dc4 = st.columns(4)
        for col_, (label_, desc_, detected_, color_) in zip([dc1, dc2, dc3, dc4], div_items):
            status  = "✅ PHÁT HIỆN" if detected_ else "—"
            bg_div  = f"{color_}22" if detected_ else "rgba(255,255,255,0.04)"
            border_ = f"1px solid {color_}" if detected_ else "1px solid rgba(255,255,255,0.1)"
            col_.markdown(
                f"<div style='text-align:center;padding:12px 8px;border-radius:10px;"
                f"background:{bg_div};border:{border_};'>"
                f"<div style='font-size:0.80rem;font-weight:700;color:{color_ if detected_ else '#888'};'>{label_}</div>"
                f"<div style='font-size:1.1rem;font-weight:800;color:{color_ if detected_ else '#555'};margin:6px 0;'>{status}</div>"
                f"<div style='font-size:0.68rem;color:#999;'>{desc_}</div></div>",
                unsafe_allow_html=True)
        if div["signals"]:
            st.markdown("<br>", unsafe_allow_html=True)
            for sname, sdesc, stype in div["signals"]:
                sc = {"bullish": "#00C853", "bearish": "#FF5252"}.get(stype, "#FFD740")
                st.markdown(
                    f"<div style='border-left:3px solid {sc};padding:8px 12px;margin:4px 0;"
                    f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                    f"<b style='color:{sc};'>{sname}</b><br>"
                    f"<span style='color:#bbb;font-size:0.82rem;'>{sdesc}</span></div>",
                    unsafe_allow_html=True)

        # ── RSI Divergence Chart (40 phiên gần nhất) ─────────────────────
        with st.expander("📊 Xem Biểu Đồ RSI Phân Kỳ (40 phiên gần nhất)", expanded=any([div["reg_bull"], div["reg_bear"], div["hid_bull"], div["hid_bear"]])):
            div_window = min(40, len(df))
            df_div = df.tail(div_window).reset_index(drop=True)
            prices_dv = df_div["close"].values
            rsi_dv    = df_div["RSI"].fillna(50).values
            n_dv      = len(prices_dv)

            fig_div = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.55, 0.45],
                                     vertical_spacing=0.04,
                                     subplot_titles=("Giá đóng cửa — Phát hiện Phân Kỳ",
                                                      "RSI(14) — Phân Kỳ RSI"))

            # Vẽ đường giá
            fig_div.add_trace(go.Scatter(x=df_div["time"], y=df_div["close"],
                                          line=dict(color="#40C4FF", width=2), name="Giá"), row=1, col=1)

            # Vẽ RSI
            fig_div.add_trace(go.Scatter(x=df_div["time"], y=df_div["RSI"],
                                          line=dict(color="#FF9800", width=2), name="RSI"), row=2, col=1)
            for lv_d, lc_d in [(70, "rgba(255,80,80,0.4)"), (30, "rgba(80,200,80,0.4)"), (50, "rgba(180,180,180,0.2)")]:
                fig_div.add_hline(y=lv_d, line_color=lc_d, line_width=1, line_dash="dash", row=2, col=1)

            # Tính lại swings để vẽ markers
            def _find_swings_chart(arr, is_high, strength=2):
                n_ = len(arr)
                out = []
                s = max(1, strength)
                for ii in range(s * 2, n_ - s * 2):
                    if is_high:
                        if all(arr[ii] > arr[ii - k] for k in range(1, s + 1)) and \
                           all(arr[ii] > arr[ii + k] for k in range(1, s + 1)):
                            out.append(ii)
                    else:
                        if all(arr[ii] < arr[ii - k] for k in range(1, s + 1)) and \
                           all(arr[ii] < arr[ii + k] for k in range(1, s + 1)):
                            out.append(ii)
                # fallback strength=1 nếu không đủ
                if len(out) < 2:
                    out = []
                    for ii in range(2, n_ - 2):
                        if is_high:
                            if arr[ii] > arr[ii-1] and arr[ii] > arr[ii+1]:
                                out.append(ii)
                        else:
                            if arr[ii] < arr[ii-1] and arr[ii] < arr[ii+1]:
                                out.append(ii)
                return out

            ph_idx = _find_swings_chart(prices_dv, True)
            pl_idx = _find_swings_chart(prices_dv, False)
            rh_idx = _find_swings_chart(rsi_dv, True)
            rl_idx = _find_swings_chart(rsi_dv, False)

            # Vẽ markers swing trên giá
            if ph_idx:
                fig_div.add_trace(go.Scatter(
                    x=[df_div["time"].iloc[i] for i in ph_idx],
                    y=[prices_dv[i] for i in ph_idx],
                    mode="markers", marker=dict(color="#FF5252", size=8, symbol="triangle-down"),
                    name="Price High", showlegend=True), row=1, col=1)
            if pl_idx:
                fig_div.add_trace(go.Scatter(
                    x=[df_div["time"].iloc[i] for i in pl_idx],
                    y=[prices_dv[i] for i in pl_idx],
                    mode="markers", marker=dict(color="#00C853", size=8, symbol="triangle-up"),
                    name="Price Low", showlegend=True), row=1, col=1)

            # Vẽ markers swing trên RSI
            if rh_idx:
                fig_div.add_trace(go.Scatter(
                    x=[df_div["time"].iloc[i] for i in rh_idx],
                    y=[rsi_dv[i] for i in rh_idx],
                    mode="markers", marker=dict(color="#FF5252", size=7, symbol="triangle-down"),
                    name="RSI High", showlegend=False), row=2, col=1)
            if rl_idx:
                fig_div.add_trace(go.Scatter(
                    x=[df_div["time"].iloc[i] for i in rl_idx],
                    y=[rsi_dv[i] for i in rl_idx],
                    mode="markers", marker=dict(color="#00C853", size=7, symbol="triangle-up"),
                    name="RSI Low", showlegend=False), row=2, col=1)

            # Vẽ đường phân kỳ nếu phát hiện
            def _nearest_idx(p_idx, swing_idxs, max_dist=10):
                candidates = [i for i in swing_idxs if abs(i - p_idx) <= max_dist]
                return min(candidates, key=lambda x: abs(x - p_idx)) if candidates else None

            # Regular Bullish: nối 2 đáy giá + 2 đáy RSI
            if div["reg_bull"] and len(pl_idx) >= 2:
                i2, i1 = pl_idx[-1], pl_idx[-2]
                r2_ = _nearest_idx(i2, rl_idx)
                r1_ = _nearest_idx(i1, rl_idx)
                if r1_ is not None and r2_ is not None:
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[i1], df_div["time"].iloc[i2]],
                        y=[prices_dv[i1], prices_dv[i2]],
                        mode="lines", line=dict(color="#00C853", width=2.5, dash="dot"),
                        name="📈 Reg Bull Div", showlegend=True), row=1, col=1)
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[r1_], df_div["time"].iloc[r2_]],
                        y=[rsi_dv[r1_], rsi_dv[r2_]],
                        mode="lines", line=dict(color="#00C853", width=2.5, dash="dot"),
                        showlegend=False), row=2, col=1)

            # Regular Bearish: nối 2 đỉnh giá + 2 đỉnh RSI
            if div["reg_bear"] and len(ph_idx) >= 2:
                i2, i1 = ph_idx[-1], ph_idx[-2]
                r2_ = _nearest_idx(i2, rh_idx)
                r1_ = _nearest_idx(i1, rh_idx)
                if r1_ is not None and r2_ is not None:
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[i1], df_div["time"].iloc[i2]],
                        y=[prices_dv[i1], prices_dv[i2]],
                        mode="lines", line=dict(color="#FF5252", width=2.5, dash="dot"),
                        name="📉 Reg Bear Div", showlegend=True), row=1, col=1)
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[r1_], df_div["time"].iloc[r2_]],
                        y=[rsi_dv[r1_], rsi_dv[r2_]],
                        mode="lines", line=dict(color="#FF5252", width=2.5, dash="dot"),
                        showlegend=False), row=2, col=1)

            # Hidden Bullish: nối 2 đáy giá + 2 đáy RSI (màu xanh nước)
            if div["hid_bull"] and len(pl_idx) >= 2:
                i2, i1 = pl_idx[-1], pl_idx[-2]
                r2_ = _nearest_idx(i2, rl_idx)
                r1_ = _nearest_idx(i1, rl_idx)
                if r1_ is not None and r2_ is not None:
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[i1], df_div["time"].iloc[i2]],
                        y=[prices_dv[i1], prices_dv[i2]],
                        mode="lines", line=dict(color="#40C4FF", width=2.5, dash="dash"),
                        name="🔒 Hid Bull Div", showlegend=True), row=1, col=1)
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[r1_], df_div["time"].iloc[r2_]],
                        y=[rsi_dv[r1_], rsi_dv[r2_]],
                        mode="lines", line=dict(color="#40C4FF", width=2.5, dash="dash"),
                        showlegend=False), row=2, col=1)

            # Hidden Bearish: nối 2 đỉnh giá + 2 đỉnh RSI (màu cam)
            if div["hid_bear"] and len(ph_idx) >= 2:
                i2, i1 = ph_idx[-1], ph_idx[-2]
                r2_ = _nearest_idx(i2, rh_idx)
                r1_ = _nearest_idx(i1, rh_idx)
                if r1_ is not None and r2_ is not None:
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[i1], df_div["time"].iloc[i2]],
                        y=[prices_dv[i1], prices_dv[i2]],
                        mode="lines", line=dict(color="#FF9100", width=2.5, dash="dash"),
                        name="🔒 Hid Bear Div", showlegend=True), row=1, col=1)
                    fig_div.add_trace(go.Scatter(
                        x=[df_div["time"].iloc[r1_], df_div["time"].iloc[r2_]],
                        y=[rsi_dv[r1_], rsi_dv[r2_]],
                        mode="lines", line=dict(color="#FF9100", width=2.5, dash="dash"),
                        showlegend=False), row=2, col=1)

            fig_div.update_layout(
                template="plotly_dark", height=550,
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                            xanchor="right", x=1, font=dict(size=9),
                            bgcolor="rgba(0,0,0,0.3)"),
                margin=dict(l=10, r=10, t=40, b=10))
            fig_div.update_xaxes(showgrid=False)
            fig_div.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_div, use_container_width=True)

            if not any([div["reg_bull"], div["reg_bear"], div["hid_bull"], div["hid_bear"]]):
                st.info("ℹ️ Chưa phát hiện phân kỳ RSI rõ ràng trong 40 phiên gần nhất. "
                        "Các điểm tam giác trên biểu đồ là các swing high/low được thuật toán nhận diện.")

        # ── Supply & Demand Zones ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("🎯 Supply & Demand Zones")
        st.caption("Demand Zone: Base+Rally → Vùng mua tiềm năng · Supply Zone: Base+Drop → Vùng kháng cự tiềm năng")
        sd_c1, sd_c2 = st.columns(2)
        with sd_c1:
            st.markdown("**📥 Demand Zones (Vùng Mua):**")
            if demand_zones:
                for dz in demand_zones:
                    fresh_label = "🟢 Fresh" if dz["fresh"] else "⚪ Tested"
                    st.markdown(
                        f"<div style='border-left:3px solid #00C853;padding:8px 12px;margin:4px 0;"
                        f"background:rgba(0,200,83,0.07);border-radius:0 8px 8px 0;'>"
                        f"<b style='color:#00C853;'>{fresh_label} Demand: {dz['bottom']:,.0f}–{dz['top']:,.0f}</b><br>"
                        f"<span style='color:#bbb;font-size:0.80rem;'>Sức mạnh: +{dz['strength']:.1f}% · phiên {dz['time']}</span></div>",
                        unsafe_allow_html=True)
            else:
                st.info("Không phát hiện Demand Zone rõ ràng")
        with sd_c2:
            st.markdown("**📤 Supply Zones (Vùng Bán):**")
            if supply_zones:
                for sz in supply_zones:
                    fresh_label = "🔴 Fresh" if sz["fresh"] else "⚪ Tested"
                    st.markdown(
                        f"<div style='border-left:3px solid #FF1744;padding:8px 12px;margin:4px 0;"
                        f"background:rgba(255,23,68,0.07);border-radius:0 8px 8px 0;'>"
                        f"<b style='color:#FF1744;'>{fresh_label} Supply: {sz['bottom']:,.0f}–{sz['top']:,.0f}</b><br>"
                        f"<span style='color:#bbb;font-size:0.80rem;'>Sức mạnh: -{sz['strength']:.1f}% · phiên {sz['time']}</span></div>",
                        unsafe_allow_html=True)
            else:
                st.info("Không phát hiện Supply Zone rõ ràng")

        # ── Smart Money Phase ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🐋 Pha Dòng Tiền Thông Minh — OBV · CMF · MFI · A/D")
        sm = sm_result
        pct_bar = max(0, min(100, (sm["score"] + sm["max_score"]) / (2 * sm["max_score"]) * 100))
        st.markdown(
            f"""<div style="border:2px solid {sm['border']};border-radius:14px;padding:20px 28px;
                background:{sm['bg']};margin-bottom:16px;">
  <div style="font-size:1.9rem;font-weight:800;color:{sm['color']};letter-spacing:2px;">
    {sm['icon']} &nbsp; GIAI ĐOẠN: {sm['phase']}
  </div>
  <div style="margin-top:8px;font-size:1rem;color:#ddd;line-height:1.6;">{sm['phase_desc']}</div>
  <div style="margin-top:12px;background:rgba(255,255,255,0.08);border-radius:8px;height:10px;width:100%;">
    <div style="height:10px;width:{pct_bar:.0f}%;border-radius:8px;
                background:linear-gradient(90deg,#ef5350,#FFD740,#00C853);"></div>
  </div>
  <div style="margin-top:8px;font-size:0.9rem;color:#aaa;">
    Điểm: <b style="color:{sm['color']};">{sm['score']:+.1f} / {sm['max_score']:.0f}</b>
  </div>
</div>""", unsafe_allow_html=True)
        sm_cols = st.columns(4)
        sm_metrics = [
            ("OBV Slope", f"{sm_result.get('obv_slope', 0):+.2f}", "#40C4FF"),
            ("CMF",       f"{sm_result.get('cmf', 0):+.3f}",       "#00C853" if (sm_result.get('cmf', 0) or 0) > 0 else "#FF5252"),
            ("MFI",       f"{sm_result.get('mfi', 50):.1f}",        "#FF9800"),
            ("Buy Ratio", f"{sm_result.get('buy_ratio', 0.5)*100:.0f}%", "#E040FB"),
        ]
        for col_, (label_, val_, col2_) in zip(sm_cols, sm_metrics):
            col_.metric(label_, val_)

        # ── Data Table ─────────────────────────────────────────────────────
        show_cols = [c for c in ["time", "open", "high", "low", "close", "volume",
                                  "RSI", "MACD", "MCDX", "ADX", "EMA20", "VWAP20", "ATR14"] if c in df.columns]
        st.write("**📋 Bảng dữ liệu 7 phiên gần nhất:**")
        st.dataframe(df.tail(7)[show_cols].round(2), hide_index=True, use_container_width=True)

        with st.expander("📐 Bảng Fibonacci Retracement"):
            fib_df = pd.DataFrame([{"Mức": k, "Giá (đ)": f"{v:,.0f}"}
                                    for k, v in fib_levels.items() if not k.startswith("_")])
            st.dataframe(fib_df, hide_index=True, use_container_width=True)

        # ── AI Analysis ────────────────────────────────────────────────────
        if not api_key:
            st.warning("💡 Nhập API Key ở thanh bên trái để nhận phân tích từ AI.")
        else:
            st.markdown("---")
            provider = "Claude (Anthropic)" if "Claude" in ai_provider else "Gemini (Google)"
            st.subheader(f"🤖 Báo Cáo AI Tổng Hợp — {provider}")
            ai_data = df.tail(10)[show_cols[:8]].round(2).to_string(index=False)
            fib_str = "\n".join([f"  • Fib {k}: {v:,.0f} đ" for k, v in fib_levels.items()
                                  if not k.startswith("_") and isinstance(v, (int, float))])
            ict_sigs = "\n".join([f"  • {s[0]}: {s[1]}" for s in ict_result["signals"]]) or "  • Không có"
            vsa_sigs = "\n".join([f"  • {p[0]} (phiên {p[1]}): {p[3]}" for p in vsa_result["patterns"]]) or "  • Không có"
            pa_sigs  = "\n".join([f"  • {s[0]}: {s[1]}" for s in pa_result["signals"]]) or "  • Không có"
            ovf_sigs = "\n".join([f"  • {s[0]}: {s[1]}" for s in ovf_result["signals"]]) or "  • Không có"
            wy_sigs  = "\n".join([f"  • {e['emoji']} {e['label']} ({e['time']}): {e['desc']}" for e in wy_result["events"]]) or "  • Không có"
            div_sigs = "\n".join([f"  • {s[0]}: {s[1]}" for s in div_result["signals"]]) or "  • Không có"
            above_cloud = None
            if pd.notna(latest.get("SpanA")) and pd.notna(latest.get("SpanB")):
                above_cloud = latest["close"] > max(latest.get("SpanA"), latest.get("SpanB"))
            ichi_status = ("trên mây (Bullish)" if above_cloud else ("dưới mây (Bearish)" if above_cloud is False else "trong mây"))
            macd_sig = "cắt lên (tăng)" if latest["MACD"] > latest["MACD_Signal"] else "cắt xuống (giảm)"
            rsi_note2 = "Quá mua" if rsi_val > 70 else ("Quá bán" if rsi_val < 30 else "Trung tính")

            prompt = f"""
Bạn là hệ thống AI định lượng cao cấp chuyên phân tích thị trường chứng khoán Việt Nam.
Sử dụng 16 phương pháp: ICT, VSA, Price Action, Overflow, Smart Money, Wyckoff, Hidden Divergence, Volume Profile, VWAP, TTM Squeeze, Heikin-Ashi, Elder Impulse, PSAR, Supply/Demand, StochRSI, ATR.

Dữ liệu 10 phiên gần nhất của mã **{ticker}**:
```
{ai_data}
```

Fibonacci: {fib_str}

Chỉ báo kỹ thuật:
- RSI: {rsi_val:.1f} — {rsi_note2}
- MACD: {macd_sig}
- ADX: {adx_val:.1f} (DI+={di_p:.1f}/DI-={di_m:.1f})
- EMA 5/20/50: {e5_v:,.0f}/{e20_v:,.0f}/{e50_v:,.0f}
- Ichimoku: {ichi_status}
- VWAP20: {vwap20:,.0f} ({'Giá trên VWAP — Bullish' if pd.notna(vwap20) and latest['close']>vwap20 else 'Giá dưới VWAP — Bearish'})
- TTM Squeeze: {'ON (đang nén)' if sq_on else f'OFF (Momentum: {sq_mom:.3f})'}
- Heikin-Ashi: {'Tăng' if ha_dir==1 else 'Giảm'} {ha_consec} phiên liên tiếp
- ATR(14): {atr14:,.0f} (SL gợi ý: {latest['close']-1.5*atr14:,.0f})
- StochRSI K/D: {stoch_k:.0f}/{stoch_d:.0f}
- Volume Profile POC: {vol_profile['poc']:,.0f} | VAH: {vol_profile['vah']:,.0f} | VAL: {vol_profile['val']:,.0f}
- Market Regime: {regime['regime']}
- Smart Money: {sm_result['phase']} (score: {sm_result['score']:+.1f})

ICT Signals:
{ict_sigs}
ICT Summary: {ict_result['summary']}

VSA Patterns:
{vsa_sigs}

Price Action:
{pa_sigs}

Overflow:
{ovf_sigs}

Wyckoff Events:
{wy_sigs}
Wyckoff Phase: {wy_result['phase']}

Divergence:
{div_sigs}

**Phân tích yêu cầu (tiếng Việt, quyết đoán, ngắn gọn súc tích):**

1. **Market Regime & Trend**: Thị trường đang ở pha nào? Xu hướng chính xác nhận chưa?
2. **Wyckoff + Smart Money**: Tiền lớn đang làm gì? Pha tích lũy/phân phối/markup?
3. **ICT + VSA + PA Confluence**: Điểm hội tụ quan trọng nhất?
4. **Volume Profile + VWAP**: Giá đang ở vị trí nào so với vùng giá trị?
5. **Divergence + TTM Squeeze**: Có tín hiệu bứt phá hoặc đảo chiều không?
6. **Kết luận & Chiến lược**:
   - ✅ Khuyến nghị: **Mua / Bán / Chờ**
   - 📥 Entry zone, 🛑 Stop Loss, 🎯 Target 1 & 2
   - ⚠️ Rủi ro chính cần chú ý
"""
            with st.spinner("⏳ AI đang phân tích đa phương pháp (16 kỹ thuật)..."):
                try:
                    result = call_claude(api_key, prompt) if "Claude" in ai_provider else call_gemini(api_key, prompt)
                    st.markdown(result)
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        st.error("❌ Hết quota API. Chờ ~1 phút hoặc đổi sang Claude.")
                    elif "401" in err or "UNAUTHENTICATED" in err:
                        st.error("❌ API Key không hợp lệ.")
                    else:
                        st.error(f"❌ Lỗi AI: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET SCANNER
# ═════════════════════════════════════════════════════════════════════════════
with _tab2:
    st.subheader("🔍 Quét Toàn Thị Trường — 16 Phương Pháp · ICT · VSA · PA · Smart Money · Wyckoff")

    with st.spinner("🔄 Đang tải danh sách cổ phiếu..."):
        _auto_universe, _source_note = build_scan_universe()

    st.caption(f"📡 Nguồn: **{_source_note}**")
    _b1, _b2, _b3 = st.columns(3)
    _b1.metric("🏆 VN30", len([t for t in VN30_TICKERS if t in _auto_universe]), "mã")
    _b2.metric("📈 Tổng thị trường", len(_auto_universe), "mã")
    _b3.metric("🔢 Không lọc", "HOSE+HNX+UPCOM", "tất cả")

    with st.expander("⚙️ Xem & chỉnh sửa danh sách quét"):
        custom_input = st.text_area("✏️ Danh sách:", value=", ".join(_auto_universe), height=120)
        _parsed = [t.strip().upper() for t in custom_input.replace("\n", ",").split(",") if t.strip()]
        scan_list = sorted(set(_parsed)) if _parsed else _auto_universe
        st.info(f"📋 Sẽ quét **{len(scan_list)} mã**")

    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    with col_f1:
        filter_phase = st.multiselect("Smart Money:", ["GOM HÀNG", "ĐẨY GIÁ (MARKUP)", "TRUNG TÍNH", "XẢ HÀNG", "ĐÈ GIÁ"], default=[])
    with col_f2:
        filter_rec = st.multiselect("Khuyến nghị:", ["✅ MUA", "⏳ THEO DÕI", "🚫 TRÁNH/BÁN"], default=[])
    with col_f3:
        filter_ict = st.multiselect("ICT Structure:", ["UPTREND", "DOWNTREND", "SIDEWAYS"], default=[])
    with col_f4:
        filter_wyckoff = st.multiselect("Wyckoff:", ["MARKUP", "ACCUMULATION", "DISTRIBUTION", "UNDEFINED"], default=[])
    with col_f5:
        filter_min_score = st.slider("Điểm tối thiểu:", min_value=-10, max_value=26, value=-5, step=1)

    if st.button("🚀 Chạy Quét Thị Trường", type="primary", key="scan_btn"):
        results = []; errs = 0
        progress_bar = st.progress(0, "Đang quét...")
        status_text  = st.empty()

        for i, sym in enumerate(scan_list):
            status_text.markdown(f"⏳ Đang phân tích **{sym}** ({i+1}/{len(scan_list)})...")
            res = scan_single_stock(sym)
            if res: results.append(res)
            else:   errs += 1
            progress_bar.progress((i + 1) / len(scan_list), text=f"Đã quét {i+1}/{len(scan_list)} mã")

        progress_bar.empty(); status_text.empty()

        if not results:
            st.error("❌ Không quét được dữ liệu."); st.stop()

        q1, q2, q3 = st.columns(3)
        q1.success(f"✅ **{len(results)} mã** quét thành công")
        q2.info(f"📊 Thất bại: **{errs} mã**")
        q3.metric("Tổng quét", len(scan_list), "mã")

        results.sort(key=lambda x: (_phase_order(x["sm_phase"]), -x["score"]))

        # Apply filters
        filtered = results
        if filter_phase:   filtered = [r for r in filtered if r["sm_phase"] in filter_phase]
        if filter_rec:     filtered = [r for r in filtered if r["recommendation"] in filter_rec]
        if filter_ict:     filtered = [r for r in filtered if any(s in r.get("ict_label","") for s in filter_ict)]
        if filter_wyckoff: filtered = [r for r in filtered if any(w in r.get("wyckoff_phase","") for w in filter_wyckoff)]
        filtered = [r for r in filtered if r["score"] >= filter_min_score]

        # Summary stats
        st.markdown("---")
        phase_counts = defaultdict(int)
        for r in results: phase_counts[r["sm_phase"]] += 1

        sum_cols = st.columns(5)
        phase_defs = [("GOM HÀNG","🏦","#00C853"), ("ĐẨY GIÁ (MARKUP)","🚀","#40C4FF"),
                      ("TRUNG TÍNH","🔍","#FFD740"), ("XẢ HÀNG","📤","#FF6D00"), ("ĐÈ GIÁ","📉","#FF1744")]
        for idx, (ph, ic, co) in enumerate(phase_defs):
            cnt = phase_counts.get(ph, 0)
            sum_cols[idx].markdown(
                f"<div style='text-align:center;padding:12px 6px;border-radius:10px;"
                f"background:{co}18;border:1px solid {co}55;'>"
                f"<div style='font-size:1.5rem;'>{ic}</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{co};'>{cnt}</div>"
                f"<div style='font-size:0.72rem;color:#aaa;'>{ph}</div></div>",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("✅ MUA",       sum(1 for r in results if r["recommendation"] == "✅ MUA"))
        m2.metric("⏳ Theo dõi",  sum(1 for r in results if r["recommendation"] == "⏳ THEO DÕI"))
        m3.metric("🚫 Tránh",     sum(1 for r in results if r["recommendation"] == "🚫 TRÁNH/BÁN"))
        m4.metric("⭐ Golden Pkt", sum(1 for r in results if "Golden Pocket" in r["fib_zone"]))
        m5.metric("🔬 BOS/CHoCH", sum(1 for r in results if "BOS" in r.get("ict_label","") or "CHoCH" in r.get("ict_label","")))
        m6.metric("🌀 SOS/Spring", sum(1 for r in results if "MARKUP" in r.get("wyckoff_phase","")))
        m7.metric("🔀 Divergence", sum(1 for r in results if r.get("div_label","—") != "—"))

        st.markdown("---")
        st.markdown(f"### 📋 Kết quả — {len(filtered)}/{len(results)} mã (sau bộ lọc)")

        PHASE_META = {
            "GOM HÀNG":        ("🏦","#00C853","rgba(0,200,83,0.10)",   "Tiền lớn bí mật tích lũy — Cơ hội sớm."),
            "ĐẨY GIÁ (MARKUP)":("🚀","#40C4FF","rgba(64,196,255,0.10)","Cá mập đang markup — Theo đà có thể."),
            "TRUNG TÍNH":      ("🔍","#FFD740","rgba(255,215,64,0.08)", "Chưa rõ — Quan sát thêm."),
            "XẢ HÀNG":         ("📤","#FF6D00","rgba(255,109,0,0.10)",  "Tiền lớn phân phối — Cẩn thận."),
            "ĐÈ GIÁ":          ("📉","#FF1744","rgba(255,23,68,0.10)",  "Tay to xả mạnh — Không nên mua."),
        }
        by_phase = defaultdict(list)
        for r in filtered: by_phase[r["sm_phase"]].append(r)

        for ph_name, (ph_icon, ph_color, ph_bg, ph_desc) in PHASE_META.items():
            group = by_phase.get(ph_name, [])
            if not group: continue
            with st.expander(f"{ph_icon} {ph_name} — {len(group)} mã", expanded=(ph_name == "GOM HÀNG")):
                st.markdown(
                    f"<div style='background:{ph_bg};border-left:4px solid {ph_color};"
                    f"padding:8px 16px;border-radius:0 8px 8px 0;margin-bottom:10px;"
                    f"color:#ddd;font-size:0.88rem;'>{ph_desc}</div>",
                    unsafe_allow_html=True)
                rows = []
                for r in group:
                    rows.append({
                        "Mã":        r["symbol"],
                        "Giá":       f"{r['close']:,.0f}",
                        "%1P":       f"{r['pct_chg']:+.2f}%",
                        "RSI":       f"{r['rsi']:.1f}",
                        "ADX":       f"{r.get('adx',0):.1f}",
                        "MACD":      "📈" if r["macd_bull"] else "📉",
                        "EMA":       "🟢" if r.get("ema_bull") else "🔴",
                        "VWAP":      "🟢" if r.get("vwap_bull") else "🔴",
                        "Squeeze":   "🔴ON" if r.get("sq_on") else ("🟢↑" if (r.get("sq_mom") or 0) > 0 else "🔴↓"),
                        "ICT":       r.get("ict_label","—")[:14],
                        "VSA":       r.get("vsa_top","—")[:14],
                        "Wyckoff":   r.get("wyckoff_phase","?")[:10],
                        "Div":       r.get("div_label","—"),
                        "SM Score":  f"{r['sm_score']:+.1f}",
                        "Fib":       r["fib_zone"][:16],
                        "Điểm":      f"{r['score']}/{r['max_score']}",
                        "KN":        r["recommendation"],
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                top_picks = [r for r in group if r["recommendation"] == "✅ MUA"]
                if top_picks and ph_name in ("GOM HÀNG", "ĐẨY GIÁ (MARKUP)"):
                    st.markdown(f"#### ⭐ Top picks — {len(top_picks)} mã khuyến nghị MUA")
                    card_cols = st.columns(min(len(top_picks), 4))
                    for ci, r in enumerate(top_picks[:8]):
                        chg_c = "#26a69a" if r["pct_chg"] >= 0 else "#ef5350"
                        fib_hl = "#FFA726" if "Golden Pocket" in r["fib_zone"] else "#00C853" if "Gần đáy" in r["fib_zone"] else ph_color
                        div_badge = f"<span style='font-size:0.68rem;color:#40C4FF;'>{r.get('div_label','—')}</span>" if r.get("div_label","—") != "—" else ""
                        wy_badge  = f"<span style='font-size:0.68rem;color:#7986CB;'>🌀{r.get('wyckoff_phase','?')}</span>"
                        with card_cols[ci % 4]:
                            st.markdown(
                                f"""<div style="border:1px solid {ph_color}55;border-top:3px solid {ph_color};
                                    border-radius:10px;padding:14px 12px;background:{ph_bg};margin-bottom:8px;">
  <div style="font-size:1.3rem;font-weight:800;color:{ph_color};">{r['symbol']}</div>
  <div style="font-size:1.05rem;font-weight:700;color:#fff;margin:4px 0;">
    {r['close']:,.0f} đ <span style="font-size:0.85rem;color:{chg_c};">{r['pct_chg']:+.2f}%</span></div>
  <div style="background:rgba(64,196,255,0.08);border-radius:6px;padding:4px 8px;margin:6px 0;font-size:0.72rem;color:#40C4FF;">
    🔬 ICT: {r.get('ict_label','—')}</div>
  <div style="font-size:0.76rem;color:#bbb;line-height:1.8;">
    RSI: <b style="color:#FF9800;">{r['rsi']:.1f}</b> · ADX: <b style="color:#E040FB;">{r.get('adx',0):.1f}</b><br>
    VSA: {r.get('vsa_top','—')[:16]}<br>
    {div_badge} {wy_badge}<br>
    <span style="color:{fib_hl};">{r['fib_zone']}</span>
  </div>
  <div style="margin-top:8px;font-size:0.82rem;font-weight:700;color:#00C853;">
    Điểm: {r['score']}/{r['max_score']} · {r['recommendation']}
  </div>
</div>""", unsafe_allow_html=True)

        # Special Screens
        st.markdown("---")

        # ICT BOS/CHoCH
        bos_stocks = [r for r in results if "BOS" in r.get("ict_label","") or "CHoCH" in r.get("ict_label","")]
        bos_stocks.sort(key=lambda x: -x["score"])
        if bos_stocks:
            st.markdown(f"### 🔬 ICT — BOS / CHoCH ({len(bos_stocks)} mã)")
            bos_rows = [{"Mã": r["symbol"], "Giá": f"{r['close']:,.0f}", "%1P": f"{r['pct_chg']:+.2f}%",
                          "ICT": r.get("ict_label",""), "Wyckoff": r.get("wyckoff_phase","?"),
                          "Div": r.get("div_label","—"), "SM": f"{r['sm_icon']} {r['sm_phase']}",
                          "KN": r["recommendation"]} for r in bos_stocks]
            st.dataframe(pd.DataFrame(bos_rows), hide_index=True, use_container_width=True)

        # Wyckoff SOS/Spring screen
        wy_strong = [r for r in results if "MARKUP" in r.get("wyckoff_phase","")]
        wy_strong.sort(key=lambda x: -x.get("wyckoff_score", 0))
        if wy_strong:
            st.markdown(f"### 🌀 Wyckoff — Tích Lũy Xong / SOS ({len(wy_strong)} mã)")
            wy_rows = [{"Mã": r["symbol"], "Giá": f"{r['close']:,.0f}", "%1P": f"{r['pct_chg']:+.2f}%",
                         "Wyckoff": r.get("wyckoff_phase","?"), "Wyckoff Score": f"{r.get('wyckoff_score',0):+d}",
                         "SM": f"{r['sm_icon']} {r['sm_phase']}", "ICT": r.get("ict_label","—"),
                         "Div": r.get("div_label","—"), "KN": r["recommendation"]} for r in wy_strong[:25]]
            st.dataframe(pd.DataFrame(wy_rows), hide_index=True, use_container_width=True)

        # Divergence screen
        div_strong = [r for r in results if r.get("div_label","—") != "—"]
        div_strong.sort(key=lambda x: -abs(x.get("div_score", 0)))
        if div_strong:
            st.markdown(f"### 🔀 Divergence — Regular & Hidden ({len(div_strong)} mã)")
            div_rows = [{"Mã": r["symbol"], "Giá": f"{r['close']:,.0f}", "%1P": f"{r['pct_chg']:+.2f}%",
                          "Divergence": r.get("div_label","—"), "Div Score": f"{r.get('div_score',0):+d}",
                          "RSI": f"{r['rsi']:.1f}", "SM": f"{r['sm_icon']} {r['sm_phase']}",
                          "KN": r["recommendation"]} for r in div_strong[:25]]
            st.dataframe(pd.DataFrame(div_rows), hide_index=True, use_container_width=True)

        # VSA strong
        vsa_bull = [r for r in results if r.get("vsa_score", 0) >= 2]
        vsa_bull.sort(key=lambda x: -x.get("vsa_score", 0))
        if vsa_bull:
            st.markdown(f"### 📊 VSA — Tín Hiệu Tích Lũy Mạnh ({len(vsa_bull)} mã)")
            vsa_rows = [{"Mã": r["symbol"], "Giá": f"{r['close']:,.0f}", "%1P": f"{r['pct_chg']:+.2f}%",
                          "VSA Score": f"{r.get('vsa_score',0):+d}", "VSA Top": r.get("vsa_top","—"),
                          "PA": r.get("pa_trend","—"), "SM": f"{r['sm_icon']} {r['sm_phase']}",
                          "KN": r["recommendation"]} for r in vsa_bull[:25]]
            st.dataframe(pd.DataFrame(vsa_rows), hide_index=True, use_container_width=True)

        # Fibonacci buy zone
        vung_mua = [r for r in results if any(kw in r["fib_zone"] for kw in ["Golden Pocket", "Gần đáy", "Hỗ trợ"])]
        vung_mua.sort(key=lambda x: -x["score"])
        if vung_mua:
            st.markdown(f"### 🛒 Về Vùng Mua Fibonacci — {len(vung_mua)} mã")
            buy_rows = [{"Mã": r["symbol"], "Giá": f"{r['close']:,.0f}", "%1P": f"{r['pct_chg']:+.2f}%",
                          "RSI": f"{r['rsi']:.1f}", "ICT": r.get("ict_label","—")[:14],
                          "Wyckoff": r.get("wyckoff_phase","?"), "Div": r.get("div_label","—"),
                          "SM": f"{r['sm_icon']} {r['sm_phase']}", "Fib": r["fib_zone"],
                          "Điểm": f"{r['score']}/{r['max_score']}", "KN": r["recommendation"]}
                         for r in vung_mua]
            st.dataframe(pd.DataFrame(buy_rows), hide_index=True, use_container_width=True)

    else:
        st.info(
            "💡 Nhấn **Chạy Quét Thị Trường** để bắt đầu.\n\n"
            "**v3.0 — 16 Phương Pháp Tích Hợp:**\n"
            "- 🔬 **ICT**: BOS · CHoCH · Order Blocks · FVG · Liquidity · OTE\n"
            "- 📊 **VSA**: Stopping · Climax · No Demand/Supply · Upthrust · Test\n"
            "- 🕯️ **PA**: Inside/Outside Bar · Key S/R · Pin Bar · Trend Structure\n"
            "- 💧 **Overflow**: Volume · BB · Breakout · Gap · Exhaustion\n"
            "- 🐋 **Smart Money**: OBV · CMF · MFI · A/D Line · Buy Ratio\n"
            "- 🌀 **Wyckoff**: SC · Spring · SOS · LPSY · No Supply\n"
            "- 🔀 **Divergence**: Regular + Hidden Divergence RSI\n"
            "- 🏔️ **Volume Profile**: POC · VAH/VAL · HVN · LVN\n"
            "- 💹 **VWAP**: Rolling 20 · ±1σ ±2σ Bands\n"
            "- ⚡ **TTM Squeeze**: BB vs Keltner Compression\n"
            "- 🌊 **Heikin-Ashi**: HA Trend Filter\n"
            "- 🔴🟢 **Elder Impulse**: EMA13 + MACD Go/No-Go\n"
            "- 🛑 **Parabolic SAR**: Trend + Trailing Stop\n"
            "- 📉 **StochRSI**: Stochastic RSI K/D\n"
            "- 🎯 **Supply & Demand**: Base+Rally/Drop Zones\n"
            "- 📏 **ATR Setup**: Auto SL/TP/R:R Calculator\n\n"
            f"**Danh sách quét:** {len(_auto_universe)} mã — HOSE · HNX · UPCOM"
        )
