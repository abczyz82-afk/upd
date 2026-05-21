import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from collections import defaultdict

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="AI Trading Dashboard 📊")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Cấu hình Hệ thống")

ai_provider = st.sidebar.radio(
    "Chọn nhà cung cấp AI:",
    ["🟠 Claude (Anthropic)", "🔵 Gemini (Google)"],
)
api_key = st.sidebar.text_input("Nhập API Key:", type="password")
if not api_key:
    st.sidebar.warning("⚠️ Nhập API Key để bật Trợ lý AI.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Chỉ báo tích hợp:**\n"
    "- RSI (14)\n"
    "- Bollinger Bands (20,2)\n"
    "- MACD (12,26,9)\n"
    "- ADX (14) + DI± *(MỚI)*\n"
    "- EMA 5 / 20 / 50 *(MỚI)*\n"
    "- Ichimoku Cloud\n"
    "- MCDX (Momentum Composite)\n"
    "- Fibonacci Retracement\n"
    "- Smart Money · OBV · CMF · MFI\n"
    "- A/D Line · Stop Hunt · Wyckoff *(MỚI)*\n"
    "- Candlestick Patterns *(MỚI)*\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Hệ Thống Phân Tích Kỹ Thuật & Trợ Lý Giao Dịch AI")
st.markdown(
    "Tích hợp **RSI · MACD · ADX · EMA · Bollinger Bands · Ichimoku · MCDX · Fibonacci · Smart Money** "
    "— Được phân tích bởi **Claude** hoặc **Gemini**."
)

_tab1, _tab2 = st.tabs(["📊 Phân Tích Đơn Lẻ", "🔍 Quét Toàn Thị Trường"])

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_clean_stock_data(symbol: str) -> pd.DataFrame | None:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)

    if symbol.startswith("VN30F"):
        try:
            url = (
                f"https://services.entrade.com.vn/chart-api/v2/ohlcs/derivative"
                f"?from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
                f"&symbol={symbol}&resolution=1D"
            )
            resp = requests.get(url, timeout=10).json()
            if "t" in resp and len(resp["t"]) > 0:
                df = pd.DataFrame({
                    "time": pd.to_datetime(resp["t"], unit="s"),
                    "open": resp["o"], "high": resp["h"],
                    "low": resp["l"], "close": resp["c"], "volume": resp["v"],
                })
                df["time"] = df["time"].dt.strftime("%Y-%m-%d")
                return df
        except Exception as e:
            st.error(f"Lỗi khi kéo dữ liệu phái sinh: {e}")
            return None

    str_start = start_date.strftime("%Y-%m-%d")
    str_end   = end_date.strftime("%Y-%m-%d")
    try:
        url = (
            f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
            f"?from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
            f"&symbol={symbol}&resolution=1D"
        )
        resp = requests.get(url, timeout=10).json()
        if "t" in resp and len(resp["t"]) > 0:
            df = pd.DataFrame({
                "time": pd.to_datetime(resp["t"], unit="s"),
                "open": resp["o"], "high": resp["h"],
                "low": resp["l"], "close": resp["c"], "volume": resp["v"],
            })
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass

    try:
        url = (
            f"https://iboard-query.ssi.com.vn/v2/stock/historical-price"
            f"?symbol={symbol}&fromDate={str_start}&toDate={str_end}&offset=0&limit=200"
        )
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10).json()
        data = resp.get("data", {}).get("items", [])
        if data:
            df = pd.DataFrame(data)
            df = df.rename(columns={
                "tradingDate": "time", "openPrice": "open", "highPrice": "high",
                "lowPrice": "low", "closePrice": "close", "totalMatchVolume": "volume",
            })
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
            return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        pass

    st.error(f"Không thể lấy dữ liệu cho mã **{symbol}** từ bất kỳ nguồn nào.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calc_ichimoku(df: pd.DataFrame) -> tuple:
    hi, lo = df["high"], df["low"]
    tenkan = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
    kijun  = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou


def calc_fibonacci(df: pd.DataFrame, lookback: int = 60) -> dict:
    window   = df.tail(lookback)
    high_val = window["high"].max()
    low_val  = window["low"].min()
    high_idx_pos = int(window["high"].values.argmax())
    low_idx_pos  = int(window["low"].values.argmin())
    is_downtrend = (high_idx_pos > low_idx_pos)
    diff = high_val - low_val
    if diff == 0:
        diff = high_val * 0.01

    if is_downtrend:
        levels = {
            "0.0% (Đỉnh)":   high_val,
            "23.6%":          high_val - 0.236 * diff,
            "38.2%":          high_val - 0.382 * diff,
            "50.0%":          high_val - 0.500 * diff,
            "61.8% ✨":       high_val - 0.618 * diff,
            "65.0% 🏅":       high_val - 0.650 * diff,
            "78.6%":          high_val - 0.786 * diff,
            "100.0% (Đáy)":   low_val,
            "127.2% 📉":      low_val  - 0.272 * diff,
            "161.8% 📉":      low_val  - 0.618 * diff,
        }
    else:
        levels = {
            "0.0% (Đáy)":    low_val,
            "23.6%":          low_val  + 0.236 * diff,
            "38.2%":          low_val  + 0.382 * diff,
            "50.0%":          low_val  + 0.500 * diff,
            "61.8% ✨":       low_val  + 0.618 * diff,
            "65.0% 🏅":       low_val  + 0.650 * diff,
            "78.6%":          low_val  + 0.786 * diff,
            "100.0% (Đỉnh)":  high_val,
            "127.2% 📈":      high_val + 0.272 * diff,
            "161.8% 📈":      high_val + 0.618 * diff,
        }
    levels["_high"]         = high_val
    levels["_low"]          = low_val
    levels["_is_downtrend"] = is_downtrend
    levels["_diff"]         = diff
    return levels


def calc_mcdx(df: pd.DataFrame) -> pd.Series:
    macd_hist = MACD(close=df["close"]).macd_diff()
    rsi       = RSIIndicator(close=df["close"], window=14).rsi()
    def _norm(s):
        mn, mx = s.min(), s.max()
        return pd.Series(0, index=s.index) if mx == mn else (s - mn) / (mx - mn) * 2 - 1
    return ((_norm(macd_hist) + _norm(rsi - 50)) / 2).rename("MCDX")


def calc_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ADX + DI+/DI- — đo sức mạnh xu hướng."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    dm_plus  = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    atr       = tr.ewm(span=window, min_periods=window, adjust=False).mean()
    di_plus   = 100 * dm_plus.ewm(span=window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus  = 100 * dm_minus.ewm(span=window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    dx        = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx       = dx.ewm(span=window, min_periods=window, adjust=False).mean()
    df["ADX"]      = adx
    df["DI_Plus"]  = di_plus
    df["DI_Minus"] = di_minus
    return df


def calc_ema_cross(df: pd.DataFrame) -> pd.DataFrame:
    """EMA 5 / 20 / 50."""
    close = df["close"]
    df["EMA5"]  = close.ewm(span=5,  adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    return df


def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    """Nhận diện nến đặc biệt trong 3 phiên gần nhất."""
    if len(df) < 3:
        return []
    patterns = []
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]
    o, h, l, c   = last["open"],  last["high"],  last["low"],  last["close"]
    po, ph, pl, pc = prev["open"], prev["high"], prev["low"], prev["close"]
    p2o, p2c      = prev2["open"], prev2["close"]
    body      = abs(c - o)
    range_    = (h - l) if (h - l) > 0 else 0.001
    upper_w   = h - max(o, c)
    lower_w   = min(o, c) - l
    prev_body = abs(pc - po)

    # Doji
    if body / range_ < 0.1:
        patterns.append(("⚖️ Doji", "Thị trường do dự — sắp có quyết định lớn", "neutral"))
    # Hammer
    elif lower_w > 2 * body and upper_w < body and c >= o:
        patterns.append(("🔨 Hammer", "Đảo chiều tăng — lực mua xuất hiện ở đáy", "bullish"))
    # Inverted Hammer
    elif upper_w > 2 * body and lower_w < body and c >= o:
        patterns.append(("🔼 Inverted Hammer", "Tín hiệu đảo chiều tăng — kiểm tra kháng cự", "bullish"))
    # Shooting Star
    elif upper_w > 2 * body and lower_w < body and c < o:
        patterns.append(("⭐ Shooting Star", "Đảo chiều giảm — lực bán xuất hiện ở đỉnh", "bearish"))
    # Hanging Man
    elif lower_w > 2 * body and upper_w < body and c < o:
        patterns.append(("🪝 Hanging Man", "Cảnh báo đảo chiều giảm ở đỉnh", "bearish"))

    # Bullish Engulfing
    if pc < po and c > o and c >= po and o <= pc:
        patterns.append(("🟢 Bullish Engulfing", "Nuốt nến đỏ — đảo chiều tăng mạnh", "bullish"))
    # Bearish Engulfing
    elif pc > po and c < o and c <= po and o >= pc:
        patterns.append(("🔴 Bearish Engulfing", "Nuốt nến xanh — đảo chiều giảm mạnh", "bearish"))

    # Morning Star
    if (p2c < p2o and abs(pc - po) < 0.3 * abs(p2c - p2o)
            and c > o and c > (p2o + p2c) / 2):
        patterns.append(("🌅 Morning Star", "3 nến đảo chiều tăng — tín hiệu rất mạnh", "bullish"))
    # Evening Star
    elif (p2c > p2o and abs(pc - po) < 0.3 * abs(p2c - p2o)
          and c < o and c < (p2o + p2c) / 2):
        patterns.append(("🌇 Evening Star", "3 nến đảo chiều giảm — tín hiệu rất mạnh", "bearish"))

    # Pin Bar Bullish
    if lower_w > 3 * body and lower_w > upper_w * 2:
        patterns.append(("📌 Pin Bar (Tăng)", "Đuôi dài dưới — từ chối vùng giá thấp mạnh", "bullish"))
    # Pin Bar Bearish
    elif upper_w > 3 * body and upper_w > lower_w * 2:
        patterns.append(("📌 Pin Bar (Giảm)", "Đuôi dài trên — từ chối vùng giá cao mạnh", "bearish"))

    # Marubozu Bullish (thân dài, không đuôi)
    if c > o and body / range_ > 0.85:
        patterns.append(("💚 Marubozu Xanh", "Nến thân dài hoàn toàn — lực mua áp đảo", "bullish"))
    elif c < o and body / range_ > 0.85:
        patterns.append(("❤️ Marubozu Đỏ", "Nến thân dài hoàn toàn — lực bán áp đảo", "bearish"))

    seen = []
    result = []
    for p in patterns:
        if p[0] not in seen:
            seen.append(p[0])
            result.append(p)
    return result[:4]


def calc_smart_money(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    open_  = df["open"]

    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = pd.array(obv, dtype=float)

    hl  = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    df["CMF"] = mfv.rolling(20).sum() / volume.rolling(20).sum()

    tp = (high + low + close) / 3
    rmf = tp * volume
    pos_mf = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df["MFI"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))

    df["Up_Vol"]   = np.where(close >= open_, volume, 0).astype(float)
    df["Down_Vol"] = np.where(close <  open_, volume, 0).astype(float)
    buy10   = pd.Series(df["Up_Vol"]).rolling(10).sum()
    sell10  = pd.Series(df["Down_Vol"]).rolling(10).sum()
    total10 = (buy10 + sell10).replace(0, np.nan)
    df["Buy_Ratio"] = buy10 / total10

    df["Vol_MA20"]  = volume.rolling(20).mean()
    df["Vol_Ratio"] = volume / df["Vol_MA20"]
    df["Body_Ratio"] = (abs(close - open_) / (high - low).replace(0, np.nan)).fillna(0)

    # A/D Line
    ad_mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    df["AD_Line"] = (ad_mfm * volume).cumsum()

    return df


def detect_smart_money_phase(df: pd.DataFrame) -> dict:
    win  = df.tail(15).copy()
    last = df.iloc[-1]
    prev5 = df.tail(5)

    obv_vals = win["OBV"].dropna().values
    if len(obv_vals) > 3:
        x = np.arange(len(obv_vals))
        obv_slope = np.polyfit(x, obv_vals, 1)[0]
        obv_slope_norm = obv_slope / (abs(obv_vals).mean() + 1e-9) * 10
    else:
        obv_slope_norm = 0

    prices = win["close"].dropna().values
    if len(prices) > 3:
        px = np.arange(len(prices))
        price_slope = np.polyfit(px, prices, 1)[0]
        price_slope_norm = price_slope / (prices.mean() + 1e-9) * 100
    else:
        price_slope_norm = 0

    cmf_val   = last["CMF"]   if pd.notna(last["CMF"])   else 0
    mfi_val   = last["MFI"]   if pd.notna(last["MFI"])   else 50
    buy_ratio = last["Buy_Ratio"] if pd.notna(last["Buy_Ratio"]) else 0.5
    vol_ratio = last["Vol_Ratio"] if pd.notna(last["Vol_Ratio"]) else 1.0
    price_range_pct = (win["high"].max() - win["low"].min()) / win["close"].mean() * 100
    is_sideways = price_range_pct < 5.0

    score   = 0.0
    signals = []

    if obv_slope_norm > 0.3:
        score += 2; signals.append(("OBV tăng",    "Tiền lớn đang mua tích lũy dần", "🟢"))
    elif obv_slope_norm < -0.3:
        score -= 2; signals.append(("OBV giảm",    "Tiền lớn đang bán/xả hàng", "🔴"))
    else:
        signals.append(("OBV phẳng", "Dòng tiền lớn chưa rõ xu hướng", "⚪"))

    if cmf_val > 0.12:
        score += 2; signals.append(("CMF cao",       f"{cmf_val:.3f} — Áp lực mua mạnh", "🟢"))
    elif cmf_val > 0:
        score += 1; signals.append(("CMF dương nhẹ", f"{cmf_val:.3f} — Mua nhiều hơn bán", "🟡"))
    elif cmf_val < -0.12:
        score -= 2; signals.append(("CMF âm mạnh",  f"{cmf_val:.3f} — Áp lực bán mạnh", "🔴"))
    else:
        score -= 1; signals.append(("CMF âm nhẹ",   f"{cmf_val:.3f} — Bán nhiều hơn mua", "🟠"))

    if mfi_val < 25:
        score += 2; signals.append(("MFI quá bán", f"{mfi_val:.1f} — Dòng tiền bán cạn kiệt", "🟢"))
    elif mfi_val > 80:
        score -= 2; signals.append(("MFI quá mua", f"{mfi_val:.1f} — Dòng tiền mua bão hoà", "🔴"))
    else:
        signals.append(("MFI trung tính", f"{mfi_val:.1f}", "⚪"))

    if buy_ratio > 0.65:
        score += 1.5; signals.append(("Khối lượng mua", f"{buy_ratio*100:.0f}% phiên xanh 10 kỳ", "🟢"))
    elif buy_ratio < 0.35:
        score -= 1.5; signals.append(("Khối lượng bán", f"{(1-buy_ratio)*100:.0f}% phiên đỏ 10 kỳ", "🔴"))
    else:
        signals.append(("Cân bằng mua/bán", f"{buy_ratio*100:.0f}% / {(1-buy_ratio)*100:.0f}%", "⚪"))

    if vol_ratio > 2.0:
        if price_slope_norm > 0:
            score += 1; signals.append(("Volume bùng nổ ↑", f"x{vol_ratio:.1f} — Cú đẩy mạnh", "🟢"))
        else:
            score -= 1; signals.append(("Volume bùng nổ ↓", f"x{vol_ratio:.1f} — Xả hàng ồ ạt", "🔴"))
    elif vol_ratio < 0.5:
        signals.append(("Volume cạn", f"x{vol_ratio:.1f} — Thị trường thiếu thanh khoản", "🟡"))

    if price_slope_norm > 0.5 and obv_slope_norm < -0.2:
        score -= 1.5; signals.append(("Phân kỳ âm",  "Giá tăng nhưng OBV giảm — Cảnh báo xả hàng", "🔴"))
    elif price_slope_norm < -0.5 and obv_slope_norm > 0.2:
        score += 1.5; signals.append(("Phân kỳ dương","Giá giảm nhưng OBV tăng — Tiền lớn đang gom", "🟢"))

    max_score = 10.0
    pct = score / max_score

    if pct >= 0.4:
        phase="GOM HÀNG"; icon="🏦"; color="#00C853"; bg="rgba(0,200,83,0.10)"; border="#00C853"
        phase_desc="Tiền lớn đang **bí mật tích lũy** cổ phiếu ở vùng giá thấp. Giá đi ngang hoặc giảm nhẹ nhưng OBV & CMF tăng — dấu hiệu **cá mập / quỹ đang gom**."
    elif pct >= 0.15:
        phase="ĐẨY GIÁ (MARKUP)"; icon="🚀"; color="#40C4FF"; bg="rgba(64,196,255,0.10)"; border="#40C4FF"
        phase_desc="Dòng tiền lớn **đang đẩy giá lên mạnh**. Volume tăng theo giá — đây là giai đoạn **theo xu hướng** với tiền thông minh."
    elif pct >= -0.15:
        phase="TRUNG TÍNH / QUAN SÁT"; icon="🔍"; color="#FFD740"; bg="rgba(255,215,64,0.08)"; border="#FFD740"
        phase_desc="Dòng tiền lớn **chưa lộ rõ ý định**. Cần thêm tín hiệu từ volume và OBV để xác nhận hướng đi."
    elif pct >= -0.40:
        phase="XẢ HÀNG (DISTRIBUTION)"; icon="📤"; color="#FF6D00"; bg="rgba(255,109,0,0.10)"; border="#FF6D00"
        phase_desc="Tiền lớn đang **lặng lẽ phân phối/xả hàng** cho nhà đầu tư nhỏ lẻ. Giá cao nhưng OBV & CMF bắt đầu suy yếu — **cảnh báo đỉnh vùng**."
    else:
        phase="ĐÈ GIÁ (MARKDOWN)"; icon="📉"; color="#FF1744"; bg="rgba(255,23,68,0.10)"; border="#FF1744"
        phase_desc="Tiền lớn đang **đè giá và bán tháo mạnh**. Tránh mua đuổi — chờ tín hiệu OBV/CMF phục hồi trở lại trước khi xem xét."

    return {
        "phase": phase, "icon": icon, "color": color, "bg": bg, "border": border,
        "phase_desc": phase_desc, "score": score, "max_score": max_score,
        "signals": signals,
        "obv_slope": obv_slope_norm, "cmf": cmf_val, "mfi": mfi_val,
        "buy_ratio": buy_ratio, "vol_ratio": vol_ratio,
        "price_slope": price_slope_norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HIDDEN SMART MONEY — Phát hiện hành vi ẩn của tiền lớn
# ─────────────────────────────────────────────────────────────────────────────
def detect_hidden_smart_money(df: pd.DataFrame, sm_result: dict) -> dict:
    """
    Phát hiện hành vi ẩn của từng nhóm tiền lớn ngay cả khi không có biến động giá lớn.
    Sử dụng: BB Compression, A/D Line, Stealth Accumulation, Stop Hunt,
             Fake Breakdown, Wyckoff Spring, Effort vs Result, Order Blocks.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    open_  = df["open"]
    obv    = df["OBV"]
    cmf_s  = df["CMF"] if "CMF" in df.columns else pd.Series(0, index=df.index)

    # ── BB Width Compression ─────────────────────────────────────────────────
    bb_compressed    = False
    bb_width_ratio   = 1.0
    bb_squeeze_pct   = 0.0
    if all(c in df.columns for c in ["BB_High", "BB_Low", "BB_Mid"]):
        bb_width    = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"].replace(0, np.nan)
        bb_width_ma = bb_width.rolling(20).mean()
        bw_last     = bb_width.iloc[-1]
        bwma_last   = bb_width_ma.iloc[-1]
        if pd.notna(bw_last) and pd.notna(bwma_last) and bwma_last > 0:
            bb_width_ratio = bw_last / bwma_last
            bb_compressed  = bb_width_ratio < 0.72
            bb_squeeze_pct = (1 - bb_width_ratio) * 100

    # ── Candle Range Compression ─────────────────────────────────────────────
    range_series    = (high - low) / close.replace(0, np.nan)
    range_ma20      = range_series.rolling(20).mean()
    range_ratio     = (range_series.iloc[-1] / range_ma20.iloc[-1]
                       if pd.notna(range_ma20.iloc[-1]) and range_ma20.iloc[-1] > 0 else 1.0)
    range_compressed = range_ratio < 0.65

    # ── Doji / Small Body count (10 phiên) ──────────────────────────────────
    body_ratio_series = df["Body_Ratio"] if "Body_Ratio" in df.columns else (
        (close - open_).abs() / (high - low).replace(0, np.nan))
    doji_count = int((body_ratio_series.tail(10) < 0.2).sum())

    # ── Order Blocks: volume cao + body nhỏ ─────────────────────────────────
    vol_ma20 = df["Vol_MA20"] if "Vol_MA20" in df.columns else volume.rolling(20).mean()
    high_vol = volume > vol_ma20 * 1.4
    small_body = body_ratio_series < 0.28
    order_block_count = int((high_vol & small_body).tail(15).sum())

    # ── A/D Line slope (5 phiên) ─────────────────────────────────────────────
    ad_vals = df["AD_Line"].dropna().tail(10).values if "AD_Line" in df.columns else np.array([0])
    ad_slope5_norm = 0.0
    if len(ad_vals) >= 5:
        ad_slope5_norm = np.polyfit(np.arange(len(ad_vals)), ad_vals, 1)[0] / (abs(ad_vals).mean() + 1e-9) * 10

    # ── OBV & Price slope 15 phiên ───────────────────────────────────────────
    obv_vals15   = obv.dropna().tail(15).values
    price_vals15 = close.dropna().tail(15).values
    obv_slope_norm15   = sm_result.get("obv_slope", 0)
    price_slope_norm15 = sm_result.get("price_slope", 0)
    stealth_accum  = obv_slope_norm15 > 0.2 and price_slope_norm15 < 0.5
    stealth_distrib = obv_slope_norm15 < -0.2 and price_slope_norm15 > -0.5

    # ── CMF Momentum (slope 5 phiên) ─────────────────────────────────────────
    cmf_slope5 = 0.0
    cmf_clean  = cmf_s.dropna()
    if len(cmf_clean) >= 5:
        cmf_arr    = cmf_clean.tail(5).values
        cmf_slope5 = float(np.polyfit(np.arange(5), cmf_arr, 1)[0])

    # ── Absorption Volume: volume x2 nhưng giá không đổi ────────────────────
    price_chg_pct  = (close.diff() / close.shift(1).replace(0, np.nan)).abs()
    absorb_mask    = (volume > vol_ma20 * 2.0) & (price_chg_pct < 0.005)
    absorption_events = int(absorb_mask.tail(10).sum())

    # ── Stop Hunt: xuyên support rồi đóng cửa trên ───────────────────────────
    recent_support = low.tail(30).quantile(0.15)
    stop_hunts = 0
    for i in range(max(-15, -len(df)), 0):
        try:
            if low.iloc[i] < recent_support * 0.997 and close.iloc[i] > recent_support:
                stop_hunts += 1
        except Exception:
            pass

    # ── Fake Breakdown: xuyên BB Lower rồi đóng vào trong ───────────────────
    fake_breakdowns = 0
    if "BB_Low" in df.columns:
        for i in range(max(-12, -len(df)), 0):
            try:
                if (low.iloc[i] < df["BB_Low"].iloc[i] * 0.999
                        and close.iloc[i] > df["BB_Low"].iloc[i]):
                    fake_breakdowns += 1
            except Exception:
                pass

    # ── Wyckoff Spring: test đáy dài hạn với volume thấp ────────────────────
    is_wyckoff_spring = False
    wyckoff_desc = ""
    if len(df) >= 20:
        long_low      = low.tail(60).min() if len(df) >= 60 else low.min()
        recent_low5   = low.tail(5).min()
        recent_vol5   = volume.tail(5).mean()
        avg_vol20     = volume.tail(20).mean()
        if recent_low5 <= long_low * 1.025 and recent_vol5 < avg_vol20 * 0.85:
            is_wyckoff_spring = True
            wyckoff_desc = (f"Giá test đáy {long_low:,.0f} với volume = {recent_vol5:,.0f} "
                            f"(thấp hơn {(1 - recent_vol5/avg_vol20)*100:.0f}% so với TB)")

    # ── Effort vs Result: volume lớn, giá không đi ──────────────────────────
    effort_absorbed = False
    effort_details  = []
    recent_5 = df.tail(5)
    vol_ma_5 = vol_ma20.tail(5)
    for idx in range(len(recent_5)):
        try:
            row     = recent_5.iloc[idx]
            vol_ref = float(vol_ma_5.iloc[idx]) if pd.notna(vol_ma_5.iloc[idx]) else 1
            if vol_ref == 0:
                continue
            candle_move = abs(row["close"] - row["open"]) / row["open"] * 100
            vol_mult    = row["volume"] / vol_ref
            if vol_mult > 1.8 and candle_move < 0.6:
                effort_absorbed = True
                effort_details.append(
                    f"Phiên {recent_5['time'].iloc[idx] if 'time' in recent_5.columns else ''}: "
                    f"Volume x{vol_mult:.1f} nhưng biến động chỉ {candle_move:.2f}%"
                )
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════════
    # MARKET MAKER ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    mm_evidence = []
    mm_score    = 0

    if bb_compressed:
        mm_score += 2
        mm_evidence.append(f"🔴 BB Width co lại còn {bb_width_ratio:.0%} so với MA20 — "
                           f"(thu hẹp {bb_squeeze_pct:.0f}%) đang NÉN LÒ XO")
    if range_compressed:
        mm_score += 1
        mm_evidence.append(f"🟡 Biên độ nến = {range_ratio:.0%} so với MA — MM đang kiểm soát spread chặt chẽ")
    if doji_count >= 4:
        mm_score += 2
        mm_evidence.append(f"🟡 {doji_count}/10 nến gần đây là Doji/Spinning Top — "
                           f"MM đang test thanh khoản hai chiều, cân bằng lệnh")
    elif doji_count >= 2:
        mm_score += 1
        mm_evidence.append(f"⚪ {doji_count}/10 nến Doji — dấu hiệu nhẹ về kiểm soát spread")
    if order_block_count >= 3:
        mm_score += 2
        mm_evidence.append(f"🔴 {order_block_count} Order Block (volume cao + body nhỏ) trong 15 phiên — "
                           f"vùng giá MM đang kiểm soát chặt")
    elif order_block_count >= 1:
        mm_score += 1
        mm_evidence.append(f"🟡 {order_block_count} Order Block phát hiện — theo dõi thêm")
    if bb_compressed and range_compressed:
        mm_score += 1
        mm_evidence.append("⚡ BB + Range đều nén → sắp có breakout/breakdown mạnh")

    if mm_score >= 5:
        mm_action = "🔄 NÉN LÒ XO MẠNH — Sắp có biến động lớn"
        mm_desc   = ("MM đang hấp thụ lệnh hai chiều cực mạnh, nén BB và spread về mức thấp nhất. "
                     "Năng lượng đang tích tụ — sắp xảy ra breakout/breakdown mạnh, hãy chuẩn bị chiến lược.")
        mm_color  = "#FF6D00"
        mm_alert  = "HIGH"
    elif mm_score >= 3:
        mm_action = "⚙️ ĐANG KIỂM SOÁT SPREAD"
        mm_desc   = "MM đang cân bằng lệnh mua/bán và thu hẹp biên độ. Giá sẽ đi ngang cho đến khi có catalyst."
        mm_color  = "#FFD740"
        mm_alert  = "MED"
    elif mm_score >= 1:
        mm_action = "⚖️ HOẠT ĐỘNG NHẸ"
        mm_desc   = "MM có dấu hiệu kiểm soát nhẹ nhưng chưa rõ ràng. Thị trường bình thường."
        mm_color  = "#90CAF9"
        mm_alert  = "LOW"
    else:
        mm_action = "🔵 HOẠT ĐỘNG BÌNH THƯỜNG"
        mm_desc   = "MM đang cung cấp thanh khoản thông thường, không có hành động đặc biệt."
        mm_color  = "#78909C"
        mm_alert  = "LOW"
    mm_confidence = min(100, mm_score * 17)

    # ════════════════════════════════════════════════════════════════════════
    # FUND (QUỸ ĐẦU TƯ) ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    fund_evidence = []
    fund_score    = 0

    if stealth_accum:
        fund_score += 3
        fund_evidence.append(f"🟢 TÍCH LŨY ÍT TIẾNG: OBV slope = {obv_slope_norm15:+.2f} tăng "
                             f"trong khi price slope = {price_slope_norm15:+.2f} — quỹ gom bí mật")
    if stealth_distrib:
        fund_score -= 3
        fund_evidence.append(f"🔴 PHÂN PHỐI ÍT TIẾNG: OBV slope = {obv_slope_norm15:+.2f} giảm "
                             f"nhưng giá vẫn cao — quỹ đang xả bí mật")
    if absorption_events >= 2:
        fund_score += 2
        fund_evidence.append(f"🟢 {absorption_events} SỰ KIỆN HẤP THỤ (volume >2x, giá <0.5%): "
                             f"quỹ đang 'nuốt' toàn bộ lệnh bán của retail")
    elif absorption_events == 1:
        fund_score += 1
        fund_evidence.append("🟡 1 sự kiện hấp thụ volume phát hiện — cần thêm xác nhận")
    if ad_slope5_norm > 0.5:
        fund_score += 2
        fund_evidence.append(f"🟢 A/D Line tăng mạnh (slope = {ad_slope5_norm:+.2f}): "
                             f"dòng tiền ẩn đang vào dù giá chưa phản ánh")
    elif ad_slope5_norm > 0.15:
        fund_score += 1
        fund_evidence.append(f"🟡 A/D Line tăng nhẹ ({ad_slope5_norm:+.2f}) — tiền đang vào từ từ")
    elif ad_slope5_norm < -0.5:
        fund_score -= 2
        fund_evidence.append(f"🔴 A/D Line giảm mạnh ({ad_slope5_norm:+.2f}) — tiền đang rút ra bí mật")
    elif ad_slope5_norm < -0.15:
        fund_score -= 1
        fund_evidence.append(f"🟠 A/D Line giảm nhẹ ({ad_slope5_norm:+.2f}) — tiền đang rút dần")
    if cmf_slope5 > 0.006:
        fund_score += 1
        fund_evidence.append(f"🟢 CMF Momentum dương ({cmf_slope5:+.5f}): áp lực mua tích lũy dần — "
                             f"quỹ đang tăng tỷ trọng từ từ")
    elif cmf_slope5 < -0.006:
        fund_score -= 1
        fund_evidence.append(f"🟠 CMF Momentum âm ({cmf_slope5:+.5f}): áp lực bán tích lũy — quỹ giảm tỷ trọng")

    if not fund_evidence:
        fund_evidence.append("⚪ Không phát hiện tín hiệu tích lũy hoặc phân phối ẩn trong 15 phiên gần nhất")

    if fund_score >= 5:
        fund_action = "📦 GOM HÀNG BÍ MẬT — Tín hiệu rất mạnh"
        fund_desc   = ("Quỹ đang tích lũy cổ phiếu một cách bí mật qua nhiều phiên. "
                       "A/D Line, OBV, và Absorption Volume đều chỉ ra dòng tiền lớn đang vào "
                       "dù giá chưa phản ánh — thường báo hiệu tăng trong vài tuần tới.")
        fund_color  = "#00E676"; fund_alert = "HIGH"
    elif fund_score >= 2:
        fund_action = "🔍 ĐANG THEO DÕI / GOM DÈ DẶT"
        fund_desc   = "Quỹ đang tích lũy từ từ, chưa đủ tín hiệu để xác nhận mạnh."
        fund_color  = "#69F0AE"; fund_alert = "MED"
    elif fund_score <= -5:
        fund_action = "🚪 THOÁT HÀNG BÍ MẬT — Cảnh báo rất cao"
        fund_desc   = ("Quỹ đang giảm tỷ trọng bí mật qua nhiều phiên. A/D Line và OBV đều giảm "
                       "trong khi giá chưa rơi — cẩn thận, giá có thể sụt mạnh sắp tới.")
        fund_color  = "#FF1744"; fund_alert = "HIGH"
    elif fund_score <= -2:
        fund_action = "📉 GIẢM TỶ TRỌNG — Thận trọng"
        fund_desc   = "Quỹ đang bán dần, chưa mạnh nhưng cần theo dõi."
        fund_color  = "#FF9100"; fund_alert = "MED"
    else:
        fund_action = "😴 CHỜ ĐỢI — Chưa có hành động rõ ràng"
        fund_desc   = "Quỹ chưa có tín hiệu tích lũy hay phân phối đặc biệt trong 15 phiên gần nhất."
        fund_color  = "#90CAF9"; fund_alert = "LOW"
    fund_confidence = min(100, abs(fund_score) * 16)

    # ════════════════════════════════════════════════════════════════════════
    # SHARK (CÁ MẬP / TAY TO) ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    shark_evidence = []
    shark_score    = 0

    if stop_hunts >= 3:
        shark_score += 3
        shark_evidence.append(f"🎯 {stop_hunts} lần GIẬT STOP LOSS trong 15 phiên — "
                              f"cá mập xuyên hỗ trợ để thu gom hàng của retail bán hoảng")
    elif stop_hunts >= 1:
        shark_score += 1
        shark_evidence.append(f"🟡 {stop_hunts} lần stop hunt phát hiện — theo dõi thêm")

    if fake_breakdowns >= 2:
        shark_score += 2
        shark_evidence.append(f"⚡ {fake_breakdowns} FAKE BREAKDOWN (xuyên BB Lower rồi đóng vào trong) — "
                              f"bẫy retail bán ra, cá mập gom hàng giá rẻ")
    elif fake_breakdowns == 1:
        shark_score += 1
        shark_evidence.append("🟡 1 Fake Breakdown — dấu hiệu nhẹ, cần thêm xác nhận")

    if is_wyckoff_spring:
        shark_score += 3
        shark_evidence.append(f"🌊 WYCKOFF SPRING — {wyckoff_desc}. "
                              f"Cá mập đang 'kiểm tra' đáy cuối cùng trước khi đẩy giá")

    if effort_absorbed:
        shark_score += 2
        for d in effort_details:
            shark_evidence.append(f"💪 EFFORT vs RESULT — {d}: lệnh bán bị hấp thụ bởi tay to")

    if bb_compressed and stealth_accum:
        shark_score += 2
        shark_evidence.append("🔥 BB nén + OBV tăng ẩn — cá mập đang hoàn tất gom hàng, sắp bắt đầu markup")

    if not shark_evidence:
        shark_evidence.append("⚪ Không phát hiện dấu hiệu gom hàng ẩn của cá mập trong 15 phiên gần nhất")

    if shark_score >= 6:
        shark_action = "🐋 GOM HÀNG CỰC KỲ TÍCH CỰC — Tín hiệu đặc biệt mạnh"
        shark_desc   = ("Nhiều tín hiệu đồng thời cho thấy cá mập đang gom hàng bí mật: "
                        "Stop hunt + Fake breakdown + Wyckoff Spring + hấp thụ volume = "
                        "giai đoạn cuối tích lũy trước đợt đẩy giá lớn.")
        shark_color  = "#00E676"; shark_alert = "HIGH"
    elif shark_score >= 3:
        shark_action = "🦈 ĐANG GOM — Tín hiệu trung bình"
        shark_desc   = "Cá mập có dấu hiệu gom hàng nhưng chưa đủ mạnh để xác nhận hoàn toàn. Theo dõi sát."
        shark_color  = "#69F0AE"; shark_alert = "MED"
    elif shark_score >= 1:
        shark_action = "👀 DẤU HIỆU YẾU — Quan sát thêm"
        shark_desc   = "Một vài tín hiệu ẩn yếu. Chưa đủ để kết luận cá mập đang hành động."
        shark_color  = "#FFD740"; shark_alert = "LOW"
    elif shark_score <= -3:
        shark_action = "📤 XẢ HÀNG MẠNH"
        shark_desc   = "Cá mập đang thoát hàng mạnh. Tránh mua vào, cân nhắc cắt lỗ nếu đang giữ."
        shark_color  = "#FF1744"; shark_alert = "HIGH"
    else:
        shark_action = "🔍 QUAN SÁT — Chưa ra tay"
        shark_desc   = "Chưa có tín hiệu rõ ràng từ cá mập. Thị trường đang bình thường."
        shark_color  = "#90CAF9"; shark_alert = "LOW"
    shark_confidence = min(100, shark_score * 15)

    # ── Hidden Signals Summary ────────────────────────────────────────────────
    hidden_signals = []
    if bb_compressed:
        hidden_signals.append(("🔴 BB Compression",
                               f"BB thu hẹp còn {bb_width_ratio:.0%} — năng lượng đang tích tụ cho biến động lớn"))
    if stealth_accum:
        hidden_signals.append(("🟢 Stealth Accumulation",
                               "OBV tăng nhưng giá không tăng — gom hàng ẩn đang diễn ra"))
    if is_wyckoff_spring:
        hidden_signals.append(("🌊 Wyckoff Spring",
                               "Test đáy với volume thấp — tín hiệu chuẩn bị đảo chiều tăng"))
    if stop_hunts >= 2:
        hidden_signals.append(("🎯 Stop Hunt",
                               f"{stop_hunts} lần giật stop → cá mập gom hàng sau hoảng loạn"))
    if fake_breakdowns >= 1:
        hidden_signals.append(("⚡ Fake Breakdown",
                               "BB Lower bị xuyên giả → bẫy retail, cá mập gom giá rẻ"))
    if absorption_events >= 2:
        hidden_signals.append(("💧 Volume Absorption",
                               "Volume cao nhưng giá không đổi → tiền lớn hấp thụ lệnh bán"))
    if ad_slope5_norm > 0.3:
        hidden_signals.append(("📊 A/D Line Tăng",
                               "Dòng tiền ẩn đang tích lũy — chưa phản ánh vào giá"))
    if effort_absorbed:
        hidden_signals.append(("💪 Effort vs Result",
                               "Volume lớn nhưng giá không đi — lực bán bị hấp thụ bởi tay to"))
    if cmf_slope5 > 0.006:
        hidden_signals.append(("📈 CMF Momentum",
                               "Áp lực mua đang tích lũy từ từ — quỹ đang tăng tỷ trọng"))

    return {
        "market_maker": {
            "action": mm_action, "desc": mm_desc, "color": mm_color,
            "evidence": mm_evidence, "confidence": mm_confidence, "alert": mm_alert,
        },
        "fund": {
            "action": fund_action, "desc": fund_desc, "color": fund_color,
            "evidence": fund_evidence, "confidence": fund_confidence, "alert": fund_alert,
        },
        "shark": {
            "action": shark_action, "desc": shark_desc, "color": shark_color,
            "evidence": shark_evidence, "confidence": shark_confidence, "alert": shark_alert,
        },
        "hidden_signals": hidden_signals,
        "bb_compressed": bb_compressed,
        "bb_width_ratio": bb_width_ratio,
        "stealth_accum": stealth_accum,
        "stealth_distrib": stealth_distrib,
        "is_wyckoff_spring": is_wyckoff_spring,
        "stop_hunts": stop_hunts,
        "fake_breakdowns": fake_breakdowns,
        "absorption_events": absorption_events,
        "effort_absorbed": effort_absorbed,
        "ad_slope": ad_slope5_norm,
        "cmf_slope": cmf_slope5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TREND STRENGTH METER
# ─────────────────────────────────────────────────────────────────────────────
def calc_trend_strength(df: pd.DataFrame, sm_result: dict) -> dict:
    """Tính điểm xu hướng tổng hợp 0–100 (0=Rất giảm, 50=Trung tính, 100=Rất tăng)."""
    score   = 50.0
    signals = []
    latest  = df.iloc[-1]
    close   = latest["close"]

    # ADX
    adx_val  = latest.get("ADX",      np.nan)
    di_plus  = latest.get("DI_Plus",  0)
    di_minus = latest.get("DI_Minus", 0)
    if pd.notna(adx_val):
        if adx_val > 35:
            adj = 12 if di_plus > di_minus else -12
            score += adj
            dir_label = "TĂNG" if di_plus > di_minus else "GIẢM"
            signals.append(f"ADX={adx_val:.1f} >35 ({dir_label} mạnh)")
        elif adx_val > 22:
            adj = 6 if di_plus > di_minus else -6
            score += adj
            dir_label = "Tăng" if di_plus > di_minus else "Giảm"
            signals.append(f"ADX={adx_val:.1f} >22 (xu hướng {dir_label})")
        else:
            signals.append(f"ADX={adx_val:.1f} <22 → Sideway, không có xu hướng rõ")

    # EMA Alignment
    ema5  = latest.get("EMA5",  np.nan)
    ema20 = latest.get("EMA20", np.nan)
    ema50 = latest.get("EMA50", np.nan)
    if pd.notna(ema5) and pd.notna(ema20) and pd.notna(ema50):
        if ema5 > ema20 > ema50 and close > ema5:
            score += 16; signals.append("EMA5>EMA20>EMA50 + Giá>EMA5: Bull Alignment hoàn hảo ✅")
        elif ema5 > ema20 and close > ema20:
            score += 8;  signals.append("EMA5>EMA20 + Giá>EMA20: Ngắn-trung hạn tăng")
        elif close > ema50:
            score += 4;  signals.append("Giá trên EMA50: Dài hạn vẫn tăng")
        elif ema5 < ema20 < ema50 and close < ema5:
            score -= 16; signals.append("EMA5<EMA20<EMA50 + Giá<EMA5: Bear Alignment hoàn hảo ❌")
        elif ema5 < ema20 and close < ema20:
            score -= 8;  signals.append("EMA5<EMA20 + Giá<EMA20: Ngắn-trung hạn giảm")
        elif close < ema50:
            score -= 4;  signals.append("Giá dưới EMA50: Dài hạn yếu")

    # Ichimoku Cloud
    span_a = latest.get("SpanA", np.nan)
    span_b = latest.get("SpanB", np.nan)
    if pd.notna(span_a) and pd.notna(span_b):
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)
        if close > cloud_top:
            score += 10; signals.append("Giá trên mây Kumo → Bullish ☁️")
        elif close < cloud_bot:
            score -= 10; signals.append("Giá dưới mây Kumo → Bearish ☁️")
        else:
            signals.append("Giá trong mây Kumo → Trung tính")
    tenkan = latest.get("Tenkan", np.nan)
    kijun  = latest.get("Kijun",  np.nan)
    if pd.notna(tenkan) and pd.notna(kijun):
        if tenkan > kijun:
            score += 4; signals.append("Tenkan > Kijun → Tín hiệu mua")
        else:
            score -= 4; signals.append("Tenkan < Kijun → Tín hiệu bán")

    # MACD
    macd_v = latest.get("MACD",        np.nan)
    sig_v  = latest.get("MACD_Signal", np.nan)
    if pd.notna(macd_v) and pd.notna(sig_v):
        if macd_v > sig_v:
            score += 5; signals.append("MACD > Signal → Momentum tăng")
        else:
            score -= 5; signals.append("MACD < Signal → Momentum giảm")

    # Smart Money phase contribution
    sm_pct  = sm_result["score"] / sm_result["max_score"]
    score   += sm_pct * 8
    signals.append(f"Smart Money score: {sm_result['score']:+.1f} ({sm_result['phase']})")

    score = max(0.0, min(100.0, score))

    if score >= 78:
        label = "RẤT MẠNH TĂNG 🚀"; color = "#00C853"; bg = "rgba(0,200,83,0.12)"
    elif score >= 62:
        label = "ĐANG TĂNG 📈";      color = "#69F0AE"; bg = "rgba(105,240,174,0.10)"
    elif score >= 42:
        label = "TRUNG TÍNH ⚖️";     color = "#FFD740"; bg = "rgba(255,215,64,0.08)"
    elif score >= 28:
        label = "ĐANG GIẢM 📉";      color = "#FF9100"; bg = "rgba(255,145,0,0.10)"
    else:
        label = "RẤT MẠNH GIẢM 🔻"; color = "#FF1744"; bg = "rgba(255,23,68,0.10)"

    return {"score": score, "label": label, "color": color, "bg": bg, "signals": signals}


# ─────────────────────────────────────────────────────────────────────────────
# AI CALL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(api_key: str, prompt: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post("https://api.anthropic.com/v1/messages",
                         headers=headers, json=body, timeout=90)
    if not resp.ok:
        st.error(f"Chi tiết lỗi: {resp.status_code} — {resp.text}")
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_gemini(api_key: str, prompt: str) -> str:
    from google import genai  # type: ignore
    import time
    MODELS = ["gemini-2.5-flash-preview-05-20", "gemini-2.0-flash", "gemini-1.5-flash-latest"]
    client = genai.Client(api_key=api_key)
    for model in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    if attempt == 0:
                        time.sleep(5); continue
                    else:
                        break
                elif "NOT_FOUND" in err_str or "404" in err_str:
                    break
                else:
                    raise
    raise RuntimeError(
        "⚠️ Đã vượt quá quota miễn phí Gemini.\n\n"
        "**Cách khắc phục:**\n- Nạp billing tại https://ai.dev/billing\n"
        "- Hoặc chờ ~1 phút rồi thử lại\n- Hoặc chuyển sang **Claude (Anthropic)**"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VN30 & UNIVERSE DATA
# ─────────────────────────────────────────────────────────────────────────────
VN30_TICKERS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","NVL","PDR","PLX","PNJ","POW","SAB","SSI",
    "STB","TCB","TPB","VCB","VHM","VIC","VJC","VNM","VPB","VRE",
]
_FALLBACK_LIQUID = [
    "EIB","LPB","MSB","NAB","OCB","SHB","VIB","BVB","ABB","PGB",
    "AGR","BSI","HCM","VCI","VIX","VND","VPI","SHS","MBS","CTS",
    "DIG","DXG","HDG","KBC","KDH","NLG","PDR","SCR","TDC","IDC",
    "IJC","ITA","BCG","CEO","DRH","NHA","TIP","VGC","HSG","NKG",
    "POM","TLH","TVN","VGS","SMC","TNA","KSB","VIS","BSR","OIL",
    "PVS","PVT","DVN","PVC","PGD","GEX","PC1","REE","DGW","FRT",
    "HAG","PAN","QNS","SAF","VNM","KDC","MCH","DPM","DCM","DGC",
    "CSV","LAS","DDV","TDN","PLC","AAA","CII","CTD","FCN","HBC",
    "HHV","LCG","SC5","VCG","HUT","CMG","ELC","SAM","SPC","VGI",
    "BWE","CNG","GMD","HAH","HVN","IMP","PAC","PHR","SZC","TCH",
    "TV2","VCS","VSH","YEG","DBC","GDT","LHG","PNJ","PVD",
]


@st.cache_data(ttl=1800, show_spinner=False)
def build_scan_universe() -> tuple[list[str], str]:
    universe    = set(VN30_TICKERS)
    source_note = "VN30 (fallback)"
    _ssi_urls = [
        "https://iboard-query.ssi.com.vn/v2/stock/board-data/exchange?exchange=HOSE&size=3000",
        "https://iboard-query.ssi.com.vn/v2/stock/snapshot?exchange=HOSE&size=3000",
    ]
    for _url in _ssi_urls:
        try:
            _h = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
            _r = requests.get(_url, headers=_h, timeout=20)
            if not _r.ok:
                continue
            _d = _r.json()
            _items = (_d if isinstance(_d, list) else
                      _d.get("data", _d.get("items", _d.get("securities", _d.get("stocks", [])))))
            if isinstance(_items, dict):
                _items = list(_items.values())[0] if _items else []
            _added = 0
            for _item in (_items or []):
                if not isinstance(_item, dict): continue
                _sym = str(_item.get("symbol", _item.get("code",
                           _item.get("ticker", _item.get("s", ""))))).upper().strip()
                if not _sym or len(_sym) > 4 or not _sym.isalpha(): continue
                _pr   = float(_item.get("lastPrice", _item.get("closePrice",
                              _item.get("close", _item.get("mp", _item.get("cp", 0))))) or 0)
                _price = _pr if _pr >= 1000 else _pr * 1000
                _vol   = float(_item.get("totalVolume", _item.get("totalMatchVolume",
                               _item.get("tradingVolume", _item.get("tv", _item.get("mv", 0))))) or 0)
                if _price > 10000 and _vol > 200000:
                    universe.add(_sym); _added += 1
            if _added > 0:
                source_note = f"SSI iBoard ({_added} mã lọc được, HOSE)"; break
        except Exception:
            continue
    if len(universe) <= len(VN30_TICKERS):
        universe.update(_FALLBACK_LIQUID)
        source_note = f"VN30 + danh sách dự phòng ({len(_FALLBACK_LIQUID)} mã)"
    return sorted(universe), source_note


@st.cache_data(ttl=300)
def scan_single_stock(symbol: str) -> dict | None:
    try:
        df = get_clean_stock_data(symbol)
        if df is None or df.empty or len(df) < 30: return None
        for col in ["open","high","low","close","volume"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df) < 30: return None

        df["RSI"]         = RSIIndicator(close=df["close"], window=14).rsi()
        bb                = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["BB_High"]     = bb.bollinger_hband()
        df["BB_Mid"]      = bb.bollinger_mavg()
        df["BB_Low"]      = bb.bollinger_lband()
        macd_obj          = MACD(close=df["close"])
        df["MACD"]        = macd_obj.macd()
        df["MACD_Signal"] = macd_obj.macd_signal()
        df["MCDX"]        = calc_mcdx(df)
        (df["Tenkan"], df["Kijun"], df["SpanA"], df["SpanB"], df["Chikou"]) = calc_ichimoku(df)
        df = calc_adx(df); df = calc_ema_cross(df)
        fib_levels = calc_fibonacci(df)
        df         = calc_smart_money(df)
        sm_result  = detect_smart_money_phase(df)

        latest  = df.iloc[-1]
        prev    = df.iloc[-2]
        pct_chg = (latest["close"] - prev["close"]) / prev["close"] * 100

        score   = 0
        rsi_now = latest["RSI"] if pd.notna(latest["RSI"]) else 50
        if rsi_now < 35:   score += 2
        elif rsi_now < 50: score += 1
        elif rsi_now > 70: score -= 2

        if pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_Signal")):
            score += 2 if latest["MACD"] > latest["MACD_Signal"] else -1

        mcdx_now = latest.get("MCDX", 0) if pd.notna(latest.get("MCDX")) else 0
        if mcdx_now > 0.2:   score += 2
        elif mcdx_now > 0:   score += 1
        else:                 score -= 1

        # ADX bonus
        adx_val = latest.get("ADX", np.nan)
        if pd.notna(adx_val) and adx_val > 25:
            if (latest.get("DI_Plus", 0) or 0) > (latest.get("DI_Minus", 0) or 0):
                score += 1
            else:
                score -= 1

        # EMA alignment
        ema5  = latest.get("EMA5",  np.nan)
        ema20 = latest.get("EMA20", np.nan)
        ema50 = latest.get("EMA50", np.nan)
        if pd.notna(ema5) and pd.notna(ema20) and pd.notna(ema50):
            if ema5 > ema20 > ema50:   score += 2
            elif ema5 < ema20 < ema50: score -= 2

        span_a_s = latest.get("SpanA", None); span_b_s = latest.get("SpanB", None)
        if span_a_s and span_b_s:
            if latest["close"] > max(span_a_s, span_b_s):   score += 2
            elif latest["close"] < min(span_a_s, span_b_s): score -= 2
        if pd.notna(latest.get("Tenkan")) and pd.notna(latest.get("Kijun")):
            score += 1 if latest["Tenkan"] > latest["Kijun"] else -1

        _is_down = fib_levels.get("_is_downtrend", True)
        _fh = fib_levels.get("_high", 0); _fl = fib_levels.get("_low", 0)
        _diff = fib_levels.get("_diff", _fh - _fl); cn = latest["close"]; fm = _diff * 0.03
        if _is_down:
            f618 = _fh - 0.618*_diff; f650 = _fh - 0.650*_diff
            f786 = _fh - 0.786*_diff; f382 = _fh - 0.382*_diff
            if f650 <= cn <= f618:            score += 3
            elif abs(cn - f618) <= fm:        score += 2
            elif f786 - fm <= cn <= f786+fm:  score += 2
            elif cn < f786:                   score -= 1
        else:
            f618 = _fl + 0.618*_diff; f382 = _fl + 0.382*_diff; f50 = _fl + 0.500*_diff
            if cn > f618:    score += 2
            elif cn > f50:   score += 1
            elif cn < f382:  score -= 1

        max_score = 16
        pct_score = score / max_score
        if pct_score >= 0.45:    recommendation = "✅ MUA"
        elif pct_score >= 0.15:  recommendation = "⏳ THEO DÕI"
        else:                    recommendation = "🚫 TRÁNH/BÁN"

        _is_down2 = fib_levels.get("_is_downtrend", True)
        _fh2 = fib_levels.get("_high", 0); _fl2 = fib_levels.get("_low", 0)
        _diff2 = fib_levels.get("_diff", _fh2 - _fl2)
        if _is_down2:
            _f618_z = _fh2 - 0.618*_diff2; _f650_z = _fh2 - 0.650*_diff2
            _f786_z = _fh2 - 0.786*_diff2; _f382_z = _fh2 - 0.382*_diff2
            if cn < _f786_z:                      fib_zone = "⛔ Dưới hỗ trợ"
            elif _f786_z <= cn <= _f650_z:        fib_zone = "⚠️ Hỗ trợ sâu (65-78.6%)"
            elif _f650_z <= cn <= _f618_z:        fib_zone = "⭐ Golden Pocket"
            elif _f618_z < cn <= _f382_z:         fib_zone = "📍 Vùng hỗ trợ (38-61.8%)"
            else:                                  fib_zone = "🔺 Trên 38.2% (cao)"
        else:
            _f236_z = _fl2 + 0.236*_diff2; _f382_z = _fl2 + 0.382*_diff2
            if cn <= _f236_z:    fib_zone = "✅ Gần đáy (tích lũy)"
            elif cn <= _f382_z:  fib_zone = "📍 Phục hồi sớm"
            else:                fib_zone = "🔺 Đã phục hồi nhiều"

        return {
            "symbol": symbol, "close": latest["close"], "pct_chg": pct_chg,
            "rsi": rsi_now, "mcdx": mcdx_now, "adx": float(adx_val) if pd.notna(adx_val) else 0,
            "macd_bull": bool(pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_Signal"))
                              and latest["MACD"] > latest["MACD_Signal"]),
            "ema_bull": bool(pd.notna(ema5) and pd.notna(ema20) and ema5 > ema20),
            "sm_phase": sm_result["phase"], "sm_icon": sm_result["icon"],
            "sm_color": sm_result["color"], "sm_score": sm_result["score"],
            "fib_zone": fib_zone, "score": score, "max_score": max_score,
            "recommendation": recommendation,
            "obv_slope": sm_result.get("obv_slope", 0), "cmf": sm_result.get("cmf", 0),
            "mfi": sm_result.get("mfi", 50), "buy_ratio": sm_result.get("buy_ratio", 0.5),
            "vol_ratio": sm_result.get("vol_ratio", 1.0),
            "avg_vol": float(latest.get("Vol_MA20", 0) or 0),
        }
    except Exception:
        return None


def _phase_order(phase: str) -> int:
    return {"GOM HÀNG": 0, "ĐẨY GIÁ (MARKUP)": 1,
            "TRUNG TÍNH / QUAN SÁT": 2, "XẢ HÀNG (DISTRIBUTION)": 3,
            "ĐÈ GIÁ (MARKDOWN)": 4}.get(phase, 5)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PHÂN TÍCH ĐƠN LẺ  (toàn bộ logic trong with _tab1:)
# ═════════════════════════════════════════════════════════════════════════════
with _tab1:
    ticker = st.text_input(
        "🔍 Nhập mã chứng khoán (Cổ phiếu hoặc VN30F1M):",
        "VN30F1M",
    ).upper().strip()

    if st.button("🚀 Lấy Dữ Liệu & Phân Tích", type="primary", key="analyze_btn"):

        with st.spinner(f"Đang trích xuất dữ liệu thị trường cho mã **{ticker}**..."):
            df = get_clean_stock_data(ticker)

        if df is None or df.empty:
            st.error("❌ Không thể lấy dữ liệu. Kiểm tra lại mã chứng khoán hoặc kết nối mạng.")
            st.stop()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        if len(df) < 30:
            st.error("❌ Không đủ dữ liệu để tính chỉ báo (cần tối thiểu 30 phiên).")
            st.stop()

        # ── Tính toán chỉ báo ─────────────────────────────────────────────
        df["RSI"]         = RSIIndicator(close=df["close"], window=14).rsi()
        bb                = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["BB_High"]     = bb.bollinger_hband()
        df["BB_Mid"]      = bb.bollinger_mavg()
        df["BB_Low"]      = bb.bollinger_lband()
        macd_obj          = MACD(close=df["close"])
        df["MACD"]        = macd_obj.macd()
        df["MACD_Signal"] = macd_obj.macd_signal()
        df["MACD_Diff"]   = macd_obj.macd_diff()
        df["MCDX"]        = calc_mcdx(df)
        (df["Tenkan"], df["Kijun"], df["SpanA"], df["SpanB"], df["Chikou"]) = calc_ichimoku(df)
        df         = calc_adx(df)
        df         = calc_ema_cross(df)
        fib_levels = calc_fibonacci(df)
        df         = calc_smart_money(df)
        sm_result  = detect_smart_money_phase(df)
        hidden_sm  = detect_hidden_smart_money(df, sm_result)
        patterns   = detect_candlestick_patterns(df)
        trend_str  = calc_trend_strength(df, sm_result)

        latest  = df.iloc[-1]
        prev    = df.iloc[-2]
        pct_chg = (latest["close"] - prev["close"]) / prev["close"] * 100

        show_cols = [c for c in
            ["time","open","high","low","close","volume","RSI","MACD","MCDX","ADX","EMA20","Tenkan","Kijun"]
            if c in df.columns]

        # ─────────────────────────────────────────────────────────────────
        # TREND STRENGTH METER
        # ─────────────────────────────────────────────────────────────────
        ts = trend_str
        st.markdown(
            f"""
<div style="background:{ts['bg']}; border-radius:14px; padding:16px 24px;
            border:1.5px solid {ts['color']}44; margin-bottom:18px; display:flex;
            align-items:center; gap:20px;">
  <div style="flex:1;">
    <div style="font-size:0.8rem; color:#aaa; letter-spacing:1px;">📡 XU HƯỚNG TỔNG HỢP</div>
    <div style="font-size:1.6rem; font-weight:800; color:{ts['color']}; margin:4px 0;">
      {ts['label']}
    </div>
    <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:10px; width:100%; margin-top:6px;">
      <div style="height:10px; width:{ts['score']:.0f}%; border-radius:6px;
                  background:linear-gradient(90deg,#FF1744,#FFD740,#00C853);"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#777; margin-top:3px;">
      <span>Rất giảm</span><span>Trung tính</span><span>Rất tăng</span>
    </div>
  </div>
  <div style="font-size:2.8rem; font-weight:900; color:{ts['color']}; min-width:70px; text-align:right;">
    {ts['score']:.0f}<span style="font-size:1rem;">/100</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        # ── Metrics row ───────────────────────────────────────────────────
        st.subheader(f"📈 Trạng thái kỹ thuật hiện tại — {ticker}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Giá Đóng Cửa", f"{latest['close']:,.0f} đ", f"{pct_chg:+.2f}%")
        rsi_val  = latest["RSI"]
        rsi_note = "⚠️ Quá mua" if rsi_val > 70 else ("⚠️ Quá bán" if rsi_val < 30 else "Trung tính")
        c2.metric("RSI (14)", f"{rsi_val:.1f}", rsi_note)
        adx_val = latest.get("ADX", 0) or 0
        di_p    = latest.get("DI_Plus",  0) or 0
        di_m    = latest.get("DI_Minus", 0) or 0
        adx_note = "Trend mạnh" if adx_val > 25 else "Sideway"
        c3.metric("ADX (14)", f"{adx_val:.1f}", f"DI+{di_p:.0f}/DI-{di_m:.0f} · {adx_note}")
        mcdx_val = latest["MCDX"]
        c4.metric("MCDX", f"{mcdx_val:.3f}", "📈 Tăng" if mcdx_val > 0 else "📉 Giảm")
        ema5_v  = latest.get("EMA5",  0) or 0
        ema20_v = latest.get("EMA20", 0) or 0
        c5.metric("EMA 5 / 20", f"{ema5_v:,.0f} / {ema20_v:,.0f}",
                  "Bullish" if ema5_v > ema20_v else "Bearish")
        c6.metric("Fib 50%", f"{fib_levels.get('50.0%', 0):,.0f} đ")

        # Candlestick patterns badge
        if patterns:
            badges = ""
            for pname, pdesc, ptype in patterns:
                col_map = {"bullish": "#00E676", "bearish": "#FF5252", "neutral": "#FFD740"}
                pcolor  = col_map.get(ptype, "#aaa")
                badges += (f"<span style='background:{pcolor}22; border:1px solid {pcolor}; "
                           f"border-radius:20px; padding:4px 12px; margin:4px; font-size:0.82rem; "
                           f"color:{pcolor}; display:inline-block;' title='{pdesc}'>{pname}</span>")
            st.markdown(
                f"<div style='margin:8px 0 16px 0;'><b style='color:#aaa;font-size:0.8rem;'>🕯️ PATTERN NẾN PHÁT HIỆN:</b><br>{badges}</div>",
                unsafe_allow_html=True)

        # ── Biểu đồ tổng hợp ─────────────────────────────────────────────
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            row_heights=[0.44, 0.13, 0.14, 0.14, 0.15],
            vertical_spacing=0.022,
            subplot_titles=(
                f"Biểu đồ nến · BB · Ichimoku · EMA · Fibonacci — {ticker}",
                "Khối lượng giao dịch",
                "RSI (14) + ADX (14)",
                "MACD + MCDX",
                "EMA 5 / 20 / 50",
            ),
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Nến",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # Bollinger Bands
        for col, color, name in [
            ("BB_High", "rgba(255,80,80,0.55)", "BB Upper"),
            ("BB_Mid",  "rgba(180,180,180,0.40)", "BB Mid"),
            ("BB_Low",  "rgba(80,220,80,0.55)", "BB Lower"),
        ]:
            fig.add_trace(go.Scatter(x=df["time"], y=df[col],
                                     line=dict(color=color, width=1), name=name), row=1, col=1)

        # EMA lines on main chart
        for ema_col, ema_color, ema_name in [
            ("EMA5",  "#FFD740", "EMA5"),
            ("EMA20", "#40C4FF", "EMA20"),
            ("EMA50", "#E040FB", "EMA50"),
        ]:
            fig.add_trace(go.Scatter(x=df["time"], y=df[ema_col],
                                     line=dict(color=ema_color, width=1.4, dash="dot"),
                                     name=ema_name, opacity=0.85), row=1, col=1)

        # Ichimoku
        fig.add_trace(go.Scatter(x=df["time"], y=df["Tenkan"],
                                  line=dict(color="#00E5FF", width=1.8), name="Tenkan-sen"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["Kijun"],
                                  line=dict(color="#FF6B6B", width=1.8), name="Kijun-sen"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["Chikou"],
                                  line=dict(color="#B39DDB", width=1.1, dash="dot"),
                                  name="Chikou Span", opacity=0.7), row=1, col=1)
        span_a = df["SpanA"].values; span_b = df["SpanB"].values; times = df["time"].values
        bullish_a = np.where(span_a >= span_b, span_a, np.nan)
        bullish_b = np.where(span_a >= span_b, span_b, np.nan)
        bearish_a = np.where(span_a <  span_b, span_a, np.nan)
        bearish_b = np.where(span_a <  span_b, span_b, np.nan)
        fig.add_trace(go.Scatter(x=times, y=bullish_a, line=dict(color="rgba(0,0,0,0)", width=0),
                                  showlegend=False, name="_bull_a"), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bullish_b, fill="tonexty",
                                  fillcolor="rgba(38,166,154,0.15)",
                                  line=dict(color="#26a69a", width=0.6), name="Kumo Bullish ☁️",
                                  legendgroup="kumo"), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bearish_b, line=dict(color="rgba(0,0,0,0)", width=0),
                                  showlegend=False, name="_bear_b"), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bearish_a, fill="tonexty",
                                  fillcolor="rgba(239,83,80,0.15)",
                                  line=dict(color="#ef5350", width=0.6), name="Kumo Bearish ☁️",
                                  legendgroup="kumo"), row=1, col=1)

        # Fibonacci
        fib_palette = {
            "0.0% (Đỉnh)": "#9E9E9E", "0.0% (Đáy)": "#9E9E9E",
            "23.6%": "#7986CB", "38.2%": "#29B6F6",
            "50.0%": "#EF5350", "61.8% ✨": "#FFA726",
            "65.0% 🏅": "#FF7043", "78.6%": "#AB47BC",
            "100.0% (Đỉnh)": "#9E9E9E", "100.0% (Đáy)": "#9E9E9E",
            "127.2% 📉": "#546E7A", "127.2% 📈": "#546E7A",
            "161.8% 📉": "#37474F", "161.8% 📈": "#37474F",
        }
        x_range = [df["time"].iloc[0], df["time"].iloc[-1]]
        x_label = df["time"].iloc[-1]
        for label, price in fib_levels.items():
            if label.startswith("_"): continue
            is_key = label in ("38.2%", "50.0%", "61.8% ✨")
            color_fib = fib_palette.get(label, "#9E9E9E")
            width_fib = 2.5 if is_key else 1.0
            fig.add_trace(go.Scatter(
                x=x_range, y=[price, price], mode="lines",
                line=dict(color=color_fib, width=width_fib, dash="dot" if not is_key else "dash"),
                name=f"Fib {label}", text=f"Fib {label}  {price:,.0f}", hoverinfo="text",
            ), row=1, col=1)
            fig.add_annotation(xref="x", yref="y", x=x_label, y=price,
                               text=f"  {label} · {price:,.0f}", showarrow=False,
                               font=dict(color=color_fib, size=9 if is_key else 8),
                               xanchor="left", row=1, col=1)

        # Volume
        bar_colors = ["#ef5350" if df["close"].iloc[i] < df["open"].iloc[i] else "#26a69a"
                      for i in range(len(df))]
        fig.add_trace(go.Bar(x=df["time"], y=df["volume"],
                              marker_color=bar_colors, name="Volume", showlegend=False), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df["time"], y=df["RSI"],
                                  line=dict(color="#FF9800", width=1.8), name="RSI"), row=3, col=1)
        for level, color in [(70, "rgba(255,80,80,0.6)"), (30, "rgba(80,200,80,0.6)")]:
            fig.add_shape(type="line", xref="paper", yref="y3",
                          x0=0, x1=1, y0=level, y1=level,
                          line=dict(color=color, width=1, dash="dash"))

        # ADX on RSI panel
        fig.add_trace(go.Scatter(x=df["time"], y=df["ADX"],
                                  line=dict(color="#E040FB", width=1.5, dash="dot"),
                                  name="ADX"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["DI_Plus"],
                                  line=dict(color="#26a69a", width=1, dash="dot"),
                                  name="DI+"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["DI_Minus"],
                                  line=dict(color="#ef5350", width=1, dash="dot"),
                                  name="DI-"), row=3, col=1)

        # MACD + MCDX
        hist_colors = np.where(df["MACD_Diff"] >= 0, "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(x=df["time"], y=df["MACD_Diff"],
                              marker_color=hist_colors, name="MACD Histogram"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["MACD"],
                                  line=dict(color="#2196F3", width=1.5), name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["MACD_Signal"],
                                  line=dict(color="#FF5722", width=1.5), name="Signal"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["MCDX"],
                                  line=dict(color="#E040FB", width=2, dash="dot"),
                                  name="MCDX"), row=4, col=1)

        # EMA panel (bottom)
        for ema_col, ema_color, ema_name in [
            ("EMA5",  "#FFD740", "EMA5"),
            ("EMA20", "#40C4FF", "EMA20"),
            ("EMA50", "#E040FB", "EMA50"),
        ]:
            fig.add_trace(go.Scatter(x=df["time"], y=df[ema_col],
                                     line=dict(color=ema_color, width=1.6),
                                     name=f"{ema_name} (panel)"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["close"],
                                  line=dict(color="rgba(255,255,255,0.5)", width=1),
                                  name="Giá (ref)"), row=5, col=1)

        fig.update_layout(
            template="plotly_dark", height=1150,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1, font=dict(size=9),
                        bgcolor="rgba(0,0,0,0.3)", borderwidth=1),
            margin=dict(l=10, r=130, t=60, b=10),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Bảng dữ liệu ─────────────────────────────────────────────────
        st.write("**📋 Bảng dữ liệu 7 phiên gần nhất (kèm chỉ báo):**")
        st.dataframe(df.tail(7)[show_cols].round(2), hide_index=True, use_container_width=True)

        # ── Fibonacci table ───────────────────────────────────────────────
        _fib_is_down = fib_levels.get("_is_downtrend", True)
        _fib_high    = fib_levels.get("_high", 0)
        _fib_low     = fib_levels.get("_low",  0)
        with st.expander("📐 Xem bảng mức Fibonacci Retracement"):
            fib_label = ("Retracement từ Đỉnh → Đáy (vùng HỖ TRỢ)" if _fib_is_down
                         else "Retracement từ Đáy → Đỉnh (vùng KHÁNG CỰ)")
            st.caption(f"📌 {fib_label} &nbsp;|&nbsp; Đỉnh: **{_fib_high:,.0f} đ** &nbsp;·&nbsp; Đáy: **{_fib_low:,.0f} đ**")
            fib_df = pd.DataFrame([{"Mức Fibonacci": k, "Giá (đ)": f"{v:,.0f}"}
                                    for k, v in fib_levels.items() if not k.startswith("_")])
            st.dataframe(fib_df, hide_index=True, use_container_width=True)

        # ─────────────────────────────────────────────────────────────────
        # SMART MONEY PHASE BANNER
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🐋 Phân Tích Dòng Tiền Thông Minh — Pha Tổng Thể")
        sm = sm_result
        pct_bar = max(0, min(100, (sm["score"] + sm["max_score"]) / (2 * sm["max_score"]) * 100))
        st.markdown(
            f"""
<div style="border:2px solid {sm['border']}; border-radius:14px; padding:20px 28px;
            background:{sm['bg']}; margin-bottom:16px;">
  <div style="font-size:1.9rem; font-weight:800; color:{sm['color']}; letter-spacing:2px;">
    {sm['icon']} &nbsp; GIAI ĐOẠN: {sm['phase']}
  </div>
  <div style="margin-top:8px; font-size:1rem; color:#ddd; line-height:1.6;">{sm['phase_desc']}</div>
  <div style="margin-top:12px; background:rgba(255,255,255,0.08); border-radius:8px; height:10px; width:100%;">
    <div style="height:10px; width:{pct_bar:.0f}%; border-radius:8px;
                background:linear-gradient(90deg,#ef5350,#FFD740,#00C853);"></div>
  </div>
  <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#999; margin-top:4px;">
    <span>📉 Đè giá / Xả</span><span>⚖️ Trung tính</span><span>📈 Gom / Đẩy</span>
  </div>
  <div style="margin-top:8px; font-size:0.9rem; color:#aaa;">
    Điểm Smart Money: <b style="color:{sm['color']};">{sm['score']:+.1f} / {sm['max_score']:.0f}</b>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        # ─────────────────────────────────────────────────────────────────
        # HIDDEN SMART MONEY — Hành vi ẩn của từng nhóm
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Hành Vi Ẩn Từng Nhóm Tiền Lớn — Phân tích chuyên sâu")
        st.caption(
            "Phát hiện hành động của **Market Maker · Quỹ đầu tư · Cá mập** ngay cả khi thị trường "
            "không có biến động lớn — dựa trên **BB Compression · A/D Line · Stealth Accumulation · "
            "Stop Hunt · Fake Breakdown · Wyckoff Spring · Effort vs Result · Order Blocks**."
        )

        # Hidden Signals Bar (nếu có)
        hsigs = hidden_sm.get("hidden_signals", [])
        if hsigs:
            st.markdown(
                "<div style='background:rgba(255,215,64,0.08); border:1px solid #FFD74044; "
                "border-radius:10px; padding:12px 18px; margin-bottom:14px;'>"
                "<div style='font-size:0.85rem; font-weight:700; color:#FFD740; margin-bottom:8px;'>"
                "⚡ TÍN HIỆU ẨN PHÁT HIỆN ĐƯỢC</div>"
                + "".join(
                    f"<div style='font-size:0.82rem; color:#ddd; padding:3px 0;'>• <b>{s[0]}</b>: {s[1]}</div>"
                    for s in hsigs
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        def _confidence_bar(pct: int, color: str) -> str:
            return (f"<div style='background:rgba(255,255,255,0.08); border-radius:4px; "
                    f"height:6px; width:100%; margin-top:6px;'>"
                    f"<div style='height:6px; width:{pct}%; border-radius:4px; "
                    f"background:{color};'></div></div>"
                    f"<div style='font-size:0.72rem; color:#888; margin-top:2px;'>"
                    f"Độ tin cậy: {pct}%</div>")

        alert_badge = {
            "HIGH": "<span style='background:#FF174444; color:#FF5252; border:1px solid #FF5252; "
                    "border-radius:12px; padding:2px 8px; font-size:0.72rem; font-weight:700;'>⚠️ CAO</span>",
            "MED":  "<span style='background:#FFD74033; color:#FFD740; border:1px solid #FFD740; "
                    "border-radius:12px; padding:2px 8px; font-size:0.72rem; font-weight:700;'>🟡 TB</span>",
            "LOW":  "<span style='background:#78909C22; color:#90CAF9; border:1px solid #78909C; "
                    "border-radius:12px; padding:2px 8px; font-size:0.72rem; font-weight:700;'>⚪ THẤP</span>",
        }

        g_mm    = hidden_sm["market_maker"]
        g_fund  = hidden_sm["fund"]
        g_shark = hidden_sm["shark"]

        col_mm, col_fund, col_shark = st.columns(3)

        with col_mm:
            ev_html = "".join(f"<div style='font-size:0.78rem; color:#ccc; padding:3px 0; "
                              f"border-bottom:1px solid rgba(255,255,255,0.05);'>{e}</div>"
                              for e in g_mm["evidence"])
            st.markdown(
                f"""<div style="border:1px solid {g_mm['color']}55; border-top:3px solid {g_mm['color']};
                    border-radius:12px; padding:16px; background:{g_mm['color']}0D; height:100%;">
  <div style="font-size:0.85rem; color:#aaa; font-weight:600;">🏦 MARKET MAKER</div>
  <div style="font-size:1.0rem; font-weight:800; color:{g_mm['color']}; margin:8px 0 4px;">
    {g_mm['action']} &nbsp; {alert_badge.get(g_mm['alert'], '')}
  </div>
  <div style="font-size:0.8rem; color:#bbb; margin-bottom:10px; line-height:1.5;">{g_mm['desc']}</div>
  <div style="font-size:0.78rem; font-weight:600; color:#aaa; margin-bottom:4px;">📋 Bằng chứng:</div>
  {ev_html}
  {_confidence_bar(g_mm['confidence'], g_mm['color'])}
</div>""",
                unsafe_allow_html=True,
            )

        with col_fund:
            ev_html = "".join(f"<div style='font-size:0.78rem; color:#ccc; padding:3px 0; "
                              f"border-bottom:1px solid rgba(255,255,255,0.05);'>{e}</div>"
                              for e in g_fund["evidence"])
            st.markdown(
                f"""<div style="border:1px solid {g_fund['color']}55; border-top:3px solid {g_fund['color']};
                    border-radius:12px; padding:16px; background:{g_fund['color']}0D; height:100%;">
  <div style="font-size:0.85rem; color:#aaa; font-weight:600;">📊 QUỸ ĐẦU TƯ</div>
  <div style="font-size:1.0rem; font-weight:800; color:{g_fund['color']}; margin:8px 0 4px;">
    {g_fund['action']} &nbsp; {alert_badge.get(g_fund['alert'], '')}
  </div>
  <div style="font-size:0.8rem; color:#bbb; margin-bottom:10px; line-height:1.5;">{g_fund['desc']}</div>
  <div style="font-size:0.78rem; font-weight:600; color:#aaa; margin-bottom:4px;">📋 Bằng chứng:</div>
  {ev_html}
  {_confidence_bar(g_fund['confidence'], g_fund['color'])}
</div>""",
                unsafe_allow_html=True,
            )

        with col_shark:
            ev_html = "".join(f"<div style='font-size:0.78rem; color:#ccc; padding:3px 0; "
                              f"border-bottom:1px solid rgba(255,255,255,0.05);'>{e}</div>"
                              for e in g_shark["evidence"])
            st.markdown(
                f"""<div style="border:1px solid {g_shark['color']}55; border-top:3px solid {g_shark['color']};
                    border-radius:12px; padding:16px; background:{g_shark['color']}0D; height:100%;">
  <div style="font-size:0.85rem; color:#aaa; font-weight:600;">🐋 CÁ MẬP / TAY TO</div>
  <div style="font-size:1.0rem; font-weight:800; color:{g_shark['color']}; margin:8px 0 4px;">
    {g_shark['action']} &nbsp; {alert_badge.get(g_shark['alert'], '')}
  </div>
  <div style="font-size:0.8rem; color:#bbb; margin-bottom:10px; line-height:1.5;">{g_shark['desc']}</div>
  <div style="font-size:0.78rem; font-weight:600; color:#aaa; margin-bottom:4px;">📋 Bằng chứng:</div>
  {ev_html}
  {_confidence_bar(g_shark['confidence'], g_shark['color'])}
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Smart Money Chart (OBV + CMF + MFI + A/D Line) ───────────────
        with st.expander("📊 Biểu đồ Smart Money chi tiết (OBV · A/D Line · CMF · MFI · Volume)"):
            fig_sm = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                row_heights=[0.28, 0.24, 0.22, 0.26],
                vertical_spacing=0.04,
                subplot_titles=(
                    "OBV & A/D Line — dòng tiền ẩn",
                    "CMF — Chaikin Money Flow (áp lực mua/bán)",
                    "MFI — Money Flow Index (RSI dòng tiền)",
                    "Volume Mua (xanh) vs Bán (đỏ) từng phiên",
                ),
            )
            fig_sm.add_trace(go.Scatter(x=df["time"], y=df["OBV"],
                                         line=dict(color="#40C4FF", width=2),
                                         fill="tozeroy", fillcolor="rgba(64,196,255,0.07)",
                                         name="OBV"), row=1, col=1)
            fig_sm.add_trace(go.Scatter(x=df["time"], y=df["OBV"].rolling(20).mean(),
                                         line=dict(color="#FF9800", width=1.2, dash="dash"),
                                         name="OBV MA20"), row=1, col=1)
            # A/D Line
            if "AD_Line" in df.columns:
                ad_norm = df["AD_Line"] / df["AD_Line"].abs().max() * df["OBV"].abs().max()
                fig_sm.add_trace(go.Scatter(x=df["time"], y=ad_norm,
                                             line=dict(color="#E040FB", width=1.5, dash="dot"),
                                             name="A/D Line (scaled)"), row=1, col=1)
            cmf_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["CMF"].fillna(0)]
            fig_sm.add_trace(go.Bar(x=df["time"], y=df["CMF"],
                                     marker_color=cmf_colors, name="CMF"), row=2, col=1)
            fig_sm.add_hline(y=0.1,  line=dict(color="#26a69a", dash="dot", width=1), row=2, col=1)
            fig_sm.add_hline(y=-0.1, line=dict(color="#ef5350", dash="dot", width=1), row=2, col=1)
            fig_sm.add_trace(go.Scatter(x=df["time"], y=df["MFI"],
                                         line=dict(color="#E040FB", width=1.8), name="MFI"), row=3, col=1)
            for lvl, cl_ in [(80, "rgba(255,80,80,0.5)"), (20, "rgba(80,200,80,0.5)")]:
                fig_sm.add_hline(y=lvl, line=dict(color=cl_, dash="dash", width=1), row=3, col=1)
            fig_sm.add_trace(go.Bar(x=df["time"], y=df["Up_Vol"],
                                     marker_color="rgba(38,166,154,0.85)", name="Volume Mua"), row=4, col=1)
            fig_sm.add_trace(go.Bar(x=df["time"], y=-df["Down_Vol"],
                                     marker_color="rgba(239,83,80,0.85)", name="Volume Bán"), row=4, col=1)
            fig_sm.add_trace(go.Scatter(x=df["time"], y=df["Vol_MA20"],
                                         line=dict(color="#FFD740", width=1.2, dash="dot"),
                                         name="Vol MA20"), row=4, col=1)
            fig_sm.update_layout(template="plotly_dark", height=740, barmode="overlay",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                              xanchor="right", x=1, font=dict(size=9)),
                                  margin=dict(l=10, r=10, t=40, b=10))
            fig_sm.update_xaxes(showgrid=False)
            fig_sm.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_sm, use_container_width=True)

        # ── Smart Money Phase detail signals ──────────────────────────────
        with st.expander("🔎 Chi tiết tín hiệu Smart Money từng chỉ báo"):
            sm_c1, sm_c2, sm_c3, sm_c4 = st.columns(4)
            sm_c1.metric("OBV Trend", f"{sm['obv_slope']:+.3f}",
                         "↑ Tích lũy" if sm["obv_slope"] > 0 else "↓ Phân phối")
            sm_c2.metric("CMF (20)", f"{sm['cmf']:.3f}",
                         "Mua trội" if sm["cmf"] > 0 else "Bán trội")
            sm_c3.metric("MFI (14)", f"{sm['mfi']:.1f}",
                         "Quá mua 🔴" if sm["mfi"] > 80 else ("Quá bán 🟢" if sm["mfi"] < 20 else "Trung tính"))
            sm_c4.metric("Mua/Bán (10P)", f"{sm['buy_ratio']*100:.0f}%/{(1-sm['buy_ratio'])*100:.0f}%",
                         "Mua trội 🟢" if sm["buy_ratio"] > 0.55 else (
                         "Bán trội 🔴" if sm["buy_ratio"] < 0.45 else "Cân bằng"))
            sig_cols = st.columns(2)
            for i, (nm_, dt_, dot_) in enumerate(sm["signals"]):
                with sig_cols[i % 2]:
                    st.markdown(f"<div style='padding:7px 12px; margin:3px 0; border-radius:8px;"
                                f"background:rgba(255,255,255,0.04);'>"
                                f"<b>{dot_} {nm_}</b><br>"
                                f"<span style='color:#bbb;font-size:0.85rem;'>{dt_}</span></div>",
                                unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────
        # SMART MONEY ALERT
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔔 Cảnh Báo Thông Minh — Tín Hiệu Gom Hàng & Vùng Mua Hợp Lý")

        _alert_close  = latest["close"]
        _alert_rsi    = latest["RSI"]
        _alert_fib_h  = fib_levels.get("_high", 0)
        _alert_fib_l  = fib_levels.get("_low",  0)
        _alert_diff   = fib_levels.get("_diff", _alert_fib_h - _alert_fib_l)
        _alert_isdown = fib_levels.get("_is_downtrend", True)

        if _alert_isdown:
            _af382 = _alert_fib_h - 0.382 * _alert_diff
            _af50  = _alert_fib_h - 0.500 * _alert_diff
            _af618 = _alert_fib_h - 0.618 * _alert_diff
            _af650 = _alert_fib_h - 0.650 * _alert_diff
            _af786 = _alert_fib_h - 0.786 * _alert_diff
            _in_golden_pocket = (_af650 <= _alert_close <= _af618)
            _near_golden      = (_af618 * 1.03 >= _alert_close >= _af618 * 0.95)
            _in_support_zone  = (_af786 <= _alert_close <= _af382)
        else:
            _af236 = _alert_fib_l + 0.236 * _alert_diff
            _af382 = _alert_fib_l + 0.382 * _alert_diff
            _af618 = _alert_fib_l + 0.618 * _alert_diff
            _in_golden_pocket = False
            _near_golden      = (_alert_close <= _af236 * 1.03)
            _in_support_zone  = (_alert_close <= _af382)

        pos_alerts = []; neg_alerts = []; neu_alerts = []
        sm_score_val = sm_result["score"]

        if sm_score_val >= 4.5:
            pos_alerts.append(("🏦 PHÁT HIỆN GOM HÀNG CỰC MẠNH!",
                               f"Điểm SM: {sm_score_val:+.1f} — OBV tăng mạnh, CMF dương.", "#00C853", True))
        elif sm_score_val >= 2.5:
            pos_alerts.append(("📊 Dấu hiệu gom hàng từ tiền lớn",
                               f"Điểm SM: {sm_score_val:+.1f} — Tín hiệu tích cực từ OBV/CMF.", "#69F0AE", False))
        elif sm_score_val <= -4.0:
            neg_alerts.append(("🚨 CẢNH BÁO XẢ HÀNG MẠNH!",
                               f"Điểm SM: {sm_score_val:+.1f} — Tiền lớn đang phân phối/thoát.", "#FF1744", True))
        elif sm_score_val <= -2.0:
            neg_alerts.append(("📤 Dấu hiệu xả hàng",
                               f"Điểm SM: {sm_score_val:+.1f} — OBV/CMF suy yếu.", "#FF5252", False))
        else:
            neu_alerts.append(("🔍 Dòng tiền lớn chưa rõ xu hướng",
                               "Điểm SM trung tính — cần thêm xác nhận.", "#FFD740"))

        # Hidden SM alerts
        if hidden_sm.get("is_wyckoff_spring"):
            pos_alerts.append(("🌊 WYCKOFF SPRING PHÁT HIỆN!",
                               "Test đáy với volume thấp — chuẩn bị đảo chiều tăng mạnh.", "#40C4FF", True))
        if hidden_sm.get("stop_hunts", 0) >= 2:
            pos_alerts.append(("🎯 STOP HUNT x" + str(hidden_sm["stop_hunts"]),
                               "Cá mập đã giật stop nhiều lần — đang gom hàng sau hoảng loạn.", "#E040FB", True))
        if hidden_sm.get("bb_compressed"):
            neu_alerts.append(("🔄 BB COMPRESSION",
                               f"BB Width = {hidden_sm['bb_width_ratio']:.0%} so với MA — sắp có biến động lớn.", "#FFD740"))
        if hidden_sm.get("absorption_events", 0) >= 2:
            pos_alerts.append(("💧 Hấp Thụ Volume Phát Hiện",
                               f"{hidden_sm['absorption_events']} sự kiện volume cao nhưng giá không đổi — tiền lớn đang gom lệnh bán.", "#69F0AE", False))

        if _in_golden_pocket:
            pos_alerts.append(("⭐ GIÁ ĐANG TRONG GOLDEN POCKET!",
                               f"Vùng 61.8%–65.0%: {_af650:,.0f}–{_af618:,.0f} đ — Vùng mua lý tưởng nhất theo Fibonacci.", "#FFA726", True))
        elif _near_golden and _alert_isdown:
            pos_alerts.append(("📍 Giá tiếp cận Golden Pocket",
                               f"Fib 61.8% tại {_af618:,.0f} đ — Chú ý vào lệnh.", "#FFD740", False))

        if _alert_rsi < 30 and sm_score_val > 0:
            pos_alerts.append(("🟢 RSI Quá Bán + Smart Money Tích Cực",
                               f"RSI={_alert_rsi:.1f} quá bán + tiền lớn vẫn mua — Xác suất đảo chiều cao.", "#00E676", True))
        elif _alert_rsi > 75 and sm_score_val < 0:
            neg_alerts.append(("🔴 RSI Quá Mua + Smart Money Rút lui",
                               f"RSI={_alert_rsi:.1f} quá mua + dòng tiền suy yếu — Rủi ro điều chỉnh.", "#FF5252", False))

        if hidden_sm.get("stealth_accum"):
            pos_alerts.append(("🟢 Tích Lũy Bí Mật (Stealth Accumulation)",
                               "OBV tăng trong khi giá không tăng — quỹ/cá mập đang gom ẩn.", "#40C4FF", False))

        # ADX trend signal
        if pd.notna(adx_val) and adx_val > 28:
            if di_p > di_m:
                pos_alerts.append(("📈 ADX Mạnh + DI+ > DI-",
                                   f"ADX={adx_val:.1f} xác nhận xu hướng tăng — Momentum đủ mạnh để theo.", "#69F0AE", False))
            else:
                neg_alerts.append(("📉 ADX Mạnh + DI- > DI+",
                                   f"ADX={adx_val:.1f} xác nhận xu hướng giảm — Tránh mua.", "#FF9100", False))

        total_pos = len(pos_alerts); total_neg = len(neg_alerts)
        if total_pos > total_neg:
            oc, ob, obr = "#00C853", "rgba(0,200,83,0.10)", "#00C853"
            ot = ("🚨 TÍN HIỆU GOM HÀNG / VÙNG MUA HỢP LÝ — NÊN CÂN NHẮC THAM GIA"
                  if any(a[3] for a in pos_alerts if len(a) == 4)
                  else "✅ Tín hiệu tích cực — Cổ phiếu về vùng mua hợp lý")
        elif total_neg > total_pos:
            oc, ob, obr = "#FF5252", "rgba(255,82,82,0.10)", "#FF5252"
            ot = "⚠️ Tín hiệu xả hàng / rủi ro — Thận trọng, chưa phải thời điểm mua"
        else:
            oc, ob, obr = "#FFD740", "rgba(255,215,64,0.08)", "#FFD740"
            ot = "🔍 Tín hiệu chưa rõ ràng — Quan sát thêm"

        def _alert_row(icon_title, detail, color, is_critical=False):
            bg_a = "0.14" if is_critical else "0.07"
            bl   = f"4px solid {color}" if is_critical else f"2px solid {color}88"
            return (f"<div style='border-left:{bl}; background:rgba(255,255,255,{bg_a}); "
                    f"border-radius:0 8px 8px 0; padding:10px 14px; margin:5px 0;'>"
                    f"<div style='font-size:0.9rem; font-weight:{'700' if is_critical else '600'}; color:{color};'>{icon_title}</div>"
                    f"<div style='font-size:0.8rem; color:#bbb; margin-top:2px; line-height:1.4;'>{detail}</div>"
                    f"</div>")

        rows_html = ""
        for a in pos_alerts:
            rows_html += _alert_row(a[0], a[1], a[2], a[3] if len(a) == 4 else False)
        for a in neg_alerts:
            rows_html += _alert_row(a[0], a[1], a[2], a[3] if len(a) == 4 else False)
        for a in neu_alerts:
            rows_html += _alert_row(a[0], a[1], a[2], False)

        st.markdown(
            f"""<div style="border:2px solid {obr}; border-radius:14px; padding:18px 22px;
                background:{ob}; margin-bottom:16px;">
  <div style="font-size:1.05rem; font-weight:800; color:{oc}; margin-bottom:12px;">{ot}</div>
  {rows_html}
</div>""",
            unsafe_allow_html=True,
        )

        # ─────────────────────────────────────────────────────────────────
        # QUICK RECOMMENDATION
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🚦 Khuyến Nghị Nhanh (Tín Hiệu Kỹ Thuật Tổng Hợp)")

        rec_score = 0; signal_details = []
        rsi_now = latest["RSI"]
        if rsi_now < 35:   rec_score += 2; signal_details.append(("RSI", f"{rsi_now:.1f} — Quá bán 📈", "🟢"))
        elif rsi_now < 50: rec_score += 1; signal_details.append(("RSI", f"{rsi_now:.1f} — Vùng tích lũy", "🟡"))
        elif rsi_now > 70: rec_score -= 2; signal_details.append(("RSI", f"{rsi_now:.1f} — Quá mua 📉", "🔴"))
        else:              signal_details.append(("RSI", f"{rsi_now:.1f} — Trung tính", "⚪"))

        if latest["MACD"] > latest["MACD_Signal"]:
            rec_score += 2; signal_details.append(("MACD", "Cắt lên — Xu hướng tăng", "🟢"))
        else:
            rec_score -= 1; signal_details.append(("MACD", "Cắt xuống — Xu hướng giảm", "🔴"))

        mcdx_now = latest["MCDX"]
        if mcdx_now > 0.2:   rec_score += 2; signal_details.append(("MCDX", f"{mcdx_now:.3f} — Động lượng mạnh", "🟢"))
        elif mcdx_now > 0:   rec_score += 1; signal_details.append(("MCDX", f"{mcdx_now:.3f} — Động lượng nhẹ", "🟡"))
        else:                 rec_score -= 1; signal_details.append(("MCDX", f"{mcdx_now:.3f} — Yếu dần", "🔴"))

        adx_now = latest.get("ADX", 0) or 0
        dip = latest.get("DI_Plus", 0) or 0; dim = latest.get("DI_Minus", 0) or 0
        if adx_now > 25:
            if dip > dim:
                rec_score += 2; signal_details.append(("ADX", f"{adx_now:.1f} — Trend tăng mạnh (DI+>{dip:.0f})", "🟢"))
            else:
                rec_score -= 2; signal_details.append(("ADX", f"{adx_now:.1f} — Trend giảm mạnh (DI->{dim:.0f})", "🔴"))
        else:
            signal_details.append(("ADX", f"{adx_now:.1f} — Sideway, không xu hướng", "⚪"))

        ema5_n = latest.get("EMA5", 0) or 0; ema20_n = latest.get("EMA20", 0) or 0; ema50_n = latest.get("EMA50", 0) or 0
        if ema5_n > ema20_n > ema50_n:
            rec_score += 2; signal_details.append(("EMA", "EMA5>EMA20>EMA50 — Bull Alignment ✅", "🟢"))
        elif ema5_n < ema20_n < ema50_n:
            rec_score -= 2; signal_details.append(("EMA", "EMA5<EMA20<EMA50 — Bear Alignment ❌", "🔴"))
        elif ema5_n > ema20_n:
            rec_score += 1; signal_details.append(("EMA", "EMA5>EMA20 — Ngắn hạn tăng", "🟡"))
        else:
            rec_score -= 1; signal_details.append(("EMA", "EMA5<EMA20 — Ngắn hạn giảm", "🟠"))

        span_a_last = latest.get("SpanA", None); span_b_last = latest.get("SpanB", None)
        if span_a_last and span_b_last:
            cloud_top = max(span_a_last, span_b_last); cloud_bot = min(span_a_last, span_b_last)
            if latest["close"] > cloud_top:
                rec_score += 2; signal_details.append(("Ichimoku", "Giá trên mây — Bullish ☁️✅", "🟢"))
            elif latest["close"] < cloud_bot:
                rec_score -= 2; signal_details.append(("Ichimoku", "Giá dưới mây — Bearish ☁️❌", "🔴"))
            else:
                signal_details.append(("Ichimoku", "Giá trong mây — Trung tính", "🟡"))
        if latest.get("Tenkan", 0) > latest.get("Kijun", 0):
            rec_score += 1; signal_details.append(("Tenkan/Kijun", "Tenkan > Kijun — Tín hiệu mua", "🟢"))
        else:
            rec_score -= 1; signal_details.append(("Tenkan/Kijun", "Tenkan < Kijun — Tín hiệu bán", "🔴"))

        _is_down = fib_levels.get("_is_downtrend", True)
        _fib_h   = fib_levels.get("_high", 0); _fib_l = fib_levels.get("_low", 0)
        _diff    = fib_levels.get("_diff", _fib_h - _fib_l)
        close_now = latest["close"]; fib_margin = _diff * 0.03
        if _is_down:
            fib_618 = _fib_h - 0.618*_diff; fib_650 = _fib_h - 0.650*_diff
            fib_50  = _fib_h - 0.500*_diff; fib_786 = _fib_h - 0.786*_diff; fib_382 = _fib_h - 0.382*_diff
            if fib_650 <= close_now <= fib_618:
                rec_score += 3; signal_details.append(("Fibonacci", f"Golden Pocket ({fib_650:,.0f}–{fib_618:,.0f}) ⭐ — Vùng mua lý tưởng!", "🟢"))
            elif abs(close_now - fib_618) <= fib_margin:
                rec_score += 2; signal_details.append(("Fibonacci", f"Tại Fib 61.8% ({fib_618:,.0f}) ✨ — Hỗ trợ vàng", "🟢"))
            elif abs(close_now - fib_50) <= fib_margin:
                rec_score += 1; signal_details.append(("Fibonacci", f"Tại Fib 50.0% ({fib_50:,.0f}) — Hỗ trợ tâm lý", "🟡"))
            elif close_now < fib_786:
                rec_score -= 1; signal_details.append(("Fibonacci", f"Phá hỗ trợ 78.6% ({fib_786:,.0f}) — Rủi ro cao", "🔴"))
            elif close_now < fib_618:
                rec_score += 1; signal_details.append(("Fibonacci", "Dưới Fib 61.8% — Vùng hỗ trợ sâu", "🟡"))
            elif close_now > fib_382:
                rec_score -= 1; signal_details.append(("Fibonacci", "Trên 38.2% — Gần kháng cự", "🔴"))
        else:
            fib_618 = _fib_l + 0.618*_diff; fib_382 = _fib_l + 0.382*_diff; fib_50 = _fib_l + 0.500*_diff
            if close_now > fib_618:
                rec_score += 2; signal_details.append(("Fibonacci", f"Vượt Fib 61.8% ({fib_618:,.0f}) ✨ — Breakout mạnh", "🟢"))
            elif close_now > fib_50:
                rec_score += 1; signal_details.append(("Fibonacci", f"Vượt Fib 50.0% ({fib_50:,.0f}) — Phục hồi tốt", "🟡"))
            elif close_now < fib_382:
                rec_score -= 1; signal_details.append(("Fibonacci", f"Dưới Fib 38.2% ({fib_382:,.0f}) — Phục hồi yếu", "🔴"))

        # Hidden SM bonus
        if hidden_sm.get("is_wyckoff_spring"):      rec_score += 2
        if hidden_sm.get("stop_hunts", 0) >= 2:     rec_score += 1
        if hidden_sm.get("absorption_events", 0) >= 2: rec_score += 1

        max_score_r = 20; pct_score_r = rec_score / max_score_r
        if pct_score_r >= 0.45:
            rl, rc, rb, rbr, rd = "✅ MUA", "#00C853", "rgba(0,200,83,0.12)", "#00C853", "Tín hiệu kỹ thuật thuận lợi — Cân nhắc mở vị thế mua."
        elif pct_score_r >= 0.15:
            rl, rc, rb, rbr, rd = "⏳ THEO DÕI", "#FFC107", "rgba(255,193,7,0.12)", "#FFC107", "Tín hiệu chưa rõ — Chờ xác nhận thêm."
        else:
            rl, rc, rb, rbr, rd = "🚫 TRÁNH / BÁN", "#FF5252", "rgba(255,82,82,0.12)", "#FF5252", "Tín hiệu yếu/tiêu cực — Không nên mua."

        stars = "⭐" * max(1, min(5, round((rec_score + max_score_r) / (2 * max_score_r) * 5)))
        st.markdown(
            f"""<div style="border:2px solid {rbr}; border-radius:12px; padding:20px 28px;
                background:{rb}; margin-bottom:16px;">
  <div style="font-size:2rem; font-weight:700; color:{rc}; letter-spacing:1px;">{rl}</div>
  <div style="font-size:1.05rem; color:#ccc; margin-top:6px;">{rd}</div>
  <div style="margin-top:10px; font-size:1rem; color:#aaa;">
    Điểm tổng hợp: <b style="color:{rc};">{rec_score}/{max_score_r}</b> &nbsp;|&nbsp; {stars}
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        with st.expander("📊 Chi tiết tín hiệu từng chỉ báo"):
            sig_col1, sig_col2 = st.columns(2)
            for i, (ind, det, dot) in enumerate(signal_details):
                with (sig_col1 if i % 2 == 0 else sig_col2):
                    st.markdown(
                        f"<div style='padding:8px 12px; margin:4px 0; border-radius:8px;"
                        f"background:rgba(255,255,255,0.04);'>"
                        f"<b>{dot} {ind}</b><br>"
                        f"<span style='color:#ccc; font-size:0.88rem;'>{det}</span></div>",
                        unsafe_allow_html=True,
                    )

        # ── Vùng mua / SL / Target ────────────────────────────────────────
        _is_down2 = fib_levels.get("_is_downtrend", True)
        _fib_h2   = fib_levels.get("_high", 0); _fib_l2 = fib_levels.get("_low", 0)
        _diff2    = fib_levels.get("_diff", _fib_h2 - _fib_l2); close_p = latest["close"]
        rr_ratio  = 0.0

        if _is_down2:
            _f236 = _fib_h2 - 0.236*_diff2; _f382 = _fib_h2 - 0.382*_diff2
            _f50  = _fib_h2 - 0.500*_diff2; _f618 = _fib_h2 - 0.618*_diff2
            _f650 = _fib_h2 - 0.650*_diff2; _f786 = _fib_h2 - 0.786*_diff2
            if close_p < _f786:
                buy_zone_low = buy_zone_high = None; zone_status = "BROKEN"
                zone_label = "⛔ Giá đã phá hỗ trợ 78.6% — Tránh mua"; zone_color = "#FF1744"
            elif _f786 <= close_p <= _f650:
                buy_zone_low = _f786*0.99; buy_zone_high = _f650; zone_status = "DEEP_SUPPORT"
                zone_label = "⚠️ Vùng hỗ trợ sâu (65%–78.6%) — Mua thận trọng, SL chặt"; zone_color = "#FF9100"
            elif _f650 <= close_p <= _f618:
                buy_zone_low = _f650; buy_zone_high = _f618; zone_status = "GOLDEN_POCKET"
                zone_label = "✅ GIÁ ĐANG TRONG GOLDEN POCKET — Vùng mua lý tưởng!"; zone_color = "#FFA726"
            elif _f618 <= close_p <= _f382:
                buy_zone_low = _f650; buy_zone_high = _f618; zone_status = "APPROACHING"
                zone_label = f"📍 Vùng mua gợi ý: {_f650:,.0f}–{_f618:,.0f} đ — Chờ về thêm"; zone_color = "#FFD740"
            else:
                buy_zone_low = _f618; buy_zone_high = _f50; zone_status = "WAIT"
                zone_label = "⏳ Chưa đến vùng mua — Chờ điều chỉnh về 50%–61.8%"; zone_color = "#90CAF9"
            stop_loss = min(_f786 * 0.980, latest.get("BB_Low", _f786 * 0.98) or _f786 * 0.98)
            target1 = _f382; target2 = _f236; fib_dir_lbl = "Downtrend (Đỉnh → Đáy)"
        else:
            _f236 = _fib_l2 + 0.236*_diff2; _f382 = _fib_l2 + 0.382*_diff2
            _f50  = _fib_l2 + 0.500*_diff2; _f618 = _fib_l2 + 0.618*_diff2
            _f650 = _fib_l2 + 0.650*_diff2; _f786 = _fib_l2 + 0.786*_diff2
            if close_p <= _f236:
                buy_zone_low = _fib_l2*0.99; buy_zone_high = _f236; zone_status = "NEAR_BASE"
                zone_label = "✅ Giá gần đáy — Vùng mua tích lũy tốt nhất"; zone_color = "#00C853"
            elif _f236 < close_p <= _f382:
                buy_zone_low = _fib_l2*0.99; buy_zone_high = _f236; zone_status = "EARLY_RECOVERY"
                zone_label = "📍 Phục hồi sớm — Vùng mua gần đáy còn hợp lý"; zone_color = "#69F0AE"
            elif _f382 < close_p <= _f618:
                buy_zone_low = _f236; buy_zone_high = _f382; zone_status = "MID_RECOVERY"
                zone_label = f"⚠️ Đã phục hồi 38–61% — Chờ pullback về {_f236:,.0f}–{_f382:,.0f}"; zone_color = "#FFD740"
            else:
                buy_zone_low = _f382; buy_zone_high = _f50; zone_status = "EXTENDED"
                zone_label = "⏳ Giá đã phục hồi nhiều — Không nên mua đuổi"; zone_color = "#FF9100"
            stop_loss = min(_fib_l2 * 0.980, latest.get("BB_Low", _fib_l2 * 0.98) or _fib_l2 * 0.98)
            target1 = _f618; target2 = _fib_h2; fib_dir_lbl = "Uptrend Recovery (Đáy → Đỉnh)"

        def _delta(price):
            if close_p > 0 and price is not None:
                return f"{((price - close_p) / close_p * 100):+.1f}%"
            return ""

        bz_low_str  = f"{buy_zone_low:,.0f}"  if buy_zone_low  is not None else "—"
        bz_high_str = f"{buy_zone_high:,.0f}" if buy_zone_high is not None else "—"
        if buy_zone_high and stop_loss and target1 and buy_zone_high > stop_loss:
            rr_risk = buy_zone_high - stop_loss; rr_reward = target1 - buy_zone_high
            rr_ratio = rr_reward / rr_risk if rr_risk > 0 else 0
        rr_label = f"R/R = 1:{rr_ratio:.1f}"
        rr_color  = "#00C853" if rr_ratio >= 2 else ("#FFD740" if rr_ratio >= 1 else "#FF5252")

        st.markdown(
            f"""<div style="background:rgba(255,255,255,0.04); border-radius:14px; padding:18px 22px;
                border:1px solid rgba(255,255,255,0.10); margin-bottom:12px;">
  <div style="font-size:0.83rem; color:#aaa; margin-bottom:4px;">
    📐 Fibonacci {fib_dir_lbl} &nbsp;·&nbsp;
    Đỉnh <b style="color:#EF5350;">{_fib_h2:,.0f}</b> đ &nbsp;·&nbsp;
    Đáy <b style="color:#26a69a;">{_fib_l2:,.0f}</b> đ &nbsp;·&nbsp;
    Hiện tại <b style="color:#FFD740;">{close_p:,.0f}</b> đ
  </div>
  <div style="font-size:0.98rem; font-weight:700; color:{zone_color}; padding:8px 12px;
              background:{zone_color}22; border-radius:8px; margin:10px 0 14px 0;
              border-left:4px solid {zone_color};">{zone_label}</div>
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr 1fr; gap:10px;">
    <div style="background:rgba(0,200,83,0.12); border:1px solid #00C853; border-radius:10px; padding:12px;">
      <div style="font-size:0.72rem; color:#aaa; margin-bottom:4px;">📥 Vùng Mua Gợi Ý</div>
      <div style="font-size:1.0rem; font-weight:700; color:#00C853;">{bz_low_str} – {bz_high_str} đ</div>
    </div>
    <div style="background:rgba(255,82,82,0.10); border:1px solid #FF5252; border-radius:10px; padding:12px;">
      <div style="font-size:0.72rem; color:#aaa; margin-bottom:4px;">🛑 Stop Loss</div>
      <div style="font-size:1.0rem; font-weight:700; color:#FF5252;">{stop_loss:,.0f} đ</div>
      <div style="font-size:0.72rem; color:#888; margin-top:4px;">{_delta(stop_loss)}</div>
    </div>
    <div style="background:rgba(33,150,243,0.10); border:1px solid #2196F3; border-radius:10px; padding:12px;">
      <div style="font-size:0.72rem; color:#aaa; margin-bottom:4px;">🎯 Target 1</div>
      <div style="font-size:1.0rem; font-weight:700; color:#2196F3;">{target1:,.0f} đ</div>
      <div style="font-size:0.72rem; color:#888; margin-top:4px;">{_delta(target1)}</div>
    </div>
    <div style="background:rgba(224,64,251,0.10); border:1px solid #E040FB; border-radius:10px; padding:12px;">
      <div style="font-size:0.72rem; color:#aaa; margin-bottom:4px;">🎯 Target 2</div>
      <div style="font-size:1.0rem; font-weight:700; color:#E040FB;">{target2:,.0f} đ</div>
      <div style="font-size:0.72rem; color:#888; margin-top:4px;">{_delta(target2)}</div>
    </div>
    <div style="background:rgba(255,255,255,0.05); border:1px solid {rr_color}; border-radius:10px; padding:12px;">
      <div style="font-size:0.72rem; color:#aaa; margin-bottom:4px;">⚖️ R/R Ratio</div>
      <div style="font-size:1.0rem; font-weight:700; color:{rr_color};">{rr_label}</div>
      <div style="font-size:0.72rem; color:#888; margin-top:4px;">
        {'✅ Tốt' if rr_ratio >= 2 else ('⚠️ Chấp nhận' if rr_ratio >= 1 else '❌ Rủi ro cao')}
      </div>
    </div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        # ─────────────────────────────────────────────────────────────────
        # AI ANALYSIS
        # ─────────────────────────────────────────────────────────────────
        if not api_key:
            st.warning("💡 Nhập API Key ở thanh bên trái để nhận phân tích từ Trợ lý AI.")
        else:
            st.markdown("---")
            provider_name = "Claude (Anthropic)" if "Claude" in ai_provider else "Gemini (Google)"
            st.subheader(f"🤖 Báo Cáo Phân Tích & Khuyến Nghị — {provider_name}")

            ai_data = df.tail(10)[show_cols].round(2).to_string(index=False)
            fib_str = "\n".join([f"  • Fib {k}: {v:,.0f} đ"
                                  for k, v in fib_levels.items()
                                  if not k.startswith("_") and isinstance(v, (int, float))])
            above_cloud = (
                latest["close"] > max(latest.get("SpanA") or 0, latest.get("SpanB") or 0)
                if pd.notna(latest.get("SpanA")) and pd.notna(latest.get("SpanB")) else None
            )
            ichi_status = ("trên mây (Bullish)" if above_cloud
                           else ("dưới mây (Bearish)" if above_cloud is False else "trong mây (trung tính)"))
            macd_signal = ("cắt lên (hội tụ tăng giá)" if latest["MACD"] > latest["MACD_Signal"]
                           else "cắt xuống (phân kỳ giảm giá)")
            adx_desc = (f"ADX={adx_val:.1f} — {'Xu hướng mạnh' if adx_val > 25 else 'Sideway'}, "
                        f"DI+={di_p:.1f} vs DI-={di_m:.1f} ({'tăng' if di_p > di_m else 'giảm'})")
            ema_desc = (f"EMA5={ema5_v:,.0f} / EMA20={ema20_v:,.0f} / EMA50={ema50_n:,.0f} — "
                        f"{'Bull Alignment' if ema5_v > ema20_v > ema50_n else ('Bear Alignment' if ema5_v < ema20_v < ema50_n else 'Hỗn hợp')}")
            hidden_desc = "\n".join([f"  • {s[0]}: {s[1]}" for s in hsigs]) if hsigs else "  • Không phát hiện tín hiệu ẩn đặc biệt"

            prompt = f"""
Bạn là hệ thống AI định lượng cao cấp chuyên phân tích thị trường chứng khoán Việt Nam.

Dữ liệu 10 phiên gần nhất của mã **{ticker}**:
```
{ai_data}
```

Fibonacci Retracement (60 phiên):
{fib_str}

Tóm tắt tín hiệu hiện tại:
- RSI: {rsi_val:.1f} → {rsi_note}
- MACD: {macd_signal}
- ADX: {adx_desc}
- EMA: {ema_desc}
- MCDX: {mcdx_val:.4f} → {"Tăng" if mcdx_val > 0 else "Giảm"}
- Ichimoku: Giá đang {ichi_status}
- Trend Strength Meter: {ts['score']:.0f}/100 — {ts['label']}
- Smart Money Phase: {sm_result['phase']} (score: {sm_result['score']:+.1f})

Tín hiệu ẩn Smart Money phát hiện:
{hidden_desc}

**Yêu cầu phân tích (viết bằng Tiếng Việt, rõ ràng và quyết đoán):**

1. **Market Regime**: Thị trường đang Trending hay Ranging? ADX và EMA alignment nói gì?
2. **RSI + MACD + MCDX**: Phân tích động lượng tổng hợp. Có phân kỳ không?
3. **ADX Analysis**: Sức mạnh xu hướng hiện tại — có nên theo xu hướng không?
4. **EMA Cross**: Tín hiệu EMA 5/20/50 — ngắn/trung/dài hạn đang ra sao?
5. **Ichimoku**: Vị trí giá vs mây, Tenkan/Kijun, Chikou Span.
6. **Fibonacci**: Giá đang ở mức nào? Vùng hỗ trợ/kháng cự tiếp theo?
7. **Smart Money Ẩn**: Đánh giá các tín hiệu ẩn — Market Maker đang làm gì, Quỹ có đang tích lũy bí mật không, Cá mập có dấu hiệu gom/xả không?
8. **Kết luận & Chiến lược**:
   - ✅ Khuyến nghị: **Mua / Bán / Chờ**
   - 📥 Vùng mua hợp lý (điểm vào)
   - 🛑 Stop Loss
   - 🎯 Target 1 & Target 2
   - ⚠️ Mức độ rủi ro: **Thấp / Trung bình / Cao**
   - 📝 Lý do tổng hợp ngắn gọn
"""
            with st.spinner("⏳ AI đang phân tích tổng hợp..."):
                try:
                    if "Claude" in ai_provider:
                        result = call_claude(api_key, prompt)
                    else:
                        result = call_gemini(api_key, prompt)
                    st.markdown(result)
                except Exception as e:
                    err_str = str(e)
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        st.error("❌ Hết quota Gemini API (429)")
                        st.warning("Chờ ~1 phút hoặc chuyển sang Claude.")
                    elif "401" in err_str or "UNAUTHENTICATED" in err_str:
                        st.error("❌ API Key không hợp lệ.")
                    else:
                        st.error(f"❌ Lỗi gọi AI: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET SCANNER
# ═════════════════════════════════════════════════════════════════════════════
with _tab2:
    st.subheader("🔍 Quét Toàn Thị Trường — Phát Hiện Hành Vi Tiền Lớn")

    with st.spinner("🔄 Đang tải danh sách cổ phiếu..."):
        _auto_universe, _source_note = build_scan_universe()

    st.caption(
        f"📡 Nguồn: **{_source_note}** · "
        f"Lọc: giá > 10,000 đ · KLGD > 200,000 CP/ngày · "
        f"**{len(_auto_universe)} mã** sẽ được quét"
    )
    _vn30_in  = [t for t in VN30_TICKERS if t in _auto_universe]
    _extra_in = [t for t in _auto_universe if t not in VN30_TICKERS]
    _b1, _b2, _b3 = st.columns(3)
    _b1.metric("🏆 VN30", len(_vn30_in), "mã")
    _b2.metric("📈 Thêm (giá>10k+KL>200k)", len(_extra_in), "mã")
    _b3.metric("🔢 Tổng mã quét", len(_auto_universe), "mã")

    with st.expander("⚙️ Xem & chỉnh sửa danh sách quét"):
        custom_input = st.text_area(
            "✏️ Chỉnh sửa danh sách (dấu phẩy/xuống dòng):",
            value=", ".join(_auto_universe), height=130,
        )
        _custom_parsed = [t.strip().upper() for t in custom_input.replace("\n", ",").split(",") if t.strip()]
        scan_list = sorted(set(_custom_parsed)) if _custom_parsed else _auto_universe
        st.info(f"📋 Sẽ quét **{len(scan_list)} mã** sau chỉnh sửa.")

    if "scan_list" not in dir():
        scan_list = _auto_universe

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_phase = st.multiselect("Lọc theo pha Smart Money:",
            ["GOM HÀNG","ĐẨY GIÁ (MARKUP)","TRUNG TÍNH / QUAN SÁT",
             "XẢ HÀNG (DISTRIBUTION)","ĐÈ GIÁ (MARKDOWN)"], default=[])
    with col_f2:
        filter_rec = st.multiselect("Lọc theo khuyến nghị:",
            ["✅ MUA","⏳ THEO DÕI","🚫 TRÁNH/BÁN"], default=[])
    with col_f3:
        filter_fib = st.multiselect("Lọc theo vùng Fibonacci:",
            ["⭐ Golden Pocket","✅ Gần đáy (tích lũy)","⚠️ Hỗ trợ sâu (65-78.6%)",
             "📍 Vùng hỗ trợ (38-61.8%)","📍 Phục hồi sớm"], default=[])

    if st.button("🚀 Chạy Quét Thị Trường", type="primary", key="scan_btn"):
        results = []; skipped_criteria = 0
        progress_bar = st.progress(0, text="Đang quét thị trường...")
        status_text  = st.empty()

        for i, sym in enumerate(scan_list):
            status_text.markdown(f"⏳ Đang phân tích **{sym}** ({i+1}/{len(scan_list)})...")
            res = scan_single_stock(sym)
            if res:
                is_vn30   = sym in VN30_TICKERS
                price_ok  = res["close"] > 10000
                avg_vol_ok = res.get("avg_vol", 0) > 200000
                if is_vn30 or (price_ok and avg_vol_ok):
                    results.append(res)
                else:
                    skipped_criteria += 1
            progress_bar.progress((i+1)/len(scan_list), text=f"Đã quét {i+1}/{len(scan_list)} mã")

        progress_bar.empty(); status_text.empty()

        if not results:
            st.error("❌ Không quét được dữ liệu. Kiểm tra kết nối mạng.")
            st.stop()

        _q1, _q2, _q3 = st.columns(3)
        _q1.success(f"✅ Quét thành công **{len(results)} mã**")
        _q2.info(f"🔍 Đã quét: **{len(scan_list)} mã**")
        if skipped_criteria:
            _q3.warning(f"⚠️ Loại: **{skipped_criteria} mã** (giá/KL không đủ)")

        results.sort(key=lambda x: (_phase_order(x["sm_phase"]), -x["score"]))

        filtered = results
        if filter_phase: filtered = [r for r in filtered if r["sm_phase"] in filter_phase]
        if filter_rec:   filtered = [r for r in filtered if r["recommendation"] in filter_rec]
        if filter_fib:   filtered = [r for r in filtered if any(fz in r["fib_zone"] for fz in filter_fib)]

        st.markdown("---")
        phase_counts = defaultdict(int)
        for r in results: phase_counts[r["sm_phase"]] += 1

        sum_cols = st.columns(5)
        phase_defs = [
            ("GOM HÀNG","🏦","#00C853"),("ĐẨY GIÁ (MARKUP)","🚀","#40C4FF"),
            ("TRUNG TÍNH / QUAN SÁT","🔍","#FFD740"),
            ("XẢ HÀNG (DISTRIBUTION)","📤","#FF6D00"),("ĐÈ GIÁ (MARKDOWN)","📉","#FF1744"),
        ]
        for idx, (ph, ic, co) in enumerate(phase_defs):
            cnt = phase_counts.get(ph, 0)
            sum_cols[idx].markdown(
                f"<div style='text-align:center; padding:12px 6px; border-radius:10px;"
                f"background:{co}18; border:1px solid {co}55;'>"
                f"<div style='font-size:1.5rem;'>{ic}</div>"
                f"<div style='font-size:1.6rem; font-weight:800; color:{co};'>{cnt}</div>"
                f"<div style='font-size:0.75rem; color:#aaa;'>{ph}</div></div>",
                unsafe_allow_html=True,
            )

        buy_cnt   = sum(1 for r in results if r["recommendation"] == "✅ MUA")
        watch_cnt = sum(1 for r in results if r["recommendation"] == "⏳ THEO DÕI")
        avoid_cnt = sum(1 for r in results if r["recommendation"] == "🚫 TRÁNH/BÁN")
        gp_cnt    = sum(1 for r in results if "Golden Pocket" in r["fib_zone"] or "Gần đáy" in r["fib_zone"])

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Khuyến nghị MUA",   buy_cnt,   f"/ {len(results)}")
        m2.metric("⏳ Theo dõi",           watch_cnt, f"/ {len(results)}")
        m3.metric("🚫 Tránh/Bán",         avoid_cnt, f"/ {len(results)}")
        m4.metric("⭐ Vùng Fib hấp dẫn",  gp_cnt,    f"/ {len(results)}")

        st.markdown("---")
        st.markdown(f"### 📋 Kết quả quét — {len(filtered)}/{len(results)} mã (sau bộ lọc)")

        PHASE_META = {
            "GOM HÀNG":              ("🏦","#00C853","rgba(0,200,83,0.10)","Tiền lớn đang bí mật tích lũy. Cơ hội mua sớm."),
            "ĐẨY GIÁ (MARKUP)":     ("🚀","#40C4FF","rgba(64,196,255,0.10)","Cá mập đang markup. Có thể theo đà nhưng quản lý rủi ro chặt."),
            "TRUNG TÍNH / QUAN SÁT":("🔍","#FFD740","rgba(255,215,64,0.08)","Chưa có tín hiệu rõ. Quan sát thêm."),
            "XẢ HÀNG (DISTRIBUTION)":("📤","#FF6D00","rgba(255,109,0,0.10)","Tiền lớn đang phân phối. Cẩn thận."),
            "ĐÈ GIÁ (MARKDOWN)":    ("📉","#FF1744","rgba(255,23,68,0.10)","Tay to xả mạnh. Không nên mua."),
        }
        by_phase = defaultdict(list)
        for r in filtered: by_phase[r["sm_phase"]].append(r)

        for ph_name, (ph_icon, ph_color, ph_bg, ph_desc) in PHASE_META.items():
            group = by_phase.get(ph_name, [])
            if not group: continue
            with st.expander(f"{ph_icon} {ph_name} — {len(group)} mã", expanded=(ph_name == "GOM HÀNG")):
                st.markdown(
                    f"<div style='background:{ph_bg}; border-left:4px solid {ph_color}; "
                    f"padding:10px 16px; border-radius:0 8px 8px 0; margin-bottom:12px; "
                    f"color:#ddd; font-size:0.9rem;'>{ph_desc}</div>",
                    unsafe_allow_html=True,
                )
                rows = []
                for r in group:
                    rows.append({
                        "Mã": r["symbol"], "Giá (đ)": f"{r['close']:,.0f}", "% 1P": f"{r['pct_chg']:+.2f}%",
                        "KL TB": f"{r['avg_vol']:,.0f}", "RSI": f"{r['rsi']:.1f}",
                        "ADX": f"{r.get('adx', 0):.1f}",
                        "MACD": "📈" if r["macd_bull"] else "📉",
                        "EMA": "🟢" if r.get("ema_bull") else "🔴",
                        "MCDX": f"{r['mcdx']:.3f}", "CMF": f"{r['cmf']:.3f}", "MFI": f"{r['mfi']:.1f}",
                        "Mua/Bán": f"{r['buy_ratio']*100:.0f}%/{(1-r['buy_ratio'])*100:.0f}%",
                        "Điểm SM": f"{r['sm_score']:+.1f}", "Vùng Fib": r["fib_zone"],
                        "KN": r["recommendation"],
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                top_picks = [r for r in group if r["recommendation"] == "✅ MUA"]
                if top_picks and ph_name in ("GOM HÀNG", "ĐẨY GIÁ (MARKUP)"):
                    st.markdown(f"#### ⭐ Cổ phiếu nổi bật ({len(top_picks)} mã)")
                    card_cols = st.columns(min(len(top_picks), 4))
                    for ci, r in enumerate(top_picks[:8]):
                        with card_cols[ci % 4]:
                            chg_color = "#26a69a" if r["pct_chg"] >= 0 else "#ef5350"
                            fib_hl = "#FFA726" if "Golden Pocket" in r["fib_zone"] else (
                                     "#00C853" if "Gần đáy" in r["fib_zone"] else ph_color)
                            st.markdown(
                                f"""<div style="border:1px solid {ph_color}55; border-top:3px solid {ph_color};
                                    border-radius:10px; padding:14px 12px; background:{ph_bg}; margin-bottom:8px;">
  <div style="font-size:1.3rem; font-weight:800; color:{ph_color};">{r['symbol']}</div>
  <div style="font-size:1.05rem; font-weight:700; color:#fff; margin:4px 0;">
    {r['close']:,.0f} đ &nbsp;
    <span style="font-size:0.85rem; color:{chg_color};">{r['pct_chg']:+.2f}%</span>
  </div>
  <div style="font-size:0.78rem; color:#bbb; line-height:1.7;">
    RSI: <b style="color:#FF9800;">{r['rsi']:.1f}</b> &nbsp;·&nbsp;
    ADX: <b style="color:#E040FB;">{r.get('adx',0):.1f}</b><br>
    MACD: {'📈' if r['macd_bull'] else '📉'} &nbsp;·&nbsp;
    EMA: {'🟢 Bull' if r.get('ema_bull') else '🔴 Bear'}<br>
    CMF: {r['cmf']:.3f} &nbsp;·&nbsp; MFI: {r['mfi']:.1f}<br>
    <span style="color:{fib_hl};">{r['fib_zone']}</span>
  </div>
  <div style="margin-top:8px; font-size:0.82rem; font-weight:700; color:#00C853;">
    Điểm: {r['score']}/{r['max_score']} &nbsp;|&nbsp; {r['recommendation']}
  </div>
</div>""",
                                unsafe_allow_html=True,
                            )

        # Vùng mua hợp lý
        st.markdown("---")
        vung_mua = [r for r in results if any(
            kw in r["fib_zone"] for kw in ["Golden Pocket", "Gần đáy", "Hỗ trợ sâu", "Phục hồi sớm"]
        )]
        vung_mua.sort(key=lambda x: -x["score"])
        st.markdown(f"### 🛒 Cổ Phiếu Về Vùng Mua Hợp Lý — {len(vung_mua)} mã")
        if vung_mua:
            buy_rows = [{"Mã": r["symbol"], "Giá (đ)": f"{r['close']:,.0f}",
                         "% 1P": f"{r['pct_chg']:+.2f}%", "KL TB": f"{r['avg_vol']:,.0f}",
                         "RSI": f"{r['rsi']:.1f}", "ADX": f"{r.get('adx',0):.1f}",
                         "Smart Money": f"{r['sm_icon']} {r['sm_phase']}",
                         "Điểm SM": f"{r['sm_score']:+.1f}", "Vùng Fib": r["fib_zone"],
                         "Điểm KT": f"{r['score']}/{r['max_score']}", "KN": r["recommendation"]}
                        for r in vung_mua]
            st.dataframe(pd.DataFrame(buy_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Hiện chưa có mã nào về vùng mua hợp lý trong danh sách quét.")

        # Cảnh báo xả hàng
        danger_list = [r for r in results if r["sm_phase"] in ("XẢ HÀNG (DISTRIBUTION)", "ĐÈ GIÁ (MARKDOWN)")]
        danger_list.sort(key=lambda x: x["sm_score"])
        if danger_list:
            st.markdown("---")
            st.markdown(f"### 🚨 Cảnh Báo Xả Hàng / Đè Giá — {len(danger_list)} mã")
            danger_rows = [{"Mã": r["symbol"], "Giá (đ)": f"{r['close']:,.0f}",
                            "% 1P": f"{r['pct_chg']:+.2f}%", "KL TB": f"{r['avg_vol']:,.0f}",
                            "RSI": f"{r['rsi']:.1f}", "ADX": f"{r.get('adx',0):.1f}",
                            "Pha": f"{r['sm_icon']} {r['sm_phase']}",
                            "Điểm SM": f"{r['sm_score']:+.1f}", "CMF": f"{r['cmf']:.3f}",
                            "MFI": f"{r['mfi']:.1f}", "KN": r["recommendation"]}
                           for r in danger_list]
            st.dataframe(pd.DataFrame(danger_rows), hide_index=True, use_container_width=True)

    else:
        st.info(
            "💡 Nhấn **Chạy Quét Thị Trường** để bắt đầu.\n\n"
            "Hệ thống phân tích từng mã và phân loại theo hành vi tiền lớn:\n"
            "- 🏦 **Gom hàng** — Tiền lớn đang tích lũy bí mật\n"
            "- 🚀 **Đẩy giá** — Cá mập đang markup sau gom xong\n"
            "- 📤 **Xả hàng** — Phân phối đỉnh cho nhà đầu tư nhỏ lẻ\n"
            "- 📉 **Đè giá** — Tay to đang bán tháo mạnh\n"
            "- 🛒 **Vùng mua hợp lý** — Giá về Fibonacci hỗ trợ\n"
            "- ⭐ **Tích hợp ADX + EMA + Wyckoff** để phát hiện tín hiệu ẩn"
        )
