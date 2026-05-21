import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests

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
    "- Ichimoku Cloud\n"
    "- MCDX (Momentum Composite)\n"
    "- Fibonacci Retracement\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Hệ Thống Phân Tích Kỹ Thuật & Trợ Lý Giao Dịch AI")
st.markdown(
    "Tích hợp **RSI · MACD · Bollinger Bands · Ichimoku · MCDX · Fibonacci** "
    "— Được phân tích bởi **Claude** hoặc **Gemini**."
)

ticker = st.text_input(
    "🔍 Nhập mã chứng khoán (Cổ phiếu hoặc VN30F1M):",
    "VN30F1M",
).upper().strip()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_clean_stock_data(symbol: str) -> pd.DataFrame | None:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)  # Extra room for Ichimoku (needs 52 bars)

    # ── Luồng 1: Phái sinh (VN30F1M, VN30F2M, ...) qua DNSE API ─────────────
    if symbol.startswith("VN30F"):
        try:
            url = (
                f"https://services.entrade.com.vn/chart-api/v2/ohlcs/derivative"
                f"?from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
                f"&symbol={symbol}&resolution=1D"
            )
            resp = requests.get(url, timeout=10).json()
            if "t" in resp and len(resp["t"]) > 0:
                df = pd.DataFrame(
                    {
                        "time": pd.to_datetime(resp["t"], unit="s"),
                        "open": resp["o"],
                        "high": resp["h"],
                        "low": resp["l"],
                        "close": resp["c"],
                        "volume": resp["v"],
                    }
                )
                df["time"] = df["time"].dt.strftime("%Y-%m-%d")
                return df
        except Exception as e:
            st.error(f"Lỗi khi kéo dữ liệu phái sinh: {e}")
            return None

    # ── Luồng 2: Cổ phiếu cơ sở qua DNSE stock API ───────────────────────────
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
            df = pd.DataFrame(
                {
                    "time": pd.to_datetime(resp["t"], unit="s"),
                    "open": resp["o"],
                    "high": resp["h"],
                    "low": resp["l"],
                    "close": resp["c"],
                    "volume": resp["v"],
                }
            )
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
            return df
    except Exception:
        pass

    # ── Luồng 3: Fallback — SSI iBoard API ───────────────────────────────────
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
    """Ichimoku Kinko Hyo — 5 thành phần chuẩn Nhật."""
    hi, lo = df["high"], df["low"]
    tenkan = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
    kijun  = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou


def calc_fibonacci(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Fibonacci Retracement chuẩn — phát hiện Swing High/Low gần nhất trong lookback phiên.
    Tự động xác định xu hướng (đang hồi từ đỉnh hay phục hồi từ đáy) để
    căn chỉnh hướng tính Fib cho đúng.
    Trả về các mức retracement + extension để dùng làm hỗ trợ/kháng cự.
    """
    window   = df.tail(lookback)
    high_val = window["high"].max()
    low_val  = window["low"].min()

    # Xác định swing point gần nhất: High hay Low xuất hiện sau cùng?
    high_idx_pos = window["high"].values.argmax()
    low_idx_pos  = window["low"].values.argmin()

    # Nếu đỉnh xuất hiện SAU đáy → xu hướng đang giảm từ đỉnh (Retracement từ High xuống)
    # Nếu đáy xuất hiện SAU đỉnh → xu hướng đang phục hồi (Retracement từ Low lên)
    is_downtrend = (high_idx_pos > low_idx_pos)

    diff = high_val - low_val

    if is_downtrend:
        # Giá đang điều chỉnh từ đỉnh → tính Fib từ HIGH xuống LOW
        # Các mức là vùng HỖ TRỢ khi giá đi xuống
        levels = {
            "0.0% (Đỉnh)":   high_val,
            "23.6%":          high_val - 0.236 * diff,
            "38.2%":          high_val - 0.382 * diff,
            "50.0%":          high_val - 0.500 * diff,
            "61.8% ✨":       high_val - 0.618 * diff,   # Golden Ratio — hỗ trợ mạnh nhất
            "78.6%":          high_val - 0.786 * diff,
            "100.0% (Đáy)":   low_val,
        }
    else:
        # Giá đang phục hồi từ đáy → tính Fib từ LOW lên HIGH
        # Các mức là vùng KHÁNG CỰ khi giá đi lên
        levels = {
            "0.0% (Đáy)":    low_val,
            "23.6%":          low_val + 0.236 * diff,
            "38.2%":          low_val + 0.382 * diff,
            "50.0%":          low_val + 0.500 * diff,
            "61.8% ✨":       low_val + 0.618 * diff,   # Golden Ratio — kháng cự mạnh nhất
            "78.6%":          low_val + 0.786 * diff,
            "100.0% (Đỉnh)":  high_val,
        }

    # Lưu metadata để các section khác dùng (không ảnh hưởng iteration)
    levels["_high"]         = high_val
    levels["_low"]          = low_val
    levels["_is_downtrend"] = is_downtrend
    return levels


def calc_mcdx(df: pd.DataFrame) -> pd.Series:
    """
    MCDX — Momentum Convergence Divergence Index (chỉ báo tổng hợp).
    Kết hợp chuẩn hóa MACD Histogram và xung lượng RSI.
    > 0 → động lượng tăng | < 0 → động lượng giảm
    """
    macd_hist = MACD(close=df["close"]).macd_diff()
    rsi       = RSIIndicator(close=df["close"], window=14).rsi()

    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return pd.Series(0, index=s.index) if mx == mn else (s - mn) / (mx - mn) * 2 - 1

    return ((_norm(macd_hist) + _norm(rsi - 50)) / 2).rename("MCDX")


def calc_smart_money(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smart Money Indicators:
    - OBV   : On-Balance Volume — xu hướng dòng tiền lớn
    - CMF   : Chaikin Money Flow (20) — áp lực mua/bán
    - MFI   : Money Flow Index (14) — RSI có tính khối lượng
    - Up/Down Volume ratio — tỷ lệ khối lượng mua/bán
    - VSA Phase: Accumulation / Markup / Distribution / Markdown
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    open_  = df["open"]

    # ── OBV ──────────────────────────────────────────────────────────────
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = pd.array(obv, dtype=float)

    # ── CMF (Chaikin Money Flow, 20 bars) ────────────────────────────────
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    df["CMF"] = mfv.rolling(20).sum() / volume.rolling(20).sum()

    # ── MFI (Money Flow Index, 14 bars) ──────────────────────────────────
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos_mf = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df["MFI"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))

    # ── Up / Down Volume (10 bars) ────────────────────────────────────────
    df["Up_Vol"]   = np.where(close >= open_, volume, 0).astype(float)
    df["Down_Vol"] = np.where(close < open_,  volume, 0).astype(float)
    buy10  = pd.Series(df["Up_Vol"]).rolling(10).sum()
    sell10 = pd.Series(df["Down_Vol"]).rolling(10).sum()
    total10 = (buy10 + sell10).replace(0, np.nan)
    df["Buy_Ratio"] = buy10 / total10   # >0.6 = mua nhiều hơn bán

    # ── Vol Ratio vs MA20 ─────────────────────────────────────────────────
    df["Vol_MA20"]  = volume.rolling(20).mean()
    df["Vol_Ratio"] = volume / df["Vol_MA20"]

    # ── Candle body ratio (body/range) ────────────────────────────────────
    df["Body_Ratio"] = (abs(close - open_) / (high - low).replace(0, np.nan)).fillna(0)

    return df


def detect_smart_money_phase(df: pd.DataFrame) -> dict:
    """
    Phân tích hành vi tiền lớn dựa trên Wyckoff + VSA + OBV.
    Trả về dict với phase, score, signals và mô tả hành vi từng nhóm.
    """
    win = df.tail(15).copy()   # 15 phiên gần nhất để phân tích
    last = df.iloc[-1]
    prev5 = df.tail(5)

    # ── OBV trend (hồi quy tuyến tính) ───────────────────────────────────
    obv_vals = win["OBV"].dropna().values
    if len(obv_vals) > 3:
        x = np.arange(len(obv_vals))
        obv_slope = np.polyfit(x, obv_vals, 1)[0]
        obv_slope_norm = obv_slope / (abs(obv_vals).mean() + 1e-9) * 10
    else:
        obv_slope_norm = 0

    # ── Price trend 15 phiên ─────────────────────────────────────────────
    prices = win["close"].dropna().values
    if len(prices) > 3:
        px = np.arange(len(prices))
        price_slope = np.polyfit(px, prices, 1)[0]
        price_slope_norm = price_slope / (prices.mean() + 1e-9) * 100
    else:
        price_slope_norm = 0

    # ── Giá trị cuối ─────────────────────────────────────────────────────
    cmf_val      = last["CMF"] if pd.notna(last["CMF"]) else 0
    mfi_val      = last["MFI"] if pd.notna(last["MFI"]) else 50
    buy_ratio    = last["Buy_Ratio"] if pd.notna(last["Buy_Ratio"]) else 0.5
    vol_ratio    = last["Vol_Ratio"] if pd.notna(last["Vol_Ratio"]) else 1.0

    # ── Volatility so với MA ──────────────────────────────────────────────
    price_range_pct = (win["high"].max() - win["low"].min()) / win["close"].mean() * 100
    is_sideways = price_range_pct < 5.0    # biên độ <5% = đi ngang

    # ── Scoring phân pha ─────────────────────────────────────────────────
    # Dương = tích lũy / đẩy giá | Âm = xả hàng / đè giá
    score = 0.0
    signals = []

    # OBV
    if obv_slope_norm > 0.3:
        score += 2; signals.append(("OBV tăng", "Tiền lớn đang mua tích lũy dần", "🟢"))
    elif obv_slope_norm < -0.3:
        score -= 2; signals.append(("OBV giảm", "Tiền lớn đang bán/xả hàng", "🔴"))
    else:
        signals.append(("OBV phẳng", "Dòng tiền lớn chưa rõ xu hướng", "⚪"))

    # CMF
    if cmf_val > 0.12:
        score += 2; signals.append(("CMF cao", f"{cmf_val:.3f} — Áp lực mua mạnh", "🟢"))
    elif cmf_val > 0:
        score += 1; signals.append(("CMF dương nhẹ", f"{cmf_val:.3f} — Mua nhiều hơn bán", "🟡"))
    elif cmf_val < -0.12:
        score -= 2; signals.append(("CMF âm mạnh", f"{cmf_val:.3f} — Áp lực bán mạnh", "🔴"))
    else:
        score -= 1; signals.append(("CMF âm nhẹ", f"{cmf_val:.3f} — Bán nhiều hơn mua", "🟠"))

    # MFI
    if mfi_val < 25:
        score += 2; signals.append(("MFI quá bán", f"{mfi_val:.1f} — Dòng tiền bán cạn kiệt", "🟢"))
    elif mfi_val > 80:
        score -= 2; signals.append(("MFI quá mua", f"{mfi_val:.1f} — Dòng tiền mua bão hoà", "🔴"))
    else:
        signals.append(("MFI trung tính", f"{mfi_val:.1f}", "⚪"))

    # Buy Ratio
    if buy_ratio > 0.65:
        score += 1.5; signals.append(("Khối lượng mua", f"{buy_ratio*100:.0f}% phiên xanh 10 kỳ", "🟢"))
    elif buy_ratio < 0.35:
        score -= 1.5; signals.append(("Khối lượng bán", f"{(1-buy_ratio)*100:.0f}% phiên đỏ 10 kỳ", "🔴"))
    else:
        signals.append(("Cân bằng mua/bán", f"{buy_ratio*100:.0f}% / {(1-buy_ratio)*100:.0f}%", "⚪"))

    # Volume spike
    if vol_ratio > 2.0:
        if price_slope_norm > 0:
            score += 1; signals.append(("Volume bùng nổ ↑", f"x{vol_ratio:.1f} trung bình — Cú đẩy mạnh", "🟢"))
        else:
            score -= 1; signals.append(("Volume bùng nổ ↓", f"x{vol_ratio:.1f} trung bình — Xả hàng ồ ạt", "🔴"))
    elif vol_ratio < 0.5:
        signals.append(("Volume cạn", f"x{vol_ratio:.1f} — Thị trường thiếu thanh khoản", "🟡"))

    # Price vs OBV divergence
    if price_slope_norm > 0.5 and obv_slope_norm < -0.2:
        score -= 1.5; signals.append(("Phân kỳ âm", "Giá tăng nhưng OBV giảm — Cảnh báo xả hàng", "🔴"))
    elif price_slope_norm < -0.5 and obv_slope_norm > 0.2:
        score += 1.5; signals.append(("Phân kỳ dương", "Giá giảm nhưng OBV tăng — Tiền lớn đang gom", "🟢"))

    # ── Xác định pha Wyckoff ──────────────────────────────────────────────
    max_score = 10.0
    pct = score / max_score

    if pct >= 0.4:
        phase = "GOM HÀNG"
        icon  = "🏦"
        color = "#00C853"
        bg    = "rgba(0,200,83,0.10)"
        border= "#00C853"
        phase_desc = (
            "Tiền lớn đang **bí mật tích lũy** cổ phiếu ở vùng giá thấp. "
            "Giá đi ngang hoặc giảm nhẹ nhưng OBV & CMF tăng — "
            "dấu hiệu **cá mập / quỹ đang gom**."
        )
    elif pct >= 0.15:
        phase = "ĐẨY GIÁ (MARKUP)"
        icon  = "🚀"
        color = "#40C4FF"
        bg    = "rgba(64,196,255,0.10)"
        border= "#40C4FF"
        phase_desc = (
            "Dòng tiền lớn **đang đẩy giá lên mạnh**. "
            "Volume tăng theo giá — đây là giai đoạn **theo xu hướng** với tiền thông minh."
        )
    elif pct >= -0.15:
        phase = "TRUNG TÍNH / QUAN SÁT"
        icon  = "🔍"
        color = "#FFD740"
        bg    = "rgba(255,215,64,0.08)"
        border= "#FFD740"
        phase_desc = (
            "Dòng tiền lớn **chưa lộ rõ ý định**. "
            "Cần thêm tín hiệu từ volume và OBV để xác nhận hướng đi."
        )
    elif pct >= -0.40:
        phase = "XẢ HÀNG (DISTRIBUTION)"
        icon  = "📤"
        color = "#FF6D00"
        bg    = "rgba(255,109,0,0.10)"
        border= "#FF6D00"
        phase_desc = (
            "Tiền lớn đang **lặng lẽ phân phối/xả hàng** cho nhà đầu tư nhỏ lẻ. "
            "Giá cao nhưng OBV & CMF bắt đầu suy yếu — **cảnh báo đỉnh vùng**."
        )
    else:
        phase = "ĐÈ GIÁ (MARKDOWN)"
        icon  = "📉"
        color = "#FF1744"
        bg    = "rgba(255,23,68,0.10)"
        border= "#FF1744"
        phase_desc = (
            "Tiền lớn đang **đè giá và bán tháo mạnh**. "
            "Tránh mua đuổi — chờ tín hiệu OBV/CMF phục hồi trở lại trước khi xem xét."
        )

    # ── Hành vi ước tính từng nhóm ────────────────────────────────────────
    groups = []

    # Market Maker
    if vol_ratio > 1.5 and abs(price_slope_norm) < 0.3 and is_sideways:
        mm_action = ("🔄 Tạo thanh khoản giả", "Tạo volume lớn nhưng giá đứng yên — kiểm soát spread")
        mm_color  = "#FFD740"
    elif score > 3:
        mm_action = ("🏗️ Hỗ trợ đà tăng", "MM đang giữ bid, không để giá rơi")
        mm_color  = "#69F0AE"
    elif score < -3:
        mm_action = ("🧨 Đè giá để gom rẻ hơn", "MM đẩy giá xuống trước khi mua vào")
        mm_color  = "#FF6D00"
    else:
        mm_action = ("⚖️ Trung lập", "MM đang cân bằng hai phía lệnh")
        mm_color  = "#90CAF9"

    # Quỹ đầu tư
    if obv_slope_norm > 0.5 and cmf_val > 0.05:
        fund_action = ("📦 Đang gom hàng", "OBV + CMF đều tăng — dấu hiệu tích lũy dài hạn")
        fund_color  = "#69F0AE"
    elif obv_slope_norm < -0.5 and cmf_val < -0.05:
        fund_action = ("🚪 Đang thoát hàng", "Giảm tỷ trọng — xả dần qua nhiều phiên")
        fund_color  = "#FF5252"
    elif buy_ratio > 0.6:
        fund_action = ("🔍 Theo dõi tích cực", "Mua nhiều hơn bán trong 10 phiên gần đây")
        fund_color  = "#FFD740"
    else:
        fund_action = ("😴 Chờ đợi", "Chưa có hành động rõ ràng")
        fund_color  = "#90CAF9"

    # Tự doanh CTCK
    if mfi_val < 30 and cmf_val > 0:
        td_action = ("🎯 Mua vào vùng quá bán", "Dòng tiền tự doanh vào khi MFI thấp")
        td_color  = "#69F0AE"
    elif mfi_val > 75 and cmf_val < 0:
        td_action = ("🏃 Chốt lời / bán khống", "Tự doanh thoát hàng ở vùng quá mua")
        td_color  = "#FF5252"
    elif pct > 0.15:
        td_action = ("📈 Theo đà tăng", "Tự doanh đang riding theo xu hướng")
        td_color  = "#FFD740"
    else:
        td_action = ("🔄 Trung lập / arb", "Hoạt động arbitrage, chưa thiên hướng")
        td_color  = "#90CAF9"

    # Cá mập / Tay to
    if pct >= 0.4:
        shark_action = ("🐋 GOM HÀNG — Tín hiệu mạnh", "Volume bùng nổ khi giá thấp, OBV tăng ngược chiều giá")
        shark_color  = "#00E676"
    elif pct >= 0.15:
        shark_action = ("🚀 ĐẨY GIÁ — Đang markup", "Cá mập đã gom xong, bắt đầu đẩy để chốt lời cao hơn")
        shark_color  = "#40C4FF"
    elif pct >= -0.15:
        shark_action = ("🔍 QUAN SÁT — Chưa ra tay", "Cá mập đứng ngoài hoặc thăm dò thanh khoản")
        shark_color  = "#FFD740"
    elif pct >= -0.40:
        shark_action = ("📤 XẢ HÀNG — Phân phối đỉnh", "Cá mập đang bán dần cho retail ở vùng cao")
        shark_color  = "#FF9100"
    else:
        shark_action = ("📉 ĐÈ GIÁ — Bán tháo mạnh", "Tay to thoát hàng nhanh, tạo panic để mua rẻ lần sau")
        shark_color  = "#FF1744"

    groups = [
        ("🏦 Market Maker", mm_action[0],   mm_action[1],   mm_color),
        ("📊 Quỹ đầu tư",  fund_action[0], fund_action[1], fund_color),
        ("🏢 Tự doanh CTCK", td_action[0], td_action[1],   td_color),
        ("🐋 Cá mập / Tay to", shark_action[0], shark_action[1], shark_color),
    ]

    return {
        "phase": phase, "icon": icon, "color": color, "bg": bg, "border": border,
        "phase_desc": phase_desc, "score": score, "max_score": max_score,
        "signals": signals, "groups": groups,
        "obv_slope": obv_slope_norm, "cmf": cmf_val, "mfi": mfi_val,
        "buy_ratio": buy_ratio, "vol_ratio": vol_ratio,
    }


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
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        timeout=90,
    )
    
    # ← THÊM ĐOẠN NÀY để debug
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
        for attempt in range(2):  # retry once per model
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    if attempt == 0:
                        time.sleep(5)  # wait 5s then retry same model
                        continue
                    else:
                        break  # try next model
                elif "NOT_FOUND" in err_str or "404" in err_str:
                    break  # model unavailable, try next
                else:
                    raise  # other errors: propagate immediately

    raise RuntimeError(
        "⚠️ Đã vượt quá quota miễn phí Gemini.\n\n"
        "**Cách khắc phục:**\n"
        "- Nạp billing tại https://ai.dev/billing để dùng tiếp\n"
        "- Hoặc chờ ~1 phút rồi thử lại (rate limit theo phút)\n"
        "- Hoặc chuyển sang **Claude (Anthropic)** ở sidebar"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — triggered on button click
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🚀 Lấy Dữ Liệu & Phân Tích AI", type="primary"):

    with st.spinner(f"Đang trích xuất dữ liệu thị trường cho mã **{ticker}**..."):
        df = get_clean_stock_data(ticker)

    if df is None or df.empty:
        st.error("❌ Không thể lấy dữ liệu. Kiểm tra lại mã chứng khoán hoặc kết nối mạng.")
        st.stop()

    # ── Làm sạch kiểu dữ liệu ─────────────────────────────────────────────
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    if len(df) < 30:
        st.error("❌ Không đủ dữ liệu để tính chỉ báo (cần tối thiểu 30 phiên).")
        st.stop()

    # ── Tính toán chỉ báo ─────────────────────────────────────────────────
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
    (
        df["Tenkan"],
        df["Kijun"],
        df["SpanA"],
        df["SpanB"],
        df["Chikou"],
    ) = calc_ichimoku(df)
    fib_levels = calc_fibonacci(df)
    df         = calc_smart_money(df)
    sm_result  = detect_smart_money_phase(df)

    latest   = df.iloc[-1]
    prev     = df.iloc[-2]
    pct_chg  = (latest["close"] - prev["close"]) / prev["close"] * 100

    # ── Metrics row ───────────────────────────────────────────────────────
    st.subheader(f"📈 Trạng thái kỹ thuật hiện tại — {ticker}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Giá Đóng Cửa",
        f"{latest['close']:,.0f} đ",
        f"{pct_chg:+.2f}%",
    )
    rsi_val  = latest["RSI"]
    rsi_note = "⚠️ Quá mua" if rsi_val > 70 else ("⚠️ Quá bán" if rsi_val < 30 else "Trung tính")
    c2.metric("RSI (14)", f"{rsi_val:.1f}", rsi_note)
    mcdx_val = latest["MCDX"]
    c3.metric("MCDX", f"{mcdx_val:.3f}", "📈 Tăng" if mcdx_val > 0 else "📉 Giảm")
    c4.metric(
        "Tenkan / Kijun",
        f"{latest['Tenkan']:,.0f} / {latest['Kijun']:,.0f}",
        "Bullish" if latest["Tenkan"] > latest["Kijun"] else "Bearish",
    )
    c5.metric("Fib 50%", f"{fib_levels['50.0%']:,.0f} đ")

    # ── Biểu đồ tổng hợp ────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.14, 0.18, 0.18],
        vertical_spacing=0.025,
        subplot_titles=(
            f"Biểu đồ nến · Bollinger Bands · Ichimoku · Fibonacci — {ticker}",
            "Khối lượng giao dịch",
            "RSI (14)",
            "MACD + MCDX",
        ),
    )

    # ── Candlestick ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="Nến", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # ── Bollinger Bands ───────────────────────────────────────────────────
    for col, color, name in [
        ("BB_High", "rgba(255,80,80,0.55)",   "BB Upper"),
        ("BB_Mid",  "rgba(180,180,180,0.40)", "BB Mid"),
        ("BB_Low",  "rgba(80,220,80,0.55)",   "BB Lower"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df[col],
                line=dict(color=color, width=1),
                name=name,
            ),
            row=1, col=1,
        )

    # ── Ichimoku ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["Tenkan"],
                   line=dict(color="#00E5FF", width=2.0),
                   name="Tenkan-sen (9)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["Kijun"],
                   line=dict(color="#FF6B6B", width=2.0),
                   name="Kijun-sen (26)"),
        row=1, col=1,
    )
    # Chikou Span
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["Chikou"],
                   line=dict(color="#B39DDB", width=1.2, dash="dot"),
                   name="Chikou Span",
                   opacity=0.7),
        row=1, col=1,
    )
    # Kumo Cloud: tô màu xanh khi SpanA > SpanB (Bullish), đỏ khi ngược lại (Bearish)
    # Vùng Bullish (SpanA >= SpanB)
    span_a = df["SpanA"].values
    span_b = df["SpanB"].values
    times  = df["time"].values

    # Vẽ hai lớp cloud: Bullish (xanh) và Bearish (đỏ)
    bullish_a = np.where(span_a >= span_b, span_a, np.nan)
    bullish_b = np.where(span_a >= span_b, span_b, np.nan)
    bearish_a = np.where(span_a < span_b,  span_a, np.nan)
    bearish_b = np.where(span_a < span_b,  span_b, np.nan)

    # Bullish cloud (xanh)
    fig.add_trace(
        go.Scatter(x=times, y=bullish_a,
                   line=dict(color="rgba(0,0,0,0)", width=0),
                   showlegend=False, name="_bull_a"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=bullish_b,
                   fill="tonexty",
                   fillcolor="rgba(38,166,154,0.18)",
                   line=dict(color="#26a69a", width=0.8),
                   name="Kumo Bullish ☁️",
                   legendgroup="kumo"),
        row=1, col=1,
    )
    # Bearish cloud (đỏ)
    fig.add_trace(
        go.Scatter(x=times, y=bearish_b,
                   line=dict(color="rgba(0,0,0,0)", width=0),
                   showlegend=False, name="_bear_b"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=bearish_a,
                   fill="tonexty",
                   fillcolor="rgba(239,83,80,0.18)",
                   line=dict(color="#ef5350", width=0.8),
                   name="Kumo Bearish ☁️",
                   legendgroup="kumo"),
        row=1, col=1,
    )

    # ── Fibonacci Retracement ─────────────────────────────────────────────
    fib_palette = {
        "0.0% (Đỉnh)":   "#9E9E9E",
        "0.0% (Đáy)":    "#9E9E9E",
        "23.6%":          "#7986CB",
        "38.2%":          "#29B6F6",
        "50.0%":          "#EF5350",
        "61.8% ✨":       "#FFA726",   # Golden Ratio
        "78.6%":          "#AB47BC",
        "100.0% (Đỉnh)":  "#9E9E9E",
        "100.0% (Đáy)":   "#9E9E9E",
    }
    fib_width = {
        "0.0% (Đỉnh)": 1, "0.0% (Đáy)": 1,
        "23.6%": 1, "38.2%": 1.5,
        "50.0%": 2.5, "61.8% ✨": 2.5,
        "78.6%": 1.5,
        "100.0% (Đỉnh)": 1, "100.0% (Đáy)": 1,
    }
    x_range = [df["time"].iloc[0], df["time"].iloc[-1]]
    x_label = df["time"].iloc[-1]  # vị trí gắn nhãn (bên phải)

    for label, price in fib_levels.items():
        if label.startswith("_"):
            continue   # bỏ qua metadata keys
        is_key = label in ("38.2%", "50.0%", "61.8% ✨")
        color_fib = fib_palette.get(label, "#9E9E9E")
        width_fib = fib_width.get(label, 1)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[price, price],
                mode="lines",
                line=dict(
                    color=color_fib,
                    width=width_fib,
                    dash="dot" if not is_key else "dash",
                ),
                name=f"Fib {label}",
                text=f"Fib {label}  {price:,.0f}",
                hoverinfo="text",
                showlegend=True,
            ),
            row=1, col=1,
        )
        # Thêm nhãn chú thích giá bên phải mỗi mức Fibonacci
        fig.add_annotation(
            xref="x", yref="y",
            x=x_label, y=price,
            text=f"  {label} · {price:,.0f}",
            showarrow=False,
            font=dict(color=color_fib, size=10 if is_key else 9),
            xanchor="left",
            row=1, col=1,
        )

    # ── Volume ────────────────────────────────────────────────────────────
    bar_colors = [
        "#ef5350" if df["close"].iloc[i] < df["open"].iloc[i] else "#26a69a"
        for i in range(len(df))
    ]
    fig.add_trace(
        go.Bar(
            x=df["time"], y=df["volume"],
            marker_color=bar_colors, name="Volume", showlegend=False,
        ),
        row=2, col=1,
    )

    # ── RSI ───────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["RSI"],
            line=dict(color="#FF9800", width=1.8), name="RSI",
        ),
        row=3, col=1,
    )
    for level, color in [(70, "rgba(255,80,80,0.6)"), (30, "rgba(80,200,80,0.6)")]:
        fig.add_shape(
            type="line", xref="paper", yref="y3",
            x0=0, x1=1, y0=level, y1=level,
            line=dict(color=color, width=1, dash="dash"),
        )

    # ── MACD + MCDX ───────────────────────────────────────────────────────
    hist_colors = np.where(df["MACD_Diff"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(
        go.Bar(
            x=df["time"], y=df["MACD_Diff"],
            marker_color=hist_colors, name="MACD Histogram", showlegend=True,
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["MACD"],
            line=dict(color="#2196F3", width=1.5), name="MACD",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["MACD_Signal"],
            line=dict(color="#FF5722", width=1.5), name="Signal",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["MCDX"],
            line=dict(color="#E040FB", width=2, dash="dot"), name="MCDX",
        ),
        row=4, col=1,
    )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=1050,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=120, t=60, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    st.plotly_chart(fig, width='stretch')

    # ── Bảng dữ liệu ─────────────────────────────────────────────────────
    st.write("**📋 Bảng dữ liệu 7 phiên gần nhất (kèm chỉ báo):**")
    show_cols = [
        c for c in
        ["time","open","high","low","close","volume","RSI","MACD","MCDX","Tenkan","Kijun"]
        if c in df.columns
    ]
    st.dataframe(
        df.tail(7)[show_cols].round(2),
        hide_index=True,
        width='stretch',
    )

    # ── Fibonacci table ───────────────────────────────────────────────────
    _fib_is_down = fib_levels.get("_is_downtrend", True)
    _fib_high    = fib_levels.get("_high", 0)
    _fib_low     = fib_levels.get("_low",  0)
    with st.expander("📐 Xem bảng mức Fibonacci Retracement"):
        fib_label = "Retracement từ Đỉnh → Đáy (vùng HỖ TRỢ)" if _fib_is_down else "Retracement từ Đáy → Đỉnh (vùng KHÁNG CỰ)"
        st.caption(f"📌 {fib_label} &nbsp;|&nbsp; Đỉnh: **{_fib_high:,.0f} đ** &nbsp;·&nbsp; Đáy: **{_fib_low:,.0f} đ**")
        fib_df = pd.DataFrame(
            [{"Mức Fibonacci": k, "Giá (đ)": f"{v:,.0f}"} for k, v in fib_levels.items() if not k.startswith("_")]
        )
        st.dataframe(fib_df, hide_index=True, width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    # SMART MONEY ANALYSIS — Hành vi tiền lớn
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🐋 Phân Tích Dòng Tiền Thông Minh — Market Maker · Quỹ · Tự Doanh · Cá Mập")
    st.caption(
        "Dựa trên **OBV · CMF · MFI · Volume Spread Analysis (Wyckoff)** — "
        "phân tích hành vi ẩn của các tay chơi lớn qua dấu vết khối lượng."
    )

    # ── Phase Banner ──────────────────────────────────────────────────────
    sm = sm_result
    pct_bar = max(0, min(100, (sm["score"] + sm["max_score"]) / (2 * sm["max_score"]) * 100))
    st.markdown(
        f"""
<div style="border:2px solid {sm['border']}; border-radius:14px; padding:20px 28px;
            background:{sm['bg']}; margin-bottom:16px;">
  <div style="font-size:2.0rem; font-weight:800; color:{sm['color']}; letter-spacing:2px;">
    {sm['icon']} &nbsp; GIA ĐOẠN: {sm['phase']}
  </div>
  <div style="margin-top:8px; font-size:1.05rem; color:#ddd; line-height:1.6;">
    {sm['phase_desc']}
  </div>
  <div style="margin-top:12px; background:rgba(255,255,255,0.08); border-radius:8px;
              height:10px; width:100%;">
    <div style="height:10px; width:{pct_bar:.0f}%; border-radius:8px;
                background:linear-gradient(90deg, #ef5350, #FFD740, #00C853);"></div>
  </div>
  <div style="display:flex; justify-content:space-between; font-size:0.75rem;
              color:#999; margin-top:4px;">
    <span>📉 Đè giá / Xả</span><span>⚖️ Trung tính</span><span>📈 Gom / Đẩy</span>
  </div>
  <div style="margin-top:8px; font-size:0.9rem; color:#aaa;">
    Điểm Smart Money: <b style="color:{sm['color']};">{sm['score']:+.1f} / {sm['max_score']:.0f}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Hành vi từng nhóm tiền lớn ───────────────────────────────────────
    st.markdown("#### 🔬 Hành vi ước tính từng nhóm tay chơi lớn")
    g_cols = st.columns(4)
    for idx, (group_name, action, detail, gcolor) in enumerate(sm["groups"]):
        with g_cols[idx]:
            st.markdown(
                f"""<div style="border:1px solid {gcolor}44; border-left:4px solid {gcolor};
                                border-radius:10px; padding:14px 12px;
                                background:{gcolor}11; height:140px;">
  <div style="font-size:0.95rem; font-weight:700; color:{gcolor};">{group_name}</div>
  <div style="font-size:0.88rem; font-weight:600; color:#eee; margin-top:6px;">{action}</div>
  <div style="font-size:0.78rem; color:#aaa; margin-top:5px; line-height:1.4;">{detail}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Smart Money Chart (OBV + CMF + Volume Breakdown) ─────────────────
    with st.expander("📊 Xem biểu đồ Smart Money chi tiết (OBV · CMF · MFI · Volume Mua/Bán)"):
        fig_sm = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.30, 0.22, 0.22, 0.26],
            vertical_spacing=0.04,
            subplot_titles=(
                "OBV — On-Balance Volume (xu hướng dòng tiền lớn)",
                "CMF — Chaikin Money Flow (áp lực mua/bán)",
                "MFI — Money Flow Index (RSI dòng tiền)",
                "Volume Mua (xanh) vs Bán (đỏ) từng phiên",
            ),
        )

        # OBV
        obv_colors = ["#26a69a" if v >= 0 else "#ef5350"
                      for v in df["OBV"].diff().fillna(0)]
        fig_sm.add_trace(
            go.Scatter(
                x=df["time"], y=df["OBV"],
                line=dict(color="#40C4FF", width=2),
                fill="tozeroy", fillcolor="rgba(64,196,255,0.08)",
                name="OBV",
            ),
            row=1, col=1,
        )
        # OBV MA20
        fig_sm.add_trace(
            go.Scatter(
                x=df["time"], y=df["OBV"].rolling(20).mean(),
                line=dict(color="#FF9800", width=1.2, dash="dash"),
                name="OBV MA20",
            ),
            row=1, col=1,
        )

        # CMF
        cmf_colors = ["#26a69a" if v >= 0 else "#ef5350"
                      for v in df["CMF"].fillna(0)]
        fig_sm.add_trace(
            go.Bar(
                x=df["time"], y=df["CMF"],
                marker_color=cmf_colors, name="CMF",
            ),
            row=2, col=1,
        )
        fig_sm.add_hline(y=0.1,  line=dict(color="#26a69a", dash="dot", width=1), row=2, col=1)
        fig_sm.add_hline(y=-0.1, line=dict(color="#ef5350", dash="dot", width=1), row=2, col=1)

        # MFI
        mfi_color_line = "#E040FB"
        fig_sm.add_trace(
            go.Scatter(
                x=df["time"], y=df["MFI"],
                line=dict(color=mfi_color_line, width=1.8),
                name="MFI",
            ),
            row=3, col=1,
        )
        for level, col_ in [(80, "rgba(255,80,80,0.5)"), (20, "rgba(80,200,80,0.5)")]:
            fig_sm.add_hline(y=level, line=dict(color=col_, dash="dash", width=1), row=3, col=1)

        # Volume Mua/Bán
        fig_sm.add_trace(
            go.Bar(
                x=df["time"], y=df["Up_Vol"],
                marker_color="rgba(38,166,154,0.85)",
                name="Volume Mua",
            ),
            row=4, col=1,
        )
        fig_sm.add_trace(
            go.Bar(
                x=df["time"], y=-df["Down_Vol"],
                marker_color="rgba(239,83,80,0.85)",
                name="Volume Bán",
            ),
            row=4, col=1,
        )
        # Vol MA20
        fig_sm.add_trace(
            go.Scatter(
                x=df["time"], y=df["Vol_MA20"],
                line=dict(color="#FFD740", width=1.2, dash="dot"),
                name="Vol MA20",
            ),
            row=4, col=1,
        )

        fig_sm.update_layout(
            template="plotly_dark",
            height=720,
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1, font=dict(size=10)),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
        )
        fig_sm.update_xaxes(showgrid=False)
        fig_sm.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_sm, width='stretch')

    # ── Signal Details ────────────────────────────────────────────────────
    with st.expander("🔎 Chi tiết tín hiệu Smart Money từng chỉ báo"):
        sig_cols = st.columns(2)
        for i, (name_, detail_, dot_) in enumerate(sm["signals"]):
            with sig_cols[i % 2]:
                st.markdown(
                    f"<div style='padding:8px 12px; margin:4px 0; border-radius:8px;"
                    f"background:rgba(255,255,255,0.05);'>"
                    f"<b>{dot_} {name_}</b><br>"
                    f"<span style='color:#bbb; font-size:0.88rem;'>{detail_}</span></div>",
                    unsafe_allow_html=True,
                )

    # Metric bar: key smart money indicators
    sm_c1, sm_c2, sm_c3, sm_c4 = st.columns(4)
    sm_c1.metric(
        "OBV Trend",
        f"{sm['obv_slope']:+.3f}",
        "↑ Tích lũy" if sm["obv_slope"] > 0 else "↓ Phân phối",
        help="Slope chuẩn hoá OBV 15 phiên — dương = tiền vào, âm = tiền ra",
    )
    sm_c2.metric(
        "CMF (20)",
        f"{sm['cmf']:.3f}",
        "Mua trội" if sm["cmf"] > 0 else "Bán trội",
        help="Chaikin Money Flow: >0.1 mua mạnh, <-0.1 bán mạnh",
    )
    sm_c3.metric(
        "MFI (14)",
        f"{sm['mfi']:.1f}",
        "Quá mua 🔴" if sm["mfi"] > 80 else ("Quá bán 🟢" if sm["mfi"] < 20 else "Trung tính"),
        help="Money Flow Index — giống RSI nhưng tính thêm khối lượng",
    )
    sm_c4.metric(
        "Tỷ lệ mua/bán (10P)",
        f"{sm['buy_ratio']*100:.0f}% / {(1-sm['buy_ratio'])*100:.0f}%",
        "Mua trội 🟢" if sm["buy_ratio"] > 0.55 else ("Bán trội 🔴" if sm["buy_ratio"] < 0.45 else "Cân bằng"),
        help="Tỷ lệ volume phiên xanh / đỏ trong 10 phiên gần nhất",
    )

    # ─────────────────────────────────────────────────────────────────────
    # QUICK RECOMMENDATION CARD (không cần API Key)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚦 Khuyến Nghị Nhanh (Tín Hiệu Kỹ Thuật Tổng Hợp)")

    # --- Tính điểm tín hiệu ---
    score = 0
    signal_details = []

    # RSI
    rsi_now = latest["RSI"]
    if rsi_now < 35:
        score += 2; signal_details.append(("RSI", f"{rsi_now:.1f} — Quá bán 📈", "🟢"))
    elif rsi_now < 50:
        score += 1; signal_details.append(("RSI", f"{rsi_now:.1f} — Vùng tích lũy", "🟡"))
    elif rsi_now > 70:
        score -= 2; signal_details.append(("RSI", f"{rsi_now:.1f} — Quá mua 📉", "🔴"))
    else:
        signal_details.append(("RSI", f"{rsi_now:.1f} — Trung tính", "⚪"))

    # MACD
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 2; signal_details.append(("MACD", "Cắt lên — Xu hướng tăng", "🟢"))
    else:
        score -= 1; signal_details.append(("MACD", "Cắt xuống — Xu hướng giảm", "🔴"))

    # MCDX
    mcdx_now = latest["MCDX"]
    if mcdx_now > 0.2:
        score += 2; signal_details.append(("MCDX", f"{mcdx_now:.3f} — Động lượng mạnh", "🟢"))
    elif mcdx_now > 0:
        score += 1; signal_details.append(("MCDX", f"{mcdx_now:.3f} — Động lượng nhẹ", "🟡"))
    else:
        score -= 1; signal_details.append(("MCDX", f"{mcdx_now:.3f} — Yếu dần", "🔴"))

    # Ichimoku
    span_a_last = latest["SpanA"] if pd.notna(latest["SpanA"]) else None
    span_b_last = latest["SpanB"] if pd.notna(latest["SpanB"]) else None
    if span_a_last and span_b_last:
        cloud_top = max(span_a_last, span_b_last)
        cloud_bot = min(span_a_last, span_b_last)
        if latest["close"] > cloud_top:
            score += 2; signal_details.append(("Ichimoku", "Giá trên mây — Bullish ☁️✅", "🟢"))
        elif latest["close"] < cloud_bot:
            score -= 2; signal_details.append(("Ichimoku", "Giá dưới mây — Bearish ☁️❌", "🔴"))
        else:
            signal_details.append(("Ichimoku", "Giá trong mây — Trung tính", "🟡"))
    if latest["Tenkan"] > latest["Kijun"]:
        score += 1; signal_details.append(("Tenkan/Kijun", "Tenkan > Kijun — Tín hiệu mua", "🟢"))
    else:
        score -= 1; signal_details.append(("Tenkan/Kijun", "Tenkan < Kijun — Tín hiệu bán", "🔴"))

    # Fibonacci — giá gần hỗ trợ
    _is_down   = fib_levels.get("_is_downtrend", True)
    _fib_h     = fib_levels.get("_high", 0)
    _fib_l     = fib_levels.get("_low",  0)
    _diff      = _fib_h - _fib_l

    if _is_down:
        fib_236_v  = _fib_h - 0.236 * _diff
        fib_382_v  = _fib_h - 0.382 * _diff
        fib_50_v   = _fib_h - 0.500 * _diff
        fib_618_v  = _fib_h - 0.618 * _diff
        fib_786_v  = _fib_h - 0.786 * _diff
    else:
        fib_236_v  = _fib_l + 0.236 * _diff
        fib_382_v  = _fib_l + 0.382 * _diff
        fib_50_v   = _fib_l + 0.500 * _diff
        fib_618_v  = _fib_l + 0.618 * _diff
        fib_786_v  = _fib_l + 0.786 * _diff

    close_now  = latest["close"]
    fib_margin = _diff * 0.03   # ±3% tolerance

    if _is_down:
        # Downtrend: các mức Fib là HỖ TRỢ → giá gần Fib thấp = tốt để mua
        if fib_618_v - fib_margin <= close_now <= fib_618_v + fib_margin:
            score += 2; signal_details.append(("Fibonacci", f"Giá tại Fib 61.8% ({fib_618_v:,.0f}) ✨ — Hỗ trợ vàng (Golden Ratio)", "🟢"))
        elif fib_50_v - fib_margin <= close_now <= fib_50_v + fib_margin:
            score += 1; signal_details.append(("Fibonacci", f"Giá tại Fib 50.0% ({fib_50_v:,.0f}) — Hỗ trợ tâm lý", "🟡"))
        elif fib_786_v - fib_margin <= close_now <= fib_786_v + fib_margin:
            score += 2; signal_details.append(("Fibonacci", f"Giá tại Fib 78.6% ({fib_786_v:,.0f}) — Hỗ trợ rất mạnh", "🟢"))
        elif close_now < fib_618_v:
            score += 1; signal_details.append(("Fibonacci", f"Giá dưới Fib 61.8% ({fib_618_v:,.0f}) — Vùng hỗ trợ sâu", "🟡"))
        elif close_now > fib_382_v:
            score -= 1; signal_details.append(("Fibonacci", f"Giá trên Fib 38.2% ({fib_382_v:,.0f}) — Gần vùng kháng cự", "🔴"))
        else:
            signal_details.append(("Fibonacci", f"Giá giữa Fib 38.2%–61.8% — Vùng trung tính", "⚪"))
    else:
        # Uptrend recovery: các mức Fib là KHÁNG CỰ → giá vượt Fib cao = tốt
        if close_now > fib_618_v:
            score += 2; signal_details.append(("Fibonacci", f"Giá vượt Fib 61.8% ({fib_618_v:,.0f}) ✨ — Breakout mạnh", "🟢"))
        elif close_now > fib_50_v:
            score += 1; signal_details.append(("Fibonacci", f"Giá vượt Fib 50.0% ({fib_50_v:,.0f}) — Phục hồi tốt", "🟡"))
        elif close_now < fib_382_v:
            score -= 1; signal_details.append(("Fibonacci", f"Giá dưới Fib 38.2% ({fib_382_v:,.0f}) — Phục hồi yếu", "🔴"))
        else:
            signal_details.append(("Fibonacci", f"Giá tại Fib 38.2%–50.0% — Đang kiểm tra kháng cự", "⚪"))

    # --- Xác định khuyến nghị tổng ---
    max_score = 12
    pct_score = score / max_score

    if pct_score >= 0.45:
        rec_label  = "✅ MUA"
        rec_color  = "#00C853"
        rec_bg     = "rgba(0,200,83,0.12)"
        rec_border = "#00C853"
        rec_desc   = "Tín hiệu kỹ thuật thuận lợi — Cân nhắc mở vị thế mua."
    elif pct_score >= 0.15:
        rec_label  = "⏳ THEO DÕI"
        rec_color  = "#FFC107"
        rec_bg     = "rgba(255,193,7,0.12)"
        rec_border = "#FFC107"
        rec_desc   = "Tín hiệu chưa rõ ràng — Chờ xác nhận thêm trước khi vào lệnh."
    else:
        rec_label  = "🚫 TRÁNH / BÁN"
        rec_color  = "#FF5252"
        rec_bg     = "rgba(255,82,82,0.12)"
        rec_border = "#FF5252"
        rec_desc   = "Tín hiệu yếu hoặc tiêu cực — Không nên mua, cân nhắc cắt lỗ nếu đang giữ."

    # Hiển thị card tổng
    stars = "⭐" * max(1, min(5, round((score + max_score) / (2 * max_score) * 5)))
    st.markdown(
        f"""
<div style="border:2px solid {rec_border}; border-radius:12px; padding:20px 28px;
            background:{rec_bg}; margin-bottom:16px;">
  <div style="font-size:2rem; font-weight:700; color:{rec_color}; letter-spacing:1px;">
    {rec_label}
  </div>
  <div style="font-size:1.1rem; color:#ccc; margin-top:6px;">{rec_desc}</div>
  <div style="margin-top:10px; font-size:1rem; color:#aaa;">
    Điểm tổng hợp: <b style="color:{rec_color};">{score}/{max_score}</b> &nbsp;|&nbsp; {stars}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Hiển thị chi tiết từng tín hiệu
    with st.expander("📊 Chi tiết tín hiệu từng chỉ báo"):
        cols = st.columns(2)
        for i, (ind, detail, dot) in enumerate(signal_details):
            with cols[i % 2]:
                st.markdown(
                    f"<div style='padding:8px 12px; margin:4px 0; border-radius:8px; "
                    f"background:rgba(255,255,255,0.05);'>"
                    f"<b>{dot} {ind}</b><br>"
                    f"<span style='color:#ccc; font-size:0.9rem;'>{detail}</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Gợi ý vùng mua / stoploss / target ──────────────────────────────
    _is_down2  = fib_levels.get("_is_downtrend", True)
    _fib_h2    = fib_levels.get("_high", 0)
    _fib_l2    = fib_levels.get("_low",  0)
    _diff2     = _fib_h2 - _fib_l2

    if _is_down2:
        _f236 = _fib_h2 - 0.236 * _diff2
        _f382 = _fib_h2 - 0.382 * _diff2
        _f50  = _fib_h2 - 0.500 * _diff2
        _f618 = _fib_h2 - 0.618 * _diff2
        _f786 = _fib_h2 - 0.786 * _diff2
        # Vùng mua: dải 50%–61.8% (hỗ trợ vàng khi điều chỉnh)
        buy_zone_low  = _f618
        buy_zone_high = _f50
        # Stop loss: dưới 78.6% hoặc BB_Low (lấy mức an toàn hơn)
        sl_fib = _f786 * 0.985   # 1.5% dưới Fib 78.6%
        sl_bb  = latest["BB_Low"]
        stop_loss = min(sl_fib, sl_bb)
        # Targets: giá trên hiện tại → 38.2% rồi 23.6%
        close_p = latest["close"]
        target1 = _f382
        target2 = _f236
    else:
        _f236 = _fib_l2 + 0.236 * _diff2
        _f382 = _fib_l2 + 0.382 * _diff2
        _f50  = _fib_l2 + 0.500 * _diff2
        _f618 = _fib_l2 + 0.618 * _diff2
        _f786 = _fib_l2 + 0.786 * _diff2
        # Vùng mua: ngay trên đáy tới 38.2% retracement (uptrend)
        buy_zone_low  = _fib_l2
        buy_zone_high = _f382
        # Stop loss: dưới đáy
        sl_fib = _fib_l2 * 0.985
        sl_bb  = latest["BB_Low"]
        stop_loss = min(sl_fib, sl_bb)
        # Targets: 61.8% và 78.6%
        target1 = _f618
        target2 = _f786

    # Xác định delta hiển thị (so với giá hiện tại)
    close_p = latest["close"]
    delta_buy  = f"{((buy_zone_high - close_p) / close_p * 100):+.1f}%" if close_p > 0 else ""
    delta_sl   = f"{((stop_loss - close_p) / close_p * 100):+.1f}%" if close_p > 0 else ""
    delta_tg1  = f"{((target1 - close_p) / close_p * 100):+.1f}%" if close_p > 0 else ""
    delta_tg2  = f"{((target2 - close_p) / close_p * 100):+.1f}%" if close_p > 0 else ""

    st.markdown(
        f"""
<div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:16px 20px;
            border:1px solid rgba(255,255,255,0.12); margin-bottom:12px;">
  <div style="font-size:0.85rem; color:#aaa; margin-bottom:10px; letter-spacing:0.5px;">
    📐 &nbsp;Fibonacci {'Retracement (Đỉnh→Đáy)' if _is_down2 else 'Recovery (Đáy→Đỉnh)'} &nbsp;·&nbsp;
    Đỉnh <b style="color:#EF5350">{_fib_h2:,.0f}</b> đ &nbsp;·&nbsp;
    Đáy <b style="color:#26a69a">{_fib_l2:,.0f}</b> đ &nbsp;·&nbsp;
    Giá hiện tại <b style="color:#FFD740">{close_p:,.0f}</b> đ
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px;">
    <div style="background:rgba(0,200,83,0.12); border:1px solid #00C853; border-radius:10px; padding:12px;">
      <div style="font-size:0.75rem; color:#aaa;">📥 Vùng mua gợi ý</div>
      <div style="font-size:1.1rem; font-weight:700; color:#00C853; margin:4px 0;">
        {buy_zone_low:,.0f} – {buy_zone_high:,.0f}
      </div>
      <div style="font-size:0.75rem; color:#888;">Fib {'61.8%→50.0%' if _is_down2 else 'Đáy→38.2%'} (vùng hỗ trợ vàng)</div>
    </div>
    <div style="background:rgba(255,82,82,0.10); border:1px solid #FF5252; border-radius:10px; padding:12px;">
      <div style="font-size:0.75rem; color:#aaa;">🛑 Stop Loss tham khảo</div>
      <div style="font-size:1.1rem; font-weight:700; color:#FF5252; margin:4px 0;">
        {stop_loss:,.0f} đ
      </div>
      <div style="font-size:0.75rem; color:#888; line-height:1.3;">
        {delta_sl} · Min(BB Lower, Fib {'78.6%-1.5%' if _is_down2 else 'Đáy-1.5%'})
      </div>
    </div>
    <div style="background:rgba(33,150,243,0.10); border:1px solid #2196F3; border-radius:10px; padding:12px;">
      <div style="font-size:0.75rem; color:#aaa;">🎯 Target 1</div>
      <div style="font-size:1.1rem; font-weight:700; color:#2196F3; margin:4px 0;">
        {target1:,.0f} đ
      </div>
      <div style="font-size:0.75rem; color:#888;">{delta_tg1} · Fib {'38.2%' if _is_down2 else '61.8%'} — kháng cự gần</div>
    </div>
    <div style="background:rgba(224,64,251,0.10); border:1px solid #E040FB; border-radius:10px; padding:12px;">
      <div style="font-size:0.75rem; color:#aaa;">🎯 Target 2</div>
      <div style="font-size:1.1rem; font-weight:700; color:#E040FB; margin:4px 0;">
        {target2:,.0f} đ
      </div>
      <div style="font-size:0.75rem; color:#888;">{delta_tg2} · Fib {'23.6%' if _is_down2 else '78.6%'} — mục tiêu xa hơn</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    # AI ANALYSIS
    # ─────────────────────────────────────────────────────────────────────
    if not api_key:
        st.warning("💡 Nhập API Key ở thanh bên trái để nhận phân tích từ Trợ lý AI.")
    else:
        st.markdown("---")
        provider_name = "Claude (Anthropic)" if "Claude" in ai_provider else "Gemini (Google)"
        st.subheader(f"🤖 Báo Cáo Phân Tích & Khuyến Nghị — {provider_name}")

        # Xây dựng context cho AI
        ai_data = df.tail(10)[show_cols].round(2).to_string(index=False)
        fib_str = "\n".join(
            [f"  • Fib {k}: {v:,.0f} đ" for k, v in fib_levels.items()]
        )

        # Ichimoku hiện tại
        above_cloud = (
            latest["close"] > max(latest["SpanA"] or 0, latest["SpanB"] or 0)
            if pd.notna(latest["SpanA"]) and pd.notna(latest["SpanB"])
            else None
        )
        ichi_status = (
            "trên mây (Bullish)" if above_cloud
            else ("dưới mây (Bearish)" if above_cloud is False else "trong mây (trung tính)")
        )

        macd_signal = (
            "cắt lên (hội tụ tăng giá)"
            if latest["MACD"] > latest["MACD_Signal"]
            else "cắt xuống (phân kỳ giảm giá)"
        )

        prompt = f"""
Bạn là một hệ thống AI định lượng cao cấp chuyên phân tích thị trường chứng khoán Việt Nam.

Dưới đây là dữ liệu 10 phiên gần nhất của mã **{ticker}** kèm đầy đủ chỉ báo kỹ thuật:

```
{ai_data}
```

Mức Fibonacci Retracement (tính từ 60 phiên gần nhất):
{fib_str}

Tóm tắt nhanh tín hiệu chỉ báo hiện tại:
- RSI: {rsi_val:.1f} → {rsi_note}
- MACD: {macd_signal}
- MCDX: {mcdx_val:.4f} → {"Động lượng tăng" if mcdx_val > 0 else "Động lượng giảm"}
- Ichimoku: Giá đang {ichi_status}
- Tenkan: {latest['Tenkan']:,.0f} | Kijun: {latest['Kijun']:,.0f}
- BB Upper: {latest['BB_High']:,.0f} | BB Lower: {latest['BB_Low']:,.0f}

**Yêu cầu phân tích chi tiết (viết bằng tiếng Việt, rõ ràng và quyết đoán):**

1. **RSI Analysis**: Đánh giá vùng quá mua/quá bán, phân kỳ RSI nếu có.
2. **MACD Analysis**: Đánh giá tín hiệu cắt lên/xuống, độ hội tụ và phân kỳ.
3. **Ichimoku Analysis**: Vị trí giá so với mây Kumo, tín hiệu Tenkan/Kijun, Chikou Span.
4. **MCDX Analysis**: Đánh giá động lượng tổng hợp và so sánh với MACD.
5. **Fibonacci Analysis**: Xác định vùng hỗ trợ/kháng cự gần nhất mà giá có thể test.
6. **Kết luận & Chiến lược giao dịch**:
   - ✅ Khuyến nghị: **Mua / Bán / Chờ**
   - 📥 Vùng mua hợp lý (điểm vào lệnh)
   - 🛑 Cắt lỗ (Stop Loss)
   - 🎯 Mục tiêu 1 (Take Profit 1)
   - 🎯 Mục tiêu 2 (Take Profit 2)
   - ⚠️ Mức độ rủi ro: **Thấp / Trung bình / Cao**
   - 📝 Lý do tổng hợp ngắn gọn (1–2 câu)
"""

        with st.spinner("⏳ AI đang phân tích tín hiệu tổng hợp và lập chiến lược giao dịch..."):
            try:
                if "Claude" in ai_provider:
                    result = call_claude(api_key, prompt)
                else:
                    result = call_gemini(api_key, prompt)
                st.markdown(result)
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
                    st.error("❌ Hết quota Gemini API (429 RESOURCE_EXHAUSTED)")
                    st.warning(
                        "**Nguyên nhân:** Tài khoản Google AI đang dùng gói miễn phí và đã hết hạn mức.\n\n"
                        "**Cách khắc phục:**\n"
                        "- 🔄 Chờ ~1 phút rồi thử lại (nếu hết giới hạn theo phút)\n"
                        "- 💳 Nạp billing tại https://ai.dev/billing để tăng hạn mức\n"
                        "- 🔀 Chuyển sang **Claude (Anthropic)** ở sidebar trái"
                    )
                elif "NOT_FOUND" in err_str or "404" in err_str:
                    st.error("❌ Model AI không tìm thấy (404 NOT_FOUND)")
                    st.info("💡 Thử chọn nhà cung cấp khác hoặc kiểm tra lại API Key.")
                elif "401" in err_str or "UNAUTHENTICATED" in err_str:
                    st.error("❌ API Key không hợp lệ hoặc hết hạn.")
                    st.info("💡 Claude: lấy key tại console.anthropic.com | Gemini: lấy key tại aistudio.google.com")
                else:
                    st.error(f"❌ Lỗi gọi AI: {e}")
                    st.info(
                        "💡 Gợi ý: Kiểm tra lại API Key và đảm bảo bạn có quyền truy cập model. "
                        "Claude dùng key từ console.anthropic.com, Gemini dùng key từ aistudio.google.com."
                    )
