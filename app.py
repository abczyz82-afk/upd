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
st.set_page_config(layout="wide", page_title="AI Trading Dashboard 📊", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────────────
# XỬ LÝ API KEY TỰ ĐỘNG BẰNG STREAMLIT SECRETS
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key_from_secrets(provider: str) -> str:
    """Lấy API Key từ cấu hình Secrets của Streamlit Cloud để không phải nhập tay"""
    try:
        if provider == "Gemini" and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        elif provider == "Claude" and "CLAUDE_API_KEY" in st.secrets:
            return st.secrets["CLAUDE_API_KEY"]
    except Exception:
        pass
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & SCANNER TÍN HIỆU GOM HÀNG
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Cấu hình Hệ thống")

ai_provider = st.sidebar.radio(
    "Chọn nhà cung cấp AI:",
    ["🔵 Gemini (Google)", "🟠 Claude (Anthropic)"],
)
provider_name = "Gemini" if "Gemini" in ai_provider else "Claude"

# Tự động điền API Key từ Secrets nếu có
default_key = get_api_key_from_secrets(provider_name)
api_key = st.sidebar.text_input(f"Nhập API Key cho {provider_name}:", value=default_key, type="password")

if not api_key:
    st.sidebar.warning("⚠️ Cấu hình API Key trong Streamlit Secrets hoặc nhập tay để bật Trợ lý AI.")

st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (Giữ nguyên nguồn DNSE & SSI cũ)
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

    return None

# ─────────────────────────────────────────────────────────────────────────────
# RADAR SCANNER VỚI NGUỒN DỮ LIỆU CŨ
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("📡 Radar Smart Money")
st.sidebar.caption("Quét tự động các mã đang được Quỹ/Cá mập gom hàng.")
watchlist_input = st.sidebar.text_input("Danh sách quét (cách nhau dấu phẩy):", "FPT, HPG, MWG, VCB, SSI")

if st.sidebar.button("Quét Tín Hiệu Ngay 🚀", use_container_width=True):
    tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    found_signals = []
    
    with st.sidebar.status("Đang quét thị trường (Nguồn: DNSE/SSI)..."):
        for t in tickers:
            df_scan = get_clean_stock_data(t)
            
            if df_scan is not None and not df_scan.empty and len(df_scan) >= 30:
                # Xử lý logic OBV nhanh để tránh load nặng
                close, volume, open_ = df_scan["close"], df_scan["volume"], df_scan["open"]
                obv = [0]
                for i in range(1, len(df_scan)):
                    if close.iloc[i] > close.iloc[i-1]: obv.append(obv[-1] + volume.iloc[i])
                    elif close.iloc[i] < close.iloc[i-1]: obv.append(obv[-1] - volume.iloc[i])
                    else: obv.append(obv[-1])
                df_scan["OBV"] = pd.array(obv, dtype=float)
                
                # CMF đơn giản
                hl = (df_scan["high"] - df_scan["low"]).replace(0, np.nan)
                mfm = ((close - df_scan["low"]) - (df_scan["high"] - close)) / hl
                mfv = mfm * volume
                cmf_val = (mfv.tail(20).sum() / volume.tail(20).sum()) if volume.tail(20).sum() > 0 else 0
                
                obv_trend = df_scan["OBV"].diff().tail(15).sum()
                
                if obv_trend > 0 and cmf_val > 0.05:
                    found_signals.append({"Mã": t, "Tín hiệu": "GOM HÀNG / Tích lũy", "CMF": round(cmf_val, 3)})
                elif cmf_val < -0.1:
                    found_signals.append({"Mã": t, "Tín hiệu": "XẢ HÀNG / Rủi ro", "CMF": round(cmf_val, 3)})
    
    if found_signals:
        st.sidebar.success("🎯 Kết quả quét:")
        st.sidebar.dataframe(pd.DataFrame(found_signals), hide_index=True)
    else:
        st.sidebar.info("Chưa tìm thấy dấu hiệu dòng tiền lớn rõ ràng ở danh sách này.")

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
# INDICATOR CALCULATIONS (Giữ nguyên toàn bộ logic cũ)
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
            "0.0% (Đỉnh)":      high_val,
            "23.6%":             high_val - 0.236  * diff,
            "38.2%":             high_val - 0.382  * diff,
            "50.0%":             high_val - 0.500  * diff,
            "61.8% ✨":         high_val - 0.618  * diff,
            "65.0% 🏅":         high_val - 0.650  * diff,
            "78.6%":             high_val - 0.786  * diff,
            "100.0% (Đáy)":      low_val,
        }
    else:
        levels = {
            "0.0% (Đáy)":        low_val,
            "23.6%":             low_val  + 0.236  * diff,
            "38.2%":             low_val  + 0.382  * diff,
            "50.0%":             low_val  + 0.500  * diff,
            "61.8% ✨":         low_val  + 0.618  * diff,
            "65.0% 🏅":         low_val  + 0.650  * diff,
            "78.6%":             low_val  + 0.786  * diff,
            "100.0% (Đỉnh)":     high_val,
        }
    levels["_high"]         = high_val
    levels["_low"]          = low_val
    levels["_is_downtrend"] = is_downtrend
    levels["_diff"]         = diff
    return levels

def calc_mcdx(df: pd.DataFrame) -> pd.Series:
    macd_hist = MACD(close=df["close"]).macd_diff()
    rsi       = RSIIndicator(close=df["close"], window=14).rsi()
    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return pd.Series(0, index=s.index) if mx == mn else (s - mn) / (mx - mn) * 2 - 1
    return ((_norm(macd_hist) + _norm(rsi - 50)) / 2).rename("MCDX")

def calc_smart_money(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]: obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]: obv.append(obv[-1] - volume.iloc[i])
        else: obv.append(obv[-1])
    df["OBV"] = pd.array(obv, dtype=float)

    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    df["CMF"] = mfv.rolling(20).sum() / volume.rolling(20).sum()

    tp = (high + low + close) / 3
    rmf = tp * volume
    pos_mf = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df["MFI"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))

    df["Up_Vol"]   = np.where(close >= open_, volume, 0).astype(float)
    df["Down_Vol"] = np.where(close < open_,  volume, 0).astype(float)
    buy10  = pd.Series(df["Up_Vol"]).rolling(10).sum()
    sell10 = pd.Series(df["Down_Vol"]).rolling(10).sum()
    total10 = (buy10 + sell10).replace(0, np.nan)
    df["Buy_Ratio"] = buy10 / total10
    df["Vol_MA20"]  = volume.rolling(20).mean()
    df["Vol_Ratio"] = volume / df["Vol_MA20"]

    return df

def detect_smart_money_phase(df: pd.DataFrame) -> dict:
    win = df.tail(15).copy()
    last = df.iloc[-1]

    obv_vals = win["OBV"].dropna().values
    obv_slope_norm = (np.polyfit(np.arange(len(obv_vals)), obv_vals, 1)[0] / (abs(obv_vals).mean() + 1e-9) * 10) if len(obv_vals) > 3 else 0

    prices = win["close"].dropna().values
    price_slope_norm = (np.polyfit(np.arange(len(prices)), prices, 1)[0] / (prices.mean() + 1e-9) * 100) if len(prices) > 3 else 0

    cmf_val = last["CMF"] if pd.notna(last["CMF"]) else 0
    mfi_val = last["MFI"] if pd.notna(last["MFI"]) else 50
    buy_ratio = last["Buy_Ratio"] if pd.notna(last["Buy_Ratio"]) else 0.5
    vol_ratio = last["Vol_Ratio"] if pd.notna(last["Vol_Ratio"]) else 1.0

    score = 0.0
    signals = []

    if obv_slope_norm > 0.3: score += 2; signals.append(("OBV tăng", "Mua tích lũy", "🟢"))
    elif obv_slope_norm < -0.3: score -= 2; signals.append(("OBV giảm", "Bán xả hàng", "🔴"))

    if cmf_val > 0.12: score += 2; signals.append(("CMF cao", "Áp lực mua mạnh", "🟢"))
    elif cmf_val < -0.12: score -= 2; signals.append(("CMF âm", "Áp lực bán mạnh", "🔴"))

    if mfi_val < 25: score += 2; signals.append(("MFI quá bán", "Dòng tiền bán cạn", "🟢"))
    elif mfi_val > 80: score -= 2; signals.append(("MFI quá mua", "Dòng tiền mua bão hoà", "🔴"))

    max_score = 10.0
    pct = score / max_score

    if pct >= 0.4: phase, icon, color = "GOM HÀNG", "🏦", "#00C853"
    elif pct >= 0.15: phase, icon, color = "ĐẨY GIÁ (MARKUP)", "🚀", "#40C4FF"
    elif pct >= -0.15: phase, icon, color = "TRUNG TÍNH", "🔍", "#FFD740"
    elif pct >= -0.40: phase, icon, color = "XẢ HÀNG (DISTRIBUTION)", "📤", "#FF6D00"
    else: phase, icon, color = "ĐÈ GIÁ (MARKDOWN)", "📉", "#FF1744"

    groups = [
        ("🏦 Market Maker", "Hỗ trợ đà" if score > 1 else "Điều tiết", "MM hoạt động", "#90CAF9"),
        ("📊 Quỹ đầu tư", "Đang gom hàng" if cmf_val > 0.05 else "Theo dõi", "Dấu hiệu tích lũy", "#69F0AE"),
        ("🐋 Cá mập", phase, "Xu hướng dòng tiền lớn", color),
    ]

    return {
        "phase": phase, "icon": icon, "color": color, "score": score, 
        "max_score": max_score, "signals": signals, "groups": groups,
        "obv_slope": obv_slope_norm, "cmf": cmf_val, "mfi": mfi_val, "buy_ratio": buy_ratio
    }

# ─────────────────────────────────────────────────────────────────────────────
# AI CALL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(api_key: str, prompt: str) -> str:
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    body = {"model": "claude-3-5-sonnet-20240620", "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=90)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

def call_gemini(api_key: str, prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Lỗi Gemini: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🚀 Lấy Dữ Liệu & Phân Tích AI", type="primary"):
    with st.spinner(f"Đang trích xuất dữ liệu thị trường cho mã **{ticker}** (Nguồn DNSE/SSI)..."):
        df = get_clean_stock_data(ticker)

    if df is None or df.empty:
        st.error("❌ Không thể lấy dữ liệu. Kiểm tra lại mã chứng khoán hoặc kết nối mạng.")
        st.stop()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    if len(df) < 30:
        st.error("❌ Không đủ dữ liệu để tính chỉ báo (cần tối thiểu 30 phiên).")
        st.stop()

    # Tính toán
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
    df["Tenkan"], df["Kijun"], df["SpanA"], df["SpanB"], df["Chikou"] = calc_ichimoku(df)
    
    fib_levels = calc_fibonacci(df)
    df         = calc_smart_money(df)
    sm_result  = detect_smart_money_phase(df)

    latest   = df.iloc[-1]
    prev     = df.iloc[-2]
    pct_chg  = (latest["close"] - prev["close"]) / prev["close"] * 100

    # Metrics row
    st.subheader(f"📈 Trạng thái kỹ thuật hiện tại — {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá Đóng Cửa", f"{latest['close']:,.0f} đ", f"{pct_chg:+.2f}%")
    c2.metric("RSI (14)", f"{latest['RSI']:.1f}")
    c3.metric("MCDX", f"{latest['MCDX']:.3f}")
    c4.metric("Fib 50%", f"{fib_levels['50.0%']:,.0f} đ")

    # Smart Money Banner
    st.markdown("---")
    st.subheader("🐋 Tín Hiệu Gom Hàng Từ Market Maker & Quỹ")
    col_sm1, col_sm2 = st.columns(2)
    with col_sm1:
        st.info(f"**Trạng thái Smart Money:** {sm_result['icon']} {sm_result['phase']}")
        st.write(f"**Điểm dòng tiền:** {sm_result['score']:+.1f} / {sm_result['max_score']}")
    with col_sm2:
        for idx, (group_name, action, detail, gcolor) in enumerate(sm_result["groups"]):
            st.markdown(f"<div style='border-left: 3px solid {gcolor}; padding-left: 10px; margin-bottom: 5px;'>"
                        f"<b>{group_name}</b>: {action}</div>", unsafe_allow_html=True)

    # Bảng dữ liệu
    st.write("**📋 Bảng dữ liệu 5 phiên gần nhất:**")
    st.dataframe(df.tail(5)[["time","open","high","low","close","volume","RSI","MACD","OBV"]].round(2), hide_index=True)

    # AI Analysis
    if not api_key:
        st.warning("💡 Nhập API Key ở thanh bên trái để nhận phân tích từ Trợ lý AI.")
    else:
        st.markdown("---")
        st.subheader(f"🤖 Báo Cáo Phân Tích & Khuyến Nghị — {provider_name}")

        ai_data = df.tail(10)[["time","close","volume","RSI","MACD","OBV","CMF"]].round(2).to_string(index=False)
        prompt = f"""
        Phân tích kỹ thuật mã {ticker}. 
        Dữ liệu 10 phiên gần nhất:
        {ai_data}
        
        Trạng thái Smart Money hiện tại: {sm_result['phase']} (Score: {sm_result['score']}).
        Fibonacci 61.8% hỗ trợ/kháng cự: {fib_levels['61.8% ✨']:,.0f}.
        
        Yêu cầu:
        1. Đánh giá Xu hướng (RSI, MACD, OBV).
        2. Dòng tiền lớn (Cá mập/Quỹ) đang làm gì?
        3. Điểm Mua / Bán / Cắt lỗ rõ ràng (ngắn gọn, quyết đoán).
        """
        
        with st.spinner("⏳ AI đang lập chiến lược giao dịch..."):
            try:
                if "Claude" in ai_provider:
                    result = call_claude(api_key, prompt)
                else:
                    result = call_gemini(api_key, prompt)
                st.markdown(result)
            except Exception as e:
                st.error(f"❌ Lỗi gọi AI: {e}")
