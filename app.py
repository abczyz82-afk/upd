import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (Tối ưu cho Web/Mobile)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="AI Trading Dashboard 📊", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────────────
# XỬ LÝ API KEY TỰ ĐỘNG BẰNG STREAMLIT SECRETS
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key_from_secrets(provider: str) -> str:
    """Lấy API Key từ cấu hình Secrets của Streamlit Cloud (nếu có)"""
    try:
        if provider == "Gemini" and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        elif provider == "Claude" and "CLAUDE_API_KEY" in st.secrets:
            return st.secrets["CLAUDE_API_KEY"]
    except Exception:
        pass
    return ""

st.sidebar.header("⚙️ Cấu hình Hệ thống")
ai_provider = st.sidebar.radio(
    "Chọn nhà cung cấp AI:",
    ["🔵 Gemini (Google)", "🟠 Claude (Anthropic)"],
)
provider_name = "Gemini" if "Gemini" in ai_provider else "Claude"

# Nếu đã cài trong Secrets, ô này sẽ tự động điền. Nếu chưa, người dùng có thể nhập tay.
default_key = get_api_key_from_secrets(provider_name)
api_key = st.sidebar.text_input(f"Nhập API Key cho {provider_name}:", value=default_key, type="password")

if not api_key:
    st.sidebar.warning("⚠️ Vui lòng nhập API Key hoặc cấu hình trong Streamlit Secrets để bật Trợ lý AI.")

# ─────────────────────────────────────────────────────────────────────────────
# RADAR QUÉT DÒNG TIỀN (Chạy ngầm, tối ưu RAM)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Radar Smart Money")
watchlist_input = st.sidebar.text_input("Quét mã (cách nhau dấu phẩy):", "FPT, HPG, TCB, MBB, SSI")

if st.sidebar.button("Quét Tín Hiệu Ngay 🚀", use_container_width=True):
    tickers = [t.strip().upper() for t in watchlist_input.split(",")]
    found_signals = []
    
    with st.sidebar.status("Đang quét thị trường..."):
        for t in tickers:
            symbol_yf = t + ".VN" if len(t) <= 4 and not t.endswith(".VN") else t
            df_scan = yf.download(symbol_yf, period="3mo", interval="1d", progress=False)
            
            if not df_scan.empty and len(df_scan) >= 30:
                if isinstance(df_scan.columns, pd.MultiIndex):
                    df_scan.columns = df_scan.columns.get_level_values(0)
                df_scan.columns = [c.lower() for c in df_scan.columns]
                
                # Logic OBV đơn giản để tránh import vòng lặp nặng
                close, volume = df_scan["close"], df_scan["volume"]
                obv = [0]
                for i in range(1, len(df_scan)):
                    if close.iloc[i] > close.iloc[i-1]: obv.append(obv[-1] + volume.iloc[i])
                    elif close.iloc[i] < close.iloc[i-1]: obv.append(obv[-1] - volume.iloc[i])
                    else: obv.append(obv[-1])
                
                df_scan["OBV"] = obv
                obv_trend = df_scan["OBV"].diff().tail(10).sum()
                rsi_last = RSIIndicator(close, window=14).rsi().iloc[-1]
                
                if obv_trend > 0 and rsi_last < 45:
                    found_signals.append({"Mã": t, "Tín hiệu": "Dòng tiền vào + Quá bán"})
                elif obv_trend > 0:
                    found_signals.append({"Mã": t, "Tín hiệu": "Đang tích lũy"})
                    
    if found_signals:
        st.sidebar.success("🎯 Kết quả:")
        st.sidebar.dataframe(pd.DataFrame(found_signals), hide_index=True)
    else:
        st.sidebar.info("Chưa có mã nào có tín hiệu gom hàng rõ rệt.")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER & MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Hệ Thống Phân Tích Kỹ Thuật AI")
st.markdown("Sử dụng dữ liệu độc lập từ **Yahoo Finance**. Không phụ thuộc vào các công ty chứng khoán (VND, SSI, TCBS...).")

ticker = st.text_input("🔍 Nhập mã cổ phiếu Việt Nam (VD: FPT, HPG, MWG):", "FPT").upper().strip()

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (Dùng TTL 15 phút để giảm tải server)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner="Đang kéo dữ liệu từ Yahoo Finance...")
def get_stock_data(symbol: str) -> pd.DataFrame | None:
    try:
        # Xử lý Hậu tố cho TT Việt Nam
        symbol_yf = symbol + ".VN" if len(symbol) <= 4 and not symbol.endswith(".VN") else symbol
        df = yf.download(symbol_yf, period="6mo", interval="1d", progress=False)
        
        if df.empty: return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"date": "time"}, inplace=True)
        return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# CÁC HÀM TÍNH TOÁN
# ─────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame):
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()
    
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]: obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]: obv.append(obv[-1] - df["volume"].iloc[i])
        else: obv.append(obv[-1])
    df["OBV"] = obv
    
    # Ichimoku
    hi, lo = df["high"], df["low"]
    df["Tenkan"] = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    df["Kijun"] = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# GỌI AI
# ─────────────────────────────────────────────────────────────────────────────
def get_ai_analysis(api_key: str, provider: str, df: pd.DataFrame, symbol: str) -> str:
    ai_data = df.tail(10)[["time","close","volume","RSI","MACD_Diff","OBV"]].round(2).to_string(index=False)
    prompt = f"""
    Bạn là chuyên gia PTKT thị trường chứng khoán. Phân tích dữ liệu 10 phiên gần nhất của {symbol}:
    {ai_data}
    
    Yêu cầu ngắn gọn:
    1. Đánh giá Xu hướng chung (từ Giá, Khối lượng, OBV).
    2. Điểm mua/Bán/Cắt lỗ hợp lý lúc này.
    3. Trả lời trực tiếp, dứt khoát, dùng tiếng Việt.
    """
    
    try:
        if provider == "Gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text
        else:
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            body = {"model": "claude-3-5-sonnet-20240620", "max_tokens": 1000, "messages": [{"role": "user", "content": prompt}]}
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"❌ Lỗi gọi AI: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🚀 Trích Xuất Dữ Liệu & Phân Tích", type="primary"):
    df = get_stock_data(ticker)
    
    if df is None or len(df) < 30:
        st.error(f"❌ Không tìm thấy dữ liệu cho mã {ticker} (hoặc dữ liệu quá ngắn). Thử thêm '.VN' (VD: FPT.VN)")
        st.stop()
        
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pct_change = ((latest["close"] - prev["close"]) / prev["close"]) * 100
    
    # Thẻ thông tin
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá Đóng Cửa", f"{latest['close']:,.0f} đ", f"{pct_change:+.2f}%")
    c2.metric("Khối lượng", f"{latest['volume']:,.0f}")
    c3.metric("RSI (14)", f"{latest['RSI']:.1f}")
    c4.metric("MACD vs Signal", f"{latest['MACD']:.1f} / {latest['MACD_Signal']:.1f}")

    # Vẽ biểu đồ tương tác Plotly
    st.subheader(f"📈 Biểu đồ Kỹ thuật — {ticker}")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], 
                        vertical_spacing=0.05, subplot_titles=("Giá & Bollinger Bands", "Khối lượng & OBV", "RSI & MACD"))
    
    # Cột Giá
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Nến"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_High"], line=dict(color='red', width=1), name="BB High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_Low"], line=dict(color='green', width=1), name="BB Low"), row=1, col=1)
    
    # Khối lượng & OBV
    colors = ['green' if df["close"].iloc[i] > df["open"].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df["time"], y=df["volume"], marker_color=colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["OBV"], line=dict(color='yellow', width=2), yaxis="y4", name="OBV"), row=2, col=1)
    
    # RSI & MACD
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI"], line=dict(color='orange', width=2), name="RSI"), row=3, col=1)
    fig.add_trace(go.Bar(x=df["time"], y=df["MACD_Diff"], marker_color=['green' if v > 0 else 'red' for v in df["MACD_Diff"]], name="MACD Hist"), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Phân tích AI
    st.markdown("---")
    st.subheader(f"🤖 Phân Tích Chiến Lược ({provider_name})")
    if api_key:
        with st.spinner("Đang chờ AI phân tích..."):
            ai_result = get_ai_analysis(api_key, provider_name, df, ticker)
            st.info(ai_result)
    else:
        st.warning("⚠️ Chưa có API Key. Hãy nhập ở Sidebar để AI lập chiến lược giao dịch.")
