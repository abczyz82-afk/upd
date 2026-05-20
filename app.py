import streamlit as st
import pandas as pd
from vnstock import stock_historical_data
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import time

# Nhập các chỉ báo kỹ thuật từ thư viện 'ta'
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

st.set_page_config(layout="wide", page_title="AI Trading Dashboard")

# --- 1. CẤU HÌNH TRỢ LÝ AI GEMINI ---
st.sidebar.header("⚙️ Cấu hình Hệ thống")
api_key = st.sidebar.text_input("Nhập Gemini API Key của bạn:", type="password")
if not api_key:
    st.sidebar.warning("⚠️ Vui lòng nhập API Key để bật tính năng Trợ lý AI.")

# --- 2. GIAO DIỆN CHÍNH & TÌM KIẾM MÃ ---
st.title("📊 Hệ Thống Phân Tích Kỹ Thuật & Trợ Lý Giao Dịch AI")
st.markdown("Hệ thống tích hợp luồng xử lý kép: Dữ liệu cơ sở qua `vnstock` và Dữ liệu phái sinh qua `DNSE API`.")

ticker = st.text_input("🔍 Nhập mã chứng khoán (Cổ phiếu hoặc VN30F1M):", "VN30F1M").upper().strip()

# Hàm lấy dữ liệu rẽ nhánh thông minh
@st.cache_data(ttl=60)
def get_clean_stock_data(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    # LUỒNG 1: XỬ LÝ RIÊNG CHO PHÁI SINH VN30F1M
    if symbol == "VN30F1M":
        try:
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            # API public của DNSE dành riêng cho phái sinh
            url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/derivative?from={start_ts}&to={end_ts}&symbol={symbol}&resolution=1D"
            resp = requests.get(url).json()
            
            if 't' in resp and len(resp['t']) > 0:
                df = pd.DataFrame({
                    'time': pd.to_datetime(resp['t'], unit='s'),
                    'open': resp['o'],
                    'high': resp['h'],
                    'low': resp['l'],
                    'close': resp['c'],
                    'volume': resp['v']
                })
                # Đổi định dạng ngày cho khớp với chuẩn của hệ thống
                df['time'] = df['time'].dt.strftime('%Y-%m-%d')
                return df
        except Exception as e:
            st.error(f"Lỗi khi kéo dữ liệu phái sinh: {e}")
            return None

    # LUỒNG 2: XỬ LÝ CỔ PHIẾU CƠ SỞ 
    str_start = start_date.strftime('%Y-%m-%d')
    str_end = end_date.strftime('%Y-%m-%d')
    
    try:
        df = stock_historical_data(symbol=symbol, start_date=str_start, end_date=str_end, source='VCI')
        if df is not None and not df.empty:
            df.columns = [col.lower() for col in df.columns]
            rename_dict = {'tradingdate': 'time', 'date': 'time', 'vol': 'volume'}
            df = df.rename(columns=rename_dict)
            return df
    except:
        pass

    try:
        df = stock_historical_data(symbol=symbol, start_date=str_start, end_date=str_end)
        if df is not None and not df.empty:
            df.columns = [col.lower() for col in df.columns]
            rename_dict = {'tradingdate': 'time', 'date': 'time', 'vol': 'volume'}
            df = df.rename(columns=rename_dict)
            return df
    except:
        return None

if st.button("Lấy Dữ Liệu & Khởi Chạy AI Analysis"):
    with st.spinner(f"Đang trích xuất dữ liệu thị trường cho mã {ticker}..."):
        df = get_clean_stock_data(ticker)
        
        if df is not None and not df.empty:
            # Ép kiểu an toàn
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['close'])
            
            # --- 3. TÍNH TOÁN CHỈ BÁO KỸ THUẬT ---
            rsi_series = RSIIndicator(close=df['close'], window=14).rsi()
            df['RSI'] = rsi_series
            
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            
            macd_obj = MACD(close=df['close'])
            df['MACD'] = macd_obj.macd()
            df['MACD_Signal'] = macd_obj.macd_signal()
            df['MACD_Diff'] = macd_obj.macd_diff()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            price_change = latest['close'] - prev['close']
            
            # Hiển thị Metrics chính
            st.subheader(f"📈 Trạng thái kỹ thuật hiện tại của {ticker}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Giá Đóng Cửa", f"{latest['close']:,.0f} đ", f"{price_change:,.0f} đ")
            m2.metric("Chỉ báo RSI (14)", f"{latest['RSI']:.2f}", "Quá mua (>70) / Quá bán (<30)" if latest['RSI'] > 70 or latest['RSI'] < 30 else "Trung tính")
            m3.metric("Bollinger Band Trên", f"{latest['BB_High']:,.0f} đ")
            m4.metric("Bollinger Band Dưới", f"{latest['BB_Low']:,.0f} đ")
            
            # --- 4. VẼ BIỂU ĐỒ NẾN ---
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Giá nến'))
            fig.add_trace(go.Scatter(x=df['time'], y=df['BB_High'], line=dict(color='rgba(250, 0, 0, 0.4)', width=1), name='BB Upper'))
            fig.add_trace(go.Scatter(x=df['time'], y=df['BB_Low'], line=dict(color='rgba(0, 250, 0, 0.4)', width=1), name='BB Lower'))
            
            fig.update_layout(title=f"Biểu đồ kỹ thuật mã {ticker}", xaxis_rangeslider_visible=False, template="plotly_dark", height=450)
            st.plotly_chart(fig, width="stretch")
            
            st.write("Bảng dữ liệu 5 phiên gần nhất tích hợp chỉ báo:")
            st.dataframe(df.tail(5)[['time', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD']], hide_index=True, width="stretch")
            
            # --- 5. GỌI API GEMINI TRỰC TIẾP TỰ ĐỘNG CHỌN MODEL ---
            if api_key:
                st.markdown("---")
                st.subheader(f"🤖 Báo Cáo Khuyến Nghị Vùng Giá Từ Trợ Lý AI")
                
                ai_context = df.tail(7)[['time', 'close', 'volume', 'RSI', 'BB_High', 'BB_Low', 'MACD']].to_string()
                
                prompt = f"""
                Bạn là một hệ thống AI định lượng cao cấp chuyên phân tích thị trường chứng khoán Việt Nam.
                Dưới đây là chuỗi dữ liệu giá kèm chỉ báo kỹ thuật được tính toán chính xác của mã {ticker}:
                {ai_context}
                
                Yêu cầu phân tích và đưa ra chiến lược:
                1. Đánh giá trạng thái giá dựa trên vị trí với dải Bollinger Bands và xung lực chỉ báo RSI hiện tại.
                2. Chỉ báo MACD đang cho tín hiệu cắt lên (hội tụ tăng giá) hay cắt xuống (phân kỳ giảm giá)?
                3. Đưa ra khuyến nghị hành động quyết đoán (Mua/Bán/Nắm giữ) kèm theo MỨC GIÁ MUA HỢP LÝ (vùng hỗ trợ cứng) và MỨC GIÁ BÁN MỤC TIÊU (vùng kháng cự gần) bằng các con số cụ thể.
                """
                
with st.spinner("AI đang tính toán điểm hội tụ chỉ báo và lập chiến lược..."):
                    try:
                        # Sử dụng chuẩn SDK mới nhất của Google
                        from google import genai
                        
                        # Khởi tạo client với API Key của bạn
                        client = genai.Client(api_key=api_key)
                        
                        # Gọi model gemini-1.5-flash (Model ổn định và phản hồi nhanh nhất hiện tại)
                        response = client.models.generate_content(
                            model='gemini-1.5-flash',
                            contents=prompt,
                        )
                        
                        # Hiển thị kết quả
                        st.info(response.text)
                        
                    except Exception as e:
                        st.error(f"Lỗi hệ thống AI (SDK Mới): {e}")
