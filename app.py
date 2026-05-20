import streamlit as st
import pandas as pd
from vnstock import stock_historical_data
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Nhập các chỉ báo kỹ thuật từ thư viện 'ta'
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

st.set_page_config(layout="wide", page_title="AI Trading Dashboard")

# --- 1. CẤU HÌNH TRỢ LÝ AI GEMINI (DÙNG REST API TRỰC TIẾP) ---
st.sidebar.header("⚙️ Cấu hình Hệ thống")
api_key = st.sidebar.text_input("Nhập Gemini API Key của bạn:", type="password")
if not api_key:
    st.sidebar.warning("⚠️ Vui lòng nhập API Key để bật tính năng Trợ lý AI.")

# --- 2. GIAO DIỆN CHÍNH & TÌM KIẾM MÃ ---
st.title("📊 Hệ Thống Phân Tích Kỹ Thuật & Trợ Lý Giao Dịch AI")
st.markdown("Hệ thống kết nối nguồn dữ liệu chuẩn Việt Nam qua `vnstock` và tự động tính toán các chỉ báo nâng cao (RSI, Bollinger Bands, MACD).")

ticker = st.text_input("🔍 Nhập mã chứng khoán (Cổ phiếu hoặc VN30F1M):", "SSI").upper().strip()

# Hàm lấy dữ liệu lịch sử thông minh đa nguồn
@st.cache_data(ttl=60)
def get_clean_stock_data(symbol):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    try:
        df = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date, source='VCI')
        if df is not None and not df.empty:
            return df
    except:
        pass

    try:
        df = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            return df
    except:
        return None

if st.button("Lấy Dữ Liệu & Khởi Chạy AI Analysis"):
    with st.spinner(f"Đang trích xuất dữ liệu thị trường cho mã {ticker}..."):
        df = get_clean_stock_data(ticker)
        
        if df is not None and not df.empty:
            df.columns = [col.lower() for col in df.columns]
            
            rename_dict = {
                'tradingdate': 'time', 'date': 'time',
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 
                'volume': 'volume', 'vol': 'volume'
            }
            df = df.rename(columns=rename_dict)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['close'])
            df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
            
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
            
            # Sửa cảnh báo use_container_width thành cú pháp mới
            st.plotly_chart(fig, width="stretch")
            
            st.write("Bảng dữ liệu 5 phiên gần nhất tích hợp chỉ báo:")
            st.dataframe(df.tail(5)[['time', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD']], hide_index=True, width="stretch")
            
            # --- 5. GỌI API GEMINI TRỰC TIẾP ---
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
                    # Danh sách các model từ mới nhất đến các bản dự phòng
                    models_to_try = [
                        "gemini-1.5-flash-latest",
                        "gemini-1.5-pro-latest",
                        "gemini-2.0-flash"
                    ]
                    
                    success = False
                    error_msg = ""
                    
                    for model_name in models_to_try:
                        try:
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                            headers = {'Content-Type': 'application/json'}
                            data = {
                                "contents": [{"parts": [{"text": prompt}]}]
                            }
                            
                            response = requests.post(url, headers=headers, data=json.dumps(data))
                            
                            if response.status_code == 200:
                                result = response.json()
                                ai_text = result['candidates'][0]['content']['parts'][0]['text']
                                st.info(f"*(Phân tích bởi model: {model_name})*\n\n" + ai_text)
                                success = True
                                break # Thành công thì thoát vòng lặp ngay
                            else:
                                error_msg = response.json().get('error', {}).get('message', 'Không rõ')
                        except Exception as e:
                            error_msg = str(e)
                            
                    if not success:
                        st.error(f"Đã thử toàn bộ danh sách model nhưng AI của Google vẫn từ chối kết nối. Lỗi cuối cùng: {error_msg}")
