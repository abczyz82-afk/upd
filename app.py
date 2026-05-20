import streamlit as st
import pandas as pd
from vnstock import stock_historical_data
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Nhập các chỉ báo kỹ thuật từ thư viện 'ta'
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

st.set_page_config(layout="wide", page_title="AI Trading Dashboard")

# --- 1. CẤU HÌNH TRỢ LÝ AI GEMINI ---
st.sidebar.header("⚙️ Cấu hình Hệ thống")
api_key = st.sidebar.text_input("Nhập Gemini API Key của bạn:", type="password")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
else:
    st.sidebar.warning("⚠️ Vui lòng nhập API Key để bật tính năng Trợ lý AI.")

# --- 2. GIAO DIỆN CHÍNH & TÌM KIẾM MÃ ---
st.title("📊 Hệ Thống Phân Tích Kỹ Thuật & Trợ Lý Giao Dịch AI")
st.markdown("Hệ thống kết nối nguồn dữ liệu chuẩn Việt Nam qua `vnstock` và tự động tính toán các chỉ báo nâng cao (RSI, Bollinger Bands, MACD).")

ticker = st.text_input("🔍 Nhập mã chứng khoán (Cổ phiếu hoặc VN30F1M):", "SSI").upper().strip()

# Hàm lấy dữ liệu lịch sử thông minh đa nguồn tránh lỗi chặn cổng kết nối
@st.cache_data(ttl=60)
def get_clean_stock_data(symbol):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    # Thử nguồn 1: VCI 
    try:
        df = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date, source='VCI')
        if df is not None and not df.empty:
            return df
    except:
        pass

    # Thử nguồn 2: Nguồn mặc định hệ thống tự điều phối (DNSE/Cafef) nếu VCI lỗi cấu trúc
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
            # Tự động viết thường toàn bộ tên cột để đồng bộ hóa đa nguồn
            df.columns = [col.lower() for col in df.columns]
            
            # Định nghĩa bảng ánh xạ đổi tên cột chuẩn hóa để tính toán chỉ báo
            rename_dict = {
                'tradingdate': 'time', 'date': 'time',
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 
                'volume': 'volume', 'vol': 'volume'
            }
            df = df.rename(columns=rename_dict)
            
            # Ép kiểu dữ liệu các cột kỹ thuật về dạng số float để tránh lỗi thư viện 'ta'
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Loại bỏ các hàng trống (NaN) nếu có
            df = df.dropna(subset=['close'])
            df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
            
            # --- 3. TÍNH TOÁN CÁC CHỈ BÁO KỸ THUẬT (Thư viện 'ta') ---
            # Tính RSI (14)
            rsi_series = RSIIndicator(close=df['close'], window=14).rsi()
            df['RSI'] = rsi_series
            
            # Tính Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            
            # Tính MACD
            macd_obj = MACD(close=df['close'])
            df['MACD'] = macd_obj.macd()
            df['MACD_Signal'] = macd_obj.macd_signal()
            df['MACD_Diff'] = macd_obj.macd_diff()
            
            # Lấy thông số phiên cuối cùng
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
            
            # --- 4. VẼ BIỂU ĐỒ NẾN KỸ THUẬT (Thư viện 'plotly') ---
            fig = go.Figure()
            # Vẽ nến Candlestick
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Giá nến'))
            # Vẽ đường Bollinger Bands
            fig.add_trace(go.Scatter(x=df['time'], y=df['BB_High'], line=dict(color='rgba(250, 0, 0, 0.4)', width=1), name='BB Upper'))
            fig.add_trace(go.Scatter(x=df['time'], y=df['BB_Low'], line=dict(color='rgba(0, 250, 0, 0.4)', width=1), name='BB Lower'))
            
            fig.update_layout(title=f"Biểu đồ kỹ thuật mã {ticker}", xaxis_rangeslider_visible=False, template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị bảng dữ liệu thô kết hợp chỉ báo
            st.write("Bảng dữ liệu 5 phiên gần nhất tích hợp chỉ báo:")
            st.dataframe(df.tail(5)[['time', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD']], hide_index=True, use_container_width=True)
            
            # --- 5. TÍCH HỢP ĐẨY DỮ LIỆU ĐA CHỈ BÁO VÀO AI GEMINI ---
            if api_key:
                st.markdown("---")
                st.subheader(f"🤖 Báo Cáo Khuyến Nghị Vùng Giá Từ Trợ Lý AI")
                
                # Trích xuất bối cảnh giàu dữ liệu chỉ báo để AI không phán đoán bừa
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
                    response = model.generate_content(prompt)
                    st.info(response.text)
            else:
                st.info("💡 Vui lòng nhập Gemini API Key ở thanh bên trái để nhận báo cáo khuyến nghị điểm mua/bán tự động từ Trợ lý AI.")
        else:
            st.error(f"Không thể tải dữ liệu cho mã {ticker}. Hệ thống tự động kiểm tra lại cổng kết nối, vui lòng bấm thử lại.")
