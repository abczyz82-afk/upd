import streamlit as st
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

# Cấu hình trang Streamlit
st.set_page_config(page_title="VN30F1M Dashboard", layout="wide")
st.title("Bảng Giá & Nhật Ký Giao Dịch Phái Sinh (VN30F1M)")

# Khởi tạo kết nối với TradingView (chế độ Guest)
@st.cache_data(ttl=60) # Cập nhật dữ liệu mỗi 60 giây
def get_vn30f1m_data():
    try:
        tv = TvDatafeed()
        # Lấy dữ liệu VN30F1M từ sàn VNINDEX trên TradingView
        df = tv.get_hist(symbol='VN30F1M', exchange='VNINDEX', interval=Interval.in_1_minute, n_bars=20)
        return df
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu từ TradingView: {e}")
        return None

# 1. BẢNG DỮ LIỆU VN30F1M
st.subheader("Bảng Dữ Liệu VN30F1M (Nguồn: TradingView)")
vn30f1m_df = get_vn30f1m_data()

if vn30f1m_df is not None and not vn30f1m_df.empty:
    # Reset index để hiển thị Datetime đẹp hơn
    vn30f1m_df = vn30f1m_df.reset_index()
    vn30f1m_df.rename(columns={'datetime': 'Thời gian', 'open': 'Mở cửa', 'high': 'Cao nhất', 
                               'low': 'Thấp nhất', 'close': 'Đóng cửa', 'volume': 'Khối lượng'}, inplace=True)
    st.dataframe(vn30f1m_df.tail(10), use_container_width=True)
else:
    st.warning("Đang chờ tải dữ liệu...")

st.markdown("---")

# 2. BẢNG NHẬT KÝ GIAO DỊCH (LỆNH ĐÃ THỰC HIỆN)
st.subheader("Bảng Lịch Sử Lệnh Đã Thực Hiện")

# Dữ liệu mô phỏng các lệnh đã đánh
trade_data = {
    "Lệnh": [1, 2, 3, 4, 5],
    "Vị thế": ["Long", "Short", "Long", "Short", "Long"],
    "Điểm vào": [1250.5, 1260.0, 1255.0, 1270.5, 1265.0],
    "Điểm ra": [1258.0, 1255.0, 1252.0, 1260.0, 1270.0],
    "Trạng thái": ["Chốt lời", "Chốt lời", "Cắt lỗ", "Chốt lời", "Chốt lời"],
    "Lãi/Lỗ (Điểm)": [7.5, 5.0, -3.0, 10.5, 5.0]
}

trade_df = pd.DataFrame(trade_data)

# Định dạng màu sắc cho cột Lãi/Lỗ
def color_profit_loss(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
    return f'color: {color}'

styled_trade_df = trade_df.style.map(color_profit_loss, subset=['Lãi/Lỗ (Điểm)'])

# Hiển thị bảng
st.dataframe(styled_trade_df, use_container_width=True)

# Hiển thị tổng kết ngắn gọn
total_profit = trade_df['Lãi/Lỗ (Điểm)'].sum()
st.metric(label="Tổng Lãi/Lỗ Trạng Thái (Điểm)", value=f"{total_profit} điểm", delta=total_profit)
