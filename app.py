import streamlit as st
import pandas as pd

# Cấu hình trang gọn nhẹ
st.set_page_config(page_title="Nhật Ký Giao Dịch", layout="centered")
st.title("Bảng Nhật Ký Giao Dịch Phái Sinh")

# Dữ liệu mô phỏng (Bạn có thể thay thế bằng file CSV hoặc Database sau này)
trade_data = {
    "Lệnh": [1, 2, 3, 4, 5, 6, 7],
    "Vị thế": ["Long", "Short", "Long", "Short", "Long", "Short", "Long"],
    "Điểm vào": [1250.5, 1260.0, 1255.0, 1270.5, 1265.0, 1280.0, 1275.5],
    "Điểm ra": [1258.0, 1255.0, 1252.0, 1260.0, 1270.0, 1283.0, 1285.0],
    "Trạng thái": ["Chốt lời", "Chốt lời", "Cắt lỗ", "Chốt lời", "Chốt lời", "Cắt lỗ", "Chốt lời"],
    "Lãi/Lỗ (Điểm)": [7.5, 5.0, -3.0, 10.5, 5.0, -3.0, 9.5]
}

df = pd.DataFrame(trade_data)

# Hàm tô màu xanh/đỏ cho cột Lãi/Lỗ để dễ nhìn
def color_profit_loss(val):
    if val > 0:
        color = '#00C853' # Xanh lá
    elif val < 0:
        color = '#FF1744' # Đỏ
    else:
        color = 'gray'
    return f'color: {color}; font-weight: bold'

# Áp dụng màu sắc vào bảng
styled_df = df.style.map(color_profit_loss, subset=['Lãi/Lỗ (Điểm)'])

# Hiển thị bảng toàn màn hình, ẩn cột số thứ tự mặc định của pandas
st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Hiển thị tổng kết nhanh
st.divider()
total_profit = df['Lãi/Lỗ (Điểm)'].sum()
st.metric(label="Tổng Lãi/Lỗ (Điểm)", value=f"{total_profit} điểm", delta=total_profit)
