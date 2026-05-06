import streamlit as st
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime

# Cấu hình trang
st.set_page_config(page_title="VN30F1M Tín Hiệu Real-time", layout="wide")
st.title("Hệ Thống Báo Tín Hiệu Phái Sinh VN30F1M")

# Hàm lấy dữ liệu (Làm mới mỗi 30 giây để cập nhật realtime)
@st.cache_data(ttl=30)
def get_live_data():
    try:
        tv = TvDatafeed()
        # Lấy 150 nến 1 phút để đủ tính toán EMA50 và phân vị BB lịch sử
        df = tv.get_hist(symbol='VN30F1M', exchange='VNINDEX', interval=Interval.in_1_minute, n_bars=150)
        if df is not None and not df.empty:
            df = df.reset_index()
            df.rename(columns={'datetime': 'Thời gian', 'open': 'Mở', 'high': 'Cao', 'low': 'Thấp', 'close': 'Đóng', 'volume': 'Khối lượng'}, inplace=True)
            return df
        return None
    except Exception as e:
        st.error(f"Lỗi kết nối dữ liệu: {e}")
        return None

# Nút Refresh thủ công
col_title, col_btn = st.columns([8, 1])
with col_btn:
    if st.button("🔄 Cập Nhật Giá"):
        st.cache_data.clear()

df = get_live_data()

if df is not None:
    # --- TÍNH TOÁN CHỈ BÁO KỸ THUẬT ---
    # EMA
    df['EMA9'] = EMAIndicator(close=df['Đóng'], window=9).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['Đóng'], window=21).ema_indicator()
    df['EMA50'] = EMAIndicator(close=df['Đóng'], window=50).ema_indicator()

    # RSI & MACD
    df['RSI'] = RSIIndicator(close=df['Đóng'], window=14).rsi()
    macd = MACD(close=df['Đóng'])
    df['MACD_Hist'] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df['Đóng'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()

    # Volume & ADX
    df['Vol_MA'] = df['Khối lượng'].rolling(window=20).mean()
    adx = ADXIndicator(high=df['Cao'], low=df['Thấp'], close=df['Đóng'], window=14)
    df['ADX'] = adx.adx()
    df['DI+'] = adx.adx_pos()
    df['DI-'] = adx.adx_neg()

    # --- ĐÁNH GIÁ REGIME ---
    cond_regime = [
        (df['ADX'] < 22),
        (df['DI+'] > df['DI-']) & (df['ADX'] >= 22),
        (df['DI-'] > df['DI+']) & (df['ADX'] >= 22)
    ]
    choices_regime = ['🔄 SIDEWAY', '🚀 UPTREND', '💥 DOWNTREND']
    df['Regime'] = np.select(cond_regime, default='KHÔNG RÕ', condlist=cond_regime, choicelist=choices_regime)

    # --- QUÉT TÍN HIỆU ---
    df['Tín Hiệu'] = ''
    
    # Tính toán percentile 15% của BB_Width để xét Squeeze
    p15_bbw = df['BB_Width'].quantile(0.15)

    for i in range(1, len(df)):
        signals = []
        
        # Điều kiện
        ema_cross_up = df['EMA9'].iloc[i] > df['EMA21'].iloc[i] and df['EMA9'].iloc[i-1] <= df['EMA21'].iloc[i-1]
        ema_cross_down = df['EMA9'].iloc[i] < df['EMA21'].iloc[i] and df['EMA9'].iloc[i-1] >= df['EMA21'].iloc[i-1]
        ema_bull = df['EMA9'].iloc[i] > df['EMA21'].iloc[i] > df['EMA50'].iloc[i] and df['Regime'].iloc[i] == '🚀 UPTREND'
        
        rsi_os = df['RSI'].iloc[i] < 30
        rsi_ob = df['RSI'].iloc[i] > 70
        macd_up = df['MACD_Hist'].iloc[i] > 0 and df['MACD_Hist'].iloc[i-1] <= 0
        
        bb_breakup = df['Đóng'].iloc[i] > df['BB_Upper'].iloc[i]
        bb_bounce = df['Thấp'].iloc[i] <= df['BB_Lower'].iloc[i] and df['Đóng'].iloc[i] > df['Mở'].iloc[i]
        bb_squeeze = df['BB_Width'].iloc[i] < p15_bbw
        vol_spike = df['Khối lượng'].iloc[i] > (2 * df['Vol_MA'].iloc[i])

        # Gán nhãn
        if ema_cross_up: signals.append("🟢 EMA 9x21 Cắt Lên (BUY)")
        if ema_cross_down: signals.append("🔴 EMA 9x21 Cắt Xuống (SELL)")
        if rsi_os: signals.append("💎 RSI Quá Bán (BUY)")
        if rsi_ob: signals.append("🔥 RSI Quá Mua (SELL)")
        if macd_up: signals.append("📈 MACD Cắt Lên (BUY)")
        if bb_breakup: signals.append("🚀 BB Break Up (BUY)")
        if bb_bounce: signals.append("🟢 BB Bounce Up (BUY)")
        if bb_squeeze: signals.append("⚡ BB Squeeze (WATCH)")
        if vol_spike: signals.append("📊 Volume Spike (WATCH)")
        # Lọc bớt tín hiệu EMA xếp Bull để tránh spam mỗi nến, chỉ ghi nhận nếu nến trước chưa có
        if ema_bull and not (df['EMA9'].iloc[i-1] > df['EMA21'].iloc[i-1] > df['EMA50'].iloc[i-1]):
            signals.append("🟢 EMA Xếp BULL (BUY)")

        if signals:
            df.at[i, 'Tín Hiệu'] = " | ".join(signals)

    # Dữ liệu nến hiện tại
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = latest['Đóng'] - prev['Đóng']

    # --- HIỂN THỊ DASHBOARD ---
    st.markdown("### Trạng Thái Thị Trường Hiện Tại")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Giá VN30F1M", f"{latest['Đóng']:.1f}", f"{price_change:.1f}")
    m2.metric("Regime", latest['Regime'])
    m3.metric("RSI (14)", f"{latest['RSI']:.1f}")
    m4.metric("ADX", f"{latest['ADX']:.1f}")

    st.markdown("---")

    col_signals, col_rules = st.columns([6, 4])
    
    with col_signals:
        st.subheader("🔔 Các Tín Hiệu Gần Nhất (Real-time)")
        # Lọc ra các nến có sinh ra tín hiệu để hiển thị
        signal_df = df[df['Tín Hiệu'] != ''][['Thời gian', 'Đóng', 'Regime', 'Tín Hiệu']].tail(10)
        
        # Định dạng thời gian cho đẹp
        signal_df['Thời gian'] = signal_df['Thời gian'].dt.strftime('%H:%M:%S %d/%m/%Y')
        
        if not signal_df.empty:
            st.dataframe(signal_df.sort_values(by='Thời gian', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có tín hiệu nào xuất hiện trong khoảng thời gian này.")

    with col_rules:
        st.subheader("📖 Chiến lược theo Regime")
        st.markdown("""
        - 🔄 **SIDEWAY** (ADX<22): Canh BB Lower mua, BB Upper bán. SL 2-3 điểm.
        - 🚀 **UPTREND** (DI+>DI-): Chỉ LONG, pullback về EMA21. Ride trend.
        - 💥 **DOWNTREND** (DI->DI+): Chỉ SHORT, hồi về EMA21. Ride trend.
        - ⚡ **BB Squeeze**: Chờ Vol Spike xác nhận hướng → vào lệnh.
        """)

else:
    st.warning("Đang tải dữ liệu từ TradingView...")
