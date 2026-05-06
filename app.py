import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Cấu hình trang Streamlit
st.set_page_config(page_title="VN30F1M Trading System", layout="wide")
st.title("Hệ Thống Tín Hiệu Phái Sinh VN30F1M")

# Hàm lấy dữ liệu
@st.cache_data(ttl=60)
def get_vn30f1m_data():
    try:
        tv = TvDatafeed()
        # Lấy 200 nến để đủ dữ liệu tính toán MA50 và các chỉ báo khác
        df = tv.get_hist(symbol='VN30F1M', exchange='VNINDEX', interval=Interval.in_1_minute, n_bars=200)
        if df is not None and not df.empty:
            df = df.reset_index()
            df.rename(columns={'datetime': 'Time'}, inplace=True)
            return df
        return None
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {e}")
        return None

# Hàm tính toán chỉ báo và tín hiệu
def apply_trading_logic(df):
    # 1. EMAs
    df['EMA9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['EMA50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()

    # 2. RSI & MACD
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['MACD_Hist'] = macd.macd_diff()

    # 3. Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()

    # 4. Volume & ADX (DMI)
    df['Vol_MA'] = df['volume'].rolling(window=20).mean()
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx.adx()
    df['DI+'] = adx.adx_pos()
    df['DI-'] = adx.adx_neg()

    # Phân loại Regime
    conditions_regime = [
        (df['ADX'] < 22),
        (df['DI+'] > df['DI-']) & (df['ADX'] >= 22),
        (df['DI-'] > df['DI+']) & (df['ADX'] >= 22)
    ]
    choices_regime = ['🔄 SIDEWAY', '🚀 UPTREND', '💥 DOWNTREND']
    df['Regime'] = np.select(conditions_regime, default='KHÔNG RÕ', condlist=conditions_regime, choicelist=choices_regime)

    # Tính toán các điều kiện tín hiệu (Signal)
    df['Signal'] = ''
    df['Marker'] = ''

    # Tạo các mask logic
    ema_cross_up = (df['EMA9'] > df['EMA21']) & (df['EMA9'].shift(1) <= df['EMA21'].shift(1))
    ema_cross_down = (df['EMA9'] < df['EMA21']) & (df['EMA9'].shift(1) >= df['EMA21'].shift(1))
    rsi_os = df['RSI'] < 30
    rsi_ob = df['RSI'] > 70
    macd_cross_up = (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
    bb_breakup = df['close'] > df['BB_Upper']
    bb_bounce = (df['low'] <= df['BB_Lower']) & (df['close'] > df['open'])
    
    # Squeeze & Vol Spike
    p15_bbw = df['BB_Width'].quantile(0.15)
    bb_squeeze = df['BB_Width'] < p15_bbw
    vol_spike = df['volume'] > (2 * df['Vol_MA'])

    # Gán Marker và Signal
    for i in range(1, len(df)):
        signals = []
        marker = ''
        
        # BUY Signals (▲)
        if ema_cross_up.iloc[i]: signals.append("EMA 9x21 Cắt Lên (BUY)")
        if df['EMA9'].iloc[i] > df['EMA21'].iloc[i] > df['EMA50'].iloc[i] and df['Regime'].iloc[i] == '🚀 UPTREND':
             # Tránh spam tín hiệu này mỗi nến, chỉ báo khi thỏa mãn
             pass 
        if rsi_os.iloc[i]: signals.append("RSI Quá Bán (BUY)")
        if macd_cross_up.iloc[i]: signals.append("MACD Cắt Lên (BUY)")
        if bb_breakup.iloc[i]: signals.append("BB Break Up (BUY)")
        if bb_bounce.iloc[i]: signals.append("BB Bounce Up (BUY)")
        
        # SELL Signals (▼)
        if ema_cross_down.iloc[i]: signals.append("EMA 9x21 Cắt Xuống (SELL)")
        if rsi_ob.iloc[i]: signals.append("RSI Quá Mua (SELL)")

        # WATCH Signals (◆)
        if bb_squeeze.iloc[i]: signals.append("BB Squeeze (WATCH)")
        if vol_spike.iloc[i]: signals.append("Vol Spike (WATCH)")

        if signals:
            df.at[i, 'Signal'] = " | ".join(signals)
            if any("BUY" in s for s in signals):
                df.at[i, 'Marker'] = '▲'
            elif any("SELL" in s for s in signals):
                df.at[i, 'Marker'] = '▼'
            elif any("WATCH" in s for s in signals):
                df.at[i, 'Marker'] = '◆'

    return df

# Xử lý luồng hiển thị
df = get_vn30f1m_data()

if df is not None:
    df = apply_trading_logic(df)
    latest = df.iloc[-1]

    # Layout Dashboard
    col1, col2, col3 = st.columns(3)
    col1.metric("Giá Hiện Tại", f"{latest['close']:.1f}", f"{latest['close'] - df['close'].iloc[-2]:.1f}")
    col2.metric("Trạng Thái Thị Trường (Regime)", latest['Regime'])
    col3.metric("RSI Hiện Tại", f"{latest['RSI']:.2f}")

    # Bảng Quy Tắc Chiến Lược
    with st.expander("📖 Xem Quy Tắc Chiến Lược & Regime", expanded=False):
        st.markdown("""
        | Tín hiệu | Điều kiện | Hành động |
        |-----------|-----------|-----------|
        | 🟢 **EMA 9×21 Cắt Lên** | EMA9 vừa cắt lên EMA21 | BUY |
        | 🔴 **EMA 9×21 Cắt Xuống** | EMA9 vừa cắt xuống EMA21 | SELL |
        | 🟢 **EMA Xếp BULL** | EMA9 > EMA21 > EMA50 + UPTREND | BUY |
        | 💎 **RSI Quá Bán** | RSI < 30 | BUY |
        | 🔥 **RSI Quá Mua** | RSI > 70 | SELL |
        | 📈 **MACD Cắt Lên** | Histogram âm → dương | BUY |
        | 🚀 **BB Break Up** | Giá > BB Upper | BUY momentum |
        | 🟢 **BB Bounce Up** | Giá hồi từ BB Lower | BUY reversion |
        | ⚡ **BB Squeeze** | BB Width < p15 lịch sử | WATCH – chờ breakout |
        | 📊 **Volume Spike** | Vol > 2× Vol MA | WATCH – xác nhận tín hiệu |
        
        **Marker trên chart:** ▲ = BUY signal · ▼ = SELL signal · ◆ = WATCH
        
        **Chiến lược theo regime:**
        - 🔄 **SIDEWAY** (ADX<22): Canh BB Lower mua, BB Upper bán. SL 2-3 điểm.
        - 🚀 **UPTREND** (DI+>DI-): Chỉ LONG, pullback về EMA21. Ride trend.
        - 💥 **DOWNTREND** (DI->DI+): Chỉ SHORT, hồi về EMA21. Ride trend.
        - ⚡ **BB Squeeze**: Chờ Vol Spike xác nhận hướng → vào lệnh.
        """)

    # Vẽ Biểu đồ Nến Nhật có gắn Marker
    st.subheader("Biểu Đồ Kỹ Thuật (1 Phút)")
    fig = go.Figure()
    
    # Nến
    fig.add_trace(go.Candlestick(x=df['Time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    
    # Thêm EMA
    fig.add_trace(go.Scatter(x=df['Time'], y=df['EMA9'], line=dict(color='blue', width=1), name='EMA9'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['EMA21'], line=dict(color='orange', width=1), name='EMA21'))
    
    # Lọc các điểm có Marker để vẽ lên chart
    buy_signals = df[df['Marker'] == '▲']
    sell_signals = df[df['Marker'] == '▼']
    watch_signals = df[df['Marker'] == '◆']

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals['Time'], y=buy_signals['low'] - 1, mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='BUY Signal'))
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals['Time'], y=sell_signals['high'] + 1, mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='SELL Signal'))
    if not watch_signals.empty:
        fig.add_trace(go.Scatter(x=watch_signals['Time'], y=watch_signals['close'], mode='markers', marker=dict(symbol='diamond', color='yellow', size=10), name='WATCH Signal'))

    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0), template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Bảng chi tiết tín hiệu gần đây
    st.subheader("Lịch Sử Tín Hiệu Gần Nhất")
    signal_df = df[df['Signal'] != ''][['Time', 'close', 'Regime', 'Signal', 'Marker']].tail(15)
    st.dataframe(signal_df.sort_values(by='Time', ascending=False), use_container_width=True)

else:
    st.warning("Đang tải dữ liệu...")
