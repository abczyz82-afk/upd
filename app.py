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
    """Fibonacci Retracement từ High/Low trong lookback phiên gần nhất."""
    window   = df.tail(lookback)
    high_val = window["high"].max()
    low_val  = window["low"].min()
    diff     = high_val - low_val
    return {
        "0.0%":   high_val,
        "23.6%":  high_val - 0.236 * diff,
        "38.2%":  high_val - 0.382 * diff,
        "50.0%":  high_val - 0.500 * diff,
        "61.8%":  high_val - 0.618 * diff,
        "78.6%":  high_val - 0.786 * diff,
        "100.0%": low_val,
    }


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
        "model": "claude-sonnet-4-5",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_gemini(api_key: str, prompt: str) -> str:
    from google import genai  # type: ignore

    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    return response.text


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
                   line=dict(color="#00BFFF", width=1.5), name="Tenkan-sen"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["Kijun"],
                   line=dict(color="#FF6347", width=1.5), name="Kijun-sen"),
        row=1, col=1,
    )
    # Kumo Cloud: fill between SpanA and SpanB
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["SpanA"],
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, name="SpanA_base",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["SpanB"],
            fill="tonexty",
            fillcolor="rgba(100,200,100,0.12)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="Kumo Cloud",
        ),
        row=1, col=1,
    )

    # ── Fibonacci Retracement ─────────────────────────────────────────────
    fib_palette = {
        "0.0%":   "#888888",
        "23.6%":  "#7986CB",
        "38.2%":  "#FFA726",
        "50.0%":  "#EF5350",
        "61.8%":  "#FFA726",
        "78.6%":  "#7986CB",
        "100.0%": "#888888",
    }
    x_range = [df["time"].iloc[0], df["time"].iloc[-1]]
    for label, price in fib_levels.items():
        fig.add_trace(
            go.Scatter(
                x=x_range, y=[price, price],
                mode="lines",
                line=dict(color=fib_palette[label], width=1, dash="dot"),
                name=f"Fib {label}",
                text=f"{label} ({price:,.0f})",
                hoverinfo="text",
                showlegend=True,
            ),
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
        height=950,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    st.plotly_chart(fig, use_container_width=True)

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
        use_container_width=True,
    )

    # ── Fibonacci table ───────────────────────────────────────────────────
    with st.expander("📐 Xem bảng mức Fibonacci Retracement"):
        fib_df = pd.DataFrame(
            [{"Mức Fibonacci": k, "Giá (đ)": f"{v:,.0f}"} for k, v in fib_levels.items()]
        )
        st.dataframe(fib_df, hide_index=True, use_container_width=True)

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
                st.error(f"❌ Lỗi gọi AI: {e}")
                st.info(
                    "💡 Gợi ý: Kiểm tra lại API Key và đảm bảo bạn có quyền truy cập model. "
                    "Claude dùng key từ console.anthropic.com, Gemini dùng key từ aistudio.google.com."
                )
