import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from collections import defaultdict

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="AI Trading Dashboard — ICT · VSA · PA · Smart Money")

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE STOCK UNIVERSE (HOSE + HNX + UPCOM hardcoded fallback)
# ─────────────────────────────────────────────────────────────────────────────
VN30_TICKERS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","NVL","PDR","PLX","PNJ","POW","SAB","SSI",
    "STB","TCB","TPB","VCB","VHM","VIC","VJC","VNM","VPB","VRE",
]

# Danh sách cổ phiếu toàn thị trường (HOSE + HNX + UPCOM)
_FULL_STOCK_UNIVERSE = sorted(set([
    # ── VN30 ──
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN",
    "MWG","NVL","PDR","PLX","PNJ","POW","SAB","SSI","STB","TCB","TPB","VCB",
    "VHM","VIC","VJC","VNM","VPB","VRE",
    # ── HOSE ──
    "AAA","AAM","AAT","ABB","ABS","ABT","ACG","ACL","ACM","ACT","ADC","AGG",
    "AGM","AGR","AHP","AIS","ALP","ALT","AMD","AMV","ANT","ANV","APG","APH",
    "APP","APT","ASG","ASM","ASP","BAF","BAL","BAX","BBC","BCG","BDB","BFC",
    "BHN","BIC","BII","BKC","BLN","BMC","BMP","BNA","BSI","BSR","BTP","BTV",
    "BVB","BVG","BVL","BWE","C4G","CAP","CAV","CCL","CEO","CHP","CII","CKG",
    "CMC","CMG","CMV","CMX","CNG","CNT","COM","CRC","CSC","CSM","CSV","CTC",
    "CTD","CTF","CTI","CTR","CTS","CVT","D2D","DAG","DAH","DAT","DBC","DCG",
    "DCL","DCM","DIG","DLG","DLT","DMC","DPG","DPM","DPS","DQC","DRC","DRH",
    "DSC","DTA","DTC","DXG","DXS","EIB","ELC","EMC","EVE","EVF","EVG","EVS",
    "FIR","FIT","FLC","FMC","FRT","GDT","GEE","GEG","GEX","GMD","GPH","GRI",
    "GSP","GTD","HAG","HAH","HAP","HAR","HAS","HAX","HBC","HCM","HDC","HDG",
    "HHP","HHS","HII","HLD","HMC","HNG","HPT","HPX","HQC","HRC","HSG","HU1",
    "HU3","HU6","HVN","HVT","ICT","IDC","IDI","IJC","IMP","INN","IPA","ITA",
    "ITC","JVC","KBC","KDC","KDH","KHD","KLF","KLS","KPF","KSB","KSH","KST",
    "LAF","LAS","LBM","LCG","LDG","LEC","LGC","LHG","LIX","LMH","LSS","LTC",
    "MCG","MCH","MCM","MCP","MDC","MDG","MIG","MIM","NAB","NAF","NAV","NBC",
    "NBT","NET","NHA","NHH","NKG","NLG","NPT","NRC","NSC","NTB","NTC","NVT",
    "OIL","OCB","OGC","OMH","PAC","PAN","PCC","PCG","PCT","PDN","PET","PGB",
    "PGD","PHR","PIC","PIT","PLC","PLX","PNJ","POW","PRC","PRE","PSH","PTC",
    "PVD","PVI","PVS","PVT","QCG","QNS","RAL","RCL","REE","ROS","SAF","SAM",
    "SAV","SBA","SC5","SCD","SCG","SCR","SDC","SDG","SDN","SFC","SFG","SFI",
    "SGN","SGR","SGT","SHB","SHI","SHP","SHS","SII","SKG","SLG","SMB","SMC",
    "SNG","SPM","SRC","SRF","SRT","SSC","SSF","STA","STG","STK","STP","SVC",
    "SVI","SVN","SVT","SZC","TBC","TCH","TCM","TCO","TDC","TDG","TDH","TDM",
    "TDN","TDT","TEG","TGG","THD","TIE","TIG","TIP","TIX","TJC","TLG","TLH",
    "TMS","TNA","TNC","TNH","TNI","TNT","TON","TPC","TPL","TRA","TRC","TSC",
    "TTA","TTF","TTH","TTP","TV2","TVB","TVD","TVS","TVT","UDC","UIC","VAB",
    "VCF","VCG","VCI","VDL","VDS","VGC","VGG","VGI","VGS","VHC","VHG","VIB",
    "VID","VIE","VIG","VIM","VIP","VIS","VIX","VKC","VKD","VLB","VMP","VMR",
    "VNA","VNC","VND","VNE","VNF","VNG","VNI","VNL","VNR","VNS","VNX","VOC",
    "VOS","VPC","VPD","VPG","VPH","VPK","VPL","VPS","VRC","VSC","VSD","VSH",
    "VSI","VST","VTC","VTG","VTK","VTL","VTM","VTO","VTS","VTV","VXB","WCS",
    "WSS","YBC","YEG",
    # ── Midcap / Smallcap thêm ──
    "AGF","AGL","AGX","BCA","BCC","BCI","BCM","BDB","BDT","BHT","BLF","BLT",
    "BMT","BNW","BPC","BSC","BTT","BUI","BVN","BVS","CAD","CBD","CCI","CDN",
    "CEL","CEN","CHC","CJC","CKG","CML","CNA","CNC","CNN","CTN","CTW","D11",
    "DAD","DAL","DAP","DBD","DBT","DDD","DDG","DDL","DDV","DGC","DGW","DHA",
    "DHB","DHC","DHG","DHT","DIC","DL1","DNC","DND","DNH","DNL","DNM","DNP",
    "DNT","DNW","DPS","DPT","DRC","DSN","DST","DTB","DTE","DTG","DTI","DTL",
    "DTM","DTP","DTS","DTT","DTV","DUS","DVD","DVG","DVN","DWT","DXL","DZM",
    "EBS","EFI","EIC","EMG","EMS","EPH","ETC","FBC","FCN","FDC","FGL","FHG",
    "FID","FIV","FNA","FRC","FRM","GAB","GIC","GIL","GKM","GLT","GMH","GPC",
    "GPN","GRS","HAD","HAI","HCC","HEV","HFX","HGM","HIO","HLG","HLT","HLY",
    "HNA","HNI","HOT","HPT","HSL","HT1","HTC","HTI","HTL","HTM","HTN","HTP",
    "HTV","HXB","HYI","ICF","ICG","IDJ","IDV","IHK","ILA","ILB","INA","IPC",
    "IRC","ITD","ITQ","ITW","KGM","KHP","KKC","KLB","KLS","KMR","KSF","KTT",
    "KVC","L10","L14","L18","L35","L43","L44","L61","L62","LAF","LCD","LCM",
    "LEC","LIC","LIG","LM3","LMI","LNC","LQN","LUT","MAC","MAF","MBA","MBS",
    "MCC","MEC","MHC","MKP","MML","MNC","MPT","MRF","MST","MTG","MTP","MVB",
    "MVN","NAG","NAP","NAT","NAW","NBB","NCT","NEL","NFL","NHT","NLC","NLS",
    "NMT","NNC","NQB","NST","NTA","NTH","NTP","NTT","NTW","NUE","NVB","OHL",
    "OPC","PAB","PBC","PFC","PGC","PGI","PGS","PGV","PHC","PHN","PIV","PKG",
    "PMG","PMP","PNC","PNG","POB","POV","PPC","PPI","PPP","PTG","PTL","PTS",
    "PTX","PVB","PVC","PVG","PVL","PVV","PXL","PXS","PXT","QBC","QHD","QST",
    "QTC","RIC","RLC","RPH","RTC","S4A","S55","S96","S99","SAP","SBB","SBT",
    "SDT","SGD","SGI","SGO","SIC","SK","SLA","SRC","SRF","SSC","SSF","TAC",
    "TAR","TAS","TAW","TCL","TCR","TCT","TCW","TET","TGN","THL","THP","THT",
    "TID","TJC","TKC","TKU","TON","TPH","TPP","TQN","TRS","TST","TTB","TTE",
    "TTS","TV1","TV3","TV4","TVH","UNI","UPC","V11","V12","VAC","VCA","VCR",
    "VDB","VDC","VFG","VFR","VGP","VGT","VHD","VKB","VLP","VMC","VNB","VNX",
    "VOC","VPK","VSD","VSP","VTJ","VXP","BCE","CCI","DPG","FPT","GMD","HAH",
    "HCM","IDC","KDH","MBB","NLG","PVS","REE","SZC","TCB","VCI","VIX","VND",
]));

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Cấu hình Hệ thống")
ai_provider = st.sidebar.radio("Chọn nhà cung cấp AI:", ["🟠 Claude (Anthropic)", "🔵 Gemini (Google)"])
api_key = st.sidebar.text_input("Nhập API Key:", type="password")
if not api_key:
    st.sidebar.warning("⚠️ Nhập API Key để bật Trợ lý AI.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Phương pháp phân tích:**\n"
    "- 🔬 **ICT** — Order Block · FVG · BOS/CHoCH · OTE\n"
    "- 📊 **VSA** — Stopping Vol · No Demand · Climax\n"
    "- 🕯️ **Price Action** — S/R · Inside Bar · Trend\n"
    "- 💧 **Overflow** — Vol Overflow · Price Overflow · Gap\n"
    "- 🐋 **Smart Money** — Wyckoff · Stop Hunt · A/D\n"
    "- 📈 **Classic** — RSI · MACD · ADX · EMA · Ichimoku\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER & TABS
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 AI Trading Dashboard — ICT · VSA · PA · Smart Money")
st.markdown(
    "Tích hợp **ICT · VSA · Price Action · Overflow · Smart Money · ADX · EMA · Ichimoku · Fibonacci** "
    "— Phân tích bởi **Claude** hoặc **Gemini**."
)
_tab1, _tab2 = st.tabs(["📊 Phân Tích Đơn Lẻ", "🔍 Quét Toàn Thị Trường"])

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_clean_stock_data(symbol: str) -> pd.DataFrame | None:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=220)

    if symbol.startswith("VN30F"):
        try:
            url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/derivative"
                   f"?from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
                   f"&symbol={symbol}&resolution=1D")
            resp = requests.get(url, timeout=10).json()
            if "t" in resp and len(resp["t"]) > 0:
                df = pd.DataFrame({"time": pd.to_datetime(resp["t"], unit="s"),
                                   "open": resp["o"], "high": resp["h"],
                                   "low": resp["l"], "close": resp["c"], "volume": resp["v"]})
                df["time"] = df["time"].dt.strftime("%Y-%m-%d")
                return df
        except Exception:
            pass

    # Source 1: entrade / DNSE
    try:
        url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
               f"?from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
               f"&symbol={symbol}&resolution=1D")
        resp = requests.get(url, timeout=10).json()
        if "t" in resp and len(resp.get("t", [])) > 0:
            df = pd.DataFrame({"time": pd.to_datetime(resp["t"], unit="s"),
                               "open": resp["o"], "high": resp["h"],
                               "low": resp["l"], "close": resp["c"], "volume": resp["v"]})
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
            # entrade trả về giá đơn vị nghìn đồng → nhân 1000
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 1000
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            return df
    except Exception:
        pass

    # Source 2: TCBS
    try:
        url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/{symbol}/bars-long-term?resolution=D&type=stock&to={(int(end_date.timestamp()))}&countBack=200"
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=12).json()
        bars = resp.get("data", [])
        if bars:
            df = pd.DataFrame(bars)
            df = df.rename(columns={"tradingDate": "time", "open": "open", "high": "high",
                                    "low": "low", "close": "close", "volume": "volume"})
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
            # TCBS trả về giá đơn vị nghìn đồng → nhân 1000
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce") * 1000
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
            return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        pass

    # Source 3: SSI iBoard
    try:
        str_start = start_date.strftime("%Y-%m-%d")
        str_end   = end_date.strftime("%Y-%m-%d")
        url = (f"https://iboard-query.ssi.com.vn/v2/stock/historical-price"
               f"?symbol={symbol}&fromDate={str_start}&toDate={str_end}&offset=0&limit=200")
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10).json()
        data = resp.get("data", {}).get("items", [])
        if data:
            df = pd.DataFrame(data)
            df = df.rename(columns={"tradingDate": "time", "openPrice": "open", "highPrice": "high",
                                    "lowPrice": "low", "closePrice": "close", "totalMatchVolume": "volume"})
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
            return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        pass

    st.error(f"Không thể lấy dữ liệu cho mã **{symbol}**.")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def build_scan_universe() -> tuple:
    """
    Lấy danh sách cổ phiếu từ nhiều nguồn + hardcoded fallback.
    Không lọc theo giá hay khối lượng.
    """
    universe = set(_FULL_STOCK_UNIVERSE)
    source_note = f"Danh sách cố định ({len(universe)} mã)"

    # Try TCBS listing
    try:
        url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/all"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).json()
        items = resp if isinstance(resp, list) else resp.get("listStock", resp.get("data", []))
        added = 0
        for item in (items or []):
            sym = str(item.get("ticker", item.get("symbol", item.get("code", "")))).upper().strip()
            if sym and 2 <= len(sym) <= 4 and sym.isalpha():
                universe.add(sym); added += 1
        if added > 0:
            source_note = f"TCBS + danh sách cố định ({len(universe)} mã)"
    except Exception:
        pass

    # Try entrade board data
    try:
        url = "https://services.entrade.com.vn/market-data/securities?type=stock&exchange=HOSE&size=3000"
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}).json()
        items = resp if isinstance(resp, list) else resp.get("data", [])
        added2 = 0
        for item in (items or []):
            sym = str(item.get("symbol", item.get("ticker", item.get("code", "")))).upper().strip()
            if sym and 2 <= len(sym) <= 4 and sym.isalpha():
                universe.add(sym); added2 += 1
        if added2 > 0:
            source_note = f"TCBS + Entrade + cố định ({len(universe)} mã)"
    except Exception:
        pass

    # Try SSI iBoard for HOSE + HNX
    for exchange in ["HOSE", "HNX"]:
        try:
            url = f"https://iboard-query.ssi.com.vn/v2/stock/board-data/exchange?exchange={exchange}&size=3000"
            headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=15).json()
            items = (resp if isinstance(resp, list) else
                     resp.get("data", resp.get("items", resp.get("securities", []))))
            for item in (items or []):
                if not isinstance(item, dict): continue
                sym = str(item.get("symbol", item.get("code", item.get("ticker", "")))).upper().strip()
                if sym and 2 <= len(sym) <= 4 and sym.isalpha():
                    universe.add(sym)
        except Exception:
            pass

    return sorted(universe), f"{source_note} — bao gồm HOSE · HNX · UPCOM"


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIC INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def calc_ichimoku(df):
    hi, lo = df["high"], df["low"]
    tenkan = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
    kijun  = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou


def calc_fibonacci(df, lookback=60):
    window = df.tail(lookback)
    high_val = window["high"].max()
    low_val  = window["low"].min()
    hi_pos   = int(window["high"].values.argmax())
    lo_pos   = int(window["low"].values.argmin())
    is_down  = (hi_pos > lo_pos)
    diff = max(high_val - low_val, high_val * 0.01)
    if is_down:
        levels = {
            "0.0% (Đỉnh)": high_val, "23.6%": high_val - 0.236*diff, "38.2%": high_val - 0.382*diff,
            "50.0%": high_val - 0.500*diff, "61.8% ✨": high_val - 0.618*diff,
            "65.0% 🏅": high_val - 0.650*diff, "78.6%": high_val - 0.786*diff,
            "100.0% (Đáy)": low_val, "127.2% 📉": low_val - 0.272*diff, "161.8% 📉": low_val - 0.618*diff,
        }
    else:
        levels = {
            "0.0% (Đáy)": low_val, "23.6%": low_val + 0.236*diff, "38.2%": low_val + 0.382*diff,
            "50.0%": low_val + 0.500*diff, "61.8% ✨": low_val + 0.618*diff,
            "65.0% 🏅": low_val + 0.650*diff, "78.6%": low_val + 0.786*diff,
            "100.0% (Đỉnh)": high_val, "127.2% 📈": high_val + 0.272*diff, "161.8% 📈": high_val + 0.618*diff,
        }
    levels["_high"] = high_val; levels["_low"] = low_val
    levels["_is_downtrend"] = is_down; levels["_diff"] = diff
    return levels


def calc_mcdx(df):
    macd_hist = MACD(close=df["close"]).macd_diff()
    rsi       = RSIIndicator(close=df["close"], window=14).rsi()
    def _n(s):
        mn, mx = s.min(), s.max()
        return pd.Series(0, index=s.index) if mx == mn else (s - mn)/(mx - mn)*2 - 1
    return ((_n(macd_hist) + _n(rsi - 50)) / 2).rename("MCDX")


def calc_adx(df, window=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr   = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    dm_p = high.diff().clip(lower=0); dm_m = (-low.diff()).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_m, 0); dm_m = dm_m.where(dm_m > dm_p, 0)
    atr  = tr.ewm(span=window, min_periods=window, adjust=False).mean()
    dip  = 100 * dm_p.ewm(span=window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    dim  = 100 * dm_m.ewm(span=window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    df["ADX"] = dx.ewm(span=window, min_periods=window, adjust=False).mean()
    df["DI_Plus"] = dip; df["DI_Minus"] = dim
    return df


def calc_ema_cross(df):
    df["EMA5"]  = df["close"].ewm(span=5,  adjust=False).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    return df


def calc_smart_money(df):
    close, high, low, vol, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
    obv = [0]
    for i in range(1, len(df)):
        obv.append(obv[-1] + vol.iloc[i] if close.iloc[i] > close.iloc[i-1]
                   else (obv[-1] - vol.iloc[i] if close.iloc[i] < close.iloc[i-1] else obv[-1]))
    df["OBV"] = pd.array(obv, dtype=float)
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    df["CMF"] = (mfm * vol).rolling(20).sum() / vol.rolling(20).sum()
    tp = (high + low + close) / 3
    pos_mf = (tp * vol).where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = (tp * vol).where(tp < tp.shift(1), 0).rolling(14).sum()
    df["MFI"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))
    df["Up_Vol"]   = np.where(close >= open_, vol, 0).astype(float)
    df["Down_Vol"] = np.where(close <  open_, vol, 0).astype(float)
    buy10 = pd.Series(df["Up_Vol"]).rolling(10).sum()
    sell10 = pd.Series(df["Down_Vol"]).rolling(10).sum()
    df["Buy_Ratio"] = buy10 / (buy10 + sell10).replace(0, np.nan)
    df["Vol_MA20"]  = vol.rolling(20).mean()
    df["Vol_Ratio"] = vol / df["Vol_MA20"]
    df["Body_Ratio"] = (abs(close - open_) / (high - low).replace(0, np.nan)).fillna(0)
    ad_mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    df["AD_Line"] = (ad_mfm * vol).cumsum()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ICT — Inner Circle Trader Analysis
# ─────────────────────────────────────────────────────────────────────────────
def calc_ict(df: pd.DataFrame) -> dict:
    """
    ICT Concepts:
    - Market Structure: BOS (Break of Structure), CHoCH (Change of Character)
    - Order Blocks: Bullish OB (last red candle before bullish move), Bearish OB (last green before bearish)
    - Fair Value Gaps (FVG): 3-candle imbalance
    - Liquidity Levels: Equal highs/lows (stop hunt zones)
    - OTE: Optimal Trade Entry (61.8–79% retracement)
    """
    if len(df) < 20:
        return {"signals": [], "order_blocks": [], "fvg_list": [], "structure": "N/A",
                "ote_zone": None, "liq_highs": [], "liq_lows": [], "score": 0, "summary": "Không đủ dữ liệu"}

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    open_ = df["open"].values
    n     = len(df)
    times = df["time"].values if "time" in df.columns else np.arange(n)

    signals = []

    # ── 1. Market Structure (BOS / CHoCH) ──────────────────────────────────
    swing_highs = []
    swing_lows  = []
    for i in range(2, n - 2):
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            swing_highs.append((i, high[i]))
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            swing_lows.append((i, low[i]))

    structure = "SIDEWAYS"
    bos_bullish = False; bos_bearish = False; choch = False

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_hh = swing_highs[-1][1]; prev_hh = swing_highs[-2][1]
        last_ll = swing_lows[-1][1];  prev_ll = swing_lows[-2][1]

        if last_hh > prev_hh and last_ll > prev_ll:
            structure = "UPTREND"
            if close[-1] > last_hh:
                bos_bullish = True
                signals.append(("🟢 BOS Bullish", f"Break of Structure: Giá phá đỉnh {last_hh:,.0f} → Cấu trúc tăng xác nhận", "bullish"))
        elif last_hh < prev_hh and last_ll < prev_ll:
            structure = "DOWNTREND"
            if close[-1] < last_ll:
                bos_bearish = True
                signals.append(("🔴 BOS Bearish", f"Break of Structure: Giá phá đáy {last_ll:,.0f} → Cấu trúc giảm xác nhận", "bearish"))

        # CHoCH: Uptrend but last swing low broken
        if structure == "UPTREND" and close[-1] < last_ll:
            choch = True
            signals.append(("⚡ CHoCH Bearish", f"Change of Character: Phá đáy swing {last_ll:,.0f} trong uptrend → Đảo chiều tiềm năng", "bearish"))
        elif structure == "DOWNTREND" and close[-1] > last_hh:
            choch = True
            signals.append(("⚡ CHoCH Bullish", f"Change of Character: Phá đỉnh swing {last_hh:,.0f} trong downtrend → Đảo chiều tiềm năng", "bullish"))

    # ── 2. Order Blocks ──────────────────────────────────────────────────────
    order_blocks = []
    lookback_ob = min(30, n - 2)
    for i in range(n - lookback_ob, n - 3):
        # Bullish OB: Last red candle before at least 2 consecutive bullish candles
        if (close[i] < open_[i] and close[i+1] > open_[i+1] and close[i+2] > open_[i+2]
                and (high[i+2] - low[i]) / max(close[i], 1) > 0.015):
            order_blocks.append({
                "type": "bullish", "top": open_[i], "bottom": close[i],
                "time": times[i], "idx": i,
                "desc": f"Bullish OB tại {close[i]:,.0f}–{open_[i]:,.0f} (phiên {times[i]})"
            })
        # Bearish OB: Last green candle before at least 2 consecutive bearish candles
        if (close[i] > open_[i] and close[i+1] < open_[i+1] and close[i+2] < open_[i+2]
                and (high[i] - low[i+2]) / max(close[i], 1) > 0.015):
            order_blocks.append({
                "type": "bearish", "top": close[i], "bottom": open_[i],
                "time": times[i], "idx": i,
                "desc": f"Bearish OB tại {open_[i]:,.0f}–{close[i]:,.0f} (phiên {times[i]})"
            })

    # Keep last 5 OBs
    order_blocks = order_blocks[-5:]

    # Check if price is at/near an OB
    current_price = close[-1]
    for ob in order_blocks:
        tolerance = (ob["top"] - ob["bottom"]) * 0.5
        if ob["bottom"] - tolerance <= current_price <= ob["top"] + tolerance:
            if ob["type"] == "bullish":
                signals.append(("🏦 Tại Bullish Order Block",
                                f"Giá {current_price:,.0f} đang tại/gần vùng mua {ob['bottom']:,.0f}–{ob['top']:,.0f}", "bullish"))
            else:
                signals.append(("🏴 Tại Bearish Order Block",
                                f"Giá {current_price:,.0f} đang tại/gần vùng kháng cự {ob['bottom']:,.0f}–{ob['top']:,.0f}", "bearish"))

    # ── 3. Fair Value Gap (FVG) ───────────────────────────────────────────────
    fvg_list = []
    for i in range(1, n - 1):
        # Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
        if low[i+1] > high[i-1]:
            gap_size = (low[i+1] - high[i-1]) / close[i] * 100
            if gap_size > 0.3:
                fvg_list.append({"type": "bullish", "top": low[i+1], "bottom": high[i-1],
                                 "time": times[i], "pct": gap_size})
        # Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
        if high[i+1] < low[i-1]:
            gap_size = (low[i-1] - high[i+1]) / close[i] * 100
            if gap_size > 0.3:
                fvg_list.append({"type": "bearish", "top": low[i-1], "bottom": high[i+1],
                                 "time": times[i], "pct": gap_size})

    fvg_list = fvg_list[-6:]

    # FVG signals
    for fvg in fvg_list[-3:]:
        if fvg["bottom"] <= current_price <= fvg["top"]:
            if fvg["type"] == "bullish":
                signals.append(("📊 Bullish FVG",
                                f"Giá đang lấp đầy Fair Value Gap tăng {fvg['bottom']:,.0f}–{fvg['top']:,.0f} ({fvg['pct']:.1f}%)", "bullish"))
            else:
                signals.append(("📉 Bearish FVG",
                                f"Giá đang lấp đầy Fair Value Gap giảm {fvg['bottom']:,.0f}–{fvg['top']:,.0f} ({fvg['pct']:.1f}%)", "bearish"))

    # ── 4. Liquidity Levels (Equal Highs / Equal Lows) ─────────────────────
    liq_highs = []; liq_lows = []
    tol_pct = 0.002
    for i in range(n - 20, n - 1):
        if i < 0: continue
        for j in range(i + 1, min(i + 8, n - 1)):
            if abs(high[i] - high[j]) / max(high[i], 1) < tol_pct:
                liq_highs.append(round((high[i] + high[j]) / 2))
            if abs(low[i] - low[j]) / max(low[i], 1) < tol_pct:
                liq_lows.append(round((low[i] + low[j]) / 2))

    liq_highs = sorted(set(liq_highs), reverse=True)[:3]
    liq_lows  = sorted(set(liq_lows))[:3]

    for lh in liq_highs:
        if abs(current_price - lh) / max(lh, 1) < 0.015:
            signals.append(("⚠️ Vùng Liquidity Cao",
                            f"Giá tiếp cận Equal High {lh:,.0f} — Vùng stop loss của người bán khống / kháng cự", "neutral"))
    for ll in liq_lows:
        if abs(current_price - ll) / max(ll, 1) < 0.015:
            signals.append(("⚠️ Vùng Liquidity Thấp",
                            f"Giá tiếp cận Equal Low {ll:,.0f} — Vùng stop loss của người mua / hỗ trợ quan trọng", "neutral"))

    # ── 5. OTE — Optimal Trade Entry (61.8–79% retracement) ────────────────
    ote_zone = None
    if len(swing_highs) >= 1 and len(swing_lows) >= 1:
        last_sh_i, last_sh_v = swing_highs[-1]
        last_sl_i, last_sl_v = swing_lows[-1]
        if last_sh_i > last_sl_i:
            # Bullish swing: từ đáy lên đỉnh, OTE là pullback 61.8–79%
            swing_range = last_sh_v - last_sl_v
            ote_low  = last_sh_v - 0.79  * swing_range
            ote_high = last_sh_v - 0.618 * swing_range
            ote_zone = {"type": "bullish", "low": ote_low, "high": ote_high,
                        "desc": f"OTE Bullish (pullback 61.8–79%): {ote_low:,.0f}–{ote_high:,.0f}"}
            if ote_low <= current_price <= ote_high:
                signals.append(("🎯 OTE Bullish Zone",
                                f"Giá đang trong vùng Optimal Trade Entry tăng ({ote_low:,.0f}–{ote_high:,.0f}) — Điểm vào mua lý tưởng theo ICT", "bullish"))
        else:
            # Bearish swing: từ đỉnh xuống đáy, OTE là pullback 61.8–79%
            swing_range = last_sh_v - last_sl_v
            ote_low  = last_sl_v + 0.618 * swing_range
            ote_high = last_sl_v + 0.79  * swing_range
            ote_zone = {"type": "bearish", "low": ote_low, "high": ote_high,
                        "desc": f"OTE Bearish (pullback 61.8–79%): {ote_low:,.0f}–{ote_high:,.0f}"}
            if ote_low <= current_price <= ote_high:
                signals.append(("🎯 OTE Bearish Zone",
                                f"Giá đang trong vùng OTE giảm ({ote_low:,.0f}–{ote_high:,.0f}) — Điểm vào bán khống theo ICT", "bearish"))

    # ── ICT Score ──────────────────────────────────────────────────────────
    score = 0
    for sig in signals:
        if sig[2] == "bullish": score += 1
        elif sig[2] == "bearish": score -= 1

    if structure == "UPTREND": score += 1
    elif structure == "DOWNTREND": score -= 1
    if bos_bullish: score += 2
    if bos_bearish: score -= 2
    if choch: score = score  # already counted

    bull_sigs = [s for s in signals if s[2] == "bullish"]
    bear_sigs = [s for s in signals if s[2] == "bearish"]
    summary = (f"Cấu trúc: {structure} · BOS: {'✅' if bos_bullish or bos_bearish else '—'} · "
               f"CHoCH: {'⚡' if choch else '—'} · OB: {len(order_blocks)} · FVG: {len(fvg_list)} · "
               f"Tín hiệu: {len(bull_sigs)}🟢 / {len(bear_sigs)}🔴")

    return {
        "signals": signals, "order_blocks": order_blocks, "fvg_list": fvg_list,
        "structure": structure, "ote_zone": ote_zone, "liq_highs": liq_highs,
        "liq_lows": liq_lows, "score": score, "summary": summary,
        "bos_bullish": bos_bullish, "bos_bearish": bos_bearish, "choch": choch,
        "swing_highs": swing_highs[-3:] if swing_highs else [],
        "swing_lows":  swing_lows[-3:]  if swing_lows  else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# VSA — Volume Spread Analysis
# ─────────────────────────────────────────────────────────────────────────────
def calc_vsa(df: pd.DataFrame) -> dict:
    """
    VSA Concepts:
    - Stopping Volume: Lực mua hấp thụ lực bán → đáy tiềm năng
    - No Demand: Giá tăng nhưng volume thấp → yếu, không nên mua
    - No Supply: Giá giảm nhưng volume thấp → seller kiệt sức → đáy gần
    - Climax Volume: Volume cực lớn + biên độ rộng → đỉnh/đáy tiềm năng
    - Upthrust: Giá xuyên lên rồi đóng cửa thấp trên volume cao → trap bull
    - Test: Giá test lại vùng cũ với volume thấp → xác nhận hỗ trợ
    """
    if len(df) < 20:
        return {"signals": [], "score": 0, "patterns": [], "summary": "Không đủ dữ liệu"}

    close = df["close"]; high = df["high"]; low = df["low"]
    open_ = df["open"]; vol   = df["volume"]
    vol_ma20 = vol.rolling(20).mean()
    spread   = (high - low) / close.replace(0, np.nan)
    spread_ma = spread.rolling(20).mean()

    patterns = []
    signals  = []

    for i in range(max(1, len(df) - 15), len(df)):
        v    = vol.iloc[i]
        vm   = vol_ma20.iloc[i] if pd.notna(vol_ma20.iloc[i]) else v
        sp   = spread.iloc[i]   if pd.notna(spread.iloc[i])   else 0
        sp_m = spread_ma.iloc[i] if pd.notna(spread_ma.iloc[i]) else sp
        c    = close.iloc[i];  o = open_.iloc[i]
        h    = high.iloc[i];   l = low.iloc[i]
        t    = df["time"].iloc[i] if "time" in df.columns else str(i)
        is_up = c >= o; body = abs(c - o); rng = h - l if h != l else 1
        upper_w = h - max(c, o); lower_w = min(c, o) - l

        if vm == 0: continue
        vol_ratio = v / vm

        # ── Stopping Volume ──────────────────────────────────────────────
        if (vol_ratio > 1.8 and sp > sp_m * 0.8 and lower_w > body * 0.5
                and c > (l + rng * 0.4)):
            patterns.append(("🛑 Stopping Volume", t, c,
                             f"Vol x{vol_ratio:.1f} · Đuôi dưới dài → Lực mua hấp thụ lực bán · Đáy tiềm năng"))
            signals.append(("🛑 Stopping Volume",
                            f"Phiên {t}: Volume x{vol_ratio:.1f} bình thường, spread rộng + đuôi dưới dài → buyer đang hấp thụ seller", "bullish"))

        # ── No Demand ────────────────────────────────────────────────────
        elif (vol_ratio < 0.7 and is_up and sp < sp_m * 0.7):
            patterns.append(("❌ No Demand", t, c,
                             f"Vol thấp x{vol_ratio:.1f} · Tăng giá yếu → Buyer không tham gia · Cẩn thận"))
            signals.append(("❌ No Demand",
                            f"Phiên {t}: Giá tăng nhẹ nhưng volume x{vol_ratio:.1f} thấp + spread hẹp → thiếu lực mua", "bearish"))

        # ── No Supply ────────────────────────────────────────────────────
        elif (vol_ratio < 0.7 and not is_up and sp < sp_m * 0.7):
            patterns.append(("✅ No Supply", t, c,
                             f"Vol thấp x{vol_ratio:.1f} · Giảm nhẹ yếu → Seller kiệt sức · Tiếp tục tăng có thể"))
            signals.append(("✅ No Supply",
                            f"Phiên {t}: Giá giảm nhẹ nhưng volume x{vol_ratio:.1f} thấp + spread hẹp → seller kiệt sức", "bullish"))

        # ── Climax Volume ────────────────────────────────────────────────
        elif vol_ratio > 3.5 and sp > sp_m * 1.4:
            if is_up:
                patterns.append(("💥 Buying Climax", t, c,
                                 f"Vol x{vol_ratio:.1f} KHỔNG LỒ · Spread rộng · Tăng mạnh → Đỉnh tiềm năng (smart money bán)"))
                signals.append(("💥 Buying Climax",
                                f"Phiên {t}: Volume x{vol_ratio:.1f} khổng lồ trên nến xanh rộng → Đỉnh phân phối tiềm năng", "bearish"))
            else:
                patterns.append(("💥 Selling Climax", t, c,
                                 f"Vol x{vol_ratio:.1f} KHỔNG LỒ · Spread rộng · Giảm mạnh → Đáy tiềm năng (smart money mua)"))
                signals.append(("💥 Selling Climax",
                                f"Phiên {t}: Volume x{vol_ratio:.1f} khổng lồ trên nến đỏ rộng → Đáy tích lũy tiềm năng", "bullish"))

        # ── Upthrust ─────────────────────────────────────────────────────
        elif (vol_ratio > 1.5 and not is_up and upper_w > body * 1.5 and sp > sp_m):
            patterns.append(("🪝 Upthrust", t, c,
                             f"Vol x{vol_ratio:.1f} · Đuôi trên dài · Đóng thấp → Bẫy bull, smart money bán"))
            signals.append(("🪝 Upthrust",
                            f"Phiên {t}: Giá xuyên lên nhưng đóng cửa thấp trên volume x{vol_ratio:.1f} cao → trap bull, giảm sắp tới", "bearish"))

        # ── Test ─────────────────────────────────────────────────────────
        elif vol_ratio < 0.6 and sp < sp_m * 0.6:
            patterns.append(("🔬 Test", t, c,
                             f"Vol thấp x{vol_ratio:.1f} · Spread hẹp → Kiểm tra thành công, hỗ trợ xác nhận"))
            signals.append(("🔬 Test (Xác nhận hỗ trợ)",
                            f"Phiên {t}: Volume x{vol_ratio:.1f} cực thấp, spread hẹp → cung cạn kiệt, hỗ trợ được xác nhận", "bullish"))

    # Recent 5 patterns only for score
    patterns = patterns[-8:]
    signals  = signals[-8:]

    score = 0
    for sig in signals:
        if sig[2] == "bullish": score += 1
        elif sig[2] == "bearish": score -= 1

    bull_p = [p for p in patterns if any(kw in p[0] for kw in ["Stopping", "No Supply", "Selling Climax", "Test", "✅"])]
    bear_p = [p for p in patterns if any(kw in p[0] for kw in ["No Demand", "Buying Climax", "Upthrust", "❌", "💥 Buy"])]

    summary = (f"VSA: {len(bull_p)} tín hiệu tăng 🟢 · {len(bear_p)} tín hiệu giảm 🔴 · "
               f"Score: {score:+d}")

    return {"signals": signals, "patterns": patterns, "score": score, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# PRICE ACTION Analysis
# ─────────────────────────────────────────────────────────────────────────────
def calc_price_action(df: pd.DataFrame) -> dict:
    """
    PA Concepts:
    - Inside Bar: Nến bên trong nến mẹ → Tích lũy / Breakout Setup
    - Outside Bar: Nến nuốt nến mẹ → Đảo chiều mạnh
    - Key S/R: Vùng hỗ trợ/kháng cự swing
    - Pin Bar: Đuôi dài → Từ chối giá mạnh
    - Trend Structure: HH/HL (uptrend), LH/LL (downtrend)
    - Consolidation: Giá đi ngang → Chuẩn bị breakout
    """
    if len(df) < 10:
        return {"signals": [], "score": 0, "key_levels": [], "trend_structure": "N/A",
                "inside_bars": 0, "is_consolidating": False, "summary": "Không đủ dữ liệu"}

    close = df["close"].values; high = df["high"].values
    low   = df["low"].values;   open_= df["open"].values
    times = df["time"].values if "time" in df.columns else np.arange(len(df))
    n     = len(df)

    signals   = []
    score     = 0
    key_levels = []

    # ── Inside Bar ─────────────────────────────────────────────────────────
    inside_bar_count = 0
    consecutive_ibs  = 0
    for i in range(1, n):
        if high[i] < high[i-1] and low[i] > low[i-1]:
            inside_bar_count += 1
            consecutive_ibs  += 1
        else:
            consecutive_ibs = 0

    if consecutive_ibs >= 2:
        signals.append(("📦 NÉN MẠNH (2+ Inside Bar liên tiếp)",
                        f"Giá bị nén chặt trong {consecutive_ibs} phiên liên tiếp → Setup breakout sắp xảy ra", "neutral"))
    elif inside_bar_count >= 1 and inside_bar_count <= 5:
        signals.append(("📦 Inside Bar",
                        f"Phát hiện {inside_bar_count} Inside Bar gần đây → Tích lũy, chờ breakout", "neutral"))

    # ── Outside Bar ─────────────────────────────────────────────────────────
    last_bar = -1
    prev_bar = -2
    if n >= 2:
        if (high[last_bar] > high[prev_bar] and low[last_bar] < low[prev_bar]):
            direction = "tăng" if close[last_bar] > open_[last_bar] else "giảm"
            color = "bullish" if close[last_bar] > open_[last_bar] else "bearish"
            signals.append(("🔥 Outside Bar",
                            f"Nến hiện tại nuốt hoàn toàn nến trước → Đảo chiều {direction} mạnh", color))
            score += 2 if color == "bullish" else -2

    # ── Trend Structure (HH/HL vs LH/LL) ──────────────────────────────────
    swing_h = []; swing_l = []
    for i in range(2, n - 2):
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            swing_h.append(high[i])
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            swing_l.append(low[i])

    trend_structure = "SIDEWAYS"
    if len(swing_h) >= 2 and len(swing_l) >= 2:
        hh = swing_h[-1] > swing_h[-2]
        hl = swing_l[-1] > swing_l[-2]
        lh = swing_h[-1] < swing_h[-2]
        ll = swing_l[-1] < swing_l[-2]
        if hh and hl:
            trend_structure = "UPTREND (HH+HL)"
            score += 2
            signals.append(("📈 Uptrend PA", "Cấu trúc: HH + HL → Xu hướng tăng rõ ràng", "bullish"))
        elif lh and ll:
            trend_structure = "DOWNTREND (LH+LL)"
            score -= 2
            signals.append(("📉 Downtrend PA", "Cấu trúc: LH + LL → Xu hướng giảm rõ ràng", "bearish"))
        elif hh and not hl:
            trend_structure = "MIXED (HH+LL — Volatile)"
            signals.append(("⚡ Biến động mạnh", "HH nhưng LL → Thị trường bất ổn", "neutral"))

    # ── Key Support / Resistance ───────────────────────────────────────────
    for sh in (swing_h[-3:] if swing_h else []):
        key_levels.append({"price": sh, "type": "resistance", "label": f"Kháng cự: {sh:,.0f}"})
    for sl in (swing_l[-3:] if swing_l else []):
        key_levels.append({"price": sl, "type": "support", "label": f"Hỗ trợ: {sl:,.0f}"})

    current_price = close[-1]
    for kl in key_levels:
        pct = abs(current_price - kl["price"]) / max(kl["price"], 1) * 100
        if pct < 1.5:
            if kl["type"] == "support":
                signals.append(("🟢 Tại vùng hỗ trợ PA",
                                f"Giá {current_price:,.0f} tiếp cận hỗ trợ {kl['price']:,.0f} (cách {pct:.1f}%) → Cơ hội mua", "bullish"))
                score += 1
            else:
                signals.append(("🔴 Tại vùng kháng cự PA",
                                f"Giá {current_price:,.0f} tiếp cận kháng cự {kl['price']:,.0f} (cách {pct:.1f}%) → Rủi ro", "bearish"))
                score -= 1

    # ── Consolidation Detection ────────────────────────────────────────────
    recent_10 = df.tail(10)
    price_range_pct = (recent_10["high"].max() - recent_10["low"].min()) / recent_10["close"].mean() * 100
    is_consolidating = price_range_pct < 5.0
    if is_consolidating:
        signals.append(("⏳ Vùng tích lũy (Consolidation)",
                        f"Biên độ 10 phiên chỉ {price_range_pct:.1f}% → Giá đang tích lũy, sắp breakout", "neutral"))

    # ── Pin Bar (recap from candlestick) ────────────────────────────────────
    last_body = abs(close[-1] - open_[-1])
    last_range = high[-1] - low[-1] if high[-1] != low[-1] else 1
    last_lower_w = min(close[-1], open_[-1]) - low[-1]
    last_upper_w = high[-1] - max(close[-1], open_[-1])
    if last_lower_w > 3 * last_body and last_lower_w > last_upper_w * 2:
        signals.append(("📌 Bullish Pin Bar (PA)", "Đuôi dài dưới → Từ chối vùng thấp mạnh → Cơ hội mua", "bullish"))
        score += 2
    elif last_upper_w > 3 * last_body and last_upper_w > last_lower_w * 2:
        signals.append(("📌 Bearish Pin Bar (PA)", "Đuôi dài trên → Từ chối vùng cao mạnh → Nguy cơ giảm", "bearish"))
        score -= 2

    summary = (f"PA: {trend_structure} · Inside Bars: {inside_bar_count} · "
               f"Key Levels: {len(key_levels)} · Score: {score:+d}")

    return {
        "signals": signals, "score": score, "key_levels": key_levels,
        "trend_structure": trend_structure, "inside_bars": inside_bar_count,
        "is_consolidating": is_consolidating, "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OVERFLOW Analysis
# ─────────────────────────────────────────────────────────────────────────────
def calc_overflow(df: pd.DataFrame) -> dict:
    """
    Overflow Concepts:
    - Volume Overflow: Vol > 3x MA20 → Institutional block trade / forced liquidation
    - Price Overflow: Giá vượt BB Upper/Lower → Over-extended, mean reversion likely
    - Breakout w/ Volume: Giá phá key level + volume lớn → Breakout xác nhận
    - Gap Analysis: Khoảng trống giá → Fill probability, momentum
    - Exhaustion: Nhiều phiên tăng/giảm liên tiếp → Kiệt sức
    """
    if len(df) < 20:
        return {"signals": [], "score": 0, "overflow_events": [], "summary": "Không đủ dữ liệu",
                "gaps": [], "consecutive_run": 0}

    close = df["close"]; high = df["high"]; low = df["low"]
    open_ = df["open"]; vol   = df["volume"]
    vol_ma20 = vol.rolling(20).mean()
    times = df["time"].values if "time" in df.columns else np.arange(len(df))

    bb      = BollingerBands(close=close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low  = bb.bollinger_lband()
    bb_mid  = bb.bollinger_mavg()

    signals = []; overflow_events = []; gaps = []
    score   = 0

    # ── Volume Overflow ─────────────────────────────────────────────────────
    for i in range(max(0, len(df) - 10), len(df)):
        vm = vol_ma20.iloc[i] if pd.notna(vol_ma20.iloc[i]) else 0
        if vm == 0: continue
        vr = vol.iloc[i] / vm
        t  = times[i]
        if vr >= 4.0:
            color = "bullish" if close.iloc[i] >= open_.iloc[i] else "bearish"
            overflow_events.append({
                "type": "EXTREME Volume Overflow",
                "time": t, "ratio": vr,
                "close": close.iloc[i], "direction": color,
                "desc": f"Volume x{vr:.1f} — CỰC ĐẠI! Tổ chức đang vào/ra ồ ạt"
            })
            signals.append((f"🌊 Volume Overflow x{vr:.1f}",
                            f"Phiên {t}: Volume x{vr:.1f} bình thường → Block trade tổ chức {'(TĂNG)' if color=='bullish' else '(GIẢM)'}",
                            color))
            score += 2 if color == "bullish" else -2
        elif vr >= 2.5:
            color = "bullish" if close.iloc[i] >= open_.iloc[i] else "bearish"
            overflow_events.append({
                "type": "Volume Overflow",
                "time": t, "ratio": vr,
                "close": close.iloc[i], "direction": color,
                "desc": f"Volume x{vr:.1f} — Tổ chức đang giao dịch"
            })

    # ── Price Overflow (BB Extension) ──────────────────────────────────────
    last_close = close.iloc[-1]
    last_bb_h  = bb_high.iloc[-1]; last_bb_l = bb_low.iloc[-1]; last_bb_m = bb_mid.iloc[-1]

    if pd.notna(last_bb_h) and last_close > last_bb_h * 1.01:
        pct_ext = (last_close / last_bb_h - 1) * 100
        signals.append(("⚡ Price Overflow — Quá Mua BB",
                        f"Giá {last_close:,.0f} vượt BB Upper {last_bb_h:,.0f} ({pct_ext:.1f}%) → Khả năng hồi về mean cao", "bearish"))
        score -= 2
        overflow_events.append({"type": "Price Overflow UP", "time": times[-1],
                                 "ratio": pct_ext, "close": last_close, "direction": "overbought",
                                 "desc": f"Giá {pct_ext:.1f}% trên BB Upper → Quá mua"})
    elif pd.notna(last_bb_l) and last_close < last_bb_l * 0.99:
        pct_ext = (1 - last_close / last_bb_l) * 100
        signals.append(("💧 Price Overflow — Quá Bán BB",
                        f"Giá {last_close:,.0f} dưới BB Lower {last_bb_l:,.0f} ({pct_ext:.1f}%) → Khả năng hồi phục cao", "bullish"))
        score += 2
        overflow_events.append({"type": "Price Overflow DOWN", "time": times[-1],
                                 "ratio": pct_ext, "close": last_close, "direction": "oversold",
                                 "desc": f"Giá {pct_ext:.1f}% dưới BB Lower → Quá bán"})

    # ── Gap Analysis ─────────────────────────────────────────────────────────
    for i in range(max(1, len(df) - 10), len(df)):
        prev_close = close.iloc[i - 1]
        curr_open  = open_.iloc[i]
        gap_pct    = (curr_open - prev_close) / prev_close * 100
        if abs(gap_pct) > 0.5:
            gaps.append({"time": times[i], "pct": gap_pct, "filled": bool(
                (gap_pct > 0 and low.iloc[i] <= prev_close) or
                (gap_pct < 0 and high.iloc[i] >= prev_close)
            )})
            if abs(gap_pct) > 1.0:
                color = "bullish" if gap_pct > 0 else "bearish"
                direction = "GẬP TĂNG" if gap_pct > 0 else "GAP GIẢM"
                signals.append((f"📐 {direction} {gap_pct:+.1f}%",
                                f"Phiên {times[i]}: Giá mở cửa lệch {gap_pct:+.1f}% so với phiên trước", color))

    # ── Consecutive Run (Exhaustion) ────────────────────────────────────────
    consecutive_run = 0
    if len(df) >= 2:
        direction_last = 1 if close.iloc[-1] > close.iloc[-2] else -1
        for i in range(len(df) - 1, max(0, len(df) - 10) - 1, -1):
            if i == 0: break
            d = 1 if close.iloc[i] > close.iloc[i-1] else -1
            if d == direction_last:
                consecutive_run += 1
            else:
                break

    if consecutive_run >= 6:
        color = "bearish" if direction_last == 1 else "bullish"
        dir_label = "tăng" if direction_last == 1 else "giảm"
        signals.append((f"😮 Kiệt sức {dir_label.upper()} ({consecutive_run} phiên liên tiếp)",
                        f"{consecutive_run} phiên {dir_label} liên tiếp → Khả năng đảo chiều hoặc điều chỉnh cao", color))
        score += 1 if color == "bullish" else -1

    # Breakout with Volume
    if len(df) >= 3:
        prev20_high = high.iloc[-21:-1].max() if len(df) >= 22 else high.iloc[:-1].max()
        prev20_low  = low.iloc[-21:-1].min()  if len(df) >= 22 else low.iloc[:-1].min()
        last_vol_r  = vol.iloc[-1] / vol_ma20.iloc[-1] if pd.notna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] > 0 else 1
        if last_close > prev20_high and last_vol_r > 1.5:
            signals.append(("🚀 BREAKOUT xác nhận",
                            f"Giá phá đỉnh 20 phiên ({prev20_high:,.0f}) với volume x{last_vol_r:.1f} → Breakout hợp lệ", "bullish"))
            score += 3
        elif last_close < prev20_low and last_vol_r > 1.5:
            signals.append(("💥 BREAKDOWN xác nhận",
                            f"Giá phá đáy 20 phiên ({prev20_low:,.0f}) với volume x{last_vol_r:.1f} → Breakdown hợp lệ", "bearish"))
            score -= 3

    summary = (f"Overflow: {len(overflow_events)} sự kiện · Gaps: {len(gaps)} · "
               f"Consecutive run: {consecutive_run} phiên · Score: {score:+d}")

    return {
        "signals": signals, "score": score, "overflow_events": overflow_events,
        "gaps": gaps, "consecutive_run": consecutive_run,
        "bb_overflow_up": (pd.notna(last_bb_h) and last_close > last_bb_h * 1.01),
        "bb_overflow_down": (pd.notna(last_bb_l) and last_close < last_bb_l * 0.99),
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SMART MONEY PHASE
# ─────────────────────────────────────────────────────────────────────────────
def detect_smart_money_phase(df: pd.DataFrame) -> dict:
    win  = df.tail(15)
    last = df.iloc[-1]

    obv_vals = win["OBV"].dropna().values
    obv_slope_norm = 0
    if len(obv_vals) > 3:
        x = np.arange(len(obv_vals))
        s = np.polyfit(x, obv_vals, 1)[0]
        obv_slope_norm = s / (abs(obv_vals).mean() + 1e-9) * 10

    prices = win["close"].dropna().values
    price_slope_norm = 0
    if len(prices) > 3:
        x = np.arange(len(prices))
        s = np.polyfit(x, prices, 1)[0]
        price_slope_norm = s / (prices.mean() + 1e-9) * 100

    cmf_val   = last["CMF"]       if pd.notna(last.get("CMF"))       else 0
    mfi_val   = last["MFI"]       if pd.notna(last.get("MFI"))       else 50
    buy_ratio = last["Buy_Ratio"] if pd.notna(last.get("Buy_Ratio")) else 0.5
    vol_ratio = last["Vol_Ratio"] if pd.notna(last.get("Vol_Ratio")) else 1.0

    score = 0; signals = []

    if obv_slope_norm > 0.3:   score += 2; signals.append(("OBV tăng","Tiền lớn tích lũy","🟢"))
    elif obv_slope_norm < -0.3: score -= 2; signals.append(("OBV giảm","Tiền lớn xả","🔴"))
    else:                        signals.append(("OBV phẳng","Chưa rõ","⚪"))

    if cmf_val > 0.12:   score += 2; signals.append(("CMF cao",   f"{cmf_val:.3f} Mua mạnh","🟢"))
    elif cmf_val > 0:    score += 1; signals.append(("CMF dương", f"{cmf_val:.3f} Mua nhẹ", "🟡"))
    elif cmf_val < -0.12: score -= 2; signals.append(("CMF âm",   f"{cmf_val:.3f} Bán mạnh","🔴"))
    else:                 score -= 1; signals.append(("CMF âm nhẹ",f"{cmf_val:.3f}","🟠"))

    if mfi_val < 25:   score += 2; signals.append(("MFI quá bán",f"{mfi_val:.1f}","🟢"))
    elif mfi_val > 80: score -= 2; signals.append(("MFI quá mua",f"{mfi_val:.1f}","🔴"))
    else:               signals.append(("MFI trung tính",f"{mfi_val:.1f}","⚪"))

    if buy_ratio > 0.65:  score += 1.5; signals.append(("KL mua trội",f"{buy_ratio*100:.0f}%","🟢"))
    elif buy_ratio < 0.35: score -= 1.5; signals.append(("KL bán trội",f"{(1-buy_ratio)*100:.0f}%","🔴"))
    else:                  signals.append(("Cân bằng",f"{buy_ratio*100:.0f}%","⚪"))

    if price_slope_norm > 0.5 and obv_slope_norm < -0.2:
        score -= 1.5; signals.append(("Phân kỳ âm","Giá tăng OBV giảm","🔴"))
    elif price_slope_norm < -0.5 and obv_slope_norm > 0.2:
        score += 1.5; signals.append(("Phân kỳ dương","Giá giảm OBV tăng","🟢"))

    max_score = 10.0; pct = score / max_score

    if pct >= 0.4:
        phase,icon,color,bg,border="GOM HÀNG","🏦","#00C853","rgba(0,200,83,0.10)","#00C853"
        phase_desc="Tiền lớn đang **bí mật tích lũy** cổ phiếu."
    elif pct >= 0.15:
        phase,icon,color,bg,border="ĐẨY GIÁ (MARKUP)","🚀","#40C4FF","rgba(64,196,255,0.10)","#40C4FF"
        phase_desc="Dòng tiền lớn **đang đẩy giá lên mạnh**."
    elif pct >= -0.15:
        phase,icon,color,bg,border="TRUNG TÍNH","🔍","#FFD740","rgba(255,215,64,0.08)","#FFD740"
        phase_desc="Dòng tiền lớn **chưa lộ rõ ý định**."
    elif pct >= -0.40:
        phase,icon,color,bg,border="XẢ HÀNG","📤","#FF6D00","rgba(255,109,0,0.10)","#FF6D00"
        phase_desc="Tiền lớn đang **lặng lẽ phân phối**."
    else:
        phase,icon,color,bg,border="ĐÈ GIÁ","📉","#FF1744","rgba(255,23,68,0.10)","#FF1744"
        phase_desc="Tiền lớn đang **đè giá và bán tháo**."

    return {
        "phase":phase,"icon":icon,"color":color,"bg":bg,"border":border,
        "phase_desc":phase_desc,"score":score,"max_score":max_score,"signals":signals,
        "obv_slope":obv_slope_norm,"cmf":cmf_val,"mfi":mfi_val,
        "buy_ratio":buy_ratio,"vol_ratio":vol_ratio,"price_slope":price_slope_norm,
    }


def detect_candlestick_patterns(df):
    if len(df) < 3: return []
    patterns = []
    last=df.iloc[-1]; prev=df.iloc[-2]; prev2=df.iloc[-3]
    o,h,l,c = last["open"],last["high"],last["low"],last["close"]
    po,ph,pl,pc = prev["open"],prev["high"],prev["low"],prev["close"]
    p2o,p2c = prev2["open"],prev2["close"]
    body=abs(c-o); range_=(h-l) if (h-l)>0 else 0.001
    upper_w=h-max(o,c); lower_w=min(o,c)-l
    if body/range_<0.1: patterns.append(("⚖️ Doji","Thị trường do dự","neutral"))
    elif lower_w>2*body and upper_w<body and c>=o: patterns.append(("🔨 Hammer","Đảo chiều tăng","bullish"))
    elif upper_w>2*body and lower_w<body and c<o: patterns.append(("⭐ Shooting Star","Đảo chiều giảm","bearish"))
    if pc<po and c>o and c>=po and o<=pc: patterns.append(("🟢 Bullish Engulfing","Nuốt nến đỏ","bullish"))
    elif pc>po and c<o and c<=po and o>=pc: patterns.append(("🔴 Bearish Engulfing","Nuốt nến xanh","bearish"))
    if p2c<p2o and abs(pc-po)<0.3*abs(p2c-p2o) and c>o and c>(p2o+p2c)/2:
        patterns.append(("🌅 Morning Star","3 nến đảo chiều tăng","bullish"))
    elif p2c>p2o and abs(pc-po)<0.3*abs(p2c-p2o) and c<o and c<(p2o+p2c)/2:
        patterns.append(("🌇 Evening Star","3 nến đảo chiều giảm","bearish"))
    if lower_w>3*body: patterns.append(("📌 Pin Bar Tăng","Đuôi dài dưới","bullish"))
    elif upper_w>3*body: patterns.append(("📌 Pin Bar Giảm","Đuôi dài trên","bearish"))
    if c>o and body/range_>0.85: patterns.append(("💚 Marubozu Xanh","Lực mua áp đảo","bullish"))
    elif c<o and body/range_>0.85: patterns.append(("❤️ Marubozu Đỏ","Lực bán áp đảo","bearish"))
    seen=[]; result=[]
    for p in patterns:
        if p[0] not in seen: seen.append(p[0]); result.append(p)
    return result[:4]


def calc_trend_strength(df, sm_result):
    score=50.0; signals=[]; latest=df.iloc[-1]; close=latest["close"]
    adx_val=latest.get("ADX",np.nan); di_p=latest.get("DI_Plus",0) or 0; di_m=latest.get("DI_Minus",0) or 0
    if pd.notna(adx_val):
        if adx_val>35:
            adj=12 if di_p>di_m else -12; score+=adj
            signals.append(f"ADX={adx_val:.1f}>35 ({'TĂNG' if di_p>di_m else 'GIẢM'} mạnh)")
        elif adx_val>22:
            adj=6 if di_p>di_m else -6; score+=adj
            signals.append(f"ADX={adx_val:.1f}>22 (xu hướng {'Tăng' if di_p>di_m else 'Giảm'})")
        else: signals.append(f"ADX={adx_val:.1f}<22 → Sideway")
    ema5=latest.get("EMA5",np.nan); ema20=latest.get("EMA20",np.nan); ema50=latest.get("EMA50",np.nan)
    if pd.notna(ema5) and pd.notna(ema20) and pd.notna(ema50):
        if ema5>ema20>ema50 and close>ema5: score+=16; signals.append("EMA5>EMA20>EMA50: Bull Alignment ✅")
        elif ema5>ema20 and close>ema20:    score+=8;  signals.append("EMA5>EMA20: Tăng ngắn-trung hạn")
        elif close>ema50:                   score+=4;  signals.append("Giá>EMA50: Dài hạn tăng")
        elif ema5<ema20<ema50 and close<ema5: score-=16; signals.append("EMA5<EMA20<EMA50: Bear Alignment ❌")
        elif ema5<ema20 and close<ema20:    score-=8;  signals.append("EMA5<EMA20: Giảm ngắn-trung hạn")
        elif close<ema50:                   score-=4;  signals.append("Giá<EMA50: Dài hạn yếu")
    span_a=latest.get("SpanA",np.nan); span_b=latest.get("SpanB",np.nan)
    if pd.notna(span_a) and pd.notna(span_b):
        ct=max(span_a,span_b); cb=min(span_a,span_b)
        if close>ct: score+=10; signals.append("Giá trên Kumo ☁️ Bullish")
        elif close<cb: score-=10; signals.append("Giá dưới Kumo ☁️ Bearish")
    macd_v=latest.get("MACD",np.nan); sig_v=latest.get("MACD_Signal",np.nan)
    if pd.notna(macd_v) and pd.notna(sig_v):
        if macd_v>sig_v: score+=5; signals.append("MACD>Signal: Momentum tăng")
        else: score-=5; signals.append("MACD<Signal: Momentum giảm")
    sm_pct=sm_result["score"]/sm_result["max_score"]; score+=sm_pct*8
    signals.append(f"Smart Money: {sm_result['score']:+.1f} ({sm_result['phase']})")
    score=max(0.0,min(100.0,score))
    if score>=78:   label="RẤT MẠNH TĂNG 🚀"; color="#00C853"; bg="rgba(0,200,83,0.12)"
    elif score>=62: label="ĐANG TĂNG 📈";      color="#69F0AE"; bg="rgba(105,240,174,0.10)"
    elif score>=42: label="TRUNG TÍNH ⚖️";     color="#FFD740"; bg="rgba(255,215,64,0.08)"
    elif score>=28: label="ĐANG GIẢM 📉";      color="#FF9100"; bg="rgba(255,145,0,0.10)"
    else:            label="RẤT MẠNH GIẢM 🔻"; color="#FF1744"; bg="rgba(255,23,68,0.10)"
    return {"score":score,"label":label,"color":color,"bg":bg,"signals":signals}


# ─────────────────────────────────────────────────────────────────────────────
# AI CALL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(api_key, prompt):
    headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"}
    body={"model":"claude-sonnet-4-6","max_tokens":2500,
          "messages":[{"role":"user","content":prompt}]}
    resp=requests.post("https://api.anthropic.com/v1/messages",headers=headers,json=body,timeout=90)
    if not resp.ok: st.error(f"Lỗi Claude: {resp.status_code} — {resp.text}")
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_gemini(api_key, prompt):
    from google import genai
    import time
    MODELS=["gemini-2.5-flash-preview-05-20","gemini-2.0-flash","gemini-1.5-flash-latest"]
    client=genai.Client(api_key=api_key)
    for model in MODELS:
        for attempt in range(2):
            try:
                return client.models.generate_content(model=model,contents=prompt).text
            except Exception as e:
                err=str(e)
                if "RESOURCE_EXHAUSTED" in err or "429" in err:
                    if attempt==0: time.sleep(5); continue
                    else: break
                elif "NOT_FOUND" in err or "404" in err: break
                else: raise
    raise RuntimeError("⚠️ Hết quota Gemini. Chờ 1 phút hoặc dùng Claude.")


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER HELPER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def scan_single_stock(symbol: str) -> dict | None:
    try:
        df = get_clean_stock_data(symbol)
        if df is None or df.empty or len(df) < 30: return None
        for col in ["open","high","low","close","volume"]:
            if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df)<30: return None

        df["RSI"]        =RSIIndicator(close=df["close"],window=14).rsi()
        bb               =BollingerBands(close=df["close"],window=20,window_dev=2)
        df["BB_High"]    =bb.bollinger_hband(); df["BB_Low"]=bb.bollinger_lband(); df["BB_Mid"]=bb.bollinger_mavg()
        macd_o           =MACD(close=df["close"])
        df["MACD"]       =macd_o.macd(); df["MACD_Signal"]=macd_o.macd_signal()
        df["MCDX"]       =calc_mcdx(df)
        (df["Tenkan"],df["Kijun"],df["SpanA"],df["SpanB"],df["Chikou"])=calc_ichimoku(df)
        df=calc_adx(df); df=calc_ema_cross(df); df=calc_smart_money(df)

        fib_levels=calc_fibonacci(df)
        sm_result =detect_smart_money_phase(df)

        # Quick ICT/VSA/PA/Overflow for scanner
        ict_r  = calc_ict(df)
        vsa_r  = calc_vsa(df)
        pa_r   = calc_price_action(df)
        ovf_r  = calc_overflow(df)

        latest=df.iloc[-1]; prev=df.iloc[-2]
        pct_chg=(latest["close"]-prev["close"])/prev["close"]*100
        rsi_now=latest["RSI"] if pd.notna(latest["RSI"]) else 50

        # Composite score
        score=0
        if rsi_now<35: score+=2
        elif rsi_now<50: score+=1
        elif rsi_now>70: score-=2
        if pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_Signal")):
            score+=(2 if latest["MACD"]>latest["MACD_Signal"] else -1)
        mcdx_now=latest.get("MCDX",0) or 0
        if mcdx_now>0.2: score+=2
        elif mcdx_now>0: score+=1
        else: score-=1
        adx_val=latest.get("ADX",0) or 0
        if adx_val>25:
            score+=(1 if (latest.get("DI_Plus",0) or 0)>(latest.get("DI_Minus",0) or 0) else -1)
        ema5=latest.get("EMA5",np.nan); ema20=latest.get("EMA20",np.nan); ema50=latest.get("EMA50",np.nan)
        if pd.notna(ema5) and pd.notna(ema20) and pd.notna(ema50):
            if ema5>ema20>ema50: score+=2
            elif ema5<ema20<ema50: score-=2
        # Add new method scores (capped)
        score += max(-3, min(3, ict_r["score"]))
        score += max(-3, min(3, vsa_r["score"]))
        score += max(-3, min(3, pa_r["score"]))
        score += max(-2, min(2, ovf_r["score"]))

        max_score=22
        pct_score=score/max_score
        if pct_score>=0.40:   recommendation="✅ MUA"
        elif pct_score>=0.12: recommendation="⏳ THEO DÕI"
        else:                  recommendation="🚫 TRÁNH/BÁN"

        # Fib zone
        _is_down=fib_levels.get("_is_downtrend",True)
        _fh=fib_levels.get("_high",0); _fl=fib_levels.get("_low",0)
        _diff=fib_levels.get("_diff",_fh-_fl); cn=latest["close"]
        if _is_down:
            _f618=_fh-0.618*_diff; _f650=_fh-0.650*_diff; _f786=_fh-0.786*_diff; _f382=_fh-0.382*_diff
            if _f650<=cn<=_f618:         fib_zone="⭐ Golden Pocket"
            elif _f786<=cn<_f650:        fib_zone="⚠️ Hỗ trợ sâu (65-78.6%)"
            elif cn<_f786:               fib_zone="⛔ Dưới hỗ trợ"
            elif _f618<cn<=_f382:        fib_zone="📍 Hỗ trợ (38-61.8%)"
            else:                         fib_zone="🔺 Trên 38.2%"
        else:
            _f236=_fl+0.236*_diff; _f382=_fl+0.382*_diff
            if cn<=_f236:   fib_zone="✅ Gần đáy (tích lũy)"
            elif cn<=_f382: fib_zone="📍 Phục hồi sớm"
            else:            fib_zone="🔺 Đã phục hồi nhiều"

        # ICT structure label
        ict_struct = ict_r.get("structure", "N/A")
        ict_bos    = "BOS✅" if ict_r.get("bos_bullish") or ict_r.get("bos_bearish") else ""
        ict_choch  = "CHoCH⚡" if ict_r.get("choch") else ""
        ict_label  = f"{ict_struct} {ict_bos}{ict_choch}".strip()

        # VSA top pattern
        top_vsa = vsa_r["patterns"][-1][0] if vsa_r["patterns"] else "—"

        return {
            "symbol":symbol,"close":latest["close"],"pct_chg":pct_chg,
            "rsi":rsi_now,"mcdx":mcdx_now,"adx":float(adx_val),
            "macd_bull":bool(pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_Signal"))
                             and latest["MACD"]>latest["MACD_Signal"]),
            "ema_bull":bool(pd.notna(ema5) and pd.notna(ema20) and ema5>ema20),
            "sm_phase":sm_result["phase"],"sm_icon":sm_result["icon"],
            "sm_color":sm_result["color"],"sm_score":sm_result["score"],
            "fib_zone":fib_zone,"score":score,"max_score":max_score,
            "recommendation":recommendation,
            "obv_slope":sm_result.get("obv_slope",0),"cmf":sm_result.get("cmf",0),
            "mfi":sm_result.get("mfi",50),"buy_ratio":sm_result.get("buy_ratio",0.5),
            "vol_ratio":sm_result.get("vol_ratio",1.0),
            "avg_vol":float(latest.get("Vol_MA20",0) or 0),
            # New method results
            "ict_score":ict_r["score"],"ict_label":ict_label,
            "vsa_score":vsa_r["score"],"vsa_top":top_vsa,
            "pa_score":pa_r["score"],"pa_trend":pa_r.get("trend_structure","N/A"),
            "ovf_score":ovf_r["score"],"ovf_events":len(ovf_r["overflow_events"]),
            "is_consolidating":pa_r.get("is_consolidating",False),
            "consecutive_run":ovf_r.get("consecutive_run",0),
        }
    except Exception:
        return None


def _phase_order(phase):
    return {"GOM HÀNG":0,"ĐẨY GIÁ (MARKUP)":1,"TRUNG TÍNH":2,"XẢ HÀNG":3,"ĐÈ GIÁ":4}.get(phase,5)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PHÂN TÍCH ĐƠN LẺ
# ═════════════════════════════════════════════════════════════════════════════
with _tab1:
    ticker = st.text_input("🔍 Nhập mã chứng khoán (VD: HPG, VCB, VN30F1M):", "VN30F1M").upper().strip()

    if st.button("🚀 Lấy Dữ Liệu & Phân Tích", type="primary", key="analyze_btn"):
        with st.spinner(f"Đang trích xuất dữ liệu **{ticker}**..."):
            df = get_clean_stock_data(ticker)

        if df is None or df.empty:
            st.error("❌ Không thể lấy dữ liệu."); st.stop()
        for col in ["open","high","low","close","volume"]:
            if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna(subset=["close"]).reset_index(drop=True)
        if len(df)<30: st.error("❌ Không đủ dữ liệu (cần ≥30 phiên)."); st.stop()

        # ── Tính toán tất cả chỉ báo ─────────────────────────────────────
        df["RSI"]        =RSIIndicator(close=df["close"],window=14).rsi()
        bb               =BollingerBands(close=df["close"],window=20,window_dev=2)
        df["BB_High"]    =bb.bollinger_hband(); df["BB_Low"]=bb.bollinger_lband(); df["BB_Mid"]=bb.bollinger_mavg()
        macd_o           =MACD(close=df["close"])
        df["MACD"]       =macd_o.macd(); df["MACD_Signal"]=macd_o.macd_signal(); df["MACD_Diff"]=macd_o.macd_diff()
        df["MCDX"]       =calc_mcdx(df)
        (df["Tenkan"],df["Kijun"],df["SpanA"],df["SpanB"],df["Chikou"])=calc_ichimoku(df)
        df=calc_adx(df); df=calc_ema_cross(df)
        fib_levels=calc_fibonacci(df)
        df=calc_smart_money(df)
        sm_result =detect_smart_money_phase(df)
        patterns  =detect_candlestick_patterns(df)
        trend_str =calc_trend_strength(df,sm_result)

        # New methods
        ict_result =calc_ict(df)
        vsa_result =calc_vsa(df)
        pa_result  =calc_price_action(df)
        ovf_result =calc_overflow(df)

        latest=df.iloc[-1]; prev=df.iloc[-2]
        pct_chg=(latest["close"]-prev["close"])/prev["close"]*100
        rsi_val=latest["RSI"]; adx_val=latest.get("ADX",0) or 0
        di_p=latest.get("DI_Plus",0) or 0; di_m=latest.get("DI_Minus",0) or 0
        ema5_v=latest.get("EMA5",0) or 0; ema20_v=latest.get("EMA20",0) or 0; ema50_v=latest.get("EMA50",0) or 0
        mcdx_val=latest["MCDX"]

        # ── Trend Strength Meter ──────────────────────────────────────────
        ts=trend_str
        st.markdown(
            f"""<div style="background:{ts['bg']};border-radius:14px;padding:16px 24px;
            border:1.5px solid {ts['color']}44;margin-bottom:18px;display:flex;align-items:center;gap:20px;">
  <div style="flex:1;">
    <div style="font-size:0.8rem;color:#aaa;letter-spacing:1px;">📡 XU HƯỚNG TỔNG HỢP</div>
    <div style="font-size:1.6rem;font-weight:800;color:{ts['color']};margin:4px 0;">{ts['label']}</div>
    <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:10px;width:100%;margin-top:6px;">
      <div style="height:10px;width:{ts['score']:.0f}%;border-radius:6px;
                  background:linear-gradient(90deg,#FF1744,#FFD740,#00C853);"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#777;margin-top:3px;">
      <span>Rất giảm</span><span>Trung tính</span><span>Rất tăng</span>
    </div>
  </div>
  <div style="font-size:2.8rem;font-weight:900;color:{ts['color']};min-width:70px;text-align:right;">
    {ts['score']:.0f}<span style="font-size:1rem;">/100</span>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Quick Method Scores ───────────────────────────────────────────
        st.subheader(f"🔬 Điểm Tổng Hợp Phương Pháp — {ticker}")
        m1,m2,m3,m4,m5 = st.columns(5)
        def _score_color(s):
            return "#00C853" if s>0 else ("#FF5252" if s<0 else "#FFD740")
        m1.markdown(f"<div style='text-align:center;padding:12px;border-radius:10px;background:rgba(64,196,255,0.1);border:1px solid #40C4FF44;'>"
                    f"<div style='font-size:0.8rem;color:#aaa;'>🔬 ICT</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{_score_color(ict_result['score'])};'>{ict_result['score']:+d}</div>"
                    f"<div style='font-size:0.72rem;color:#bbb;'>{ict_result['structure']}</div></div>",unsafe_allow_html=True)
        m2.markdown(f"<div style='text-align:center;padding:12px;border-radius:10px;background:rgba(255,167,38,0.1);border:1px solid #FFA72644;'>"
                    f"<div style='font-size:0.8rem;color:#aaa;'>📊 VSA</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{_score_color(vsa_result['score'])};'>{vsa_result['score']:+d}</div>"
                    f"<div style='font-size:0.72rem;color:#bbb;'>{len(vsa_result['patterns'])} patterns</div></div>",unsafe_allow_html=True)
        m3.markdown(f"<div style='text-align:center;padding:12px;border-radius:10px;background:rgba(224,64,251,0.1);border:1px solid #E040FB44;'>"
                    f"<div style='font-size:0.8rem;color:#aaa;'>🕯️ PA</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{_score_color(pa_result['score'])};'>{pa_result['score']:+d}</div>"
                    f"<div style='font-size:0.72rem;color:#bbb;'>{pa_result.get('trend_structure','N/A')[:12]}</div></div>",unsafe_allow_html=True)
        m4.markdown(f"<div style='text-align:center;padding:12px;border-radius:10px;background:rgba(255,23,68,0.1);border:1px solid #FF174444;'>"
                    f"<div style='font-size:0.8rem;color:#aaa;'>💧 Overflow</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{_score_color(ovf_result['score'])};'>{ovf_result['score']:+d}</div>"
                    f"<div style='font-size:0.72rem;color:#bbb;'>{len(ovf_result['overflow_events'])} sự kiện</div></div>",unsafe_allow_html=True)
        m5.markdown(f"<div style='text-align:center;padding:12px;border-radius:10px;background:{sm_result['bg']};border:1px solid {sm_result['color']}44;'>"
                    f"<div style='font-size:0.8rem;color:#aaa;'>🐋 Smart Money</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{sm_result['color']};'>{sm_result['score']:+.1f}</div>"
                    f"<div style='font-size:0.72rem;color:#bbb;'>{sm_result['phase'][:12]}</div></div>",unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Classic Metrics ───────────────────────────────────────────────
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Giá Đóng Cửa",f"{latest['close']:,.0f} đ",f"{pct_chg:+.2f}%")
        rsi_note="⚠️ Quá mua" if rsi_val>70 else ("⚠️ Quá bán" if rsi_val<30 else "Trung tính")
        c2.metric("RSI (14)",f"{rsi_val:.1f}",rsi_note)
        c3.metric("ADX (14)",f"{adx_val:.1f}",f"DI+{di_p:.0f}/DI-{di_m:.0f}")
        c4.metric("MCDX",f"{mcdx_val:.3f}","📈 Tăng" if mcdx_val>0 else "📉 Giảm")
        c5.metric("EMA 5/20/50",f"{ema5_v:,.0f}/{ema20_v:,.0f}","Bull" if ema5_v>ema20_v else "Bear")
        c6.metric("ICT Structure",ict_result["structure"],
                  "BOS✅" if ict_result.get("bos_bullish") or ict_result.get("bos_bearish") else "—")

        # Candlestick patterns badges
        if patterns:
            badges=""
            for pn,pd_,pt in patterns:
                col_m={"bullish":"#00E676","bearish":"#FF5252","neutral":"#FFD740"}
                pc=col_m.get(pt,"#aaa")
                badges+=f"<span style='background:{pc}22;border:1px solid {pc};border-radius:20px;padding:4px 12px;margin:4px;font-size:0.82rem;color:{pc};display:inline-block;' title='{pd_}'>{pn}</span>"
            st.markdown(f"<div style='margin:8px 0 16px 0;'><b style='color:#aaa;font-size:0.8rem;'>🕯️ PATTERN NẾN:</b><br>{badges}</div>",unsafe_allow_html=True)

        # ── Main Chart ────────────────────────────────────────────────────
        fig=make_subplots(rows=5,cols=1,shared_xaxes=True,
            row_heights=[0.44,0.13,0.14,0.14,0.15],vertical_spacing=0.022,
            subplot_titles=(f"Nến · BB · Ichimoku · EMA · Fibonacci — {ticker}",
                            "Khối lượng","RSI + ADX","MACD + MCDX","EMA 5/20/50"))
        fig.add_trace(go.Candlestick(x=df["time"],open=df["open"],high=df["high"],
            low=df["low"],close=df["close"],name="Nến",
            increasing_line_color="#26a69a",decreasing_line_color="#ef5350"),row=1,col=1)
        for col_,color_,name_ in [("BB_High","rgba(255,80,80,0.55)","BB Upper"),
                                   ("BB_Mid","rgba(180,180,180,0.4)","BB Mid"),
                                   ("BB_Low","rgba(80,220,80,0.55)","BB Lower")]:
            fig.add_trace(go.Scatter(x=df["time"],y=df[col_],line=dict(color=color_,width=1),name=name_),row=1,col=1)
        for ec,eco,en in [("EMA5","#FFD740","EMA5"),("EMA20","#40C4FF","EMA20"),("EMA50","#E040FB","EMA50")]:
            fig.add_trace(go.Scatter(x=df["time"],y=df[ec],line=dict(color=eco,width=1.4,dash="dot"),
                                     name=en,opacity=0.85),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["Tenkan"],line=dict(color="#00E5FF",width=1.8),name="Tenkan"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["Kijun"],line=dict(color="#FF6B6B",width=1.8),name="Kijun"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["Chikou"],line=dict(color="#B39DDB",width=1.1,dash="dot"),name="Chikou",opacity=0.7),row=1,col=1)
        spa=df["SpanA"].values; spb=df["SpanB"].values; tms=df["time"].values
        bull_a=np.where(spa>=spb,spa,np.nan); bull_b=np.where(spa>=spb,spb,np.nan)
        bear_a=np.where(spa<spb,spa,np.nan);  bear_b=np.where(spa<spb,spb,np.nan)
        fig.add_trace(go.Scatter(x=tms,y=bull_a,line=dict(color="rgba(0,0,0,0)",width=0),showlegend=False,name="_ba"),row=1,col=1)
        fig.add_trace(go.Scatter(x=tms,y=bull_b,fill="tonexty",fillcolor="rgba(38,166,154,0.15)",
                                 line=dict(color="#26a69a",width=0.6),name="Kumo Bull"),row=1,col=1)
        fig.add_trace(go.Scatter(x=tms,y=bear_b,line=dict(color="rgba(0,0,0,0)",width=0),showlegend=False,name="_bb"),row=1,col=1)
        fig.add_trace(go.Scatter(x=tms,y=bear_a,fill="tonexty",fillcolor="rgba(239,83,80,0.15)",
                                 line=dict(color="#ef5350",width=0.6),name="Kumo Bear"),row=1,col=1)

        # ICT Order Blocks on chart
        for ob in ict_result["order_blocks"]:
            fill_c="rgba(0,200,83,0.12)" if ob["type"]=="bullish" else "rgba(255,23,68,0.12)"
            line_c="#00C853" if ob["type"]=="bullish" else "#FF1744"
            x0=df["time"].iloc[max(0,ob["idx"]-1)] if ob["idx"]<len(df) else df["time"].iloc[-5]
            fig.add_shape(type="rect",xref="x",yref="y",
                          x0=x0,x1=df["time"].iloc[-1],y0=ob["bottom"],y1=ob["top"],
                          fillcolor=fill_c,line=dict(color=line_c,width=1.2,dash="dot"),row=1,col=1)
            fig.add_annotation(xref="x",yref="y",x=df["time"].iloc[-1],y=(ob["top"]+ob["bottom"])/2,
                               text=f" OB {'🟢' if ob['type']=='bullish' else '🔴'}",
                               showarrow=False,font=dict(color=line_c,size=8),xanchor="left",row=1,col=1)

        # ICT FVG on chart
        for fvg in ict_result["fvg_list"][-3:]:
            fill_c="rgba(0,229,118,0.08)" if fvg["type"]=="bullish" else "rgba(255,82,82,0.08)"
            line_c="#00E676" if fvg["type"]=="bullish" else "#FF5252"
            fig.add_shape(type="rect",xref="x",yref="y",
                          x0=fvg["time"],x1=df["time"].iloc[-1],
                          y0=fvg["bottom"],y1=fvg["top"],
                          fillcolor=fill_c,line=dict(color=line_c,width=0.8,dash="dash"),row=1,col=1)

        # OTE zone
        if ict_result["ote_zone"]:
            ote=ict_result["ote_zone"]
            ote_c="rgba(255,167,38,0.15)" if ote["type"]=="bullish" else "rgba(224,64,251,0.12)"
            fig.add_shape(type="rect",xref="paper",yref="y",x0=0,x1=1,
                          y0=ote["low"],y1=ote["high"],
                          fillcolor=ote_c,line=dict(color="#FFA726",width=1.2,dash="dot"),row=1,col=1)
            fig.add_annotation(xref="paper",yref="y",x=1,y=(ote["low"]+ote["high"])/2,
                               text=" OTE Zone",showarrow=False,
                               font=dict(color="#FFA726",size=9),xanchor="left",row=1,col=1)

        # Fibonacci
        fib_pal={"0.0% (Đỉnh)":"#9E9E9E","0.0% (Đáy)":"#9E9E9E","23.6%":"#7986CB","38.2%":"#29B6F6",
                 "50.0%":"#EF5350","61.8% ✨":"#FFA726","65.0% 🏅":"#FF7043","78.6%":"#AB47BC",
                 "100.0% (Đỉnh)":"#9E9E9E","100.0% (Đáy)":"#9E9E9E",
                 "127.2% 📉":"#546E7A","127.2% 📈":"#546E7A","161.8% 📉":"#37474F","161.8% 📈":"#37474F"}
        x_range=[df["time"].iloc[0],df["time"].iloc[-1]]; x_label=df["time"].iloc[-1]
        for label,price in fib_levels.items():
            if label.startswith("_"): continue
            is_key=label in ("38.2%","50.0%","61.8% ✨")
            fc=fib_pal.get(label,"#9E9E9E")
            fig.add_trace(go.Scatter(x=x_range,y=[price,price],mode="lines",
                line=dict(color=fc,width=2.5 if is_key else 1.0,dash="dash" if is_key else "dot"),
                name=f"Fib {label}",text=f"Fib {label} {price:,.0f}",hoverinfo="text"),row=1,col=1)
            fig.add_annotation(xref="x",yref="y",x=x_label,y=price,
                text=f"  {label}·{price:,.0f}",showarrow=False,
                font=dict(color=fc,size=9 if is_key else 8),xanchor="left",row=1,col=1)

        # Volume
        bc=["#ef5350" if df["close"].iloc[i]<df["open"].iloc[i] else "#26a69a" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df["time"],y=df["volume"],marker_color=bc,name="Volume",showlegend=False),row=2,col=1)

        # RSI + ADX
        fig.add_trace(go.Scatter(x=df["time"],y=df["RSI"],line=dict(color="#FF9800",width=1.8),name="RSI"),row=3,col=1)
        for lv,lc in [(70,"rgba(255,80,80,0.6)"),(30,"rgba(80,200,80,0.6)")]:
            fig.add_shape(type="line",xref="paper",yref="y3",x0=0,x1=1,y0=lv,y1=lv,
                          line=dict(color=lc,width=1,dash="dash"))
        fig.add_trace(go.Scatter(x=df["time"],y=df["ADX"],line=dict(color="#E040FB",width=1.5,dash="dot"),name="ADX"),row=3,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["DI_Plus"],line=dict(color="#26a69a",width=1,dash="dot"),name="DI+"),row=3,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["DI_Minus"],line=dict(color="#ef5350",width=1,dash="dot"),name="DI-"),row=3,col=1)

        # MACD + MCDX
        hc=np.where(df["MACD_Diff"]>=0,"#26a69a","#ef5350")
        fig.add_trace(go.Bar(x=df["time"],y=df["MACD_Diff"],marker_color=hc,name="MACD Hist"),row=4,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["MACD"],line=dict(color="#2196F3",width=1.5),name="MACD"),row=4,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["MACD_Signal"],line=dict(color="#FF5722",width=1.5),name="Signal"),row=4,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["MCDX"],line=dict(color="#E040FB",width=2,dash="dot"),name="MCDX"),row=4,col=1)

        # EMA panel
        for ec,eco,en in [("EMA5","#FFD740","EMA5"),("EMA20","#40C4FF","EMA20"),("EMA50","#E040FB","EMA50")]:
            fig.add_trace(go.Scatter(x=df["time"],y=df[ec],line=dict(color=eco,width=1.6),name=f"{en}(p)"),row=5,col=1)
        fig.add_trace(go.Scatter(x=df["time"],y=df["close"],line=dict(color="rgba(255,255,255,0.5)",width=1),name="Giá"),row=5,col=1)

        fig.update_layout(template="plotly_dark",height=1200,xaxis_rangeslider_visible=False,
            legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                        font=dict(size=9),bgcolor="rgba(0,0,0,0.3)",borderwidth=1),
            margin=dict(l=10,r=140,t=60,b=10))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True,gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig,use_container_width=True)

        # ── ICT SECTION ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Phân Tích ICT — Inner Circle Trader")
        st.caption("Market Structure · Order Blocks · Fair Value Gaps · Liquidity · OTE")

        ict_col = {"bullish":"#00E676","bearish":"#FF5252","neutral":"#FFD740"}
        ict_c = "#00E676" if ict_result["score"]>0 else ("#FF5252" if ict_result["score"]<0 else "#FFD740")
        st.markdown(
            f"""<div style="background:rgba(64,196,255,0.07);border:1px solid #40C4FF44;border-left:4px solid #40C4FF;
                border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.75rem;color:#aaa;">🏗️ Market Structure</div>
         <div style="font-size:1.1rem;font-weight:700;color:#40C4FF;">{ict_result['structure']}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">📦 Order Blocks</div>
         <div style="font-size:1.1rem;font-weight:700;color:#FFA726;">{len(ict_result['order_blocks'])} vùng</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">📊 FVG</div>
         <div style="font-size:1.1rem;font-weight:700;color:#E040FB;">{len(ict_result['fvg_list'])} gap</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">🎯 ICT Score</div>
         <div style="font-size:1.1rem;font-weight:700;color:{ict_c};">{ict_result['score']:+d}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

        if ict_result["signals"]:
            ict_s1, ict_s2 = st.columns(2)
            for i, (sname, sdesc, stype) in enumerate(ict_result["signals"]):
                sc = ict_col.get(stype, "#aaa")
                with (ict_s1 if i%2==0 else ict_s2):
                    st.markdown(f"<div style='border-left:3px solid {sc};padding:8px 12px;margin:4px 0;"
                                f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                                f"<b style='color:{sc};'>{sname}</b><br>"
                                f"<span style='color:#bbb;font-size:0.82rem;'>{sdesc}</span></div>",
                                unsafe_allow_html=True)

        with st.expander("📋 Chi tiết Order Blocks & FVG"):
            ob_c1, ob_c2 = st.columns(2)
            with ob_c1:
                st.markdown("**📦 Order Blocks gần nhất:**")
                for ob in ict_result["order_blocks"]:
                    c_="🟢" if ob["type"]=="bullish" else "🔴"
                    st.markdown(f"- {c_} {ob['desc']}")
                if not ict_result["order_blocks"]: st.info("Không phát hiện OB trong 30 phiên")
            with ob_c2:
                st.markdown("**📊 Fair Value Gaps gần nhất:**")
                for fvg in ict_result["fvg_list"][-5:]:
                    c_="🟢" if fvg["type"]=="bullish" else "🔴"
                    st.markdown(f"- {c_} FVG {fvg['type']} tại {fvg['bottom']:,.0f}–{fvg['top']:,.0f} ({fvg['pct']:.1f}%) — phiên {fvg['time']}")
                if not ict_result["fvg_list"]: st.info("Không phát hiện FVG đáng kể")

        if ict_result["ote_zone"]:
            ote=ict_result["ote_zone"]
            ote_c_="#FFA726" if ote["type"]=="bullish" else "#E040FB"
            st.info(f"🎯 **{ote['desc']}** — {'Vùng vào mua lý tưởng theo ICT' if ote['type']=='bullish' else 'Vùng vào bán lý tưởng theo ICT'}")

        if ict_result["liq_highs"] or ict_result["liq_lows"]:
            st.markdown(f"**⚠️ Vùng Liquidity (Stop Hunt Zones):** "
                        f"Equal Highs: {', '.join([f'{h:,.0f}' for h in ict_result['liq_highs']])} | "
                        f"Equal Lows: {', '.join([f'{l:,.0f}' for l in ict_result['liq_lows']])}")

        # ── VSA SECTION ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Phân Tích VSA — Volume Spread Analysis")
        st.caption("Stopping Volume · No Demand · No Supply · Climax · Upthrust · Test")

        vsa_c = "#00E676" if vsa_result["score"]>0 else ("#FF5252" if vsa_result["score"]<0 else "#FFD740")
        bull_vsa=[p for p in vsa_result["patterns"] if any(k in p[0] for k in ["Stopping","No Supply","Selling Climax","Test","✅"])]
        bear_vsa=[p for p in vsa_result["patterns"] if any(k in p[0] for k in ["No Demand","Buying Climax","Upthrust","❌","💥 Buy"])]

        st.markdown(
            f"""<div style="background:rgba(255,167,38,0.07);border:1px solid #FFA72644;border-left:4px solid #FFA726;
                border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.75rem;color:#aaa;">🟢 Tín hiệu Tăng</div>
         <div style="font-size:1.5rem;font-weight:800;color:#00E676;">{len(bull_vsa)}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">🔴 Tín hiệu Giảm</div>
         <div style="font-size:1.5rem;font-weight:800;color:#FF5252;">{len(bear_vsa)}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">📋 Tổng Patterns</div>
         <div style="font-size:1.5rem;font-weight:800;color:#FFA726;">{len(vsa_result['patterns'])}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">⚖️ VSA Score</div>
         <div style="font-size:1.5rem;font-weight:800;color:{vsa_c};">{vsa_result['score']:+d}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

        if vsa_result["patterns"]:
            for pname, pt, pc_, pdesc in vsa_result["patterns"]:
                is_bull = any(k in pname for k in ["Stopping","No Supply","Selling Climax","Test","✅"])
                is_bear = any(k in pname for k in ["No Demand","Buying Climax","Upthrust","❌","💥 Buy"])
                color_ = "#00E676" if is_bull else ("#FF5252" if is_bear else "#FFD740")
                st.markdown(f"<div style='border-left:3px solid {color_};padding:8px 14px;margin:4px 0;"
                            f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                            f"<b style='color:{color_};'>{pname}</b> "
                            f"<span style='color:#888;font-size:0.78rem;'>phiên {pt} · {pc_:,.0f} đ</span><br>"
                            f"<span style='color:#bbb;font-size:0.82rem;'>{pdesc}</span></div>",
                            unsafe_allow_html=True)
        else:
            st.info("Không phát hiện VSA pattern đáng kể trong 15 phiên gần nhất")

        # ── PRICE ACTION SECTION ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("🕯️ Phân Tích Price Action")
        st.caption("Market Structure · Inside/Outside Bar · Key S/R · Pin Bar · Consolidation")

        pa_c = "#00E676" if pa_result["score"]>0 else ("#FF5252" if pa_result["score"]<0 else "#FFD740")
        st.markdown(
            f"""<div style="background:rgba(224,64,251,0.07);border:1px solid #E040FB44;border-left:4px solid #E040FB;
                border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.75rem;color:#aaa;">📐 Cấu trúc PA</div>
         <div style="font-size:1.0rem;font-weight:700;color:#E040FB;">{pa_result.get('trend_structure','N/A')}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">📦 Inside Bars</div>
         <div style="font-size:1.5rem;font-weight:800;color:#FFA726;">{pa_result.get('inside_bars',0)}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">🔑 Key Levels</div>
         <div style="font-size:1.5rem;font-weight:800;color:#40C4FF;">{len(pa_result.get('key_levels',[]))}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">⚖️ PA Score</div>
         <div style="font-size:1.5rem;font-weight:800;color:{pa_c};">{pa_result['score']:+d}</div></div>
  </div>
  {'<div style="margin-top:10px;font-size:0.82rem;color:#FFD740;">⏳ ĐANG TÍCH LŨY — Giá bị nén, sắp breakout</div>' if pa_result.get("is_consolidating") else ''}
</div>""", unsafe_allow_html=True)

        if pa_result["signals"]:
            pa_c1, pa_c2 = st.columns(2)
            for i, (sn, sd, st_) in enumerate(pa_result["signals"]):
                sc = {"bullish":"#00E676","bearish":"#FF5252","neutral":"#FFD740"}.get(st_,"#aaa")
                with (pa_c1 if i%2==0 else pa_c2):
                    st.markdown(f"<div style='border-left:3px solid {sc};padding:8px 12px;margin:4px 0;"
                                f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                                f"<b style='color:{sc};'>{sn}</b><br>"
                                f"<span style='color:#bbb;font-size:0.82rem;'>{sd}</span></div>",
                                unsafe_allow_html=True)

        if pa_result.get("key_levels"):
            st.markdown("**🔑 Key S/R Levels (PA):**")
            kl_html=""
            for kl in pa_result["key_levels"]:
                c_="#00E676" if kl["type"]=="support" else "#FF5252"
                kl_html+=f"<span style='background:{c_}22;border:1px solid {c_};border-radius:12px;padding:3px 10px;margin:3px;font-size:0.8rem;color:{c_};display:inline-block;'>{kl['label']}</span>"
            st.markdown(kl_html, unsafe_allow_html=True)

        # ── OVERFLOW SECTION ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💧 Phân Tích Overflow — Volume · Price · Breakout · Gap")
        st.caption("Volume Overflow · BB Extension · Breakout Confirmation · Gap Analysis · Exhaustion")

        ovf_c = "#00E676" if ovf_result["score"]>0 else ("#FF5252" if ovf_result["score"]<0 else "#FFD740")
        st.markdown(
            f"""<div style="background:rgba(255,23,68,0.07);border:1px solid #FF174444;border-left:4px solid #FF1744;
                border-radius:10px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;">
    <div><div style="font-size:0.75rem;color:#aaa;">🌊 Volume Events</div>
         <div style="font-size:1.5rem;font-weight:800;color:#FF9100;">{len(ovf_result['overflow_events'])}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">📐 Gaps</div>
         <div style="font-size:1.5rem;font-weight:800;color:#40C4FF;">{len(ovf_result['gaps'])}</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">🔁 Consecutive Run</div>
         <div style="font-size:1.5rem;font-weight:800;color:#FFD740;">{ovf_result.get('consecutive_run',0)} phiên</div></div>
    <div><div style="font-size:0.75rem;color:#aaa;">⚖️ Overflow Score</div>
         <div style="font-size:1.5rem;font-weight:800;color:{ovf_c};">{ovf_result['score']:+d}</div></div>
  </div>
  {'<div style="margin-top:8px;color:#FF5252;font-size:0.82rem;">⚡ GIÁ ĐANG VƯỢT BB UPPER — Quá mua, hồi về trung bình</div>' if ovf_result.get("bb_overflow_up") else ''}
  {'<div style="margin-top:8px;color:#00E676;font-size:0.82rem;">💧 GIÁ ĐANG DƯỚI BB LOWER — Quá bán, hồi phục khả năng cao</div>' if ovf_result.get("bb_overflow_down") else ''}
</div>""", unsafe_allow_html=True)

        if ovf_result["signals"]:
            ovf_c1, ovf_c2 = st.columns(2)
            for i, (sn, sd, st_) in enumerate(ovf_result["signals"]):
                sc={"bullish":"#00E676","bearish":"#FF5252","neutral":"#FFD740"}.get(st_,"#aaa")
                with (ovf_c1 if i%2==0 else ovf_c2):
                    st.markdown(f"<div style='border-left:3px solid {sc};padding:8px 12px;margin:4px 0;"
                                f"background:rgba(255,255,255,0.04);border-radius:0 8px 8px 0;'>"
                                f"<b style='color:{sc};'>{sn}</b><br>"
                                f"<span style='color:#bbb;font-size:0.82rem;'>{sd}</span></div>",
                                unsafe_allow_html=True)

        # ── SMART MONEY PHASE ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🐋 Pha Dòng Tiền Thông Minh — Wyckoff & Smart Money")
        sm=sm_result
        pct_bar=max(0,min(100,(sm["score"]+sm["max_score"])/(2*sm["max_score"])*100))
        st.markdown(
            f"""<div style="border:2px solid {sm['border']};border-radius:14px;padding:20px 28px;
                background:{sm['bg']};margin-bottom:16px;">
  <div style="font-size:1.9rem;font-weight:800;color:{sm['color']};letter-spacing:2px;">
    {sm['icon']} &nbsp; GIAI ĐOẠN: {sm['phase']}
  </div>
  <div style="margin-top:8px;font-size:1rem;color:#ddd;line-height:1.6;">{sm['phase_desc']}</div>
  <div style="margin-top:12px;background:rgba(255,255,255,0.08);border-radius:8px;height:10px;width:100%;">
    <div style="height:10px;width:{pct_bar:.0f}%;border-radius:8px;
                background:linear-gradient(90deg,#ef5350,#FFD740,#00C853);"></div>
  </div>
  <div style="margin-top:8px;font-size:0.9rem;color:#aaa;">
    Điểm: <b style="color:{sm['color']};">{sm['score']:+.1f} / {sm['max_score']:.0f}</b>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Data table ────────────────────────────────────────────────────
        show_cols=[c for c in ["time","open","high","low","close","volume","RSI","MACD","MCDX","ADX","EMA20"] if c in df.columns]
        st.write("**📋 Bảng dữ liệu 7 phiên gần nhất:**")
        st.dataframe(df.tail(7)[show_cols].round(2),hide_index=True,use_container_width=True)

        with st.expander("📐 Bảng Fibonacci Retracement"):
            fib_df=pd.DataFrame([{"Mức":k,"Giá (đ)":f"{v:,.0f}"} for k,v in fib_levels.items() if not k.startswith("_")])
            st.dataframe(fib_df,hide_index=True,use_container_width=True)

        # ── AI ANALYSIS ───────────────────────────────────────────────────
        if not api_key:
            st.warning("💡 Nhập API Key ở thanh bên trái để nhận phân tích từ AI.")
        else:
            st.markdown("---")
            provider="Claude (Anthropic)" if "Claude" in ai_provider else "Gemini (Google)"
            st.subheader(f"🤖 Báo Cáo AI Tổng Hợp — {provider}")

            ai_data=df.tail(10)[show_cols].round(2).to_string(index=False)
            fib_str="\n".join([f"  • Fib {k}: {v:,.0f} đ" for k,v in fib_levels.items() if not k.startswith("_") and isinstance(v,(int,float))])
            ict_sigs="\n".join([f"  • {s[0]}: {s[1]}" for s in ict_result["signals"]]) or "  • Không có tín hiệu ICT đặc biệt"
            vsa_sigs="\n".join([f"  • {p[0]} (phiên {p[1]}): {p[3]}" for p in vsa_result["patterns"]]) or "  • Không có VSA pattern đặc biệt"
            pa_sigs ="\n".join([f"  • {s[0]}: {s[1]}" for s in pa_result["signals"]]) or "  • Không có PA signal đặc biệt"
            ovf_sigs="\n".join([f"  • {s[0]}: {s[1]}" for s in ovf_result["signals"]]) or "  • Không có Overflow signal"
            above_cloud=None
            if pd.notna(latest.get("SpanA")) and pd.notna(latest.get("SpanB")):
                above_cloud=latest["close"]>max(latest.get("SpanA"),latest.get("SpanB"))
            ichi_status=("trên mây (Bullish)" if above_cloud else ("dưới mây (Bearish)" if above_cloud is False else "trong mây"))
            macd_sig="cắt lên (tăng)" if latest["MACD"]>latest["MACD_Signal"] else "cắt xuống (giảm)"

            prompt=f"""
Bạn là hệ thống AI định lượng cao cấp chuyên phân tích thị trường chứng khoán Việt Nam.
Sử dụng đồng thời: ICT, VSA, Price Action, Overflow, Smart Money, RSI, MACD, ADX, EMA, Ichimoku, Fibonacci.

Dữ liệu 10 phiên gần nhất của mã **{ticker}**:
```
{ai_data}
```

Fibonacci:
{fib_str}

Tóm tắt chỉ báo:
- RSI: {rsi_val:.1f} — {rsi_note}
- MACD: {macd_sig}
- ADX: {adx_val:.1f} (DI+={di_p:.1f}/DI-={di_m:.1f})
- EMA: {ema5_v:,.0f}/{ema20_v:,.0f}/{ema50_v:,.0f} — {'Bull' if ema5_v>ema20_v else 'Bear'} Alignment
- Ichimoku: {ichi_status}
- MCDX: {mcdx_val:.4f}
- Smart Money: {sm_result['phase']} (score: {sm_result['score']:+.1f})

ICT Signals:
{ict_sigs}
ICT Summary: {ict_result['summary']}

VSA Patterns:
{vsa_sigs}

Price Action:
{pa_sigs}
PA: {pa_result.get('trend_structure','N/A')} · Inside Bars: {pa_result.get('inside_bars',0)}

Overflow:
{ovf_sigs}
Overflow: {ovf_result['summary']}

**Phân tích yêu cầu (tiếng Việt, quyết đoán):**

1. **ICT Analysis**: Market Structure (BOS/CHoCH), Order Blocks, FVG, OTE — đánh giá tổng thể
2. **VSA Analysis**: Các patterns volume phát hiện nói lên điều gì về ý định tiền lớn?
3. **Price Action**: Cấu trúc PA, key levels, setup hiện tại — breakout hay reversal?
4. **Overflow**: Có volume overflow hay price overflow không? Breakout xác nhận hay false break?
5. **Confluence**: Điểm hội tụ của ICT + VSA + PA + Overflow + Smart Money — tín hiệu nào mạnh nhất?
6. **Kết luận & Chiến lược**:
   - ✅ Khuyến nghị: **Mua / Bán / Chờ**
   - 📥 Vùng vào lệnh (Entry zone)
   - 🛑 Stop Loss (dựa trên Order Block / key level)
   - 🎯 Target 1 & Target 2 (FVG fill / swing high/low)
   - ⚖️ Risk/Reward ratio
   - ⚠️ Rủi ro chính cần chú ý
"""
            with st.spinner("⏳ AI đang phân tích đa phương pháp..."):
                try:
                    result = call_claude(api_key, prompt) if "Claude" in ai_provider else call_gemini(api_key, prompt)
                    st.markdown(result)
                except Exception as e:
                    err=str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        st.error("❌ Hết quota API. Chờ ~1 phút hoặc đổi sang Claude.")
                    elif "401" in err or "UNAUTHENTICATED" in err:
                        st.error("❌ API Key không hợp lệ.")
                    else:
                        st.error(f"❌ Lỗi AI: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET SCANNER
# ═════════════════════════════════════════════════════════════════════════════
with _tab2:
    st.subheader("🔍 Quét Toàn Thị Trường — ICT · VSA · PA · Smart Money")

    with st.spinner("🔄 Đang tải danh sách cổ phiếu từ nhiều nguồn..."):
        _auto_universe, _source_note = build_scan_universe()

    st.caption(f"📡 Nguồn: **{_source_note}**")
    _b1,_b2,_b3=st.columns(3)
    _b1.metric("🏆 VN30",len([t for t in VN30_TICKERS if t in _auto_universe]),"mã")
    _b2.metric("📈 Tổng thị trường",len(_auto_universe),"mã")
    _b3.metric("🔢 HOSE + HNX + UPCOM","Tất cả","không lọc")

    with st.expander("⚙️ Xem & chỉnh sửa danh sách quét"):
        custom_input=st.text_area("✏️ Chỉnh sửa danh sách:",value=", ".join(_auto_universe),height=130)
        _parsed=[t.strip().upper() for t in custom_input.replace("\n",",").split(",") if t.strip()]
        scan_list=sorted(set(_parsed)) if _parsed else _auto_universe
        st.info(f"📋 Sẽ quét **{len(scan_list)} mã**")

    # Filter options
    col_f1,col_f2,col_f3,col_f4=st.columns(4)
    with col_f1:
        filter_phase=st.multiselect("Lọc Smart Money:",["GOM HÀNG","ĐẨY GIÁ (MARKUP)","TRUNG TÍNH","XẢ HÀNG","ĐÈ GIÁ"],default=[])
    with col_f2:
        filter_rec=st.multiselect("Lọc khuyến nghị:",["✅ MUA","⏳ THEO DÕI","🚫 TRÁNH/BÁN"],default=[])
    with col_f3:
        filter_ict=st.multiselect("Lọc ICT Structure:",["UPTREND","DOWNTREND","SIDEWAYS"],default=[])
    with col_f4:
        filter_min_score=st.slider("Điểm tối thiểu:",min_value=-10,max_value=22,value=-5,step=1)

    if st.button("🚀 Chạy Quét Thị Trường", type="primary", key="scan_btn"):
        results=[]; errs=0
        progress_bar=st.progress(0,"Đang quét...")
        status_text=st.empty()

        for i,sym in enumerate(scan_list):
            status_text.markdown(f"⏳ Đang phân tích **{sym}** ({i+1}/{len(scan_list)})...")
            res=scan_single_stock(sym)
            if res: results.append(res)
            else: errs+=1
            progress_bar.progress((i+1)/len(scan_list),text=f"Đã quét {i+1}/{len(scan_list)} mã")

        progress_bar.empty(); status_text.empty()

        if not results: st.error("❌ Không quét được dữ liệu."); st.stop()

        q1,q2,q3=st.columns(3)
        q1.success(f"✅ **{len(results)} mã** quét thành công")
        q2.info(f"📊 Thất bại: **{errs} mã** (không có dữ liệu)")
        q3.metric("Tổng quét",len(scan_list),"mã")

        results.sort(key=lambda x:(_phase_order(x["sm_phase"]),-x["score"]))

        # Apply filters
        filtered=results
        if filter_phase: filtered=[r for r in filtered if r["sm_phase"] in filter_phase]
        if filter_rec:   filtered=[r for r in filtered if r["recommendation"] in filter_rec]
        if filter_ict:
            filtered=[r for r in filtered if any(ict_s in r.get("ict_label","") for ict_s in filter_ict)]
        filtered=[r for r in filtered if r["score"]>=filter_min_score]

        # Summary counts
        st.markdown("---")
        phase_counts=defaultdict(int)
        for r in results: phase_counts[r["sm_phase"]]+=1

        sum_cols=st.columns(5)
        phase_defs=[("GOM HÀNG","🏦","#00C853"),("ĐẨY GIÁ (MARKUP)","🚀","#40C4FF"),
                    ("TRUNG TÍNH","🔍","#FFD740"),("XẢ HÀNG","📤","#FF6D00"),("ĐÈ GIÁ","📉","#FF1744")]
        for idx,(ph,ic,co) in enumerate(phase_defs):
            cnt=phase_counts.get(ph,0)
            sum_cols[idx].markdown(
                f"<div style='text-align:center;padding:12px 6px;border-radius:10px;background:{co}18;border:1px solid {co}55;'>"
                f"<div style='font-size:1.5rem;'>{ic}</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{co};'>{cnt}</div>"
                f"<div style='font-size:0.75rem;color:#aaa;'>{ph}</div></div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("✅ MUA",sum(1 for r in results if r["recommendation"]=="✅ MUA"))
        m2.metric("⏳ Theo dõi",sum(1 for r in results if r["recommendation"]=="⏳ THEO DÕI"))
        m3.metric("🚫 Tránh",sum(1 for r in results if r["recommendation"]=="🚫 TRÁNH/BÁN"))
        m4.metric("⭐ Fib hấp dẫn",sum(1 for r in results if "Golden Pocket" in r["fib_zone"] or "Gần đáy" in r["fib_zone"]))
        m5.metric("🔬 BOS/CHoCH",sum(1 for r in results if "BOS" in r.get("ict_label","") or "CHoCH" in r.get("ict_label","")))
        m6.metric("💧 Vol Overflow",sum(1 for r in results if r.get("ovf_events",0)>0))

        st.markdown("---")
        st.markdown(f"### 📋 Kết quả — {len(filtered)}/{len(results)} mã (sau bộ lọc)")

        # Display by phase
        PHASE_META={
            "GOM HÀNG":("🏦","#00C853","rgba(0,200,83,0.10)","Tiền lớn bí mật tích lũy — Cơ hội sớm."),
            "ĐẨY GIÁ (MARKUP)":("🚀","#40C4FF","rgba(64,196,255,0.10)","Cá mập đang markup — Theo đà có thể."),
            "TRUNG TÍNH":("🔍","#FFD740","rgba(255,215,64,0.08)","Chưa rõ — Quan sát thêm."),
            "XẢ HÀNG":("📤","#FF6D00","rgba(255,109,0,0.10)","Tiền lớn phân phối — Cẩn thận."),
            "ĐÈ GIÁ":("📉","#FF1744","rgba(255,23,68,0.10)","Tay to xả mạnh — Không nên mua."),
        }
        by_phase=defaultdict(list)
        for r in filtered: by_phase[r["sm_phase"]].append(r)

        for ph_name,(ph_icon,ph_color,ph_bg,ph_desc) in PHASE_META.items():
            group=by_phase.get(ph_name,[])
            if not group: continue
            with st.expander(f"{ph_icon} {ph_name} — {len(group)} mã",expanded=(ph_name=="GOM HÀNG")):
                st.markdown(f"<div style='background:{ph_bg};border-left:4px solid {ph_color};"
                            f"padding:8px 16px;border-radius:0 8px 8px 0;margin-bottom:10px;"
                            f"color:#ddd;font-size:0.88rem;'>{ph_desc}</div>",unsafe_allow_html=True)
                rows=[]
                for r in group:
                    rows.append({
                        "Mã":r["symbol"],"Giá":f"{r['close']:,.0f}","%1P":f"{r['pct_chg']:+.2f}%",
                        "RSI":f"{r['rsi']:.1f}","ADX":f"{r.get('adx',0):.1f}",
                        "MACD":"📈" if r["macd_bull"] else "📉","EMA":"🟢" if r.get("ema_bull") else "🔴",
                        "ICT":r.get("ict_label","—")[:18],
                        "VSA Top":r.get("vsa_top","—")[:16],
                        "PA":r.get("pa_trend","—")[:14],
                        "Vol OVF":f"x{r.get('ovf_events',0)}" if r.get("ovf_events",0)>0 else "—",
                        "SM Score":f"{r['sm_score']:+.1f}",
                        "Fib":r["fib_zone"][:18],"Điểm":f"{r['score']}/{r['max_score']}",
                        "KN":r["recommendation"],
                    })
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

                # Top picks cards
                top_picks=[r for r in group if r["recommendation"]=="✅ MUA"]
                if top_picks and ph_name in ("GOM HÀNG","ĐẨY GIÁ (MARKUP)"):
                    st.markdown(f"#### ⭐ Top picks — {len(top_picks)} mã khuyến nghị MUA")
                    card_cols=st.columns(min(len(top_picks),4))
                    for ci,r in enumerate(top_picks[:8]):
                        chg_c="#26a69a" if r["pct_chg"]>=0 else "#ef5350"
                        fib_hl="#FFA726" if "Golden Pocket" in r["fib_zone"] else "#00C853" if "Gần đáy" in r["fib_zone"] else ph_color
                        ict_bg="#40C4FF22" if "UPTREND" in r.get("ict_label","") else "rgba(255,255,255,0.04)"
                        with card_cols[ci%4]:
                            st.markdown(
                                f"""<div style="border:1px solid {ph_color}55;border-top:3px solid {ph_color};
                                    border-radius:10px;padding:14px 12px;background:{ph_bg};margin-bottom:8px;">
  <div style="font-size:1.3rem;font-weight:800;color:{ph_color};">{r['symbol']}</div>
  <div style="font-size:1.05rem;font-weight:700;color:#fff;margin:4px 0;">
    {r['close']:,.0f} đ <span style="font-size:0.85rem;color:{chg_c};">{r['pct_chg']:+.2f}%</span></div>
  <div style="background:{ict_bg};border-radius:6px;padding:4px 8px;margin:6px 0;font-size:0.75rem;color:#40C4FF;">
    🔬 ICT: {r.get('ict_label','—')}</div>
  <div style="font-size:0.78rem;color:#bbb;line-height:1.7;">
    RSI: <b style="color:#FF9800;">{r['rsi']:.1f}</b> · ADX: <b style="color:#E040FB;">{r.get('adx',0):.1f}</b><br>
    VSA: {r.get('vsa_top','—')[:18]}<br>
    PA: {r.get('pa_trend','—')[:14]}<br>
    <span style="color:{fib_hl};">{r['fib_zone']}</span>
  </div>
  <div style="margin-top:8px;font-size:0.82rem;font-weight:700;color:#00C853;">
    Điểm: {r['score']}/{r['max_score']} · {r['recommendation']}
  </div>
</div>""", unsafe_allow_html=True)

        # ── Special Screens ───────────────────────────────────────────────
        st.markdown("---")
        # ICT BOS/CHoCH screen
        bos_stocks=[r for r in results if "BOS" in r.get("ict_label","") or "CHoCH" in r.get("ict_label","")]
        bos_stocks.sort(key=lambda x:-x["score"])
        if bos_stocks:
            st.markdown(f"### 🔬 ICT — Cổ Phiếu Có BOS / CHoCH ({len(bos_stocks)} mã)")
            bos_rows=[{"Mã":r["symbol"],"Giá":f"{r['close']:,.0f}","%1P":f"{r['pct_chg']:+.2f}%",
                       "ICT":r.get("ict_label",""),"RSI":f"{r['rsi']:.1f}",
                       "Smart Money":f"{r['sm_icon']} {r['sm_phase']}","KN":r["recommendation"]}
                      for r in bos_stocks]
            st.dataframe(pd.DataFrame(bos_rows),hide_index=True,use_container_width=True)

        # VSA anomaly screen
        vsa_bull=[r for r in results if r.get("vsa_score",0)>=2]
        vsa_bull.sort(key=lambda x:-x.get("vsa_score",0))
        if vsa_bull:
            st.markdown(f"### 📊 VSA — Tín Hiệu Tích Lũy Mạnh ({len(vsa_bull)} mã)")
            vsa_rows=[{"Mã":r["symbol"],"Giá":f"{r['close']:,.0f}","%1P":f"{r['pct_chg']:+.2f}%",
                       "VSA Score":f"{r.get('vsa_score',0):+d}","VSA Top":r.get("vsa_top","—"),
                       "PA":r.get("pa_trend","—"),"Smart Money":f"{r['sm_icon']} {r['sm_phase']}",
                       "KN":r["recommendation"]}
                      for r in vsa_bull[:30]]
            st.dataframe(pd.DataFrame(vsa_rows),hide_index=True,use_container_width=True)

        # Overflow screen
        ovf_stocks=[r for r in results if r.get("ovf_events",0)>=2 or r.get("ovf_score",0)>=3]
        ovf_stocks.sort(key=lambda x:-x.get("ovf_score",0))
        if ovf_stocks:
            st.markdown(f"### 💧 OVERFLOW — Cổ Phiếu Có Volume/Price Bất Thường ({len(ovf_stocks)} mã)")
            ovf_rows=[{"Mã":r["symbol"],"Giá":f"{r['close']:,.0f}","%1P":f"{r['pct_chg']:+.2f}%",
                       "Vol Events":r.get("ovf_events",0),"Ovf Score":f"{r.get('ovf_score',0):+d}",
                       "Run":f"{r.get('consecutive_run',0)}P","KN":r["recommendation"]}
                      for r in ovf_stocks[:25]]
            st.dataframe(pd.DataFrame(ovf_rows),hide_index=True,use_container_width=True)

        # Fib buy zone
        vung_mua=[r for r in results if any(kw in r["fib_zone"] for kw in ["Golden Pocket","Gần đáy","Hỗ trợ"])]
        vung_mua.sort(key=lambda x:-x["score"])
        if vung_mua:
            st.markdown(f"### 🛒 Về Vùng Mua Fibonacci Hợp Lý — {len(vung_mua)} mã")
            buy_rows=[{"Mã":r["symbol"],"Giá":f"{r['close']:,.0f}","%1P":f"{r['pct_chg']:+.2f}%",
                       "RSI":f"{r['rsi']:.1f}","ICT":r.get("ict_label","—")[:16],
                       "SM":f"{r['sm_icon']} {r['sm_phase']}","Fib":r["fib_zone"],
                       "Điểm":f"{r['score']}/{r['max_score']}","KN":r["recommendation"]}
                      for r in vung_mua]
            st.dataframe(pd.DataFrame(buy_rows),hide_index=True,use_container_width=True)

    else:
        st.info(
            "💡 Nhấn **Chạy Quét Thị Trường** để bắt đầu.\n\n"
            "**Phương pháp phân tích tích hợp:**\n"
            "- 🔬 **ICT**: Market Structure (BOS/CHoCH) · Order Blocks · FVG · Liquidity · OTE\n"
            "- 📊 **VSA**: Stopping Volume · No Demand/Supply · Climax · Upthrust · Test\n"
            "- 🕯️ **Price Action**: Inside/Outside Bar · Key S/R · Pin Bar · Trend Structure\n"
            "- 💧 **Overflow**: Volume Overflow · BB Extension · Breakout · Gap · Exhaustion\n"
            "- 🐋 **Smart Money**: Wyckoff · OBV · CMF · MFI · A/D Line\n"
            "- 📈 **Classic**: RSI · MACD · ADX · EMA 5/20/50 · Ichimoku · Fibonacci\n\n"
            f"**Danh sách quét:** {len(_auto_universe)} mã — HOSE · HNX · UPCOM (không lọc theo giá/KL)"
        )
