import requests
def get_vn30f_data(symbol="VN30F2506", resolution="1"):
    url = f"https://iboard.ssi.com.vn/dchart/api/1.1/chart/history"
    params = {"symbol": symbol, "resolution": resolution, ...}
    r = requests.get(url, params=params)
    return r.json()
