import requests
api_endpoint = "https://api.binance.com/api/v3/klines"


def get_candles(symbol, num_candles, interval="1h"):
    get_params = {"symbol": symbol, "interval": interval, "limit": num_candles}
    r = requests.get(api_endpoint, params=get_params)
    return r.json()

# print(get_candles("BNBBUSD", 24))