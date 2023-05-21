import requests
import pandas as pd
api_endpoint = "https://api.binance.com/api/v3/klines"


def get_candles(symbol, num_candles, interval="1h"):
    get_params = {"symbol": symbol, "interval": interval, "limit": num_candles}
    r = requests.get(api_endpoint, params=get_params)
    if "code" in r.json():
        raise Exception("Error in Binance API Call")
    candles_dataframe = pd.read_json(r.text).drop([5, 6, 7, 8, 9, 10, 11], axis=1)
    candles_dataframe.columns = ["Time_Open", "Open", "High", "Low", "Close"]
    return  candles_dataframe



def format_candles_for_plotly(pandas_candles):

    plotly_trace = {
        "x": pandas_candles["Time_Open"].values.tolist(),
        "open": pandas_candles["Open"].values.tolist(),
        "high": pandas_candles["High"].values.tolist(),
        "low": pandas_candles["Low"].values.tolist(),
        "close": pandas_candles["Close"].values.tolist(),
        
        "type": "candlestick"
    }

    return plotly_trace
