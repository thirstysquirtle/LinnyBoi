import requests
import pandas as pd
import pymongo

REGRESSION_UNITS = 10
SNAPSHOT_TIMEPADDING = 5
RISK_PER_TRADE_IN_PERCENT = 2
CANDLES_BACK_STOP_LOSS = 2


# BNBBUSD_params = {"symbol": "BNBBUSD", "interval": "1h", "limit": ""}
# endpoint = "https://api.binance.com/api/v3/klines"
# r = requests.get(endpoint, params=BNBBUSD_params)
# candles = r.text

# candles =  pd.read_json(candles)
# print(candles)


client = pymongo.MongoClient("mongodb+srv://bob:w1w2w3w4@freetier.lenkq.mongodb.net/?retryWrites=true&w=majority")
db = client.test

print(db["test"])