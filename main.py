import pandas as pd


#custom Modules
import regression_model as reg
import binance_functions as bn
import my_logging as log
NUM_CANDLES = 12

TIME_INTERVAL = "1h"

SNAPSHOT_TIMEPADDING = 5
RISK_PER_TRADE_IN_PERCENT = 2
CANDLES_BACK_STOP_LOSS = 2

print(bn.get_candles("BNBBUSD", NUM_CANDLES))

print(reg.model([1.4,6,1,9]))




print(reg.model)