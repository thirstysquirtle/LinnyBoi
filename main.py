NUM_CANDLES = 3
SYMBOL = "BTCUSDT"

TIME_INTERVAL = "1h"

RISK_PER_TRADE_IN_PERCENT = 2
CANDLES_BACK_STOP_LOSS = 2

# my modules
import binance_functions as bn
import regression_model as reg

# import my_logging as log


candles = bn.get_candles(SYMBOL, NUM_CANDLES, interval=TIME_INTERVAL)
reg.model.train_model(
    candles["Time_Open"], reg.get_candle_middle(candles), num_epochs=80000
)

print(reg.get_future_prediction_xy(reg.model, candles["Time_Open"]))
