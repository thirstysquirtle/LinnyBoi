from binance.um_futures import UMFutures
from binance.error import ClientError
from dotenv import load_dotenv
import os

load_dotenv()

um_futures_client = UMFutures(
    key=os.environ["futures_key"],
    secret=os.environ["futures_secret"],
    base_url="https://testnet.binancefuture.com",
)


def check_for_position(symbol, account=um_futures_client):
    position = next(
        filter(
            lambda position: position["symbol"] == symbol,
            um_futures_client.get_position_risk(),
        )
    )
    return False if float(position["positionAmt"]) == 0 else True


def bn_order(
    symbol: str,
    side: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    quantity: float = 0.001,
):
    if check_for_position(symbol=symbol):
        return "Position Already Open"
    oposite = "SELL" if side == "BUY" else "BUY"
    params = [
        {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": f"{quantity}",
            "price": f"{price + 0.5}",
            "timeInForce": "IOC",
        },
        {
            "symbol": symbol,
            "side": oposite,
            "type": "TAKE_PROFIT_MARKET",
            "closePosition": "True",
            "stopPrice": f"{take_profit}",
        },
        {
            "symbol": symbol,
            "side": oposite,
            "type": "STOP_MARKET",
            "closePosition": "True",
            "stopPrice": f"{stop_loss}",
        },
    ]
    um_futures_client.cancel_open_orders(symbol=symbol)
    try:
        response = um_futures_client.new_batch_order(params)
        print(response)
    except ClientError as error:
        print(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


def verify_stop_loss(entry_price, stop_loss, side, max_risk=0.02):
    if side == "BUY":
        percent_loss = 1 - (stop_loss / entry_price)
        if percent_loss < max_risk and percent_loss > 0:
            return True
    if side == "SELL":
        percent_loss = (stop_loss / entry_price) - 1
        if percent_loss < max_risk and percent_loss > 0:
            return True
    return False


print(
    bn_order(
        symbol="BTCUSDT",
        side="BUY",
        price=25285.0,
        stop_loss=24000,
        take_profit=26000,
        quantity=0.01,
    )
)
