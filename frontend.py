import flask as fsk
import binance_functions as bn
import regression_model as reg
import json


app = fsk.Flask(__name__)


@app.route("/")
def serve_index():
    return fsk.send_from_directory("frontend", "index.html")


@app.route("/<string:filename>")
def serve_static_files(filename):
    return fsk.send_from_directory("frontend", filename)


@app.route("/test")
def serve_plotly_json():
    candles = bn.get_candles("BTCUSDT", 50)
    candles_trace = bn.format_candles_for_plotly(candles)
    regression_trace = reg.format_regression_for_plotly(candles["Time_Open"].values)
    return fsk.jsonify([candles_trace, regression_trace])


app.run(port=1234)
