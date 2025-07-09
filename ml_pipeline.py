import os
import logging
from flask import Flask, jsonify, request

# ─── Guarded MT5 import ───────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logging.warning("MetaTrader5 module not installed or not on this platform; MT5 fetches will be skipped.")

import joblib
import numpy   as np
import pandas  as pd
import xgboost as xgb

# ─── Logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ─── S3 PERSISTENCE ──────────────────────────────────────────────────
AWS_KEY    = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
BUCKET     = os.getenv("S3_BUCKET")
MODEL_KEY  = "models/xgb_model.pkl"
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/xgb_model.pkl")

s3_client = None
if AWS_KEY and AWS_SECRET and BUCKET:
    import boto3
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
else:
    logging.info("No S3 credentials; skipping S3 persistence.")

def upload_model():
    if not s3_client:
        return False
    s3_client.upload_file(MODEL_PATH, BUCKET, MODEL_KEY)
    logging.info("Model uploaded to S3: %s/%s", BUCKET, MODEL_KEY)
    return True

def download_model():
    if not s3_client:
        return False
    try:
        s3_client.download_file(BUCKET, MODEL_KEY, MODEL_PATH)
        logging.info("Model downloaded from S3 into %s", MODEL_PATH)
        return True
    except Exception as e:
        logging.warning("Failed to download model: %s", e)
        return False

# ─── INITIAL MODEL LOAD ──────────────────────────────────────────────
logging.info("Trying to load existing model from %s …", MODEL_PATH)
if not download_model() and not os.path.exists(MODEL_PATH):
    logging.warning("No model present on disk; call POST /train to create one.")

# ─── MT5 CONFIG ───────────────────────────────────────────────────────
MT5_SERVER   = os.getenv("MT5_SERVER")
MT5_LOGIN    = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
SYMBOL       = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME    = os.getenv("TIMEFRAME", "TIMEFRAME_H1")
LOOKBACK     = int(os.getenv("LOOKBACK", "1000"))

# ─── FEATURE ENGINEERING STUBS ────────────────────────────────────────
def add_fvg_features(df):         ...
def add_trailing_sl(df, **kw):    ...
def add_supply_demand(df, **kw):  ...
def label_signals(df, **kw):      ...

def build_features(df):
    df = add_fvg_features(df)
    df = add_trailing_sl(df)
    df = add_supply_demand(df)
    df["return"] = df["close"].pct_change()
    df = label_signals(df)
    return df.dropna()

# ─── MT5 OHLC FETCH ───────────────────────────────────────────────────
def fetch_ohlc(symbol, timeframe, lookback):
    if not mt5:
        raise RuntimeError("MT5 unavailable")
    if not mt5.initialize(server=MT5_SERVER, login=int(MT5_LOGIN), password=MT5_PASSWORD):
        raise RuntimeError("MT5 init failed: " + mt5.last_error()[2])
    rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, timeframe), 0, lookback)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open","high","low","close","tick_volume"]]

FEATURE_COLS = ["return","fvg_up","fvg_down","tsl","dist_supply","dist_demand"]

# ─── FLASK SETUP ─────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, LOOKBACK)
    except Exception as e:
        logging.error("MT5 fetch failed: %s", e)
        return jsonify(error="MT5 failed", details=str(e)), 503

    df_feat = build_features(df)
    X, y = df_feat[FEATURE_COLS], df_feat["signal"].astype(int)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    upload_model()
    return jsonify(status="retrained"), 200

@app.route("/signal.json", methods=["GET"])
def signal():
    if not os.path.exists(MODEL_PATH):
        return jsonify(error="Model not available. Please /train first."), 503
    model = joblib.load(MODEL_PATH)

    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, LOOKBACK)
    except Exception as e:
        logging.error("MT5 fetch failed: %s", e)
        return jsonify(error="MT5 failed", details=str(e)), 503

    df_feat = build_features(df).tail(1)
    proba   = model.predict_proba(df_feat[FEATURE_COLS])[0]
    labels  = {0: "sell", 1: "hold", 2: "buy"}

    resp = {
        "symbol":       SYMBOL,
        "timestamp":    df_feat.index[0].isoformat(),
        "signal":       labels[int(np.argmax(proba))],
        "probabilities": {
            "sell": round(float(proba[0]), 4),
            "hold": round(float(proba[1]), 4),
            "buy":  round(float(proba[2]), 4)
        }
    }
    logging.info("Signal→ %s", resp)
    return jsonify(resp), 200

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify(status="healthy"), 200

if __name__ == "__main__":
    # production WSGI server will be invoked by Render via gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
