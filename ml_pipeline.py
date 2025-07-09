import os
import logging
import joblib

import pandas  as pd
import numpy   as np
import xgboost as xgb

from flask import Flask, jsonify, request

# ─── Optional AWS/S3 persistence ─────────────────────────────────────────
import boto3
AWS_KEY    = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
BUCKET     = os.getenv("S3_BUCKET")
MODEL_KEY  = "models/xgb_model.pkl"
LOCAL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(
        os.getenv("HOME", "/tmp"),
        "xgb_model.pkl"
    )
)

s3_client = None
if AWS_KEY and AWS_SECRET and BUCKET:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
else:
    logging.warning("S3 credentials not found. Skipping S3 persistence.")

def upload_model():
    if not s3_client:
        return False
    s3_client.upload_file(LOCAL_PATH, BUCKET, MODEL_KEY)
    logging.info("Model uploaded to S3 bucket %s", BUCKET)
    return True

def download_model():
    if not s3_client:
        return False
    try:
        s3_client.download_file(BUCKET, MODEL_KEY, LOCAL_PATH)
        return True
    except Exception as e:
        logging.warning("Failed to download model from S3: %s", e)
        return False

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

# ─── Attempt to load existing model on start ─────────────────────────────
logging.info("Trying to download initial model from S3...")
if download_model():
    logging.info("Model loaded from S3.")
else:
    logging.warning("No model found locally; POST /train first.")

# ─── MT5 CONFIG ──────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 module not installed; MT5 fetches will fail.")

MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH")  # e.g. "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
MT5_SERVER        = os.getenv("MT5_SERVER")
MT5_LOGIN_STR     = os.getenv("MT5_LOGIN")
MT5_LOGIN         = int(MT5_LOGIN_STR) if MT5_LOGIN_STR and MT5_LOGIN_STR.isdigit() else None
MT5_PASSWORD      = os.getenv("MT5_PASSWORD")

SYMBOL       = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME    = getattr(mt5, os.getenv("TIMEFRAME", "TIMEFRAME_H1"), None)
LOOKBACK     = int(os.getenv("LOOKBACK", "1000"))

# ─── Feature engineering placeholders ────────────────────────────────────
def add_fvg_features(df):
    # … your implementation …
    return df

def add_trailing_sl(df, period=14, multiplier=1.5):
    # … your implementation …
    return df

def add_supply_demand(df, window=20):
    # … your implementation …
    return df

def label_signals(df, hold_thr=0.0005):
    # … your implementation …
    return df

def build_features(df):
    df = add_fvg_features(df)
    df = add_trailing_sl(df)
    df = add_supply_demand(df)
    df["return"] = df["close"].pct_change()
    df = label_signals(df)
    return df.dropna()

feature_cols = [
    "return", "fvg_up", "fvg_down", "tsl",
    "dist_supply", "dist_demand"
]

# ─── Data fetching from MT5 ──────────────────────────────────────────────
def fetch_ohlc(symbol, timeframe, lookback):
    if not MT5_AVAILABLE:
        raise RuntimeError("MT5 module not available")
    # build initialize kwargs
    init_kwargs = {}
    if MT5_TERMINAL_PATH:
        init_kwargs["path"] = MT5_TERMINAL_PATH
    init_kwargs.update({
        "server":   MT5_SERVER,
        "login":    MT5_LOGIN,
        "password": MT5_PASSWORD
    })
    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, lookback)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]]

# ─── Retrain endpoint ───────────────────────────────────────────────────
@app.route("/train", methods=["POST"])
def train():
    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, LOOKBACK)
    except Exception as e:
        logging.error("MT5 fetch failed: %s", e)
        return jsonify({"error": "MT5 failed", "details": str(e)}), 503

    df_feat = build_features(df)
    X = df_feat[feature_cols]
    y = df_feat["signal"].astype(int)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X, y)

    joblib.dump(model, LOCAL_PATH)
    upload_model()

    return jsonify({"status": "retrained"}), 200

# ─── Signal endpoint ────────────────────────────────────────────────────
@app.route("/signal.json", methods=["GET"])
def signal():
    if not os.path.exists(LOCAL_PATH):
        return jsonify({"error":"Model not available. Please /train first."}), 503
    model = joblib.load(LOCAL_PATH)

    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, LOOKBACK)
    except Exception as e:
        logging.error("MT5 fetch failed: %s", e)
        return jsonify({"error":"MT5 failed","details":str(e)}), 503

    df_feat = build_features(df).tail(1)
    X_latest = df_feat[feature_cols]

    proba  = model.predict_proba(X_latest)[0]
    labels = {0:"sell",1:"hold",2:"buy"}
    resp = {
        "symbol": SYMBOL,
        "timestamp": df_feat.index[0].isoformat(),
        "signal": labels[int(proba.argmax())],
        "probabilities":{
            "sell": round(float(proba[0]),4),
            "hold": round(float(proba[1]),4),
            "buy" : round(float(proba[2]),4)
        }
    }
    logging.info("Signal: %s", resp)
    return jsonify(resp), 200

# ─── Health check ───────────────────────────────────────────────────────
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status":"healthy"}), 200

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT","5000")))
