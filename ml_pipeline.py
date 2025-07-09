import os
import sys
import logging
import joblib

import numpy  as np
import pandas as pd
import xgboost as xgb

from flask import Flask, jsonify
from sklearn.model_selection import TimeSeriesSplit

# ─── Optional S3 persistence ───────────────────────────────────────────
import boto3
AWS_KEY    = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
S3_BUCKET  = os.getenv("S3_BUCKET")
MODEL_KEY  = "models/xgb_model.pkl"
LOCAL_PATH = os.getenv("MODEL_PATH", "/tmp/xgb_model.pkl")

if AWS_KEY and AWS_SECRET and S3_BUCKET:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
else:
    s3 = None
    logging.warning("S3 credentials not configured; skipping S3 persistence")

def upload_model():
    if not s3: return False
    s3.upload_file(LOCAL_PATH, S3_BUCKET, MODEL_KEY)
    logging.info("Uploaded model to S3 %s/%s", S3_BUCKET, MODEL_KEY)
    return True

def download_model():
    if not s3: return False
    try:
        s3.download_file(S3_BUCKET, MODEL_KEY, LOCAL_PATH)
        logging.info("Downloaded model from S3")
        return True
    except Exception:
        logging.warning("No model found in S3; skipping download")
        return False

# ─── MetaTrader5 import (Windows only) ─────────────────────────────────
try:
    if sys.platform == "win32":
        import MetaTrader5 as mt5
    else:
        raise ImportError
except ImportError:
    mt5 = None
    logging.warning("MetaTrader5 unavailable on this platform; MT5 calls will fail")

# ─── Flask setup ────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── Read core settings from env ────────────────────────────────────────
MT5_SERVER   = os.getenv("MT5_SERVER")
MT5_LOGIN    = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
SYMBOL       = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME    = getattr(mt5, os.getenv("TIMEFRAME","TIMEFRAME_H1"), None)
LOOKBACK     = int(os.getenv("LOOKBACK","1000"))

feature_cols = ["return","fvg_up","fvg_down","tsl","dist_supply","dist_demand"]

# ─── Feature Engineering (your existing functions) ─────────────────────
def add_fvg_features(df):
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"]  = df["low"].shift(1)
    df["fvg_up"]   = np.where(df["low"] > df["prev_high"],
                              df["prev_high"] - df["low"], 0.0)
    df["fvg_down"] = np.where(df["high"] < df["prev_low"],
                              df["prev_low"] - df["high"], 0.0)
    return df

def add_trailing_sl(df, period=14, multiplier=1.5):
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"]  - df["close"].shift()).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(period).mean()
    df["tsl"] = df["close"] - multiplier * df["atr"]
    return df

def add_supply_demand(df, window=20):
    df["zone_high"] = df["high"].rolling(window).max()
    df["zone_low"]  = df["low"].rolling(window).min()
    df["dist_supply"]  = df["zone_high"]  - df["close"]
    df["dist_demand"]  = df["close"] - df["zone_low"]
    return df

def label_signals(df, hold_thr=0.0005):
    ret = df["close"].pct_change().shift(-1)
    conditions = [ret > hold_thr, ret < -hold_thr]
    choices    = [2, 0]
    df["signal"] = np.select(conditions, choices, default=1)
    return df

def build_features(df):
    df = add_fvg_features(df)
    df = add_trailing_sl(df)
    df = add_supply_demand(df)
    df["return"] = df["close"].pct_change()
    df = label_signals(df)
    return df.dropna()

# ─── Data Fetching via MT5 ──────────────────────────────────────────────
def fetch_ohlc(symbol, timeframe, lookback):
    if not mt5:
        raise RuntimeError("MT5 unavailable")
    if not mt5.initialize(server=MT5_SERVER, login=int(MT5_LOGIN), password=MT5_PASSWORD):
        raise RuntimeError("MT5 init failed")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, lookback)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]]

# ─── Attempt initial model download ────────────────────────────────────
logging.info("Trying to download model from S3...")
if not os.path.exists(LOCAL_PATH):
    download_model()

# ─── /train endpoint ───────────────────────────────────────────────────
@app.route("/train", methods=["POST"])
def train():
    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, lookback=int(os.getenv("LOOKBACK","5000")))
    except Exception as e:
        logging.error("MT5 fetch failed: %s", e)
        return jsonify({"error":"MT5 failed", "details":str(e)}), 503

    df_feat = build_features(df)
    X, y = df_feat[feature_cols], df_feat["signal"].astype(int)

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
    logging.info("Retraining complete; model saved to %s", LOCAL_PATH)
    return jsonify({"status":"retrained"}), 200

# ─── /signal.json endpoint ──────────────────────────────────────────────
@app.route("/signal.json", methods=["GET"])
def signal():
    if not os.path.exists(LOCAL_PATH):
        return jsonify({"error":"Model not available. Please /train first."}), 503

    model = joblib.load(LOCAL_PATH)

    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, lookback=int(os.getenv("LOOKBACK","1000")))
    except Exception as e:
        logging.error("MT5 fetch failed: %s", e)
        return jsonify({"error":"MT5 failed","details":str(e)}), 503

    feat = build_features(df).tail(1)
    proba = model.predict_proba(feat[feature_cols])[0]
    labels = {0:"sell",1:"hold",2:"buy"}

    return jsonify({
        "symbol": SYMBOL,
        "timestamp": feat.index[0].isoformat(),
        "signal": labels[int(np.argmax(proba))],
        "probabilities": {
            "sell": round(float(proba[0]),4),
            "hold": round(float(proba[1]),4),
            "buy":  round(float(proba[2]),4)
        }
    }), 200

# ─── /healthz endpoint ──────────────────────────────────────────────────
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status":"healthy"}), 200

# ─── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # in-production we use gunicorn; locally you can use this
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")), debug=True)
