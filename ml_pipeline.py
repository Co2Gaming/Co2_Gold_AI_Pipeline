import os
import joblib
import logging

import numpy   as np
import pandas  as pd
import xgboost as xgb
import boto3

# MetaTrader5 only available on Windows
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logging.warning("MetaTrader5 module not installed; MT5 data fetches will fail on this platform.")

from flask                     import Flask, jsonify
from sklearn.model_selection   import TimeSeriesSplit

# — configure logging —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

# ─── S3 PERSISTENCE ──────────────────────────────────────────────────────
AWS_KEY    = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
BUCKET     = os.getenv("S3_BUCKET")
MODEL_KEY  = "models/xgb_model.pkl"
LOCAL_PATH = "/tmp/xgb_model.pkl"
 
s3_client = None
if all([AWS_KEY, AWS_SECRET, BUCKET]):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
else:
    logging.warning("S3 credentials not found. File operations will be skipped.")
 
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
        logging.warning("Failed to download model from S3: %s", e.__class__.__name__)
        return False

# ─── INITIAL MODEL LOAD ──────────────────────────────────────────────────
logging.info("Attempting to download initial model from S3...")
if download_model():
    logging.info("Model successfully downloaded and loaded at startup.")
else:
    logging.warning("Could not download model at startup. The /train endpoint must be called first.")

# ─── MT5 CONFIG ──────────────────────────────────────────────────────────
MT5_SERVER    = os.getenv("MT5_SERVER")
MT5_LOGIN_STR = os.getenv("MT5_LOGIN")
MT5_LOGIN     = int(MT5_LOGIN_STR) if MT5_LOGIN_STR and MT5_LOGIN_STR.isdigit() else None
MT5_PASSWORD  = os.getenv("MT5_PASSWORD")
SYMBOL        = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME     = getattr(mt5, os.getenv("TIMEFRAME", "TIMEFRAME_H1")) if mt5 else None

# ─── FEATURE ENGINEERING (stubs) ─────────────────────────────────────────
def add_fvg_features(df): ...
def add_trailing_sl(df, period=14, multiplier=1.5): ...
def add_supply_demand(df, window=20): ...
def label_signals(df, hold_thr=0.0005): ...

def build_features(df):
    df = add_fvg_features(df)
    df = add_trailing_sl(df)
    df = add_supply_demand(df)
    df["return"] = df["close"].pct_change()
    df = label_signals(df)
    return df.dropna()

# ─── DATA FETCHING ────────────────────────────────────────────────────────
def fetch_ohlc(symbol, timeframe, lookback):
    if mt5 is None:
        raise RuntimeError("MT5 module not available on this platform")
    if not all([MT5_SERVER, MT5_LOGIN, MT5_PASSWORD]) or not mt5.initialize(
        server=MT5_SERVER, login=MT5_LOGIN, password=MT5_PASSWORD
    ):
        raise RuntimeError("MT5 init failed")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, lookback)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]]

feature_cols = [
    "return","fvg_up","fvg_down","tsl",
    "dist_supply","dist_demand"
]

# ─── RE-TRAIN ENDPOINT ──────────────────────────────────────────────────
@app.route("/train", methods=["POST"])
def train():
    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, lookback=5000)
    except RuntimeError as e:
        logging.error("Failed to fetch data from MT5 during training: %s", e)
        return jsonify({"error": "MT5 connection failed", "details": str(e)}), 503

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

# ─── SIGNAL ENDPOINT ────────────────────────────────────────────────────
@app.route("/signal.json", methods=["GET"])
def signal():
    if not os.path.exists(LOCAL_PATH):
        logging.error("Model file not found. Must train first.")
        return jsonify({"error": "Model not available"}), 503

    model = joblib.load(LOCAL_PATH)
    try:
        df = fetch_ohlc(SYMBOL, TIMEFRAME, lookback=1000)
    except RuntimeError as e:
        logging.error("MT5 fetch failed: %s", e)
        return jsonify({"error": "MT5 connection failed", "details": str(e)}), 503

    df_feat = build_features(df).tail(1)
    X_latest = df_feat[feature_cols]
    proba = model.predict_proba(X_latest)[0]
    labels = {0: "sell", 1: "hold", 2: "buy"}

    response = {
        "symbol":    SYMBOL,
        "timestamp": df_feat.index[0].isoformat(),
        "signal":    labels[int(np.argmax(proba))],
        "probabilities": {
            "sell": round(float(proba[0]),4),
            "hold": round(float(proba[1]),4),
            "buy":  round(float(proba[2]),4)
        }
    }
    logging.info("Generated signal: %s", response)
    return jsonify(response), 200

# ─── HEALTH CHECK ───────────────────────────────────────────────────────
@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
