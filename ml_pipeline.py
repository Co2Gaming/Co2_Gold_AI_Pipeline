import os
import joblib
import logging

import numpy   as np
import pandas  as pd
import xgboost as xgb

# optional imports
try:
    import boto3
except ImportError:
    boto3 = None

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from flask                   import Flask, jsonify
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

# ─── MODEL PATH (iCloud) ────────────────────────────────────────────────
LOCAL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Users\The_R\iCloudDrive\MyPythonBot Storage\xgb_model.pkl"
)
logging.info("Model file path: %s", LOCAL_PATH)

# ─── OPTIONAL S3 SETUP ───────────────────────────────────────────────────
AWS_KEY    = os.getenv("AWS_KEY")
AWS_SECRET = os.getenv("AWS_SECRET")
BUCKET     = os.getenv("S3_BUCKET")
MODEL_KEY  = "models/xgb_model.pkl"
s3_client  = None

if boto3 and all([AWS_KEY, AWS_SECRET, BUCKET]):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET
    )
else:
    logging.warning("Skipping S3 (no creds or missing boto3)")

def upload_model():
    if not s3_client: return False
    s3_client.upload_file(LOCAL_PATH, BUCKET, MODEL_KEY)
    logging.info("Uploaded model to S3")
    return True

def download_model():
    if not s3_client: return False
    try:
        s3_client.download_file(BUCKET, MODEL_KEY, LOCAL_PATH)
        return True
    except Exception as e:
        logging.warning("S3 download failed: %s", e)
        return False

logging.info("Try downloading model from S3...")
if download_model():
    logging.info("Model loaded at startup")
else:
    logging.warning("No local model; POST /train first")

# ─── MT5 SETTINGS ───────────────────────────────────────────────────────
MT5_SERVER   = os.getenv("MT5_SERVER")
MT5_LOGIN    = int(os.getenv("MT5_LOGIN","0") or 0)
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
SYMBOL       = os.getenv("SYMBOL","XAUUSD")
TIMEFRAME    = getattr(mt5, os.getenv("TIMEFRAME","TIMEFRAME_H1"), None)

# ─── STUB FEATURE FUNCTIONS ─────────────────────────────────────────────
def add_fvg_features(df):
    df["fvg_up"]   = 0.0
    df["fvg_down"] = 0.0
    return df

def add_trailing_sl(df, period=14, multiplier=1.5):
    df["tsl"] = 0.0
    return df

def add_supply_demand(df, window=20):
    df["dist_supply"] = 0.0
    df["dist_demand"] = 0.0
    return df

def label_signals(df, hold_thr=0.0005):
    # 0=sell,1=hold,2=buy
    df["signal"] = 1
    return df

def build_features(df):
    df = add_fvg_features(df)
    df = add_trailing_sl(df)
    df = add_supply_demand(df)
    df["return"] = df["close"].pct_change()
    df = label_signals(df)
    return df.dropna()

feature_cols = ["return","fvg_up","fvg_down","tsl","dist_supply","dist_demand"]

# ─── DATA FETCH ─────────────────────────────────────────────────────────
def fetch_ohlc(symbol, timeframe, lookback):
    if not mt5 or not all([MT5_SERVER,MT5_LOGIN,MT5_PASSWORD]):
        raise RuntimeError("MT5 unavailable")
    if not mt5.initialize(server=MT5_SERVER, login=MT5_LOGIN, password=MT5_PASSWORD):
        raise RuntimeError("MT5 init failed")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, lookback)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"],unit="s")
    df.set_index("time", inplace=True)
    return df[["open","high","low","close","tick_volume"]]

# ─── /train ENDPOINT ────────────────────────────────────────────────────
@app.route("/train", methods=["POST"])
def train():
    try:
        df = fetch_ohlc(SYMBOL,TIMEFRAME,lookback=int(os.getenv("LOOKBACK",5000)))
    except Exception as e:
        logging.error("Train fetch error: %s", e)
        return jsonify({"error":"MT5 failed","details":str(e)}),503

    feats = build_features(df)
    X = feats[feature_cols]
    y = feats["signal"].astype(int)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X,y)

    os.makedirs(os.path.dirname(LOCAL_PATH),exist_ok=True)
    joblib.dump(model,LOCAL_PATH)
    upload_model()
    return jsonify({"status":"retrained"}),200

# ─── /signal.json ENDPOINT ─────────────────────────────────────────────
@app.route("/signal.json", methods=["GET"])
def signal():
    if not os.path.exists(LOCAL_PATH):
        return jsonify({"error":"Model not available. Please /train first."}),503

    model = joblib.load(LOCAL_PATH)
    try:
        df = fetch_ohlc(SYMBOL,TIMEFRAME,lookback=int(os.getenv("LOOKBACK",1000)))
    except Exception as e:
        logging.error("Signal fetch error: %s", e)
        return jsonify({"error":"MT5 failed","details":str(e)}),503

    latest = build_features(df).tail(1)
    proba = model.predict_proba(latest[feature_cols])[0]
    labels = {0:"sell",1:"hold",2:"buy"}

    resp = {
        "symbol": SYMBOL,
        "timestamp": latest.index[0].isoformat(),
        "signal":   labels[int(np.argmax(proba))],
        "probabilities": {
            "sell": round(float(proba[0]),4),
            "hold": round(float(proba[1]),4),
            "buy":  round(float(proba[2]),4)
        }
    }
    logging.info("Signal → %s",resp)
    return jsonify(resp),200

# ─── HEALTH CHECK ────────────────────────────────────────────────────────
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status":"healthy"}),200

if __name__=="__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT","5000")))
