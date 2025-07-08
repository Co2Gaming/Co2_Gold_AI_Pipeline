import os
import logging
import pandas as pd
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Your existing ai_input endpoint (example)
@app.route("/ai_input.json")
def ai_input():
    # Load your data from wherever you have it
    # For example, if you fetch a CSV from an online URL:
    # df = pd.read_csv("https://example.com/gold_price.csv")
    df = pd.read_csv("static/gold_price.csv")  # or local path

    # Feature engineering: price returns, rolling stats, etc.
    df["return"] = df["close"].pct_change()
    df = df.dropna().tail(50)

    # Train a simple RandomForest
    X = df[["return"]]
    y = (df["return"].shift(-1) > 0).astype(int)  # next-step up/down
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:-1], y[:-1])

    # Predict on the most recent point
    latest = X.iloc[[-1]]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0, pred]

    signal = "buy" if pred == 1 else "sell"
    result = {
        "signal": signal,
        "confidence": float(prob),
        "timestamp": df.index[-1].isoformat()
    }

    logging.info(f"Generated signal: {result}")
    return jsonify(result)

@app.route("/")
def index():
    return jsonify({"message": "FX AI Pipeline is running. Use /ai_input.json"}), 200

@app.route("/healthz")
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Local debug; on Render this will use gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
