import numpy as np
import pandas as pd
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

def add_fvg_features(df, lookback=3):
    """
    Fair Value Gap: difference between consecutive candles' highs/lows.
    We’ll mark a gap when the current candle’s low is above the prior candle’s high, etc.
    """
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"]  = df["low"].shift(1)
    # Up-gap  
    df["fvg_up"]   = np.where(df["low"] > df["prev_high"],
                               df["low"] - df["prev_high"], 0.0)
    # Down-gap  
    df["fvg_down"] = np.where(df["high"] < df["prev_low"],
                               df["prev_low"] - df["high"], 0.0)
    return df

def add_trailing_sl_feature(df, atr_period=14, multiplier=1.5):
    """
    Trailing stop‐loss proxy: use ATR‐based band.
    We’ll compute an ATR band below price as a volatility‐adjusted stop.
    """
    df["tr"] = df[["high", "low", "close"]].diff().abs().max(axis=1)
    df["atr"] = df["tr"].rolling(atr_period).mean()
    # Trailing SL level = close - multiplier * ATR
    df["tsl_level"] = df["close"] - multiplier * df["atr"]
    return df

def add_supply_demand_zones(df, window=20):
    """
    Supply & Demand: local maxima/minima over sliding window.
    We'll mark distance from nearest recent zone.
    """
    df["zone_high"] = df["high"].rolling(window).max()
    df["zone_low"]  = df["low"].rolling(window).min()
    # How far from supply zone?
    df["dist_supply"] = df["zone_high"] - df["close"]
    df["dist_demand"] = df["close"] - df["zone_low"]
    return df

@app.route("/ai_input.json")
def ai_input():
    # 1) Load your raw OHLC data (replace with your source)
    df = pd.read_csv("static/gold_price.csv", parse_dates=["timestamp"], index_col="timestamp")

    # 2) Base features
    df["return"] = df["close"].pct_change()

    # 3) Add advanced features
    df = add_fvg_features(df)
    df = add_trailing_sl_feature(df)
    df = add_supply_demand_zones(df)

    # 4) Drop NaNs & select last N points
    df = df.dropna().tail(100)

    # 5) Prepare X/y
    X = df[[
        "return",
        "fvg_up", "fvg_down",
        "tsl_level",
        "dist_supply", "dist_demand"
    ]]
    # Next‐step up/down
    y = (df["close"].shift(-1) > df["close"]).astype(int)

    # 6) Train/test split on rolling window
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:-1]
    y_train, _       = y.iloc[:split], y.iloc[split:-1]

    # 7) Fit a quick RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 8) Predict on latest point
    latest = X.iloc[[-1]]
    pred   = model.predict(latest)[0]
    prob   = model.predict_proba(latest)[0, pred]

    signal = "buy" if pred == 1 else "sell"
    return jsonify({
        "signal":     signal,
        "confidence": round(float(prob), 4),
        "features":   latest.iloc[0].to_dict(),
        "timestamp":  df.index[-1].isoformat(),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
