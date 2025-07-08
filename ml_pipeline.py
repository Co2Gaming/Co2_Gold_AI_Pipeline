import json
from flask import Flask, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

CSV_URL = "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv"

@app.route("/ai_input.json")
def generate_signal():
    try:
        df = pd.read_csv(CSV_URL)

        # Prepare simple ML features
        df = df.tail(50).copy()
        df["Price_Change"] = df["Price"].diff()
        df["Direction"] = df["Price_Change"].apply(lambda x: 1 if x > 0 else 0)
        df["Lag1"] = df["Price"].shift(1)
        df["Lag2"] = df["Price"].shift(2)
        df.dropna(inplace=True)

        X = df[["Lag1", "Lag2"]]
        y = df["Direction"]

        model = RandomForestClassifier()
        model.fit(X, y)

        latest = df.iloc[-1][["Lag1", "Lag2"]].values.reshape(1, -2)
        pred = model.predict(latest)[0]
        conf = model.predict_proba(latest).max()

        recommendation = "buy" if pred == 1 else "sell"

        result = {
            "recommendation": recommendation,
            "confidence": round(float(conf), 4),
            "top_feature": "Lag2"
        }

        with open("static/ai_input.json", "w") as f:
            json.dump(result, f)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
