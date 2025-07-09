import os
import logging
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from flask import Flask, jsonify

# Configure logging
event_fmt = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=event_fmt)

app = Flask(__name__)

# --- Basic Endpoints --------------------------------------------------------

@app.route('/')
def index():
    """Root endpoint to verify app is up."""
    return 'OK', 200

@app.route('/healthz')
def healthz():
    """Health check for Render or any load-balancer."""
    return 'healthy', 200

# --- Feature Engineering Functions ----------------------------------------

def add_fvg_features(df):
    df['prev_high'] = df['high'].shift(1)
    df['prev_low']  = df['low'].shift(1)
    df['fvg_up']   = np.where(df['low'] > df['prev_high'],
                              df['prev_high'] - df['low'], 0.0)
    df['fvg_down'] = np.where(df['high'] < df['prev_low'],
                              df['prev_low'] - df['high'], 0.0)
    return df

def add_trailing_sl(df, period=14, multiplier=1.5):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low']  - df['close'].shift()).abs()
    df['tr']  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(period).mean()
    df['tsl'] = df['close'] - multiplier * df['atr']
    return df

def add_supply_demand(df, window=20):
    df['zone_high']  = df['high'].rolling(window).max()
    df['zone_low']   = df['low'].rolling(window).min()
    df['dist_supply'] = df['zone_high'] - df['close']
    df['dist_demand'] = df['close'] - df['zone_low']
    return df

def label_signals(df, hold_thr=0.0005):
    ret = df['close'].pct_change().shift(-1)
    conditions = [ret > hold_thr, ret < -hold_thr]
    choices    = [2, 0]  # 2=Buy, 0=Sell
    df['signal'] = np.select(conditions, choices, default=1)  # 1=Hold
    return df

def fetch_ohlc(symbol='XAUUSD', timeframe=mt5.TIMEFRAME_H1, lookback=2000):
    if not mt5.initialize():
        logging.error('MT5 initialization failed')
        return pd.DataFrame()
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
    mt5.shutdown()

    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open','high','low','close','tick_volume']]

def build_features(df):
    df = add_fvg_features(df)
    df = add_trailing_sl(df)
    df = add_supply_demand(df)
    df['return'] = df['close'].pct_change()
    df = label_signals(df)
    df = df.dropna()
    return df

# --- Signal Generation Endpoint -------------------------------------------

@app.route('/signal.json')
def generate_signal():
    # 1) Load data from MT5
    df = fetch_ohlc(
        symbol=os.getenv('SYMBOL','XAUUSD'),
        timeframe=mt5.TIMEFRAME_H1,
        lookback=int(os.getenv('LOOKBACK','1000'))
    )
    if df.empty:
        return jsonify({'error':'Failed to fetch data'}), 500

    # 2) Feature engineering
    df_feat = build_features(df).tail(500)
    feature_cols = ['return','fvg_up','fvg_down','tsl','dist_supply','dist_demand']
    X = df_feat[feature_cols]
    y = df_feat['signal']

    # 3) Train/test split (rolling) & model list
    tss = TimeSeriesSplit(n_splits=3)
    models = []
    for train_idx, test_idx in tss.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=200,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        models.append(model)

    # 4) Predict latest bar with last model
    latest_X = X.iloc[[-1]]
    proba    = models[-1].predict_proba(latest_X)[0]
    pred     = int(np.argmax(proba))
    labels   = {0:'sell',1:'hold',2:'buy'}

    response = {
        'symbol':    os.getenv('SYMBOL','XAUUSD'),
        'timestamp': latest_X.index[0].isoformat(),
        'signal':    labels[pred],
        'probabilities': {
            'sell': round(float(proba[0]),4),
            'hold': round(float(proba[1]),4),
            'buy':  round(float(proba[2]),4)
        }
    }

    logging.info(f"Generated signal: {response}")
    return jsonify(response)

# --- App Runner ------------------------------------------------------------

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.getenv('PORT','5000')))
