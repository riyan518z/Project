from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import pickle
import os
import random
import yfinance as yf   # ⭐ REAL TIME DATA

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs("uploads", exist_ok=True)

# ------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------
US_MODEL_PATH = "us_xgb_direction_model.pkl"
NIFTY_MODEL_PATH = "modelnif.pkl"
APPLE_MODEL_PATH = "lstm_model.keras"
APPLE_SCALER_PATH = "scaler.pkl"
HISTORICAL_CSV = "final.csv"

# ------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------
us_model = joblib.load(US_MODEL_PATH)
nifty_model = joblib.load(NIFTY_MODEL_PATH)

apple_lstm = tf.keras.models.load_model(APPLE_MODEL_PATH)
with open(APPLE_SCALER_PATH, "rb") as f:
    apple_scaler = pickle.load(f)

SEQ_LEN = 60

# ------------------------------------------------------
# MODEL FEATURE LIST
# ------------------------------------------------------
FEATURES = [
    "Open","High","Low","Close","Volume",
    "SMA_10","SMA_20","RSI","MACD","Volatility",
    "Close_lag_1","Return_lag_1","Close_lag_2","Return_lag_2",
    "Close_lag_3","Return_lag_3","Close_lag_4","Return_lag_4",
    "Close_lag_5","Return_lag_5","SMA_5","SMA_10_Close","Volatility_5"
]

# ------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------
def compute_features(open_, high, low, close, volume):

    if not os.path.exists(HISTORICAL_CSV):
        raise FileNotFoundError(f"{HISTORICAL_CSV} not found.")

    hist = pd.read_csv(HISTORICAL_CSV)

    required = {"Open","High","Low","Close","Volume"}
    if not required.issubset(hist.columns):
        raise ValueError(f"CSV must have columns {required}")

    base = hist.copy().reset_index(drop=True)

    if len(base) < 10:
        base = pd.concat([base] * 6, ignore_index=True)

    new_row = pd.DataFrame({
        "Open": [open_],
        "High": [high],
        "Low": [low],
        "Close": [close],
        "Volume": [volume]
    })

    df = pd.concat([base, new_row], ignore_index=True)

    # --- Indicators ---
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    diff = df["Close"].diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"].fillna(0, inplace=True)

    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(5).std()

    for lag in range(1, 6):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"Return_lag_{lag}"] = df["Close"].pct_change(lag)

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10_Close"] = df["Close"].rolling(10).mean()
    df["Volatility_5"] = df["Close"].pct_change().rolling(5).std()

    df.fillna(0, inplace=True)

    latest = df.iloc[-1:][FEATURES]

    return latest

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

# ------------------------------------------------------
# US — XGBOOST DIRECTION MODEL
# ------------------------------------------------------
@app.route("/us", methods=["GET", "POST"])
def us_stock():
    direction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            open_ = float(request.form.get("open"))
            high = float(request.form.get("high"))
            low = float(request.form.get("low"))
            close = float(request.form.get("close"))
            volume = float(request.form.get("volume"))

            features_df = compute_features(open_, high, low, close, volume)

            # ⭐ FIX: convert to numpy
            X = features_df.to_numpy()

            pred = us_model.predict(X)[0]

            try:
                prob = us_model.predict_proba(X)[0][int(pred)]
            except:
                prob = 0.5

            direction = "UP" if int(pred) == 1 else "DOWN"
            confidence = f"{prob*100:.2f}%"

        except Exception as e:
            error = f"{e}"

    return render_template("us.html", direction=direction, confidence=confidence, error=error)

# ------------------------------------------------------
# NIFTY MODEL (UNTOUCHED)
# ------------------------------------------------------
@app.route("/nifty", methods=["GET", "POST"])
def nifty():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = [
                request.form.get("Open", 0),
                request.form.get("High", 0),
                request.form.get("Low", 0),
                request.form.get("Volume", 0),
                request.form.get("SMA_10", 0),
                request.form.get("SMA_20", 0),
                request.form.get("RSI", 0),
                request.form.get("MACD", 0),
                request.form.get("Volatility", 0),
            ]

            df = pd.DataFrame([features], columns=[
                "Open","High","Low","Volume","SMA_10","SMA_20","RSI","MACD","Volatility"
            ])

            prediction = float(nifty_model.predict(df)[0])
            prediction = round(prediction, 2)

        except Exception as e:
            error = str(e)

    return render_template("nifty.html", prediction=prediction, error=error)

# ------------------------------------------------------
# APPLE LSTM MODEL (UNTOUCHED)
# ------------------------------------------------------
def process_apple_csv(path):
    df = pd.read_csv(path)
    df["Price"] = df["Price"].astype(str).str.replace(",", "").astype(float)
    if len(df) < 60:
        raise ValueError("CSV must contain at least 60 rows.")
    return df["Price"].tail(60).values.reshape(-1, 1)

@app.route("/apple", methods=["GET","POST"])
def apple():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            file = request.files["file"]
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

            seq = process_apple_csv(save_path)
            scaled = apple_scaler.transform(seq)
            scaled = np.expand_dims(scaled, 0)

            p_scaled = apple_lstm.predict(scaled)[0][0]
            prediction = apple_scaler.inverse_transform([[p_scaled]])[0][0]
            prediction = round(float(prediction), 2)

        except Exception as e:
            error = str(e)

    return render_template("apple.html", prediction=prediction, error=error)

# ------------------------------------------------------
# ⭐ REAL-TIME STOCK TICKER (NSE LIVE DATA)
# ------------------------------------------------------
@app.route("/live_prices")
def live_prices():

    tickers = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "INFY": "INFY.NS",
        "ICICIBANK": "ICICIBANK.NS"
    }

    prices = {}

    for name, symbol in tickers.items():
        try:
            info = yf.Ticker(symbol).fast_info

            price = info.get("last_price") or info.get("previous_close")

            prices[name] = round(float(price), 2) if price else "N/A"

        except:0+2
    prices[name] = "N/A"

    return prices

# ------------------------------------------------------
# RUN
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
