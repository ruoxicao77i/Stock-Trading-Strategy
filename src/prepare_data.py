"""
Data preparation script: fetches news, runs FinBERT sentiment, builds features.
Run this ONCE before model.ipynb. Results are cached to data/sentiment_cache/.
"""

from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import requests

# ── Path Configuration ─────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
CLEAN_DIR  = f"{DATA_ROOT}/clean"
NEWS_DIR   = f"{DATA_ROOT}/news"
CACHE_DIR  = f"{DATA_ROOT}/sentiment_cache"
MASTER_DIR = f"{DATA_ROOT}/master"

for _d in [CLEAN_DIR, NEWS_DIR, CACHE_DIR, MASTER_DIR]:
    os.makedirs(_d, exist_ok=True)

START_DATE = datetime(2025, 4, 1)
END_DATE   = datetime(2026, 3, 31)
SYMBOLS = [
    "AAPL","AMZN","BAC","CVX","GOOG","GS","HD","JNJ","JPM",
    "MSFT","PFE","TSLA","UNH","XOM"
]

API_KEY = "d7crg5pr01qv03etebegd7crg5pr01qv03etebf0"

# ── Load FinBERT ───────────────────────────────────────────────────────
from transformers import pipeline

print("Loading FinBERT ...")
finbert_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("FinBERT loaded.")

def get_finbert_score(text):
    try:
        if not text or pd.isna(text):
            return 0.0
        result = finbert_pipe(str(text)[:512])[0]
        label, score = result['label'], result['score']
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        return 0.0
    except Exception:
        return 0.0

# ── Helper functions ───────────────────────────────────────────────────
def fetch_stock(symbol, start, end):
    path = f"{CLEAN_DIR}/{symbol}_clean.csv"
    if not os.path.exists(path):
        return {"s": "no_data"}
    df = pd.read_csv(path)
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] < end)]
    if df.empty:
        return {"s": "no_data"}
    return {"s": "ok", "t": [int(d.timestamp()) for d in df["date"]], "c": df["Close"].tolist()}

def fetch_news(symbol, start_date, end_date):
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    return response.json()

# ── Section 2: Sentiment cache per symbol ─────────────────────────────
print("\n=== Section 2: News + FinBERT sentiment ===")
for symbol in SYMBOLS:
    cache_path = f"{CACHE_DIR}/{symbol}_processed.csv"
    if os.path.exists(cache_path):
        print(f"{symbol}: already cached, skipping")
        continue

    print(f"Processing {symbol} ...")
    try:
        news_path = f"{NEWS_DIR}/{symbol}_news.csv"
        if os.path.exists(news_path):
            df_news = pd.read_csv(news_path)
        else:
            all_news = []
            current = START_DATE
            while current < END_DATE:
                next_date = current + timedelta(days=7)
                news_data = fetch_news(symbol, current.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d"))
                for item in news_data:
                    all_news.append({
                        "date": datetime.fromtimestamp(item["datetime"]).strftime("%Y-%m-%d"),
                        "headline": item.get("headline", ""),
                        "summary": item.get("summary", "")
                    })
                current = next_date
                time.sleep(0.5)

            if not all_news:
                print(f"  {symbol}: no news, skipping")
                continue

            df_news = pd.DataFrame(all_news)
            df_news.to_csv(news_path, index=False)

        df_news['full_text'] = df_news['headline'].fillna('') + ". " + df_news['summary'].fillna('')
        df_news['sentiment_score'] = df_news['full_text'].apply(get_finbert_score)
        daily_sentiment = df_news.groupby('date')['sentiment_score'].mean().reset_index()

        stock_data = fetch_stock(symbol, START_DATE, END_DATE)
        if stock_data.get("s") != "ok":
            print(f"  {symbol}: no stock data, skipping")
            continue

        df_stock = pd.DataFrame({
            "date": [datetime.fromtimestamp(t).strftime("%Y-%m-%d") for t in stock_data["t"]],
            "close": stock_data["c"]
        })
        df_final = pd.merge(df_stock, daily_sentiment, on="date", how="left").fillna(0)
        df_final["label"] = df_final["close"].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)
        df_final['sentiment_score_shifted'] = df_final['sentiment_score'].shift(1)
        df_final["symbol"] = symbol
        df_final = df_final.dropna()

        df_final.to_csv(cache_path, index=False)
        print(f"  {symbol}: done")

    except Exception as e:
        print(f"  {symbol} error: {e}")

# ── Section 3: Mega feature dataset ───────────────────────────────────
print("\n=== Section 3: Feature engineering ===")
all_stocks_master = []

for symbol in SYMBOLS:
    cache_path = f"{CACHE_DIR}/{symbol}_mega_features.csv"
    if os.path.exists(cache_path):
        all_stocks_master.append(pd.read_csv(cache_path))
        continue

    print(f"Processing {symbol} ...")
    try:
        q_path = f"{CLEAN_DIR}/{symbol}_clean.csv"
        if not os.path.exists(q_path):
            print(f"  {symbol}_clean.csv not found, skipping")
            continue

        df_rich = pd.read_csv(q_path)
        df_rich.rename(columns={'Date': 'date'}, inplace=True)
        df_rich['date'] = pd.to_datetime(df_rich['date']).dt.strftime('%Y-%m-%d')
        df_rich = df_rich[
            (df_rich['date'] >= START_DATE.strftime('%Y-%m-%d')) &
            (df_rich['date'] <= END_DATE.strftime('%Y-%m-%d'))
        ]

        news_path = f"{NEWS_DIR}/{symbol}_news.csv"
        if os.path.exists(news_path):
            df_news_sym = pd.read_csv(news_path)
            df_news_sym['full_text'] = df_news_sym['headline'].fillna('') + ". " + df_news_sym['summary'].fillna('')
            df_news_sym['sentiment_score'] = df_news_sym['full_text'].apply(get_finbert_score)
            daily_sentiment = df_news_sym.groupby('date')['sentiment_score'].mean().reset_index()
        else:
            daily_sentiment = pd.DataFrame(columns=['date', 'sentiment_score'])

        merged = pd.merge(df_rich, daily_sentiment, on="date", how="left").fillna(0)

        cols_to_lag = ['Volume', 'Return', 'Market_Return', 'Excess_Return', 'Corr_30', 'Beta_30', 'sentiment_score']
        for col in cols_to_lag:
            if col in merged.columns:
                merged[f'{col}_lag1'] = merged[col].shift(1)

        merged["symbol"] = symbol
        merged = merged.dropna()

        merged.to_csv(cache_path, index=False)
        all_stocks_master.append(merged)
        print(f"  {symbol}: done")

    except Exception as e:
        print(f"  {symbol} failed: {e}")

if all_stocks_master:
    from sklearn.preprocessing import LabelEncoder
    mega_df = pd.concat(all_stocks_master, axis=0, ignore_index=True)
    le = LabelEncoder()
    mega_df['symbol_id'] = le.fit_transform(mega_df['symbol'])
    mega_df.to_csv(f"{MASTER_DIR}/Final_Mega_Dataset.csv", index=False)
    print(f"\nDone. {len(all_stocks_master)} stocks, {len(mega_df)} rows → {MASTER_DIR}/Final_Mega_Dataset.csv")
