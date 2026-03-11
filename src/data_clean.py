import pandas as pd
import numpy as np
import os
import pandas_market_calendars as mcal

"""
Stock Data Cleaning & Feature Engineering

This script loads raw stock data downloaded from Yahoo Finance,
cleans the data, and generates additional financial features.

Pipeline:
    1. Raw CSV files (data/raw)
    2. Data cleaning & feature engineering
    3. Cleaned dataset (data/clean)

Generated features include:
    - Daily stock return
    - Market return (SPY)
    - Excess return
    - 30-day rolling correlation with the market
    - 30-day rolling beta
"""


# ==============================
# Directory Configuration
# ==============================

RAW_DIR = "../data/raw"
CLEAN_DIR = "../data/clean"
TICKERS = ["AAPL", "GOOG", "MSFT", "TSLA", "NVDA", "QQQ"]

def load_and_clean_stock(filepath):
    """
    Load a raw stock CSV file and perform basic cleaning.

    Steps:
    1. Remove incorrect first row (Yahoo Finance artifact)
    2. Convert Date column to datetime
    3. Convert numeric columns to numeric types
    4. Sort data by date
    5. Compute daily return

    Parameters
    ----------
    filepath : str
        Path to the raw stock CSV file

    Returns
    -------
    pandas.DataFrame
        Cleaned stock data with daily return
    """
    df = pd.read_csv(filepath)

    # Remove the first row (sometimes contains metadata or invalid values)
    df = df.iloc[1:]

    # Convert Date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Convert price and volume columns to numeric values
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    print(df.isna().sum())
    df[numeric_cols] = df[numeric_cols].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    
    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(
        start_date=df.index.min(),
        end_date=df.index.max()
    )

    trading_days = schedule.index

    missing = trading_days.difference(df.index)
    print(missing)

    # Sort rows by date to ensure chronological order
    df = df.sort_values("Date")

    # Compute daily stock return (计算收益率)
    # Return_t = (Price_t - Price_t-1) / Price_t-1
    df["Return"] = df["Close"].pct_change()

    return df


def load_spy():
    """
    Load SPY (S&P 500 ETF) data and compute market return.

    SPY is used as the market benchmark.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing:
        - Date
        - Market_Return
    """
    spy = pd.read_csv(f"{RAW_DIR}/SPY.csv")

    # Remove first invalid row
    spy = spy.iloc[1:]

    spy["Date"] = pd.to_datetime(spy["Date"])

    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    spy[numeric_cols] = spy[numeric_cols].apply(pd.to_numeric)

    spy = spy.sort_values("Date")

    spy["Market_Return"] = spy["Close"].pct_change()

    return spy[["Date", "Market_Return"]]


def add_features(stock_df, spy_df):

    # 按日期合并
    df = pd.merge(stock_df, spy_df, on="Date", how="inner")

    # 超额收益
    df["Excess_Return"] = df["Return"] - df["Market_Return"]

    # 30天滚动相关性
    df["Corr_30"] = df["Return"].rolling(30).corr(df["Market_Return"])

    # Rolling Beta
    cov = df["Return"].rolling(30).cov(df["Market_Return"])
    var = df["Market_Return"].rolling(30).var()

    df["Beta_30"] = cov / var

    return df


def main():

    os.makedirs(CLEAN_DIR, exist_ok=True)

    spy = load_spy()

    for ticker in TICKERS:

        print(f"Cleaning {ticker}...")

        stock_path = f"{RAW_DIR}/{ticker}.csv"

        stock = load_and_clean_stock(stock_path)

        stock = add_features(stock, spy)

        save_path = f"{CLEAN_DIR}/{ticker}_clean.csv"

        stock.to_csv(save_path, index=False)

        print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()