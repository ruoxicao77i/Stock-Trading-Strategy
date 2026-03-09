import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"

TICKERS = ["AAPL", "GOOG", "MSFT"]


def load_and_clean_stock(filepath):

    df = pd.read_csv(filepath)

    # 删除第一行错误数据
    df = df.iloc[1:]

    # 日期转换
    df["Date"] = pd.to_datetime(df["Date"])

    # 数值列转换
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    # 排序
    df = df.sort_values("Date")

    # 计算收益率
    df["Return"] = df["Close"].pct_change()

    return df


def load_spy():

    spy = pd.read_csv(f"{RAW_DIR}/SPY.csv")

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