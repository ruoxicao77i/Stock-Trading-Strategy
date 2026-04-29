import pandas as pd
import numpy as np
import os

# ✅ 所有数据统一来源
RAW_DIR = "data/raw"

CLEAN_DIR = "data/clean"

TICKERS = [
    "AAPL", "MSFT", "GOOG",
    "JPM", "BAC", "GS",
    "JNJ", "PFE", "UNH",
    "XOM", "CVX",
    "AMZN", "TSLA", "HD"
]

# 股票 → 行业ETF
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOG": "XLK",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF",
    "JNJ": "XLV", "PFE": "XLV", "UNH": "XLV",
    "XOM": "XLE", "CVX": "XLE",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY"
}


def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    df = df.iloc[1:]
    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    df = df.sort_values("Date")
    df["Return"] = df["Close"].pct_change()

    return df


def load_spy():
    spy = load_and_clean(f"{RAW_DIR}/SPY.csv")
    spy.rename(columns={"Return": "Market_Return"}, inplace=True)
    return spy[["Date", "Market_Return"]]


def load_sector(etf):
    sector = load_and_clean(f"{RAW_DIR}/{etf}.csv")
    sector.rename(columns={"Return": "Sector_Return"}, inplace=True)
    return sector[["Date", "Sector_Return"]]


def add_features(stock_df, spy_df, sector_df):

    # ===== merge SPY =====
    df = pd.merge(stock_df, spy_df, on="Date", how="inner")

    # ===== vs SPY =====
    df["Excess_Return"] = df["Return"] - df["Market_Return"]

    df["Corr_30"] = df["Return"].rolling(30).corr(df["Market_Return"])

    cov = df["Return"].rolling(30).cov(df["Market_Return"])
    var = df["Market_Return"].rolling(30).var()
    df["Beta_30"] = cov / var

    # ===== merge sector =====
    df = pd.merge(df, sector_df, on="Date", how="inner")

    # ===== vs sector =====
    df["Excess_Return_sector"] = df["Return"] - df["Sector_Return"]

    df["Corr_30_sector"] = df["Return"].rolling(30).corr(df["Sector_Return"])

    cov_s = df["Return"].rolling(30).cov(df["Sector_Return"])
    var_s = df["Sector_Return"].rolling(30).var()
    df["Beta_30_sector"] = cov_s / var_s

    return df


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    spy = load_spy()

    for ticker in TICKERS:
        print(f"Cleaning {ticker}...")

        stock = load_and_clean(f"{RAW_DIR}/{ticker}.csv")

        sector_etf = SECTOR_MAP[ticker]
        sector = load_sector(sector_etf)

        stock = add_features(stock, spy, sector)

        save_path = f"{CLEAN_DIR}/{ticker}_clean.csv"

        stock.to_csv(save_path, index=False)

        print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()