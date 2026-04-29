import yfinance as yf
import pandas as pd
import os


"""
Stock Data Downloader

Run this script in the src directory:
    python data_download.py

This script downloads historical stock data using Yahoo Finance (yfinance) 
and saves the data as CSV files under the data/raw directory.

You can modify the following parameters:
    - stock tickers
    - start/end date
    - data interval
    - save directory
"""

# ==============================
# Configuration Section
# ==============================

# List of stock/ETF tickers to download
# AAPL (Apple) | GOOG (Google) | MSFT (Microsoft)
# TSLA (Tesla) | NVDA (NVIDIA)
# QQQ (NASDAQ-100 ETF) | SPY (S&P500 ETF)

TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOG", "NVDA", "QQQ","XLK",

    # Financial
    "JPM", "BAC", "GS", "XLF",

    # Healthcare
    "JNJ", "PFE", "UNH", "XLV",

    # Energy
    "XOM", "CVX", "XLE",

    # Consumer
    "AMZN", "TSLA", "HD", "XLY",
    
     "SPY"
]

# Time range of historical data
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# Data interval
INTERVAL = "1d"

# Directory to store downloaded CSV files
SAVE_DIR = "data/raw"
#-------設定的地方-------



def download_stock(ticker, start_date, end_date, interval):
    """
    Download historical stock data for a given ticker
    and save it as a CSV file.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., AAPL, MSFT)
    """

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False
    )

    # If no data is returned, print error message
    if df.empty:
        print(f"Error: {ticker}")
        return

    # Convert index (Date) into a column
    df.reset_index(inplace=True)
    # Build file path and save df to csv
    filepath = f"{SAVE_DIR}/{ticker}.csv"
    df.to_csv(filepath, index=False)

    print(f"Downloaded and saved: {ticker}")


def main():
    """
    Main function:
    - create data directory
    - download data for all tickers
    """

    # Create save directory if it does not exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    for ticker in TICKERS:
        try:
            download_stock(ticker, START_DATE, END_DATE, INTERVAL)
        except Exception as e:
            print(f"Error: {ticker}: {e}")



if __name__ == "__main__":
    main()