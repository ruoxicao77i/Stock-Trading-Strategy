import yfinance as yf
import pandas as pd
import os

# 剎src執行python data_download.py
# 設定的地方自己調整時間、股票代碼
# 下載的資料會存在data/raw資料夾裡面
# AAPL(apple) GOOG(Google) MSFT(Microsoft) AMZN(Amazon) TSLA(Tesla) NVDA(NVIDIA)

#-------設定的地方-------
TICKERS = [
    "NVDA",
    "TSLA"
]

START_DATE = "2019-05-01"
END_DATE = "2022-10-17"
INTERVAL = "1d"
SAVE_DIR = "../data/raw"
#-------設定的地方-------


def download_stock(ticker):

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        progress=False
    )


    if df.empty:
        print(f"Error: {ticker}")
        return

    df.reset_index(inplace=True)

    filepath = f"{SAVE_DIR}/{ticker}.csv"
    df.to_csv(filepath, index=False)


def main():

    os.makedirs(SAVE_DIR, exist_ok=True)

    for ticker in TICKERS:
        try:
            download_stock(ticker)
        except Exception as e:
            print(f"Error: {ticker}: {e}")



if __name__ == "__main__":
    main()