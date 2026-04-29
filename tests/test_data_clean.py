import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_clean import load_and_clean, add_features


def test_load_and_clean_creates_return_column(tmp_path):
    csv_path = tmp_path / "AAPL.csv"

    df = pd.DataFrame({
        "Date": ["Ticker", "2020-01-01", "2020-01-02", "2020-01-03"],
        "Close": ["AAPL", "100", "110", "121"],
        "High": ["AAPL", "101", "111", "122"],
        "Low": ["AAPL", "99", "109", "120"],
        "Open": ["AAPL", "100", "105", "115"],
        "Volume": ["AAPL", "1000", "1200", "1500"],
    })

    df.to_csv(csv_path, index=False)

    cleaned = load_and_clean(csv_path)

    assert "Return" in cleaned.columns
    assert len(cleaned) == 3
    assert round(cleaned["Return"].iloc[1], 2) == 0.10


def test_add_features_creates_market_and_sector_features():
    dates = pd.date_range("2020-01-01", periods=40)

    stock_df = pd.DataFrame({
        "Date": dates,
        "Return": [0.01] * 40,
    })

    spy_df = pd.DataFrame({
        "Date": dates,
        "Market_Return": [0.005] * 40,
    })

    sector_df = pd.DataFrame({
        "Date": dates,
        "Sector_Return": [0.004] * 40,
    })

    result = add_features(stock_df, spy_df, sector_df)

    expected_columns = [
        "Excess_Return",
        "Corr_30",
        "Beta_30",
        "Excess_Return_sector",
        "Corr_30_sector",
        "Beta_30_sector",
    ]

    for col in expected_columns:
        assert col in result.columns

    assert round(result["Excess_Return"].iloc[0], 3) == 0.005
    assert round(result["Excess_Return_sector"].iloc[0], 3) == 0.006