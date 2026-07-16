"""Tests for price-history persistence."""

import pandas as pd

from src.data.storage import StockDatabase


def test_save_price_history_accepts_datetime_index(tmp_path):
    """DatetimeIndex price frames should persist with real dates, not NaT."""
    db = StockDatabase(f"sqlite:///{tmp_path / 'stocks.db'}")
    prices = pd.DataFrame(
        {
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1_000_000, 1_100_000],
        },
        index=pd.DatetimeIndex(['2024-01-02', '2024-01-03'], name='Date')
    )

    db.save_price_history('AAPL', prices)
    saved = db.get_price_history('AAPL', '2024-01-01', '2024-01-31')

    assert len(saved) == 2
    assert saved['Date'].isna().sum() == 0
    assert saved['Close'].tolist() == [101.0, 102.0]


def test_save_price_history_skips_invalid_dates(tmp_path):
    """Invalid Date rows should be dropped before SQLite date binding."""
    db = StockDatabase(f"sqlite:///{tmp_path / 'stocks.db'}")
    prices = pd.DataFrame({
        'Date': ['2024-01-02', None],
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1_000_000, 1_100_000],
    })

    db.save_price_history('AAPL', prices)
    saved = db.get_price_history('AAPL', '2024-01-01', '2024-01-31')

    assert len(saved) == 1
    assert saved['Close'].tolist() == [101.0]
