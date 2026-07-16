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


def test_save_price_history_updates_existing_dates_on_rerun(tmp_path):
    """Saving an overlapping date range twice should update, not duplicate."""
    db = StockDatabase(f"sqlite:///{tmp_path / 'stocks.db'}")
    first = pd.DataFrame({
        'Date': ['2024-01-02', '2024-01-03'],
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1_000_000, 1_100_000],
    })
    second = pd.DataFrame({
        'Date': ['2024-01-03', '2024-01-04'],
        'Open': [201.0, 104.0],
        'High': [203.0, 106.0],
        'Low': [200.0, 103.0],
        'Close': [202.0, 105.0],
        'Volume': [2_100_000, 1_200_000],
    })

    db.save_price_history('AAPL', first)
    db.save_price_history('AAPL', second)
    saved = db.get_price_history('AAPL', '2024-01-01', '2024-01-31')

    assert len(saved) == 3
    assert saved['Date'].dt.strftime('%Y-%m-%d').tolist() == [
        '2024-01-02', '2024-01-03', '2024-01-04'
    ]
    assert saved['Close'].tolist() == [101.0, 202.0, 105.0]


def test_save_price_history_deduplicates_input_dates(tmp_path):
    """Duplicate dates in one input frame should keep the latest row."""
    db = StockDatabase(f"sqlite:///{tmp_path / 'stocks.db'}")
    prices = pd.DataFrame({
        'Date': ['2024-01-02', '2024-01-02'],
        'Open': [100.0, 200.0],
        'High': [102.0, 202.0],
        'Low': [99.0, 199.0],
        'Close': [101.0, 201.0],
        'Volume': [1_000_000, 2_000_000],
    })

    db.save_price_history('AAPL', prices)
    saved = db.get_price_history('AAPL', '2024-01-01', '2024-01-31')

    assert len(saved) == 1
    assert saved['Open'].tolist() == [200.0]
    assert saved['Close'].tolist() == [201.0]


def test_save_price_history_rerun_with_timezone_aware_dates(tmp_path):
    """Timezone-aware Yahoo dates should match existing naive SQLite dates."""
    db = StockDatabase(f"sqlite:///{tmp_path / 'stocks.db'}")
    first = pd.DataFrame({
        'Date': ['2024-01-02', '2024-01-03'],
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1_000_000, 1_100_000],
    })
    second = pd.DataFrame({
        'Date': pd.DatetimeIndex(['2024-01-03', '2024-01-04'], tz='America/New_York'),
        'Open': [201.0, 104.0],
        'High': [203.0, 106.0],
        'Low': [200.0, 103.0],
        'Close': [202.0, 105.0],
        'Volume': [2_100_000, 1_200_000],
    })

    db.save_price_history('AAPL', first)
    db.save_price_history('AAPL', second)
    saved = db.get_price_history('AAPL', '2024-01-01', '2024-01-31')

    assert len(saved) == 3
    assert saved['Date'].dt.strftime('%Y-%m-%d').tolist() == [
        '2024-01-02', '2024-01-03', '2024-01-04'
    ]
    assert saved['Close'].tolist() == [101.0, 202.0, 105.0]
