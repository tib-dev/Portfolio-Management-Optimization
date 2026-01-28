from pathlib import Path
from typing import Dict, List
import pandas as pd
import pytest

from pmo_forcasting.data.pipeline import MarketDataDownloader


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_config() -> Dict:
    return {
        "date_range": {
            "start": "2020-01-01",
            "end": "2020-01-10",
        },
        "fields": {
            "prices": ["open", "high", "low", "close", "adj_close"],
            "volume": True,
        },
        "assets": [
            {
                "ticker": "TSLA",
                "asset_class": "equity",
                "risk_profile": "high",
            },
            {
                "ticker": "SPY",
                "asset_class": "equity",
                "risk_profile": "medium",
            },
        ],
    }


@pytest.fixture
def fake_yfinance_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Adj Close": [1.4, 2.4, 3.4, 4.4, 5.4],
            "Volume": [100, 200, 300, 400, 500],
        },
        index=dates,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def mock_yfinance_download(monkeypatch, df: pd.DataFrame):
    def _mock_download(*args, **kwargs):
        return df.copy()

    monkeypatch.setattr("yfinance.download", _mock_download)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_run_downloads_and_combines_data(
    tmp_path: Path,
    sample_config: Dict,
    fake_yfinance_df: pd.DataFrame,
    monkeypatch,
):
    """Downloader should return a combined DataFrame with all tickers."""
    mock_yfinance_download(monkeypatch, fake_yfinance_df)

    downloader = MarketDataDownloader(
        config=sample_config,
        output_dir=tmp_path,
    )

    df = downloader.run(force=True)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df["ticker"].unique()) == {"TSLA", "SPY"}

    expected_cols = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ticker",
        "asset_class",
        "risk_profile",
    }
    assert expected_cols.issubset(df.columns)


def test_csv_persistence_and_reload(
    tmp_path: Path,
    sample_config: Dict,
    fake_yfinance_df: pd.DataFrame,
    monkeypatch,
):
    """Downloader should save CSVs and reload them when force=False."""
    mock_yfinance_download(monkeypatch, fake_yfinance_df)

    downloader = MarketDataDownloader(sample_config, tmp_path)

    df_first = downloader.run(force=True)
    df_second = downloader.run(force=False)

    assert len(df_first) == len(df_second)
    assert (tmp_path / "TSLA.csv").exists()
    assert (tmp_path / "SPY.csv").exists()


def test_missing_data_raises_error(
    tmp_path: Path,
    sample_config: Dict,
    monkeypatch,
):
    """Empty yfinance response should trigger a RuntimeError per implementation."""
    def _empty_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("yfinance.download", _empty_download)

    downloader = MarketDataDownloader(sample_config, tmp_path)

    with pytest.raises(RuntimeError, match="Yahoo Finance returned no data."):
        downloader.run(force=True)


def test_schema_guard_enforced(
    tmp_path: Path,
    sample_config: Dict,
    fake_yfinance_df: pd.DataFrame,
    monkeypatch,
):
    """Missing required OHLC columns should result in an empty DataFrame (graceful fail)."""
    broken_df = fake_yfinance_df.drop(columns=["Close"])

    mock_yfinance_download(monkeypatch, broken_df)

    downloader = MarketDataDownloader(sample_config, tmp_path)

    df = downloader.run(force=True)

    assert df.empty


def test_single_asset_failure_does_not_break_others(
    tmp_path: Path,
    sample_config: Dict,
    fake_yfinance_df: pd.DataFrame,
    monkeypatch,
):
    """
    Failure for one ticker (TSLA) should not prevent others (SPY) from loading.
    The mock ensures TSLA is treated as empty data.
    """
    def _conditional_download(tickers, *args, **kwargs):
        # Normalize tickers to a list
        ticker_list = [tickers] if isinstance(tickers, str) else tickers

        if ticker_list == ["TSLA"]:
            return pd.DataFrame()

        return fake_yfinance_df.copy()

    monkeypatch.setattr("yfinance.download", _conditional_download)

    downloader = MarketDataDownloader(sample_config, tmp_path)

    df = downloader.run(force=True)

    assert not df.empty

    # Assert that TSLA is not in the final consolidated data
    assert "TSLA" not in df["ticker"].values
    assert "SPY" in df["ticker"].values
    assert df["ticker"].nunique() == 1
