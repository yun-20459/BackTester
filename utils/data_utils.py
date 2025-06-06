from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date
import pandas as pd
import yfinance as yf

from common import data
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


def _check_table_exists(conn: sqlite3.Connection, ticker: str) -> bool:
  """Checks if a table for the given ticker exists in the database."""
  cursor = conn.cursor()
  table_exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ticker}';"
  cursor.execute(table_exists_query)
  return cursor.fetchone() is not None


def _get_db_date_range(conn: sqlite3.Connection,
                       ticker: str) -> tuple[date | None, date | None]:
  """Gets the min and max dates from a stock table in the database."""
  cursor = conn.cursor()
  date_range_query = f"SELECT MIN(Date), MAX(Date) FROM {ticker};"
  cursor.execute(date_range_query)
  min_db_date_str, max_db_date_str = cursor.fetchone()

  min_db_date = pd.to_datetime(
      min_db_date_str).date() if min_db_date_str else None
  max_db_date = pd.to_datetime(
      max_db_date_str).date() if max_db_date_str else None
  return min_db_date, max_db_date


def _get_tickers_to_download(conn: sqlite3.Connection, tickers: Sequence[str],
                             start_date: date, end_date: date) -> list[str]:
  """Determines which tickers need to be downloaded based on existing DB data."""
  tickers_to_download = []
  req_start_date_obj = pd.to_datetime(start_date).date()
  req_end_date_obj = pd.to_datetime(end_date).date()

  for ticker in tickers:
    if _check_table_exists(conn, ticker):
      min_db_date, max_db_date = _get_db_date_range(conn, ticker)

      if min_db_date and max_db_date and \
         min_db_date <= req_start_date_obj and max_db_date >= req_end_date_obj:
        logger.info(
            "Data for %s from %s to %s already exists in DB. Skipping download.",
            ticker, start_date, end_date)
        continue
      logger.info(
          "Partial data for %s in DB or range not fully covered for %s to %s. Will download.",
          ticker, start_date, end_date)
      tickers_to_download.append(ticker)
    else:
      logger.info("Table for %s does not exist in DB. Will download.", ticker)
      tickers_to_download.append(ticker)
  return tickers_to_download


def _download_from_yfinance(tickers: list[str], start_date: date,
                            end_date: date) -> pd.DataFrame:
  """Downloads stock data from yfinance."""
  logger.info("Downloading data for missing tickers: %s from %s to %s",
              ', '.join(tickers), start_date, end_date)

  yf_start_date_str = start_date.strftime('%Y-%m-%d')
  yf_end_date_obj = end_date + pd.Timedelta(days=1)
  yf_end_date_str = yf_end_date_obj.strftime('%Y-%m-%d')

  try:
    raw_stock_data = yf.download(tickers=tickers,
                                 start=yf_start_date_str,
                                 end=yf_end_date_str,
                                 progress=False)
    return raw_stock_data
  except Exception as e:
    logger.error("Error downloading stock data from yfinance: %s", e)
    raise RuntimeError(
        f"Error downloading stock data from yfinance: {e}") from e


def _save_data_to_db(
    conn: sqlite3.Connection, raw_data: pd.DataFrame,
    tickers_to_download: list[str]) -> dict[str, pd.DataFrame]:
  """Saves downloaded stock data to the SQLite database."""
  downloaded_dfs = {}
  if isinstance(raw_data.columns, pd.MultiIndex):
    for ticker in tickers_to_download:
      if ticker in raw_data.columns.levels[1]:
        stock_df = raw_data.loc[:, (slice(None), ticker)]
        stock_df.columns = stock_df.columns.droplevel(1)
        stock_df.index = pd.to_datetime(stock_df.index)
        stock_df.to_sql(name=ticker, con=conn, if_exists='replace', index=True)
        downloaded_dfs[ticker] = stock_df
      else:
        logger.warning(
            "Data for %s was requested but not found in yfinance download result.",
            ticker)
  elif not raw_data.empty and tickers_to_download:
    # This case handles a single ticker download where raw_data is not MultiIndex
    ticker = tickers_to_download[0]
    raw_data.index = pd.to_datetime(raw_data.index)
    raw_data.to_sql(name=ticker, con=conn, if_exists='replace', index=True)
    downloaded_dfs[ticker] = raw_data
  else:
    logger.warning(
        "yfinance download returned empty data for the requested tickers.")
  return downloaded_dfs


def download_stock_data(tickers: Sequence[str], start_date: date,
                        end_date: date) -> dict[str, pd.DataFrame]:
  """
  Downloads stock data for specified tickers and date range,
  only if the data is not already present in the SQLite database for the full period.
  Saves newly downloaded data to the SQLite database.

  Args:
    tickers (Sequence[str]): List of stock tickers to download.
    start_date (date): Start date for the data (inclusive).
    end_date (date): End date for the data (inclusive for DB check, exclusive for yfinance).

  Returns:
    dict[str, pd.DataFrame]: A dictionary of newly downloaded DataFrames, keyed by ticker.
                             Returns an empty dictionary if no new data was downloaded.
  """
  downloaded_data_dfs = {}
  conn: sqlite3.Connection | None = None
  try:
    conn = sqlite3.connect(data.DATA_DB_NAME)
  except sqlite3.Error as e:
    logger.error("Error connecting to database: %s", e)
    raise RuntimeError(f"Error connecting to database: {e}") from e

  tickers_to_download = _get_tickers_to_download(conn, tickers, start_date,
                                                 end_date)

  if not tickers_to_download:
    logger.info(
        "All requested stock data already exists in the database for the specified period. No new downloads needed."
    )
    conn.close()
    return {}

  raw_stock_data = _download_from_yfinance(tickers_to_download, start_date,
                                           end_date)

  if raw_stock_data.empty:
    logger.warning(
        "No data downloaded from yfinance for the requested tickers.")
    conn.close()
    return {}

  downloaded_data_dfs = _save_data_to_db(conn, raw_stock_data,
                                         tickers_to_download)

  conn.close()
  if downloaded_data_dfs:
    logger.info("Successfully downloaded and saved data for %d tickers.",
                len(downloaded_data_dfs))
  else:
    logger.info("No new data was downloaded or saved.")

  return downloaded_data_dfs


def download_and_save_latest_data(symbol: str,
                                  db_path: str = data.DATA_DB_NAME
                                  ) -> float | None:
  """
    Downloads the latest daily data for a single symbol from yfinance and saves it to SQLite.
    Returns the latest closing price if successful, otherwise None.
    """
  logger.info("Attempting to download latest data for %s from yfinance...",
              symbol)
  conn: sqlite3.Connection | None = None
  today = date.today()
  try:
    raw_data = yf.download(tickers=symbol,
                           period='2d',
                           interval='1d',
                           progress=False)
  except Exception as e:
    logger.error("Error downloading latest data for %s from yfinance: %s",
                 symbol, e)
    return None

  if raw_data.empty:
    logger.warning(
        "No latest data found for %s from yfinance for the last 2 days.",
        symbol)
    return None

  raw_data.index = pd.to_datetime(raw_data.index)
  raw_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

  latest_data_row = raw_data.iloc[[-1]]  # Get the very last available row

  if latest_data_row.empty:
    logger.warning(
        "No valid latest data found for %s to determine current price.",
        symbol)
    return None

  current_price = latest_data_row['Close'].iloc[0]
  latest_data_date = latest_data_row.index[0].date()

  if latest_data_date == today:
    logger.info(
        "Successfully downloaded today's data for %s. Current Close: %.2f",
        symbol, current_price)
  else:
    logger.info("Latest data for %s is from %s. Using its Close: %.2f", symbol,
                latest_data_date.strftime('%Y-%m-%d'), current_price)

  try:
    conn = sqlite3.connect(db_path)
    if _check_table_exists(conn, symbol):
      _, max_db_date = _get_db_date_range(conn, symbol)
      if max_db_date == latest_data_date:
        logger.info(
            "Latest data for %s on %s already exists in DB. Skipping append.",
            symbol, latest_data_date)
        return current_price

    latest_data_row.to_sql(name=symbol,
                           con=conn,
                           if_exists='append',
                           index=True)
    logger.info("Latest data for %s on %s saved to %s.", symbol,
                latest_data_date.strftime('%Y-%m-%d'), db_path)

    return current_price

  except sqlite3.Error as e:
    logger.error(
        "Database error while downloading or saving latest data for %s: %s",
        symbol, e)
    return None
  finally:
    if conn:
      conn.close()


def fetch_stock_data_from_sqlite(db_path: str, symbol: str, start_date: str,
                                 end_date: str) -> pd.DataFrame | None:
  """
    Retrieves OHLCV data for a given stock and date range from an SQLite database.

    Args:
      db_path (str): Path to the SQLite database file.
      symbol (str): Stock ticker, which is also the table name in the database.
      start_date (str): Start date (YYYY-MM-DD).
      end_date (str): End date (YYYY-MM-DD).

    Returns:
      pd.DataFrame: Pandas DataFrame containing OHLCV data, with Date as index.
                    Returns None if data does not exist or is empty.
    """
  conn: sqlite3.Connection | None = None
  query = f"""
      SELECT Date, Open, High, Low, Close, Volume
      FROM {symbol}
      WHERE Date BETWEEN '{start_date}' AND '{end_date}'
      ORDER BY Date ASC;
      """
  logger.info("Fetching data for %s from %s to %s from database '%s'...",
              symbol, start_date, end_date, db_path)
  try:
    conn = sqlite3.connect(db_path)

    stock_data = pd.read_sql_query(query,
                                   conn,
                                   index_col='Date',
                                   parse_dates=['Date'])

  except sqlite3.Error as e:
    logger.error("Database operation error for %s: %s", symbol, e)
    return None
  except pd.errors.DatabaseError as e:
    logger.error("Pandas database reading error for %s: %s", symbol, e)
    return None
  finally:
    if conn:
      conn.close()

  if stock_data.empty:
    logger.warning("No data found for %s in the database or data is empty.",
                   symbol)
    return None

  stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

  logger.info("Successfully fetched %d data points.", len(stock_data))
  return stock_data


def fetch_multiple_stock_data_from_sqlite(
    db_path: str, symbols: list[str], start_date: str,
    end_date: str) -> dict[str, pd.DataFrame] | None:
  """
    Fetches OHLCV data for multiple stocks from an SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        symbols (list[str]): List of stock symbols.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).

    Returns:
        dict[str, pd.DataFrame]: Dictionary where keys are stock symbols and values are their OHLCV DataFrames.
                                If data retrieval for a certain stock fails, it will not be included in the dictionary.
    """
  all_stock_data = {}
  logger.info("Fetching data for multiple stocks (%s) from %s to %s...",
              ', '.join(symbols), start_date, end_date)
  for symbol in symbols:
    stock_df = fetch_stock_data_from_sqlite(db_path, symbol, start_date,
                                            end_date)
    if stock_df is not None and not stock_df.empty:
      all_stock_data[symbol] = stock_df
    else:
      logger.warning(
          "Could not retrieve valid data for %s, skipping this stock.", symbol)

  if not all_stock_data:
    logger.error("No stock data was successfully retrieved.")
    return None

  logger.info("Successfully retrieved data for %d stocks.",
              len(all_stock_data))
  return all_stock_data
