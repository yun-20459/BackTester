"""Ticker downloading util functions."""

from collections.abc import Sequence

import yfinance as yf
import pandas as pd
from datetime import date
import sqlite3

from common import data
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


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
  conn = None
  try:
    conn = sqlite3.connect(data.DATA_DB_NAME)
    cursor = conn.cursor()
  except Exception as e:
    logger.error(f"Error connecting to database: {e}")
    raise RuntimeError(f"Error connecting to database: {e}")

  tickers_to_download = []
  for ticker in tickers:
    # Check if table exists
    table_exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ticker}';"
    cursor.execute(table_exists_query)
    table_exists = cursor.fetchone()

    if table_exists:
      # Check date range coverage
      date_range_query = f"SELECT MIN(Date), MAX(Date) FROM {ticker};"
      cursor.execute(date_range_query)
      min_db_date_str, max_db_date_str = cursor.fetchone()

      if min_db_date_str and max_db_date_str:
        # Convert DB dates to date objects for comparison
        min_db_date = pd.to_datetime(min_db_date_str).date()
        max_db_date = pd.to_datetime(max_db_date_str).date()

        # Convert input dates to date objects for comparison
        req_start_date_obj = pd.to_datetime(start_date).date()
        req_end_date_obj = pd.to_datetime(end_date).date()

        logger.debug(f"min db date: {min_db_date}, max db date: {max_db_date}")
        logger.debug(
            f"req start date: {req_start_date_obj}, req end date: {req_end_date_obj}"
        )
        # Check if the database data fully covers the requested period
        if min_db_date <= req_start_date_obj and max_db_date >= req_end_date_obj:
          logger.info(
              f"Data for {ticker} from {start_date} to {end_date} already exists in DB. Skipping download."
          )
          continue  # Skip download for this ticker
        else:
          logger.info(
              f"Partial data for {ticker} in DB or range not fully covered for {start_date} to {end_date}. Will download."
          )
          tickers_to_download.append(ticker)
      else:  # Table exists but no data
        logger.info(f"Table for {ticker} exists but is empty. Will download.")
        tickers_to_download.append(ticker)
    else:
      logger.info(f"Table for {ticker} does not exist in DB. Will download.")
      tickers_to_download.append(ticker)

  if not tickers_to_download:
    logger.info(
        "All requested stock data already exists in the database for the specified period. No new downloads needed."
    )
    conn.close()
    return {}  # Return empty dict if nothing downloaded

  try:
    logger.info(
        f"Downloading data for missing tickers: {', '.join(tickers_to_download)} from {start_date} to {end_date}"
    )

    # yfinance's 'end' parameter is exclusive, so we need to add a day to it
    # to include the data for the `end_date` itself.
    yf_start_date_str = start_date.strftime('%Y-%m-%d')
    yf_end_date_obj = end_date + pd.Timedelta(days=1)
    yf_end_date_str = yf_end_date_obj.strftime('%Y-%m-%d')

    raw_stock_data = yf.download(tickers=tickers_to_download,
                                 start=yf_start_date_str,
                                 end=yf_end_date_str)
  except Exception as e:
    logger.error(f"Error downloading stock data: {e}")
    raise RuntimeError(f"Error downloading stock data: {e}")

  if isinstance(raw_stock_data.columns, pd.MultiIndex):
    # Multi-ticker download
    for ticker in tickers_to_download:
      if ticker in raw_stock_data.columns.levels[
          1]:  # Check if ticker data was actually downloaded
        stock_df = raw_stock_data.loc[:, (slice(None), ticker)]
        stock_df.columns = stock_df.columns.droplevel(1)
        # Ensure 'Date' index is datetime for consistency
        stock_df.index = pd.to_datetime(stock_df.index)
        stock_df.to_sql(name=ticker, con=conn, if_exists='replace',
                        index=True)  # Replace existing data for this ticker
        downloaded_data_dfs[ticker] = stock_df
      else:
        logger.warning(
            f"Data for {ticker} was requested but not found in yfinance download result."
        )
  elif not raw_stock_data.empty:  # Single ticker download case
    # This block handles the case where yfinance returns a single DataFrame (e.g., if only one ticker was requested)
    if tickers_to_download:  # Ensure there was at least one ticker
      ticker = tickers_to_download[0]  # Assume only one ticker in this case
      # Ensure 'Date' index is datetime for consistency
      raw_stock_data.index = pd.to_datetime(raw_stock_data.index)
      raw_stock_data.to_sql(
          name=ticker, con=conn, if_exists='replace',
          index=True)  # Replace existing data for this ticker
      downloaded_data_dfs[ticker] = raw_stock_data
  else:
    logger.warning(
        "yfinance download returned empty data for the requested tickers.")

  conn.close()
  if downloaded_data_dfs:
    logger.info(
        f"Successfully downloaded and saved data for {len(downloaded_data_dfs)} tickers."
    )
  else:
    logger.info("No new data was downloaded or saved.")

  return downloaded_data_dfs


def _fetch_stock_data_from_sqlite(db_path: str, symbol: str, start_date: str,
                                  end_date: str):
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
  conn = None
  try:
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT Date, Open, High, Low, Close, Volume
        FROM {symbol}
        WHERE Date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY Date ASC;
        """
    logger.info(
        f"Fetching data for {symbol} from {start_date} to {end_date} from database '{db_path}'..."
    )

    data = pd.read_sql_query(query,
                             conn,
                             index_col='Date',
                             parse_dates=['Date'])

    if data.empty:
      logger.warning(
          f"No data found for {symbol} in the database or data is empty.")
      return None

    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    logger.info(f"Successfully fetched {len(data)} data points.")
    return data

  except sqlite3.Error as e:
    logger.error(f"Database operation error: {e}")
    return None
  except pd.errors.DatabaseError as e:
    logger.error(f"Pandas database reading error: {e}")
    return None
  except Exception as e:
    logger.error(f"Unexpected Error: {e}")
    return None
  finally:
    if conn:
      conn.close()


def fetch_multiple_stock_data_from_sqlite(db_path: str, symbols: list[str],
                                          start_date: str, end_date: str):
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
  logger.info(
      f"Fetching data for multiple stocks ({', '.join(symbols)}) from {start_date} to {end_date}..."
  )
  for symbol in symbols:
    data = _fetch_stock_data_from_sqlite(db_path, symbol, start_date, end_date)
    if data is not None and not data.empty:
      all_stock_data[symbol] = data
    else:
      logger.info(
          f"Warning: Could not retrieve valid data for {symbol}, skipping this stock."
      )

  if not all_stock_data:
    logger.error("Error: No stock data was successfully retrieved.")
    return None

  logger.info(f"Successfully retrieved data for {len(all_stock_data)} stocks.")
  return all_stock_data
