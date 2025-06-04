# utils/data_utils.py

from collections.abc import Sequence
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import sqlite3

from common import data  # Import the data module for DB name
from utils import logger_utils  # Import our custom logger utility

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
  except Exception as e:
    logger.error(f"Error connecting to database: {e}")
    raise RuntimeError(f"Error connecting to database: {e}")

  tickers_to_download = []
  for ticker in tickers:
    if _check_table_exists(conn, ticker):
      min_db_date, max_db_date = _get_db_date_range(conn, ticker)

      req_start_date_obj = pd.to_datetime(start_date).date()
      req_end_date_obj = pd.to_datetime(end_date).date()

      if min_db_date and max_db_date and \
         min_db_date <= req_start_date_obj and max_db_date >= req_end_date_obj:
        logger.info(
            f"Data for {ticker} from {start_date} to {end_date} already exists in DB. Skipping download."
        )
        continue
      else:
        logger.info(
            f"Partial data for {ticker} in DB or range not fully covered for {start_date} to {end_date}. Will download."
        )
        tickers_to_download.append(ticker)
    else:
      logger.info(f"Table for {ticker} does not exist in DB. Will download.")
      tickers_to_download.append(ticker)

  if not tickers_to_download:
    logger.info(
        "All requested stock data already exists in the database for the specified period. No new downloads needed."
    )
    conn.close()
    return {}

  try:
    logger.info(
        f"Downloading data for missing tickers: {', '.join(tickers_to_download)} from {start_date} to {end_date}"
    )

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
    for ticker in tickers_to_download:
      if ticker in raw_stock_data.columns.levels[1]:
        stock_df = raw_stock_data.loc[:, (slice(None), ticker)]
        stock_df.columns = stock_df.columns.droplevel(1)
        stock_df.index = pd.to_datetime(stock_df.index)
        # Use if_exists='replace' to ensure full range is replaced if partial existed
        stock_df.to_sql(name=ticker, con=conn, if_exists='replace', index=True)
        downloaded_data_dfs[ticker] = stock_df
      else:
        logger.warning(
            f"Data for {ticker} was requested but not found in yfinance download result."
        )
  elif not raw_stock_data.empty:
    if tickers_to_download:
      ticker = tickers_to_download[0]
      raw_stock_data.index = pd.to_datetime(raw_stock_data.index)
      raw_stock_data.to_sql(name=ticker,
                            con=conn,
                            if_exists='replace',
                            index=True)
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


def download_and_save_latest_data(symbol: str,
                                  db_path: str = data.DATA_DB_NAME
                                  ) -> float | None:
  """
    Downloads the latest daily data for a single symbol from yfinance and saves it to SQLite.
    Returns the latest closing price if successful, otherwise None.
    """
  logger.info(
      f"Attempting to download latest data for {symbol} from yfinance...")
  try:
    today = date.today()
    raw_data = yf.download(
        tickers=symbol, period='2d',
        interval='1d')  # Get last 2 days to ensure latest close

    if raw_data.empty:
      logger.warning(
          f"No latest data found for {symbol} from yfinance for the last 2 days."
      )
      return None

    raw_data.index = pd.to_datetime(raw_data.index)
    raw_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    latest_data_row = raw_data.iloc[[-1]]  # Get the very last available row

    if latest_data_row.empty:
      logger.warning(
          f"No valid latest data found for {symbol} to determine current price."
      )
      return None

    current_price = latest_data_row['Close'].iloc[0]
    latest_data_date = latest_data_row.index[0].date()

    if latest_data_date == today:
      logger.info(
          f"Successfully downloaded today's data for {symbol}. Current Close: {current_price:.2f}"
      )
    else:
      logger.info(
          f"Latest data for {symbol} is from {latest_data_date.strftime('%Y-%m-%d')}. Using its Close: {current_price:.2f}"
      )

    # Save to SQLite
    conn = sqlite3.connect(db_path)
    # Check if today's data already exists to avoid duplicates if running multiple times on same day
    if _check_table_exists(conn, symbol):
      min_db_date, max_db_date = _get_db_date_range(conn, symbol)
      if max_db_date == latest_data_date:
        logger.info(
            f"Latest data for {symbol} on {latest_data_date} already exists in DB. Skipping append."
        )
        conn.close()
        return current_price

    # Append only the latest row to avoid re-appending older data
    latest_data_row.to_sql(name=symbol,
                           con=conn,
                           if_exists='append',
                           index=True)
    conn.close()
    logger.info(
        f"Latest data for {symbol} on {latest_data_date.strftime('%Y-%m-%d')} saved to {db_path}."
    )

    return current_price

  except Exception as e:
    logger.error(f"Error downloading or saving latest data for {symbol}: {e}")
    return None


def fetch_stock_data_from_sqlite(db_path: str, symbol: str, start_date: str,
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
    data = fetch_stock_data_from_sqlite(db_path, symbol, start_date, end_date)
    if data is not None and not data.empty:
      all_stock_data[symbol] = data
    else:
      logger.warning(
          f"Could not retrieve valid data for {symbol}, skipping this stock.")

  if not all_stock_data:
    logger.error("No stock data was successfully retrieved.")
    return None

  logger.info(f"Successfully retrieved data for {len(all_stock_data)} stocks.")
  return all_stock_data
