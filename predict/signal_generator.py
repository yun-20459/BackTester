import argparse
import importlib
import json
import logging
import sys
from datetime import date, timedelta

import pandas as pd

from common import data, market
from core import account
from utils import data_utils, logger_utils

logger = logger_utils.get_logger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("predict/predict.log", mode='w', encoding='utf-8')
    ])


def _parse_arguments():
  """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
  parser = argparse.ArgumentParser(
      description="Generate trading signals for today based on a config file.")
  parser.add_argument(
      '--config_path',
      type=str,
      required=True,
      help="Path to the JSON configuration file (e.g., 'config.json').")

  return parser.parse_args()


def _load_config(config_path: str) -> dict:
  """
    Loads and parses the JSON configuration file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Loaded configuration dictionary.

    Raises:
        SystemExit: If the config file cannot be loaded or is invalid.
    """
  try:
    with open(config_path, 'r', encoding='utf-8') as f:
      config = json.load(f)
    logger.info("Configuration loaded from: %s", config_path)
    return config
  except FileNotFoundError:
    logger.critical("Config file not found at: %s", config_path)
    sys.exit(1)
  except json.JSONDecodeError as e:
    logger.critical("Error parsing config file %s: %s", config_path, e)
    sys.exit(1)
  except Exception as e:
    logger.critical("An unexpected error occurred while loading config: %s", e)
    sys.exit(1)


def _prepare_historical_data(symbol: str, current_date: date,
                             historical_period_days: int,
                             db_path: str) -> pd.DataFrame:
  """
    Fetches historical data for a single symbol.

    Args:
        symbol (str): The stock symbol.
        current_date (date): The current date (for historical end date).
        historical_period_days (int): Number of historical days to fetch.
        db_path (str): Path to the SQLite database.

    Returns:
        pd.DataFrame: Historical data, or None if data cannot be fetched.
    """
  end_date_hist = current_date - timedelta(days=1)
  start_date_hist = end_date_hist - timedelta(days=historical_period_days)

  historical_dfs = data_utils.fetch_multiple_stock_data_from_sqlite(
      db_path, [symbol], start_date_hist.strftime('%Y-%m-%d'),
      end_date_hist.strftime('%Y-%m-%d'))

  if not historical_dfs or symbol not in historical_dfs:
    logger.error(
        "Could not fetch sufficient historical data for %s. Cannot generate signal.",
        symbol)
    return None

  return historical_dfs[symbol]


def _load_strategy(strategy_config: dict, broker: account.AccountSimulator):
  """
    Dynamically loads and instantiates the trading strategy.

    Args:
        strategy_config (dict): Dictionary containing 'name', 'module', and 'params' for the strategy.
        broker (account.AccountSimulator): The broker instance to pass to the strategy.

    Returns:
        object: An instantiated strategy object.

    Raises:
        ImportError: If the strategy module or class cannot be found.
        AttributeError: If the strategy class cannot be found in the module.
    """
  strategy_name = strategy_config['name']
  strategy_module_name = strategy_config['module']
  strategy_params = strategy_config.get('params', {})

  strategy_module = importlib.import_module(f"strategy.{strategy_module_name}")
  strategy_class = getattr(strategy_module, strategy_name)
  return strategy_class(broker, **strategy_params)


def _get_current_bar_data(today: date, current_price: float) -> dict:
  """
    Prepares today's market data in the expected format.

    Args:
        today (date): The current date.
        current_price (float): The current closing price.

    Returns:
        dict: A dictionary representing today's bar data.
    """
  return {
      'Date': pd.Timestamp(today),
      'Open': current_price,
      'High': current_price,
      'Low': current_price,
      'Close': current_price,
      'Volume': 0
  }


def get_today_signal(
    symbol: str,
    strategy_config: dict,
    historical_period_days: int,
    initial_dummy_capital: float,
    db_path: str = data.DATA_DB_NAME) -> tuple[market.Signal, int]:
  """
    Generates a trading signal for a given stock using a specified strategy.
    Automatically fetches today's price from yfinance.

    Args:
        symbol (str): The stock symbol to get the signal for.
        strategy_config (dict): Dictionary containing 'name', 'module', and 'params' for the strategy.
        historical_period_days (int): Number of historical days to fetch for strategy calculation.
        initial_dummy_capital (float): Initial capital for the dummy broker (used for position checks).
        db_path (str): Path to the SQLite database file containing historical data.

    Returns:
        tuple[market.Signal, int]: A tuple containing the suggested action (Signal.BUY, Signal.SELL, Signal.HOLD)
                                 and the quantity. If Signal.HOLD, quantity will be 0.
    """
  logger.info("Generating signal for %s using strategy '%s'...", symbol,
              strategy_config['name'])

  today = date.today()

  current_price = data_utils.download_and_save_latest_data(symbol, db_path)
  if current_price is None:
    logger.error("Could not get current price for %s. Cannot generate signal.",
                 symbol)
    return market.Signal.HOLD, 0

  historical_data = _prepare_historical_data(symbol, today,
                                             historical_period_days, db_path)
  if historical_data is None:
    return market.Signal.HOLD, 0

  dummy_broker = account.AccountSimulator(initial_dummy_capital,
                                          commission_rate=0.0)

  try:
    strategy = _load_strategy(strategy_config, dummy_broker)
  except (ImportError, AttributeError) as e:
    logger.error(
        "Failed to load strategy '%s' from module '%s': %s",
        strategy_config['name'],
        strategy_config['module'],
        e,
    )
    return market.Signal.HOLD, 0

  current_bar_data = _get_current_bar_data(today, current_price)
  current_data_for_combined_df = pd.DataFrame([current_bar_data
                                               ]).set_index('Date')
  combined_data = pd.concat([historical_data, current_data_for_combined_df])

  action, quantity = strategy.on_bar(symbol, current_bar_data, combined_data)

  logger.info("Signal for %s on %s: %s %d shares.", symbol,
              today.strftime('%Y-%m-%d'), action.value, quantity)
  return action, quantity


def main():
  """
    Main function to parse arguments, load config, and generate trading signals for multiple stocks.
    """
  args = _parse_arguments()
  config = _load_config(args.config_path)

  strategy_config = config.get('strategy')
  signal_generation_params = config.get('signal_generation_params', {})

  config_stock_list = config.get('stock_list', [])

  if not strategy_config or 'name' not in strategy_config or 'module' not in strategy_config:
    logger.critical(
        "Config error: 'strategy' section or 'name'/'module' missing.")
    sys.exit(1)
  if not config_stock_list:
    logger.critical("Config error: 'stock_list' section is missing or empty.")
    sys.exit(1)

  target_symbols = [symbol.upper() for symbol in config_stock_list]

  if not target_symbols:
    logger.critical(
        "Config error: No valid stock symbols found in 'stock_list' section.")
    sys.exit(1)

  all_signals = {}
  for symbol in target_symbols:
    action, quantity = get_today_signal(
        symbol=symbol,
        strategy_config=strategy_config,
        historical_period_days=signal_generation_params.get(
            'historical_period_days', 200),
        initial_dummy_capital=signal_generation_params.get(
            'initial_dummy_capital', 100_000.0))
    all_signals[symbol] = {'action': action.value, 'quantity': quantity}

  logger.info("\n--- Today's Trading Signals Summary (%d stocks) ---",
              len(all_signals))
  for symbol, signal_info in all_signals.items():
    logger.info("  %s: Action: %s, Quantity: %d", symbol,
                signal_info['action'], signal_info['quantity'])
  logger.info("------------------------------------------")


if __name__ == '__main__':
  main()
