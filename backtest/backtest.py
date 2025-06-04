import matplotlib.pyplot as plt
import argparse
import pandas as pd
from datetime import datetime, date
import logging
import json  # Import json for config file parsing
import importlib  # Import importlib for dynamic module loading

from utils import data_utils
from common import data
from core import engine as backtesting_engine
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


def _parse_arguments():
  """
  Parses command-line arguments.

  Returns:
      argparse.Namespace: Parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description="Run a stock backtest based on a config file.")
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
    logger.info(f"Configuration loaded from: {config_path}")
    return config
  except FileNotFoundError:
    logger.critical(f"Config file not found at: {config_path}")
    exit(1)
  except json.JSONDecodeError as e:
    logger.critical(f"Error parsing config file {config_path}: {e}")
    exit(1)
  except Exception as e:
    logger.critical(f"An unexpected error occurred while loading config: {e}")
    exit(1)


def main():
  args = _parse_arguments()
  config = _load_config(args.config_path)

  # Extract backtest parameters from config
  backtest_params = config.get('backtest_params', {})
  strategy_config = config.get('strategy',
                               {})  # Get strategy config from the same JSON

  start_date_str = backtest_params.get('start_date')
  end_date_str = backtest_params.get('end_date')
  symbols = backtest_params.get('symbols')
  initial_capital = backtest_params.get('initial_capital')
  commission_rate = backtest_params.get('commission_rate')

  # Validate essential backtest parameters
  if not all([
      start_date_str, end_date_str, symbols, initial_capital is not None,
      commission_rate is not None
  ]):
    logger.critical(
        "Config error: Missing essential parameters in 'backtest_params' (start_date, end_date, symbols, initial_capital, commission_rate)."
    )
    exit(1)
  if not strategy_config or 'name' not in strategy_config or 'module' not in strategy_config:
    logger.critical(
        "Config error: 'strategy' section or 'name'/'module' missing.")
    exit(1)

  # Convert date strings to date objects
  try:
    START_DATE = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    END_DATE = datetime.strptime(end_date_str, '%Y-%m-%d').date()
  except ValueError as e:
    logger.critical(
        f"Config error: Invalid date format in 'backtest_params' (expected YYYY-MM-DD): {e}"
    )
    exit(1)

  # 1. Download missing stock data
  logger.info("Starting data download/check...")
  data_utils.download_stock_data(symbols, START_DATE, END_DATE)
  logger.info("Data download/check completed.")

  # 2. Fetch all required stock data from SQLite
  all_stock_data = data_utils.fetch_multiple_stock_data_from_sqlite(
      data.DATA_DB_NAME, symbols, START_DATE.strftime('%Y-%m-%d'),
      END_DATE.strftime('%Y-%m-%d'))

  if all_stock_data is None:
    logger.critical(
        "No stock data was successfully retrieved for backtest, program will exit."
    )
    exit(1)

  # 3. Initialize backtesting engine
  engine_instance = backtesting_engine.BacktestingEngine(
      initial_capital=initial_capital, commission_rate=commission_rate)
  engine_instance.set_data(all_stock_data)

  # 4. Dynamically load and set the strategy
  strategy_name = strategy_config['name']
  strategy_module_name = strategy_config['module']
  strategy_params = strategy_config.get('params', {})

  try:
    strategy_module = importlib.import_module(
        f"strategy.{strategy_module_name}")
    strategy_class = getattr(strategy_module, strategy_name)
  except (ImportError, AttributeError) as e:
    logger.critical(
        f"Failed to load strategy '{strategy_name}' from module '{strategy_module_name}': {e}"
    )
    exit(1)

  engine_instance.set_strategy(strategy_class, **strategy_params)

  # 5. Run the backtest
  logger.info("\n--- Starting Backtest ---")
  engine_instance.run()
  logger.info("\n--- Backtest Finished ---")


if __name__ == '__main__':
  main()
