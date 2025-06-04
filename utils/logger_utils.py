import logging


def get_logger(name: str):
  """
    Returns a logger instance for the given module name.
    """
  logger = logging.getLogger(name)
  # Ensure the logger does not add handlers multiple times if called repeatedly
  if not logger.handlers:
    # Default to INFO level, can be overridden by root logger configuration in main.py
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.FileHandler('backtest/backtest.log', 'w')

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)
  return logger
