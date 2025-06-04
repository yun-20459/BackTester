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

  return logger
