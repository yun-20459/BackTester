import pandas as pd
from utils import logger_utils
from common.market import Signal  # Import the Signal enum

logger = logger_utils.get_logger(__name__)


class BaseStrategy:
  """
    Base class for all trading strategies.
    Defines the interface that strategies must implement.
    """

  def __init__(self, broker):
    """
        Initializes the strategy.

        Args:
            broker: An instance of the simulated account, used for querying account information (e.g., current positions).
                    Note: The strategy will no longer directly execute orders; it will return a suggested action.
        """
    self.broker = broker
    logger.info(f"Strategy '{self.__class__.__name__}' initialized.")

  def on_bar(
      self, symbol: str, current_data: dict, historical_data: pd.DataFrame
  ) -> tuple[Signal, int]:  # Updated return type hint
    """
        Called when new bar data arrives.
        All trading logic for the strategy will be implemented in this method.
        This method should return a suggested action and quantity.

        Args:
            symbol (str): The stock symbol to which the current bar belongs.
            current_data (dict): Data for the current bar, including 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
            historical_data (pd.DataFrame): DataFrame containing all historical bars up to and including the current bar.
                                            Can be used to calculate technical indicators.

        Returns:
            tuple[Signal, int]: A tuple containing the suggested action (Signal.BUY, Signal.SELL, Signal.HOLD)
                                and the quantity. If Signal.HOLD, quantity should be 0.
        """
    # This is an abstract method, concrete strategies need to override it.
    return Signal.HOLD, 0  # Default to hold
