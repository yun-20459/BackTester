import pandas as pd
import talib
import numpy as np

from strategy import base
from utils import logger_utils
from common import market

logger = logger_utils.get_logger(__name__)


class SimpleMovingAverageStrategy(base.BaseStrategy):
  """
    An example strategy based on Simple Moving Average.
    Buys when the short-term moving average crosses above the long-term moving average, and sells when it crosses below.
    Now uses TA-Lib to calculate moving averages.
    """

  def __init__(self, broker, short_window: int = 20, long_window: int = 50):
    super().__init__(broker)
    self.short_window = short_window
    self.long_window = long_window
    logger.info(
        f"  SMA Strategy Parameters: Short Window={short_window}, Long Window={long_window}"
    )

  def on_bar(
      self, symbol: str, current_data: dict, historical_data: pd.DataFrame
  ) -> tuple[market.Signal, int]:  # Updated return type hint
    """
        Implements the trading logic for the SMA strategy.
        Returns the suggested action and quantity.
        """
    data_for_calc = historical_data.copy()

    if len(data_for_calc) < self.long_window:
      return market.Signal.HOLD, 0

    data_for_calc['SMA_Short'] = talib.SMA(data_for_calc['Close'].values,
                                           timeperiod=self.short_window)
    data_for_calc['SMA_Long'] = talib.SMA(data_for_calc['Close'].values,
                                          timeperiod=self.long_window)

    if np.isnan(data_for_calc['SMA_Short'].iloc[-1]) or np.isnan(
        data_for_calc['SMA_Long'].iloc[-1]):
      return market.Signal.HOLD, 0

    latest_short_sma = data_for_calc['SMA_Short'].iloc[-1]
    latest_long_sma = data_for_calc['SMA_Long'].iloc[-1]

    current_close_price = current_data['Close']
    current_date = current_data['Date']

    current_position_quantity = self.broker.positions.get(symbol, {}).get(
        'quantity', 0)
    in_position = current_position_quantity > 0

    # Buy signal: Short-term SMA crosses above long-term SMA (Golden Cross)
    if latest_short_sma > latest_long_sma and not in_position:
      if len(data_for_calc) > self.long_window + 1:
        prev_short_sma = data_for_calc['SMA_Short'].iloc[-2]
        prev_long_sma = data_for_calc['SMA_Long'].iloc[-2]

        if not np.isnan(prev_short_sma) and not np.isnan(prev_long_sma) and (
            prev_short_sma <= prev_long_sma):
          buy_quantity = int(self.broker.current_cash / current_close_price *
                             0.95)
          if buy_quantity > 0:
            logger.info(
                f"  {current_date.strftime('%Y-%m-%d')} - {symbol}: Golden Cross, suggesting BUY {buy_quantity} shares."
            )
            return market.Signal.BUY, buy_quantity

    # Sell signal: Short-term SMA crosses below long-term SMA (Death Cross)
    elif latest_short_sma < latest_long_sma and in_position:
      if len(data_for_calc) > self.long_window + 1:
        prev_short_sma = data_for_calc['SMA_Short'].iloc[-2]
        prev_long_sma = data_for_calc['SMA_Long'].iloc[-2]

        if not np.isnan(prev_short_sma) and not np.isnan(prev_long_sma) and (
            prev_short_sma >= prev_long_sma):
          sell_quantity = current_position_quantity
          if sell_quantity > 0:
            logger.info(
                f"  {current_date.strftime('%Y-%m-%d')} - {symbol}: Death Cross, suggesting SELL {sell_quantity} shares."
            )
            return market.Signal.SELL, sell_quantity

    return market.Signal.HOLD, 0
