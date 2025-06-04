import pandas as pd
import logging
import talib
import numpy as np

from strategy import base
from utils import logger_utils
from common import market

# Get a logger instance for this module
logger = logger_utils.get_logger(__name__)


class BBRSIStrategy(base.BaseStrategy):
  """
    A trading strategy based on Bollinger Bands and RSI.
    Buys when price breaks lower Bollinger Band and RSI is oversold.
    Exits based on stop loss, take profit, or reverse signal (price breaks upper BB and RSI is overbought).
    """

  def __init__(self,
               broker,
               bb_period: int = 20,
               bb_dev: float = 2.0,
               rsi_period: int = 14,
               rsi_oversold: int = 30,
               rsi_overbought: int = 70,
               stop_loss_pct: float = 0.05,
               take_profit_pct: float = 0.15,
               sell_percentage: float = 0.7):
    super().__init__(broker)
    self.bb_period = bb_period
    self.bb_dev = bb_dev
    self.rsi_period = rsi_period
    self.rsi_oversold = rsi_oversold
    self.rsi_overbought = rsi_overbought
    self.stop_loss_pct = stop_loss_pct
    self.take_profit_pct = take_profit_pct
    self.sell_percentage = sell_percentage

    # For multi-symbol backtesting, 'in_position' and 'entry_price' should be per-symbol.
    self.positions_status = {
    }  # {'symbol': {'in_position': bool, 'entry_price': float}}

    logger.info(f"  BBRSI Strategy Parameters:")
    logger.info(f"    BB Period: {self.bb_period}, BB Dev: {self.bb_dev}")
    logger.info(
        f"    RSI Period: {self.rsi_period}, RSI Oversold: {self.rsi_oversold}, RSI Overbought: {self.rsi_overbought}"
    )
    logger.info(f"    Stop Loss Percent: {self.stop_loss_pct * 100:.2f}%")
    logger.info(f"    Take Profit Percent: {self.take_profit_pct * 100:.2f}%")

  def on_bar(self, symbol: str, current_data: dict,
             historical_data: pd.DataFrame) -> tuple[market.Signal, int]:
    """
        Implements the BBRSI strategy's trading logic for each bar and returns a signal.

        Args:
            symbol (str): The stock symbol for the current bar.
            current_data (dict): OHLCV data for the current bar.
            historical_data (pd.DataFrame): Historical OHLCV data up to and including the current bar.

        Returns:
            tuple[market.Signal, int]: A tuple containing the recommended signal (BUY, SELL, HOLD)
                                       and the quantity for the action.
        """
    # Initialize position status for the symbol if not present
    if symbol not in self.positions_status:
      self.positions_status[symbol] = {
          'in_position': False,
          'entry_price': None
      }

    in_position = self.positions_status[symbol]['in_position']
    entry_price = self.positions_status[symbol][
        'entry_price']  # This will be None if not in position

    # Make a copy of historical_data to avoid SettingWithCopyWarning
    data_for_calc = historical_data.copy()

    # Ensure enough data for indicator calculation
    # BBANDS needs at least bb_period bars
    # RSI needs at least rsi_period bars
    required_data_length = max(self.bb_period,
                               self.rsi_period) + 1  # +1 for current bar
    if len(data_for_calc) < required_data_length:
      return market.Signal.HOLD, 0

    # --- Calculate Bollinger Bands and RSI using TA-Lib ---
    close_prices = data_for_calc['Close'].astype(np.float64).values

    # BBANDS returns (upperband, middleband, lowerband)
    upper_band, middle_band, lower_band = talib.BBANDS(
        close_prices,
        timeperiod=self.bb_period,
        nbdevup=self.bb_dev,
        nbdevdn=self.bb_dev,
        matype=0  # 0 for SMA (Simple Moving Average)
    )

    rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)

    # Get latest indicator values
    current_upper_band = upper_band[-1]
    current_lower_band = lower_band[-1]
    current_rsi = rsi[-1]

    current_close_price = current_data['Close']
    current_date = current_data['Date']

    # Ensure latest indicator values are not NaN before proceeding
    if np.isnan(current_upper_band) or np.isnan(
        current_lower_band) or np.isnan(current_rsi):
      return market.Signal.HOLD, 0

    # --- Exit Conditions (Long Position) - Check first to prioritize exits ---
    if in_position:
      current_held_quantity = self.broker.positions.get(symbol,
                                                        {}).get('quantity', 0)

      # Calculate sell quantity based on sell_percentage
      # Ensure it's an integer and at least 1 if a sell is intended
      calculated_sell_quantity = int(current_held_quantity *
                                     self.sell_percentage)
      sell_quantity = max(
          1, calculated_sell_quantity) if calculated_sell_quantity > 0 else 0

      if sell_quantity == 0:  # If calculated quantity is 0, no sell action
        return market.Signal.HOLD, 0

      # 1. Stop Loss Exit
      stop_loss_price = entry_price * (1 - self.stop_loss_pct)
      if current_close_price <= stop_loss_price:
        self.positions_status[symbol][
            'in_position'] = False  # Assume full exit for simplicity in state
        self.positions_status[symbol]['entry_price'] = None
        return market.Signal.SELL, sell_quantity

      # 2. Fixed Percentage Take Profit Exit
      take_profit_price = entry_price * (1 + self.take_profit_pct)
      if current_close_price >= take_profit_price:
        self.positions_status[symbol][
            'in_position'] = False  # Assume full exit for simplicity in state
        self.positions_status[symbol]['entry_price'] = None
        return market.Signal.SELL, sell_quantity

      # 3. Reverse Signal Exit (Price breaks upper BB AND RSI is overbought)
      if current_close_price > current_upper_band and current_rsi >= self.rsi_overbought:
        self.positions_status[symbol][
            'in_position'] = False  # Assume full exit for simplicity in state
        self.positions_status[symbol]['entry_price'] = None
        return market.Signal.SELL, sell_quantity

      return market.Signal.HOLD, 0  # If in position but no sell conditions met

    # --- Entry Conditions (Long Position) - Check only if not in position ---
    if not in_position:
      # 1. Price breaks lower Bollinger Band
      price_breaks_lower_bb = (current_close_price < current_lower_band)

      # 2. RSI enters oversold area
      rsi_oversold_condition = (current_rsi <= self.rsi_oversold)

      if price_breaks_lower_bb and rsi_oversold_condition:
        # Calculate quantity to buy (e.g., use 95% of current cash)
        buy_quantity = int(self.broker.current_cash / current_close_price *
                           0.95)
        if buy_quantity > 0:
          self.positions_status[symbol]['in_position'] = True
          self.positions_status[symbol]['entry_price'] = current_close_price
          return market.Signal.BUY, buy_quantity
        else:
          return market.Signal.HOLD, 0
    return market.Signal.HOLD, 0  # If no action is taken by any of the conditions
