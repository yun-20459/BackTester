import numpy as np
import pandas as pd
import talib

from common import market
from strategy import base
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


class ADXStrategy(base.BaseStrategy):
  """
    A trading strategy based on ADX, +DI, and -DI indicators for trend following,
    with multiple exit conditions for risk management.
    """

  def __init__(
      self,
      broker,
      adx_period: int = 14,
      adx_rising_lookback: int = 5,
      adx_trend_threshold: int = 25,
      stop_loss_pct: float = 0.05,
      take_profit_pct: float = 0.15,
      adx_exit_threshold: int = 20,
      sell_percentage: float = 0.2,
  ):
    super().__init__(broker)
    self.adx_period = adx_period
    self.adx_rising_lookback = adx_rising_lookback
    self.adx_trend_threshold = adx_trend_threshold
    self.stop_loss_pct = stop_loss_pct
    self.take_profit_pct = take_profit_pct
    self.adx_exit_threshold = adx_exit_threshold
    self.sell_percentage = sell_percentage

    self.positions_status = {}

    logger.info(f"  ADX Strategy Parameters:")
    logger.info(f"    ADX Period: {self.adx_period}")
    logger.info(f"    ADX Rising Lookback: {self.adx_rising_lookback}")
    logger.info(f"    ADX Trend Threshold: {self.adx_trend_threshold}")
    logger.info(f"    Stop Loss Percent: {self.stop_loss_pct * 100:.2f}%")
    logger.info(f"    Take Profit Percent: {self.take_profit_pct * 100:.2f}%")
    logger.info(f"    ADX Exit Threshold: {self.adx_exit_threshold}")
    logger.info(f"    Sell Percentage: {self.sell_percentage * 100:.2f}%")

  def on_bar(self, symbol: str, current_data: dict,
             historical_data: pd.DataFrame) -> tuple[market.Signal, int]:
    """
        Implements the ADX strategy's trading logic for each bar and returns a signal.

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

    # Ensure enough data for ADX calculation
    if len(data_for_calc) < max(self.adx_period * 2 + 1, self.adx_period +
                                self.adx_rising_lookback + 1):
      return market.Signal.HOLD, 0  # Use market.Signal

    # --- Calculate ADX, +DI, -DI using TA-Lib ---
    high = data_for_calc['High'].values
    low = data_for_calc['Low'].values
    close = data_for_calc['Close'].values

    adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

    # Ensure latest values are not NaN before proceeding
    if np.isnan(adx[-1]) or np.isnan(plus_di[-1]) or np.isnan(minus_di[-1]):
      return market.Signal.HOLD, 0  # Use market.Signal

    current_adx = adx[-1]
    current_plus_di = plus_di[-1]
    current_minus_di = minus_di[-1]

    current_close_price = current_data['Close']

    # Get previous day's indicator values for cross-over detection
    # Ensure there's a previous day's data for comparison
    if len(adx) < 2 or np.isnan(adx[-2]) or np.isnan(plus_di[-2]) or np.isnan(
        minus_di[-2]):
      prev_adx = np.nan
      prev_plus_di = np.nan
      prev_minus_di = np.nan
    else:
      prev_adx = adx[-2]
      prev_plus_di = plus_di[-2]
      prev_minus_di = minus_di[-2]

    # --- Exit Conditions (Long Position) - Check first to prioritize exits ---
    if in_position:
      # 1. Stop Loss Exit
      stop_loss_price = entry_price * (1 - self.stop_loss_pct)
      if current_close_price <= stop_loss_price:
        sell_quantity = self.broker.positions.get(symbol,
                                                  {}).get('quantity', 0)
        if sell_quantity > 0:
          self.positions_status[symbol]['in_position'] = False
          self.positions_status[symbol]['entry_price'] = None
          return market.Signal.SELL, sell_quantity  # Use market.Signal
        else:  # No position to sell, but condition met, so hold
          return market.Signal.HOLD, 0  # Use market.Signal

      # 2. Take Profit Exit (Fixed Percentage OR ADX Weakening)
      take_profit_price = entry_price * (1 + self.take_profit_pct)
      adx_weakening = (not np.isnan(prev_adx) and current_adx
                       < prev_adx) or (current_adx < self.adx_exit_threshold)

      if current_close_price >= take_profit_price or adx_weakening:
        sell_quantity = self.broker.positions.get(symbol,
                                                  {}).get('quantity', 0)
        if sell_quantity > 0:
          exit_reason = ""
          if current_close_price >= take_profit_price:
            exit_reason += "FIXED TAKE PROFIT"
          if adx_weakening:
            if exit_reason: exit_reason += " OR "
            exit_reason += f"ADX WEAKENING (Current ADX: {current_adx:.2f})"

          self.positions_status[symbol]['in_position'] = False
          self.positions_status[symbol]['entry_price'] = None
          return market.Signal.SELL, sell_quantity  # Use market.Signal
        else:  # No position to sell, but condition met, so hold
          return market.Signal.HOLD, 0  # Use market.Signal

      # 3. Trend Reversal Exit: -DI crosses above +DI
      minus_di_cross_above_plus_di = (current_minus_di > current_plus_di) and \
                                     (not np.isnan(prev_minus_di) and not np.isnan(prev_plus_di) and prev_minus_di <= prev_plus_di)

      if minus_di_cross_above_plus_di:
        sell_quantity = self.broker.positions.get(symbol,
                                                  {}).get('quantity', 0)
        if sell_quantity > 0:
          self.positions_status[symbol]['in_position'] = False
          self.positions_status[symbol]['entry_price'] = None
          return market.Signal.SELL, sell_quantity  # Use market.Signal
        else:  # No position to sell, but condition met, so hold
          return market.Signal.HOLD, 0  # Use market.Signal

    # --- Entry Conditions (Long Position) - Check only if not in position ---
    if not in_position:
      # 1. Trend Strength Confirmation: ADX current value > average of ADX over adx_rising_lookback periods
      #    And ADX current value > adx_trend_threshold

      # Ensure enough data for ADX rising average calculation
      if len(adx) < self.adx_rising_lookback + 1:
        return market.Signal.HOLD, 0  # Use market.Signal

      # Calculate average of ADX over the lookback period (excluding current day)
      adx_lookback_avg = np.mean(
          adx[max(0,
                  len(adx) - self.adx_rising_lookback - 1):-1])

      # Check if ADX is rising and above threshold
      adx_is_rising = (current_adx > adx_lookback_avg)
      adx_is_strong = (current_adx > self.adx_trend_threshold)

      # 2. Trend Direction Confirmation: +DI crosses above -DI
      plus_di_cross_above_minus_di = (current_plus_di > current_minus_di) and \
                                     (not np.isnan(prev_plus_di) and not np.isnan(prev_minus_di) and prev_plus_di <= prev_minus_di)

      if adx_is_rising and adx_is_strong and plus_di_cross_above_minus_di:
        # Calculate quantity to buy (e.g., use 95% of current cash)
        buy_quantity = int(self.broker.current_cash / current_close_price *
                           self.sell_percentage)  # Use 95% of available cash
        if buy_quantity > 0:
          self.positions_status[symbol]['in_position'] = True
          self.positions_status[symbol]['entry_price'] = current_close_price
          return market.Signal.BUY, buy_quantity  # Use market.Signal
        else:  # Cannot buy (e.g., not enough cash for 1 share), so hold
          return market.Signal.HOLD, 0  # Use market.Signal

    return market.Signal.HOLD, 0  # If no action is taken by any of the conditions
