# strategy/my_ml_strategy.py

import pandas as pd
import logging
import numpy as np
import talib  # For feature engineering examples

from strategy.ml_base import MLBase  # Import the abstract MLBase
from utils import logger_utils
from common import market

logger = logger_utils.get_logger(__name__)


class MyMLStrategy(MLBase):  # Inherit from MLBase
  """
    A concrete Machine Learning based trading strategy implementing specific
    feature engineering and trading decision logic.
    """

  def __init__(
      self,
      broker,
      model_path: str,
      model_architecture_name: str,
      model_input_dim: int,
      feature_cols: list,
      # Trading decision parameters are now part of this concrete strategy
      buy_threshold: float = 0.55,
      sell_threshold: float = 0.45,
      stop_loss_pct: float = 0.05,
      take_profit_pct: float = 0.15,
      **kwargs):
    # Pass ML model related parameters to MLBase
    super().__init__(broker, model_path, model_architecture_name,
                     model_input_dim, feature_cols, **kwargs)

    # Store trading decision parameters here
    self.buy_threshold = buy_threshold
    self.sell_threshold = sell_threshold
    self.stop_loss_pct = stop_loss_pct
    self.take_profit_pct = take_profit_pct

    logger.info(f"  MyMLStrategy Specific Parameters:")
    logger.info(
        f"    Buy Threshold: {self.buy_threshold}, Sell Threshold: {self.sell_threshold}"
    )
    logger.info(
        f"    Stop Loss: {self.stop_loss_pct * 100:.2f}%, Take Profit: {self.take_profit_pct * 100:.2f}%"
    )

  def _engineer_features(self,
                         historical_data: pd.DataFrame) -> pd.DataFrame | None:
    """
        Implementation of feature engineering for this specific ML strategy.
        This must match the features used during the model's training.
        """
    data_for_features = historical_data.copy()
    data_for_features[['Open', 'High', 'Low', 'Close',
                       'Volume']] = data_for_features[[
                           'Open', 'High', 'Low', 'Close', 'Volume'
                       ]].astype(np.float64)

    try:
      # --- Your specific feature engineering logic goes here ---
      # Example (same as previous MLBase for now, but you can customize)
      data_for_features['SMA_10'] = talib.SMA(
          data_for_features['Close'].values, timeperiod=10)
      data_for_features['SMA_20'] = talib.SMA(
          data_for_features['Close'].values, timeperiod=20)
      data_for_features['RSI_14'] = talib.RSI(
          data_for_features['Close'].values, timeperiod=14)
      macd, macdsignal, macdhist = talib.MACD(
          data_for_features['Close'].values,
          fastperiod=12,
          slowperiod=26,
          signalperiod=9)
      data_for_features['MACD'] = macd
      data_for_features['MACD_Signal'] = macdsignal
      data_for_features['Daily_Return'] = data_for_features[
          'Close'].pct_change()
      data_for_features['Prev_Close'] = data_for_features['Close'].shift(1)
      # --- End of example feature engineering ---

    except Exception as e:
      logger.warning(f"Error during feature engineering in MyMLStrategy: {e}.")
      return None

    features_df = data_for_features[self.feature_cols].dropna()

    if features_df.empty or features_df.shape[1] != self.model_input_dim:
      logger.debug(
          f"Not enough or incorrect number of features ({features_df.shape[1]} vs {self.model_input_dim}) for prediction in MyMLStrategy."
      )
      return None

    return features_df

  def _make_trading_decision(
      self, symbol: str, current_data: dict, prediction_output: float | None,
      in_position: bool,
      entry_price: float | None) -> tuple[market.Signal, int]:
    """
        Implementation of the trading decision logic for MyMLStrategy.
        """
    current_close_price = current_data['Close']
    current_date = current_data['Date']

    # --- Trading Logic based on ML Prediction and Risk Management ---
    # Exit conditions (prioritize exits)
    if in_position:
      current_held_quantity = self.broker.positions.get(symbol,
                                                        {}).get('quantity', 0)
      sell_quantity = current_held_quantity

      if sell_quantity == 0:
        return market.Signal.HOLD, 0

      # 1. Stop Loss Exit
      stop_loss_price = entry_price * (1 - self.stop_loss_pct)
      if current_close_price <= stop_loss_price:
        logger.info(
            f"  {current_date.strftime('%Y-%m-%d')} - {symbol}: SELL Signal - STOP LOSS (MyMLStrategy). Price: {current_close_price:.2f}, Entry: {entry_price:.2f}, SL Price: {stop_loss_price:.2f}"
        )
        return market.Signal.SELL, sell_quantity

      # 2. Fixed Percentage Take Profit Exit
      take_profit_price = entry_price * (1 + self.take_profit_pct)
      if current_close_price >= take_profit_price:
        logger.info(
            f"  {current_date.strftime('%Y-%m-%d')} - {symbol}: SELL Signal - TAKE PROFIT (MyMLStrategy). Price: {current_close_price:.2f}, Entry: {entry_price:.2f}, TP Price: {take_profit_price:.2f}"
        )
        return market.Signal.SELL, sell_quantity

      # 3. ML Model recommends Sell/Weakening (e.g., probability below sell_threshold)
      if prediction_output is not None and prediction_output < self.sell_threshold:
        logger.info(
            f"  {current_date.strftime('%Y-%m-%d')} - {symbol}: SELL Signal - ML MODEL (proba: {prediction_output:.2f} < {self.sell_threshold}). Price: {current_close_price:.2f}"
        )
        return market.Signal.SELL, sell_quantity

      return market.Signal.HOLD, 0

    # Entry conditions (if not in position)
    if not in_position:
      # ML Model recommends Buy (e.g., probability above buy_threshold)
      if prediction_output is not None and prediction_output > self.buy_threshold:
        buy_quantity = int(self.broker.current_cash / current_close_price *
                           0.95)
        if buy_quantity > 0:
          logger.info(
              f"  {current_date.strftime('%Y-%m-%d')} - {symbol}: BUY Signal - ML MODEL (proba: {prediction_output:.2f} > {self.buy_threshold}). Suggesting BUY {buy_quantity} shares."
          )
          return market.Signal.BUY, buy_quantity
        else:
          return market.Signal.HOLD, 0

    return market.Signal.HOLD, 0
