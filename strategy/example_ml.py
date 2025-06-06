from __future__ import annotations

import numpy as np
import pandas as pd
import talib
import torch

from common import market
from strategy import ml_base
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


class MyMLStrategy(ml_base.MLBase):  # Inherit from MLBase
  """
    A concrete Machine Learning based trading strategy implementing specific
    feature engineering and trading decision logic based on MLBase.
    """

  def __init__(self,
               broker,
               model_path: str,
               model_architecture_name: str,
               model_input_dim: int,
               feature_cols: list,
               buy_threshold: float = 0.55,
               sell_threshold: float = 0.45,
               stop_loss_pct: float = 0.05,
               take_profit_pct: float = 0.15,
               **kwargs):
    # Pass ML model related parameters to MLBase's constructor
    super().__init__(broker, model_path, model_architecture_name,
                     model_input_dim, **kwargs)

    # Store parameters specific to this concrete strategy
    self.feature_cols = feature_cols  # Store feature_cols here
    self.buy_threshold = buy_threshold
    self.sell_threshold = sell_threshold
    self.stop_loss_pct = stop_loss_pct
    self.take_profit_pct = take_profit_pct

    logger.info("  MyMLStrategy Specific Parameters:")
    logger.info("    Features: %s", self.feature_cols)
    logger.info("    Buy Threshold: %.2f", self.buy_threshold)
    logger.info("    Sell Threshold: %.2f", self.sell_threshold)
    logger.info("    Stop Loss: %.2f%%", self.stop_loss_pct * 100)
    logger.info("    Take Profit: %.2f%%", self.take_profit_pct * 100)

  def _engineer_features(self,
                         historical_data: pd.DataFrame) -> torch.Tensor | None:
    """
        Implementation of feature engineering for this specific ML strategy.
        Transforms historical_data into a PyTorch tensor ready for the model.
        """
    data_for_features = historical_data.copy()
    data_for_features[['Open', 'High', 'Low', 'Close',
                       'Volume']] = data_for_features[[
                           'Open', 'High', 'Low', 'Close', 'Volume'
                       ]].astype(np.float64)

    try:
      # --- Your specific feature engineering logic goes here ---
      # This logic must match the features used during your model's training.
      # Example features:
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
      logger.warning("Error during feature engineering in MyMLStrategy: %s.",
                     e)
      return None

    features_df = data_for_features[self.feature_cols].dropna()

    if features_df.empty or features_df.shape[1] != self.model_input_dim:
      logger.debug(
          "Not enough or incorrect number of features (%s vs %s) for prediction in MyMLStrategy.",
          features_df.shape[1], self.model_input_dim)
      return None

    # Convert the last row of engineered features to a PyTorch tensor
    # Ensure it's a 2D tensor (batch_size=1, features_count)
    features_tensor = torch.tensor(features_df.iloc[[-1]].values,
                                   dtype=torch.float32)
    return features_tensor

  def _make_trading_decision(
      self, symbol: str, current_data: dict,
      raw_prediction_output: float | None, in_position: bool,
      entry_price: float | None) -> tuple[market.Signal, int]:
    """
        Implementation of the trading decision logic for MyMLStrategy.
        Interprets the raw model prediction and applies trading rules.
        """
    current_close_price = current_data['Close']
    current_date = current_data['Date']

    # Interpret the raw prediction output here
    # Assuming raw_prediction_output is a probability (0.0 to 1.0)
    prediction_proba = raw_prediction_output
    if prediction_proba is None:  # Should not happen if _predict_raw works
      return market.Signal.HOLD, 0

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
            "  %s - %s: SELL Signal - STOP LOSS (MyMLStrategy). Price: %.2f, Entry: %.2f, SL Price: %.2f",
            current_date.strftime('%Y-%m-%d'), symbol, current_close_price,
            entry_price, stop_loss_price)
        return market.Signal.SELL, sell_quantity

      # 2. Fixed Percentage Take Profit Exit
      take_profit_price = entry_price * (1 + self.take_profit_pct)
      if current_close_price >= take_profit_price:
        logger.info(
            "  %s - %s: SELL Signal - TAKE PROFIT (MyMLStrategy). Price: %.2f, Entry: %.2f, TP Price: %.2f",
            current_date.strftime('%Y-%m-%d'), symbol, current_close_price,
            entry_price, take_profit_price)
        return market.Signal.SELL, sell_quantity

      # 3. ML Model recommends Sell/Weakening (e.g., probability below sell_threshold)
      if prediction_proba < self.sell_threshold:
        logger.info(
            "  %s - %s: SELL Signal - ML MODEL (proba: %.2f < %.2f). Price: %.2f",
            current_date.strftime('%Y-%m-%d'), symbol, prediction_proba,
            self.sell_threshold, current_close_price)
        return market.Signal.SELL, sell_quantity

      return market.Signal.HOLD, 0

    # Entry conditions (if not in position)
    if not in_position:
      # ML Model recommends Buy (e.g., probability above buy_threshold)
      if prediction_proba > self.buy_threshold:
        buy_quantity = int(self.broker.current_cash / current_close_price *
                           0.95)
        if buy_quantity > 0:
          logger.info(
              "  %s - %s: BUY Signal - ML MODEL (proba: %.2f > %.2f). Suggesting BUY %s shares.",
              current_date.strftime('%Y-%m-%d'), symbol, prediction_proba,
              self.buy_threshold, buy_quantity)
          return market.Signal.BUY, buy_quantity
        else:
          return market.Signal.HOLD, 0

    return market.Signal.HOLD, 0
