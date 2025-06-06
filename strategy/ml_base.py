from __future__ import annotations

import abc

import pandas as pd
import torch

from common import market
from ml_models.model_loader import model_loader
from strategy.base import BaseStrategy
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


class MLBase(BaseStrategy, abc.ABC):
  """
    An abstract base class for Machine Learning based trading strategies.
    It handles loading the pre-trained PyTorch model and performing raw inference.
    Subclasses must implement:
    - _engineer_features(historical_data): for specific feature engineering.
    - _make_trading_decision(...): for specific buy/sell/hold logic based on raw model prediction.
    """

  def __init__(
      self,
      broker,
      model_path: str,
      model_architecture_name: str,
      model_input_dim: int,
  ):
    super().__init__(broker)
    self.model_path = model_path
    self.model_architecture_name = model_architecture_name
    self.model_input_dim = model_input_dim

    self.device = model_loader.device

    try:
      self.model = model_loader.load_model(self.model_path,
                                           self.model_architecture_name,
                                           self.model_input_dim)
    except Exception as e:
      logger.critical(
          "MLBase: Failed to initialize due to model loading error: %s", e)
      raise

    self.positions_status = {
    }  # {'symbol': {'in_position': bool, 'entry_price': float}}

    logger.info("  MLBase Strategy Base Initialized:")
    logger.info("    Model Path: %s", self.model_path)
    logger.info("    Model Architecture: %s", self.model_architecture_name)
    logger.info("    Model Input Dim: %s", self.model_input_dim)

  @abc.abstractmethod
  def _engineer_features(self,
                         historical_data: pd.DataFrame) -> torch.Tensor | None:
    """
        Abstract method: Subclasses must implement this for specific feature engineering.
        This method should transform historical_data into a PyTorch tensor (e.g., torch.float32)
        ready for the model's prediction.
        Ensures no look-ahead bias.
        """

  def _predict_raw(self, features_tensor: torch.Tensor) -> float | None:
    """
        Performs raw inference using the loaded PyTorch model.
        Returns the raw prediction output (e.g., a single probability, or a logit).
        """
    if features_tensor is None or features_tensor.numel(
    ) == 0:  # Check if tensor is empty
      logger.debug("No features tensor provided for prediction.")
      return None

    input_tensor_on_device = features_tensor.to(self.device)

    prediction_output = None
    try:
      with torch.no_grad():
        self.model.eval()  # Ensure model is in eval mode before inference
        output = self.model(input_tensor_on_device)

        # --- This part is now more generic for raw model output ---
        # It returns the raw output, leaving interpretation to the subclass.
        if output.dim() == 0:  # Scalar output (e.g., a single probability)
          prediction_output = output.item()
        elif output.dim() == 1 and output.numel(
        ) == 1:  # Tensor with one element
          prediction_output = output.item()
        elif output.dim(
        ) > 1 and output.shape[1] == 1:  # Output shape like [batch_size, 1]
          prediction_output = output.squeeze().item()  # Squeeze and get scalar
        elif output.dim() > 1 and output.shape[
            1] == 2:  # Output shape like [batch_size, 2] (e.g., logits for 2 classes)
          # Return the logits or probabilities for both classes, subclass decides
          # For simplicity, returning proba of class 1 as before, but can be customized
          prediction_output = torch.softmax(output, dim=1)[:, 1].item()
        else:
          logger.warning(
              "Unexpected PyTorch model output shape: %s. Returning None.",
              output.shape)
          return None

      return prediction_output

    except Exception as e:
      logger.error("Error during ML model raw prediction: %s. Returning None.",
                   e)
      return None

  @abc.abstractmethod
  def _make_trading_decision(
      self, symbol: str, current_data: dict,
      raw_prediction_output: float | None, in_position: bool,
      entry_price: float | None) -> tuple[market.Signal, int]:
    """
        Abstract method: Subclasses must implement this for specific trading logic.
        Based on the raw model prediction and current state, decides BUY, SELL, or HOLD.
        """

  def on_bar(self, symbol: str, current_data: dict,
             historical_data: pd.DataFrame) -> tuple[market.Signal, int]:
    """
        The main method called by the backtesting engine for each bar.
        Orchestrates feature engineering, raw prediction, and delegates to trading decision.
        """
    if symbol not in self.positions_status:
      self.positions_status[symbol] = {
          'in_position': False,
          'entry_price': None
      }

    in_position = self.positions_status[symbol]['in_position']
    entry_price = self.positions_status[symbol]['entry_price']

    # 1. Feature Engineering (implemented by subclass)
    # Should return a PyTorch tensor ready for the model
    features_tensor = self._engineer_features(historical_data)

    if features_tensor is None:
      logger.debug(
          "  %s - %s: Features missing or could not be engineered. Holding.",
          current_data['Date'].strftime('%Y-%m-%d'), symbol)
      return market.Signal.HOLD, 0

    # 2. Raw Model Prediction
    raw_prediction_output = self._predict_raw(features_tensor)

    if raw_prediction_output is None:
      logger.debug("  %s - %s: Raw prediction failed. Holding.",
                   current_data['Date'].strftime('%Y-%m-%d'), symbol)
      return market.Signal.HOLD, 0

    # 3. Make Trading Decision (implemented by subclass)
    action, quantity = self._make_trading_decision(
        symbol=symbol,
        current_data=current_data,
        raw_prediction_output=raw_prediction_output,
        in_position=in_position,
        entry_price=entry_price)

    # 4. Update Strategy's Internal Position Status based on intended action
    if action == market.Signal.BUY and quantity > 0:
      self.positions_status[symbol]['in_position'] = True
      self.positions_status[symbol]['entry_price'] = current_data['Close']
    elif action == market.Signal.SELL and quantity > 0:
      self.positions_status[symbol]['in_position'] = False
      self.positions_status[symbol]['entry_price'] = None

    logger.debug("  %s - %s: Raw P: %.2f -> Action: %s, Qty: %s",
                 current_data['Date'].strftime('%Y-%m-%d'), symbol,
                 raw_prediction_output, action.value, quantity)
    return action, quantity
