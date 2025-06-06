import pandas as pd
import numpy as np
import torch
import abc

from strategy.base import BaseStrategy
from utils import logger_utils
from common import market
from ml_models.model_loader import model_loader

logger = logger_utils.get_logger(__name__)


class MLBase(BaseStrategy, abc.ABC):
  """
    An abstract base class for Machine Learning based trading strategies.
    It handles loading the pre-trained PyTorch model and performing inference.
    Subclasses must implement:
    - _engineer_features(historical_data): for specific feature engineering.
    - _make_trading_decision(...): for specific buy/sell/hold logic based on model prediction.
    """

  def __init__(
      self,
      broker,
      model_path: str,  # Path to the saved ML model state_dict
      model_architecture_name: str,  # Name of the PyTorch model class
      model_input_dim: int,  # Input dimension for the PyTorch model
      feature_cols: list,  # List of feature columns expected by the model
      **kwargs  # For any other optional base parameters
  ):
    super().__init__(broker)
    self.model_path = model_path
    self.model_architecture_name = model_architecture_name
    self.model_input_dim = model_input_dim
    self.feature_cols = feature_cols

    self.device = model_loader.device

    try:
      self.model = model_loader.load_model(self.model_path,
                                           self.model_architecture_name,
                                           self.model_input_dim)
    except Exception as e:
      logger.critical(
          f"MLBase: Failed to initialize due to model loading error: {e}")
      raise  # Re-raise to stop the application if model can't be loaded

    self.positions_status = {
    }  # {'symbol': {'in_position': bool, 'entry_price': float}}

    logger.info(f"  MLBase Strategy Base Initialized:")
    logger.info(f"    Model Path: {self.model_path}")
    logger.info(f"    Model Architecture: {self.model_architecture_name}")
    logger.info(f"    Model Input Dim: {self.model_input_dim}")
    logger.info(f"    Expected Features: {self.feature_cols}")

  @abc.abstractmethod
  def _engineer_features(self,
                         historical_data: pd.DataFrame) -> pd.DataFrame | None:
    """
        Abstract method: Subclasses must implement this for specific feature engineering.
        Ensures no look-ahead bias.
        """
    pass  # To be implemented by concrete subclasses

  def _make_prediction(self, features_df: pd.DataFrame) -> float | None:
    """
        Performs inference using the loaded PyTorch model.
        Returns the prediction probability (or raw output), or None if prediction fails.
        """
    if features_df is None or features_df.empty:
      logger.debug("No features provided for prediction.")
      return None

    latest_features_row = features_df.iloc[[-1]]

    prediction_proba = None
    try:
      input_tensor = torch.tensor(latest_features_row.values,
                                  dtype=torch.float32).to(self.device)

      with torch.no_grad():
        self.model.eval()  # Ensure model is in eval mode before inference
        output = self.model(input_tensor)

        # --- Adjust this part based on your specific model's output ---
        # Example: Binary classification with sigmoid output
        if output.dim() > 1 and output.shape[1] == 1:
          prediction_proba = torch.sigmoid(output).item()
        elif output.dim(
        ) == 1:  # E.g., single output neuron with sigmoid already applied
          prediction_proba = output.item()
        elif output.dim() > 1 and output.shape[
            1] == 2:  # E.g., logits for 2 classes (apply softmax)
          prediction_proba = torch.softmax(output,
                                           dim=1)[:,
                                                  1].item()  # Prob of class 1
        else:
          logger.warning(
              f"Unexpected PyTorch model output shape: {output.shape}. Returning None."
          )
          return None

      return prediction_proba

    except Exception as e:
      logger.error(f"Error during ML model prediction: {e}. Returning None.")
      return None

  @abc.abstractmethod
  def _make_trading_decision(
      self, symbol: str, current_data: dict, prediction_output: float | None,
      in_position: bool,
      entry_price: float | None) -> tuple[market.Signal, int]:
    """
        Abstract method: Subclasses must implement this for specific trading logic.
        Based on the model's prediction and current state, decides BUY, SELL, or HOLD.
        """
    pass  # To be implemented by concrete subclasses

  def on_bar(self, symbol: str, current_data: dict,
             historical_data: pd.DataFrame) -> tuple[market.Signal, int]:
    """
        The main method called by the backtesting engine for each bar.
        Orchestrates feature engineering, prediction, and delegates to trading decision.
        """
    if symbol not in self.positions_status:
      self.positions_status[symbol] = {
          'in_position': False,
          'entry_price': None
      }

    in_position = self.positions_status[symbol]['in_position']
    entry_price = self.positions_status[symbol]['entry_price']

    # 1. Feature Engineering (implemented by subclass)
    features_df = self._engineer_features(historical_data)

    if features_df is None:  # Features could not be engineered
      logger.debug(
          f"  {current_data['Date'].strftime('%Y-%m-%d')} - {symbol}: Features missing. Holding."
      )
      return market.Signal.HOLD, 0

    # 2. Model Inference
    prediction_output = self._make_prediction(features_df)

    if prediction_output is None:  # Prediction failed
      logger.debug(
          f"  {current_data['Date'].strftime('%Y-%m-%d')} - {symbol}: Prediction failed. Holding."
      )
      return market.Signal.HOLD, 0

    # 3. Make Trading Decision (implemented by subclass)
    action, quantity = self._make_trading_decision(
        symbol=symbol,
        current_data=current_data,
        prediction_output=prediction_output,
        in_position=in_position,
        entry_price=entry_price)

    # 4. Update Strategy's Internal Position Status based on intended action
    # This is a critical step for state management *within* the strategy.
    if action == market.Signal.BUY and quantity > 0:
      self.positions_status[symbol]['in_position'] = True
      self.positions_status[symbol]['entry_price'] = current_data[
          'Close']  # Record entry price
    elif action == market.Signal.SELL and quantity > 0:
      # If sell_percentage is less than 1.0, you'd need more complex state management here
      # For now, assuming any sell signal means closing the position for this strategy's state.
      self.positions_status[symbol]['in_position'] = False
      self.positions_status[symbol]['entry_price'] = None

    logger.debug(
        f"  {current_data['Date'].strftime('%Y-%m-%d')} - {symbol}: Predicted P: {prediction_output:.2f} -> Action: {action.value}, Qty: {quantity}"
    )
    return action, quantity
