import os

import torch
from torch import nn

from utils import logger_utils

logger = logger_utils.get_logger(__name__)


# --- Define your PyTorch model architectures here ---
# It's best practice to define your model architectures directly in your project
# or in a dedicated 'models.py' file within ml_models, so they can be imported.
# For this example, we'll define SimpleBinaryClassifier here for simplicity.
# In a real project, consider: from .models import SimpleBinaryClassifier
class SimpleBinaryClassifier(nn.Module):

  def __init__(self, input_dim):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, 32)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(32, 1)

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


# --- End of model architecture definitions ---


class MLModelLoader:
  """
    Handles loading pre-trained ML models and providing their metadata.
    Supports only PyTorch models saved as state_dict.
    """

  def __init__(self):
    self.device = self._get_device()
    logger.info("MLModelLoader initialized. Using device: %s", self.device)

  def _get_device(self):
    """Detects and returns the appropriate PyTorch device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
      return torch.device("cuda")
    if torch.backends.mps.is_available():
      return torch.device("mps")
    return torch.device("cpu")

  def load_model(self, model_path: str, model_architecture_name: str,
                 model_input_dim: int):
    """
        Loads a pre-trained PyTorch model's state_dict into its architecture.

        Args:
            model_path (str): The file path to the saved model's state_dict (.pt or .pth).
            model_architecture_name (str): The name of the model class (e.g., 'SimpleBinaryClassifier').
            model_input_dim (int): The input dimension required to instantiate the model architecture.

        Returns:
            torch.nn.Module: The loaded and initialized PyTorch model.

        Raises:
            FileNotFoundError: If the model state_dict file does not exist.
            ValueError: If the model architecture name is not found.
            RuntimeError: For other loading errors.
        """
    if not os.path.exists(model_path):
      logger.error("Model state_dict file not found at: %s", model_path)
      raise FileNotFoundError(
          f"Model state_dict file not found at: {model_path}")

    try:
      # Dynamically get the model architecture class
      model_class = globals().get(model_architecture_name)
      if model_class is None:
        raise ValueError(
            f"Model architecture class '{model_architecture_name}' not found.")

      # Instantiate the model architecture
      model = model_class(input_dim=model_input_dim)

      # Load the state_dict
      model.load_state_dict(torch.load(model_path, map_location=self.device))

      model.eval()  # Set model to evaluation mode
      model.to(self.device)  # Move model to detected device
      logger.info(
          "Successfully loaded PyTorch model state_dict from: %s to device: %s",
          model_path, self.device)

      return model
    except (FileNotFoundError, ValueError) as e:
      raise e
    except Exception as e:
      logger.critical(
          "Error loading PyTorch model from %s (architecture: %s): %s",
          model_path, model_architecture_name, e)
      raise RuntimeError(
          f"Failed to load PyTorch model from {model_path}: {e}") from e

  # The get_model_metadata method remains largely conceptual for simplicity.
  def get_model_metadata(self, model_path: str) -> dict:
    logger.info("Retrieving metadata for model at: %s", model_path)
    return {"model_path": model_path}


# Create a singleton instance to avoid re-initializing
model_loader = MLModelLoader()
