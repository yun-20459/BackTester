import torch

from ml_models import architectures
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


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
                 **model_args):
    """
    Loads a PyTorch model's state_dict and initializes the model.
    It dynamically gets the model class from the architectures module.

    Args:
        model_path (str): Path to the saved ML model state_dict.
        model_architecture_name (str): Name of the PyTorch model class (e.g., "DualTransformerClassifier").
        **model_args: Additional arguments to pass to the model's constructor.

    Returns:
        torch.nn.Module: Loaded PyTorch model instance.
    
    Raises:
        ValueError: If the model architecture class is not found in the architectures module.
        Exception: If model state_dict loading fails.
    """
    model_class = None
    try:
      model_class = getattr(architectures, model_architecture_name)
      logger.info(
          f"Dynamically loaded model class: {model_architecture_name} from architectures.py"
      )
    except AttributeError:
      raise ValueError(
          f"Model architecture class '{model_architecture_name}' not found in ml_models/architectures.py. "
          "Please ensure the class is correctly defined there.")

    try:
      model = model_class(**model_args).to(self.device)
      model.load_state_dict(torch.load(model_path, map_location=self.device))
      model.eval()
      logger.info("Model '%s' loaded successfully from %s.",
                  model_architecture_name, model_path)
    except Exception as e:
      logger.critical("Failed to load model state_dict for %s from %s: %s",
                      model_architecture_name, model_path, e)

    return model

  def get_model_metadata(self, model_path: str) -> dict:
    logger.info("Retrieving metadata for model at: %s", model_path)
    return {"model_path": model_path}


model_loader = MLModelLoader()
