import torch
import torch.nn as nn
import os


# 1. Define a simple PyTorch model architecture
# This model takes 6 features (matching example feature_cols) and outputs 1 logit (for binary classification)
class SimpleBinaryClassifier(nn.Module):

  def __init__(self, input_dim):
    super(SimpleBinaryClassifier, self).__init__()
    self.fc1 = nn.Linear(input_dim, 32)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(32,
                         1)  # Output a single logit for binary classification

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


# 2. Instantiate the model
# Assuming 'feature_cols' from config: ["SMA_10", "RSI_14", "MACD", "MACD_Signal", "Daily_Return", "Prev_Close"]
# So, input_dim should be 6.
input_dim = 6  # This must match the number of features your _engineer_features generates and your model expects
model = SimpleBinaryClassifier(input_dim=input_dim)

# 3. (Optional) Dummy training/fitting (to have some weights)
# This is crucial for some modules to have actual weights
dummy_input = torch.randn(1, input_dim)  # Batch size 1, input_dim features
dummy_output = model(dummy_input)
# print(f"Dummy output: {dummy_output}")

# 4. Save the model's state_dict
model_dir = "ml_models"
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

model_path = os.path.join(
    model_dir,
    "my_pytorch_model_state_dict.pt")  # Changed filename to reflect state_dict

# Save only the state_dict
torch.save(model.state_dict(), model_path)
print(f"Dummy PyTorch model state_dict saved to: {model_path}")

# To check if it loads:
# loaded_state_dict = torch.load(model_path)
# new_model = SimpleBinaryClassifier(input_dim=input_dim) # Instantiate architecture first
# new_model.load_state_dict(loaded_state_dict)
# new_model.eval()
# print(new_model)
