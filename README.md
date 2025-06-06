# **User Guide**

This guide provides instructions on how to set up and run the stock trading backtest system and the daily signal generator.

## **Project Structure Overview**

The project is organized into modular folders to promote clarity and maintainability:

* `backtest.py`: The main script for executing historical backtests.  
* `common/`: Contains common utilities such as the database name definition (`data.py`) and trading signals (`market.py`).  
* `core/`: Holds the core backtesting engine (`engine.py`) and account simulation logic (`account_simulator.py`).  
* `ml_models/`: Dedicated to storing machine learning models and the model_loader.py for dynamic model loading.  
* `predict/`: Contains the signal_generator.py for generating daily trading signals, along with its specific config.json.  
* `strategy/`: Defines various trading strategies, including abstract base classes (`base.py`, `ml_base.py`) and concrete implementations (`adx_strategy.py`, `bb_rsi_strategy.py`, `simple_moving_average.py`, `my_ml_strategy.py`).  
* `utils/`: Provides general utility functions for data handling (data_utils.py) and logging (logger_utils.py).  
* `stock_data.db`: Your SQLite database file for storing historical stock data.

## **Prerequisites**

Before running the system, ensure you have Python 3.x installed and all necessary libraries.

```shell
pip install pandas yfinance matplotlib talib torch
```

**Note on TA-Lib:** talib requires the underlying TA-Lib C library to be installed. Please refer to external documentation or your operating system's package manager for specific installation steps (e.g., brew install ta-lib on macOS, sudo apt-get install libta-lib-dev on Linux).

## **1. Setting Up Your ML Model (PyTorch Only)**

This system is exclusively configured to load and utilize pre-trained PyTorch models.

### **1.1 Define Your PyTorch Model Architecture**

Your PyTorch model's class definition must be accessible by the model_loader.py. For basic cases, you can define it directly within `ml_models/model_loader.py`. For larger projects, consider defining your model architectures in a separate file (e.g., `ml_models/architectures.py`) and importing them into `model_loader.py`.

Example of a simple model architecture (`SimpleBinaryClassifier`):

```python
# Part of ml_models/model_loader.py or ml_models/architectures.py  
import torch.nn as nn

class SimpleBinaryClassifier(nn.Module):  
    def __init__(self, input_dim):  
        super(SimpleBinaryClassifier, self).__init__()  
        self.fc1 = nn.Linear(input_dim, 32)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(32, 1) # Output a single logit for binany classification

    def forward(self, x):  
        return self.fc2(self.relu(self.fc1(x)))
```

### **1.2 Train and Save Your Model's State Dictionary**

Train your PyTorch model independently and save its state_dict(). The system's model_loader expects the model's weights to be saved in this format (.pt or .pth file).

Example script to save a dummy model's state dictionary:

create_dummy_pytorch_model.py (Run this script once from your project root)  

```python
import torch  
import torch.nn as nn  
import os  
import numpy as np

# Make sure this matches your model architecture and expected input features  
class SimpleBinaryClassifier(nn.Module):  
    def __init__(self, input_dim):  
        super(SimpleBinaryClassifier, self).__init__()  
        self.fc1 = nn.Linear(input_dim, 32)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):  
        return self.fc2(self.relu(self.fc1(x)))

input_dim = 6 # This must match your feature_cols count and model's input  
model = SimpleBinaryClassifier(input_dim=input_dim)

# (Optional) Dummy fitting (to have some weights)  
dummy_input = torch.randn(1, input_dim)  
_ = model(dummy_input)

model_dir = "ml_models"  
if not os.path.exists(model_dir):  
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "my_pytorch_model_state_dict.pt")  
torch.save(model.state_dict(), model_path)  
print(f"Dummy PyTorch model state_dict saved to: {model_path}")
```

## **2. Defining Your Trading Strategies**

The system supports two main categories of trading strategies. All strategies must inherit from `strategy/base.py`'s `BaseStrategy`.

### **2.1 Rule-Based Strategies**

These strategies employ explicitly defined mathematical indicators and logical rules to generate buy/sell signals (e.g., moving average crossovers, RSI overbought/oversold).

* **Definition:** Create a new Python file under the strategy/ folder (e.g., adx_strategy.py, bb_rsi_strategy.py, simple_moving_average.py).  
* **Implementation:** Your strategy class should inherit from BaseStrategy and override its on_bar method. Inside on_bar, you will calculate indicators and return a (`market.Signal`, `int`) tuple (action and quantity).

Example parameters for ADXStrategy (defined in `strategy/adx_strategy.py`):

```json
{  
  "name": "ADXStrategy",  
  "module": "adx_strategy",  
  "params": {  
    "adx_period": 14,  
    "adx_rising_lookback": 5,  
    "adx_trend_threshold": 25,  
    "stop_loss_pct": 0.05,  
    "take_profit_pct": 0.15,  
    "adx_exit_threshold": 20  
  }  
}
```

### **2.2 ML-Based Strategies**

These strategies utilize pre-trained machine learning models (currently only PyTorch) to predict future trends or generate signals.

* **Base Class (MLBase):** The strategy/ml_base.py defines MLBase as an **abstract base class**. It handles the universal aspects of ML strategies: model loading (model_loader), common inference logic, and manages internal position status. It requires subclasses to implement specific methods.  
* **Concrete Implementation (MyMLStrategy):** You will define your specific ML strategy by inheriting from MLBase.  
  * **Definition:** Create a new Python file under the `strategy/` folder (e.g., my_ml_strategy.py).  
  * **Implementation:**  
    * Override `_engineer_features(self, historical_data)`: This is where you implement the exact feature engineering steps required by your specific ML model. This must precisely match the features used during your model's training.  
    * Override `_make_trading_decision(self, symbol, current_data, prediction_output, in_position, entry_price)`: This method contains your specific trading logic, including how you interpret the model's prediction_output (e.g., probability thresholds) and how you apply risk management rules like stop-loss and take-profit.

Example parameters for `MyMLStrategy` (defined in `strategy/example_ml.py`):

```json
{  
  "name": "MyMLStrategy",  
  "module": "example_ml",  
  "params": {  
    "model_path": "ml_models/my_pytorch_model_state_dict.pt",  
    "model_architecture_name": "SimpleBinaryClassifier",  
    "model_input_dim": 6,  
    "feature_cols": ["SMA_10", "RSI_14", "MACD", "MACD_Signal", "Daily_Return", "Prev_Close"],  
    "buy_threshold": 0.55,  
    "sell_threshold": 0.45,  
    "stop_loss_pct": 0.05,  
    "take_profit_pct": 0.15  
  }  
}
```

## **3. Running Backtest**

The backtest script allows you to evaluate your chosen trading strategy on historical data.

### **3.1 Define Backtest Configuration (config.json)**

Create or modify a config.json file (e.g., `backtest_config.json` in your project root) specifically for backtesting. In the "strategy" block, specify the name and module of the strategy you wish to use (either rule-based or ML-based), along with its params.

**Example: Rule-Based Strategy (e.g., ADXStrategy)**

```json
{  
  "strategy": {  
    "name": "ADXStrategy",  
    "module": "adx_strategy",  
    "params": {  
      "adx_period": 14,  
      "adx_rising_lookback": 5,  
      "adx_trend_threshold": 25,  
      "stop_loss_pct": 0.05,  
      "take_profit_pct": 0.15,  
      "adx_exit_threshold": 20  
    }  
  },  
  "backtest_params": {  
    "start_date": "2020-01-01",  
    "end_date": "2024-12-31",  
    "initial_capital": 100000.0,  
    "commission_rate": 0.015,  
    "symbols": ["AAPL", "MSFT", "GOOG"]  
  }  
}
```

**Example: ML-Based Strategy (e.g., MyMLStrategy)**

```json
{  
  "strategy": {  
    "name": "MyMLStrategy",  
    "module": "my_ml_strategy",  
    "params": {  
      "model_path": "ml_models/my_pytorch_model_state_dict.pt",  
      "model_architecture_name": "SimpleBinaryClassifier",  
      "model_input_dim": 6,  
      "feature_cols": ["SMA_10", "RSI_14", "MACD", "MACD_Signal", "Daily_Return", "Prev_Close"],  
      "buy_threshold": 0.55,  
      "sell_threshold": 0.45,  
      "stop_loss_pct": 0.05,  
      "take_profit_pct": 0.15  
    }  
  },  
  "backtest_params": {  
    "start_date": "2020-01-01",  
    "end_date": "2024-12-31",  
    "initial_capital": 100000.0,  
    "commission_rate": 0.015,  
    "symbols": ["AAPL", "MSFT", "GOOG"]  
  }  
}
```

### **3.2 Run the Backtest**

Navigate to your project's root directory. Then, execute the backtest.py script as a module, passing the path to your backtest configuration file.

```shell
python -m backtest.backtest --config_path backtest/backtest_config.json
```

## **4. Running Today's Signal Generation**

The signal generator script provides daily trading signals for a list of stocks. It automatically fetches the latest stock prices from yfinance and saves them to your SQLite database before generating signals.

### **4.1 Define Signal Generator Configuration (predict/config.json)**

Create or modify a config.json file (e.g., `predict/signal_config.json`) within the `predict/` folder. This config file now takes a simple list of stock symbols. In the "strategy" block, specify the name and module of the strategy you wish to use.

**Example: Rule-Based Strategy (e.g., ADXStrategy)**

```json
{  
  "strategy": {  
    "name": "ADXStrategy",  
    "module": "adx_strategy",  
    "params": {  
      "adx_period": 14,  
      "adx_rising_lookback": 5,  
      "adx_trend_threshold": 25,  
      "stop_loss_pct": 0.05,  
      "take_profit_pct": 0.15,  
      "adx_exit_threshold": 20  
    }  
  },  
  "signal_generation_params": {  
    "historical_period_days": 200,  
    "initial_dummy_capital": 100000.0  
  },  
  "stock_list": [  
    "AAPL",  
    "MSFT",  
    "GOOG"  
  ]  
}
```

**Example: ML-Based Strategy (e.g., MyMLStrategy)**

```json
{  
  "strategy": {  
    "name": "MyMLStrategy",  
    "module": "example_ml",  
    "params": {  
      "model_path": "ml_models/my_pytorch_model_state_dict.pt",  
      "model_architecture_name": "SimpleBinaryClassifier",  
      "model_input_dim": 6,  
      "feature_cols": ["SMA_10", "RSI_14", "MACD", "MACD_Signal", "Daily_Return", "Prev_Close"],  
      "buy_threshold": 0.55,  
      "sell_threshold": 0.45,  
      "stop_loss_pct": 0.05,  
      "take_profit_pct": 0.15  
    }  
  },  
  "signal_generation_params": {  
    "historical_period_days": 200,  
    "initial_dummy_capital": 100000.0  
  },  
  "stock_list": [  
    "AAPL",  
    "MSFT",  
    "GOOG"  
  ]  
}
```

### **4.2 Run the Signal Generator**

Navigate to your project's root directory (e.g., backtest/). Then, execute the signal_generator.py script as a module, passing the path to its configuration file.

```shell
python -m predict.signal_generator --config_path predict/signal_config.json  
```
