# User Guide

## Running backtest

1. Define strategy in `strategy/` folder first.

2. Then define the config.json in `backtest/` folder

```json
{
  "strategy": {
    "name": "ADXStrategy",
    "module": "adx_strategy",
    "params": {
      "adx_period": 8,
      "adx_rising_lookback": 2,
      "stop_loss_pct": 0.05,
      "take_profit_pct": 0.4
    }
  },
  "backtest_params": {
    "start_date": "2020-01-01",
    "end_date": "2025-05-31",
    "initial_capital": 100000.0,
    "commission_rate": 0.015,
    "symbols": [
      "AAPL",
      "MSFT",
      "GOOGL",
      "NVDA",
      "TSLA",
      "PANW",
      "CRWD",
      "AMZN",
      "AMD"
    ]
  }
}
```

```bash
python -m backtest.backtest --config backtest/config.json
```

## Running today signal

For example ADX strategy

```bash
python -m predict.signal_generator --config_path predict/config.json
```

```json
{
  "strategy": {
    "name": "ADXStrategy",
    "module": "adx_strategy",
    "params": {
      "adx_period": 8,
      "adx_rising_lookback": 2,
      "take_profit_pct": 0.45,
      "stop_loss_pct": 0.05,
      "adx_exit_threshold": 20,
      "sell_percentage": 0.7
    }
  },
  "signal_generation_params": {
    "historical_period_days": 200,
    "initial_dummy_capital": 100000.0
  },
  "stocks_info": [
    {
      "symbol": "AAPL",
      "current_price": 180.50
    },
    {
      "symbol": "MSFT",
      "current_price": 420.00
    },
    {
      "symbol": "GOOG",
      "current_price": 170.00
    }
  ]
}
```