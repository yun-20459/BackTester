# User Guide

## Running backtest

```bash
python -m  backtest.backtest --start_date 2020-06-04 --end_date 2025-06-03 --tickers_list AAPL GOOGL MSFT AMZN TSLA META NVDA PANW CRWD 
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
  "stock_info": {
    "symbol": "GOOGL",
    "current_price": 166.18
  }
}
```