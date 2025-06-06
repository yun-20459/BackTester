import numpy as np
import pandas as pd

from common import market
from core import account
from strategy import base
from utils import logger_utils

logger = logger_utils.get_logger(__name__)


class BacktestingEngine:
  """
    Backtesting engine, responsible for coordinating data, strategies, and account simulation.
    Supports multi-stock backtesting.
    """

  def __init__(self, initial_capital: float, commission_rate: float = 0.001):
    """
        Initializes the backtesting engine.

        Args:
            initial_capital (float): Initial capital.
            commission_rate (float): Transaction commission rate.
        """
    self.broker = account.AccountSimulator(initial_capital, commission_rate)
    self.strategy = None
    self.data = {}
    self.symbols = []
    self.all_dates = pd.DatetimeIndex([])

  def set_data(self, data_dict: dict[str, pd.DataFrame]):
    """
        Sets the backtesting data. Now accepts a dictionary containing data for multiple stocks.

        Args:
            data_dict (dict[str, pd.DataFrame]): Dictionary where keys are stock symbols and values are OHLCV DataFrames for that stock.
                                                The index of each DataFrame should be datetime, containing 'Open', 'High', 'Low', 'Close', 'Volume' columns.
        """
    if not isinstance(data_dict, dict) or not data_dict:
      logger.error(
          "set_data must receive a non-empty dictionary, where keys are stock symbols and values are DataFrames."
      )
      raise ValueError(
          "set_data must receive a non-empty dictionary, where keys are stock symbols and values are DataFrames."
      )

    self.data = {}
    all_dates_set = set()

    for symbol, df in data_dict.items():
      if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning(
            "Data for stock %s is empty or in incorrect format, skipping.",
            symbol)
        continue

      df_sorted = df.sort_index()
      self.data[symbol] = df_sorted
      self.symbols.append(symbol)
      all_dates_set.update(df_sorted.index.tolist())

      logger.info("Backtesting data loaded, stock: %s, data range: %s to %s",
                  symbol,
                  df_sorted.index.min().strftime('%Y-%m-%d'),
                  df_sorted.index.max().strftime('%Y-%m-%d'))

    self.symbols.sort()
    self.all_dates = pd.DatetimeIndex(sorted(list(all_dates_set)))
    logger.info("Total backtesting date range: %s to %s",
                self.all_dates.min().strftime('%Y-%m-%d'),
                self.all_dates.max().strftime('%Y-%m-%d'))

  def set_strategy(self, strategy_class: base.BaseStrategy, *args, **kwargs):
    """
        Sets the backtesting strategy.

        Args:
            strategy_class: Strategy class (e.g., MyStrategy).
            *args: Variable length argument list to pass to the strategy's initialization function.
            **kwargs: Arbitrary keyword arguments to pass to the strategy's initialization function.
        """
    self.strategy = strategy_class(self.broker, *args, **kwargs)
    logger.info("Strategy '%s' has been set.", strategy_class.__name__)

  def _calculate_max_drawdown(self) -> float:
    """
        Calculates the maximum drawdown from the equity curve.

        Returns:
            float: The maximum drawdown as a percentage.
        """
    if not self.broker.equity_curve:
      return 0.0

    equity_values = pd.Series([d['equity'] for d in self.broker.equity_curve])

    running_max = equity_values.expanding().max()
    drawdown = (equity_values - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    return max_drawdown

  def _calculate_sharpe_ratio(self,
                              annual_risk_free_rate: float = 0.0) -> float:
    """
        Calculates the Sharpe Ratio from the equity curve.

        Args:
            annual_risk_free_rate (float): The annual risk-free rate (e.g., 0.02 for 2%).

        Returns:
            float: The Sharpe Ratio. Returns 0 if standard deviation is zero.
        """
    if len(self.broker.equity_curve) < 2:
      return 0.0

    equity_series = pd.Series([d['equity'] for d in self.broker.equity_curve])

    daily_returns = equity_series.pct_change().dropna()

    if daily_returns.empty:
      return 0.0

    avg_daily_return = daily_returns.mean()
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1 / 252) - 1
    excess_daily_returns = avg_daily_return - daily_risk_free_rate
    std_dev_daily_returns = daily_returns.std()

    if std_dev_daily_returns == 0:
      return 0.0

    sharpe_ratio = (excess_daily_returns /
                    std_dev_daily_returns) * np.sqrt(252)

    return sharpe_ratio

  def run(self):
    """
        Runs the backtest.
        """
    if not self.data or self.strategy is None:
      logger.error("Please set backtesting data and strategy first.")
      return

    for _, current_date in enumerate(self.all_dates):
      daily_closing_prices = {}

      for symbol in self.symbols:
        symbol_data = self.data[symbol]

        if current_date in symbol_data.index:
          row = symbol_data.loc[current_date]
          current_data = row.to_dict()
          current_data['Date'] = current_date

          historical_data = symbol_data.loc[:current_date]

          action, quantity = self.strategy.on_bar(symbol, current_data,
                                                  historical_data)

          # Execute the order based on the strategy's suggestion using Signal enum
          if action == market.Signal.BUY:  # Changed from 'BUY' to Signal.BUY
            self.broker.execute_order(current_date, symbol, quantity,
                                      current_data['Close'], 'MARKET')
          elif action == market.Signal.SELL:  # Changed from 'SELL' to Signal.SELL
            self.broker.execute_order(current_date, symbol, -quantity,
                                      current_data['Close'], 'MARKET')
          # If Signal.HOLD, do nothing

          daily_closing_prices[symbol] = current_data['Close']

      daily_equity = self.broker.get_current_equity(daily_closing_prices)
      self.broker.equity_curve.append({
          'date': current_date,
          'equity': daily_equity
      })

    logger.info("\n--- Backtest Ended ---")

    final_prices = {
        symbol: self.data[symbol]['Close'].iloc[-1]
        for symbol in self.symbols if not self.data[symbol].empty
    }
    final_equity = self.broker.get_current_equity(final_prices)
    total_return = (final_equity - self.broker.initial_capital
                    ) / self.broker.initial_capital * 100

    logger.info("Final Equity: %.2f", final_equity)
    logger.info("Total Return: %.2f%%", total_return)

    max_drawdown = self._calculate_max_drawdown()
    sharpe_ratio = self._calculate_sharpe_ratio(annual_risk_free_rate=0.02)

    logger.info("Max Drawdown: %.2f%%", max_drawdown)
    logger.info("Sharpe Ratio: %.4f", sharpe_ratio)
    logger.info(
        "  (Note: Sharpe Ratio calculation assumes an annualized risk-free rate of 2%% and 252 trading days per year.)"
    )

    logger.info("\n--- Transaction Log (%d records) ---",
                len(self.broker.get_transaction_log()))
    if self.broker.get_transaction_log():
      for log in self.broker.get_transaction_log():
        logger.info(
            "  %s | %s %d shares of %s | Price: %.2f | Fee: %.2f | Cash: %.2f",
            log['timestamp'].strftime('%Y-%m-%d'), log['type'],
            log['quantity'], log['symbol'], log['price'], log['fee'],
            log['cash_after_trade'])
    else:
      logger.info("No transactions occurred.")
