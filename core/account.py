import pandas as pd

from utils import logger_utils

logger = logger_utils.get_logger(__name__)


class AccountSimulator:
  """
    Simulates a trading account, managing capital, positions, and transaction records.
    """

  def __init__(self, initial_capital: float, commission_rate: float = 0.001):
    """
        Initializes the simulated account.

        Args:
            initial_capital (float): Initial capital.
            commission_rate (float): Commission rate per transaction (e.g., 0.001 represents 0.1%).
        """
    if initial_capital <= 0:
      raise ValueError("Initial capital must be greater than zero.")

    self.initial_capital = initial_capital
    self.current_cash = initial_capital  # Current available cash
    self.positions = {
    }  # Positions: {'stock_symbol': {'quantity': quantity, 'avg_cost': average_cost}}
    self.transaction_log = []  # Transaction log
    self.equity_curve = []  # Equity curve
    self.commission_rate = commission_rate  # Commission rate

    logger.info("Simulated account initialized, initial capital: %.2f",
                self.initial_capital)

  def _record_transaction(self, timestamp, symbol, trade_type, price, quantity,
                          amount, fee):
    """
        Records transaction log.
        """
    self.transaction_log.append({
        'timestamp': timestamp,
        'symbol': symbol,
        'type': trade_type,  # 'BUY' or 'SELL'
        'price': price,
        'quantity': quantity,
        'amount': amount,
        'fee': fee,
        'cash_after_trade': self.current_cash,
        'positions_after_trade': {
            s: p['quantity']
            for s, p in self.positions.items()
        }  # Simplified position recording
    })

  def execute_order(self,
                    timestamp: pd.Timestamp,
                    symbol: str,
                    quantity: int,
                    price: float,
                    order_type: str = 'MARKET'):
    """
        Executes an order. Currently only supports market orders.

        Args:
            timestamp (pd.Timestamp): Timestamp of the transaction.
            symbol (str): Stock symbol.
            quantity (int): Quantity to trade (positive for buy, negative for sell).
            price (float): Execution price.
            order_type (str): Order type (currently only 'MARKET' is supported).
        """
    if order_type != 'MARKET':
      logger.warning(
          "Warning: Only MARKET orders are currently supported. Order type '{%s}' was not processed.",
          order_type)
      return

    if quantity == 0:
      return

    # Calculate total trade amount (excluding commission)
    trade_amount = quantity * price
    # Calculate commission (charged for both buy and sell)
    fee = abs(trade_amount) * self.commission_rate

    if quantity > 0:  # Buy
      if self.current_cash >= (trade_amount + fee):
        self.current_cash -= (trade_amount + fee)
        # Update positions
        if symbol not in self.positions:
          self.positions[symbol] = {'quantity': 0, 'avg_cost': 0.0}

        # Calculate new average cost
        current_total_cost = self.positions[symbol][
            'quantity'] * self.positions[symbol]['avg_cost']
        new_total_cost = current_total_cost + trade_amount
        new_quantity = self.positions[symbol]['quantity'] + quantity

        self.positions[symbol]['quantity'] = new_quantity
        self.positions[symbol][
            'avg_cost'] = new_total_cost / new_quantity if new_quantity > 0 else 0.0

        self._record_transaction(timestamp, symbol, 'BUY', price, quantity,
                                 trade_amount, fee)
      else:
        logger.info(
            "  %s - Insufficient funds to buy %s shares of %s. Required: %.2f, Cash: %.2f",
            timestamp.strftime('%Y-%m-%d'),
            quantity,
            symbol,
            trade_amount + fee,
            self.current_cash,
        )

    elif quantity < 0:  # Sell
      abs_quantity = abs(quantity)
      if symbol in self.positions and self.positions[symbol][
          'quantity'] >= abs_quantity:
        self.current_cash += (abs(trade_amount) - fee
                              )  # Sell proceeds minus commission
        self.positions[symbol]['quantity'] -= abs_quantity

        # If position becomes zero, remove the stock
        if self.positions[symbol]['quantity'] == 0:
          del self.positions[symbol]

        self._record_transaction(timestamp, symbol, 'SELL', price,
                                 abs_quantity, abs(trade_amount), fee)
      else:
        logger.info(
            "  %s - Insufficient position to sell %s shares of %s. Current position: %s shares",
            timestamp.strftime('%Y-%m-%d'), abs_quantity, symbol,
            self.positions.get(symbol, {}).get('quantity', 0))

  def get_current_equity(self, current_prices: dict) -> float:
    """
        Calculates current total equity (cash + market value of positions).

        Args:
            current_prices (dict): Current prices of all held stocks, e.g., {'stock_symbol': price}.

        Returns:
            float: Current total equity.
        """
    portfolio_value = 0.0
    for symbol, pos_info in self.positions.items():
      if symbol in current_prices:
        portfolio_value += pos_info['quantity'] * current_prices[symbol]
      else:
        logger.warning(
            "Warning: Could not get current price for stock %s, its position value will not be included in total equity.",
            symbol)

    total_equity = self.current_cash + portfolio_value
    return total_equity

  def get_transaction_log(self) -> list:
    """
        Returns the complete transaction log.
        """
    return self.transaction_log

  def get_equity_curve(self) -> list:
    """
        Returns the equity curve data.
        """
    return self.equity_curve
