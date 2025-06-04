from enum import Enum


class Signal(Enum):
  """
    Represents a trading signal.
    """
  BUY = 'BUY'
  SELL = 'SELL'
  HOLD = 'HOLD'
