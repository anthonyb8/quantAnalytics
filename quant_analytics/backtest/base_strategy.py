from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
import numpy as np


class SymbolMap:
    def __init__(self):
        self.map = {}

    def append_symbol(self, symbol: str, quantity_mp: int, price_mp: float):
        self.map[symbol] = {
            "quantity_multiplier": quantity_mp,
            "price_multiplier": price_mp,
        }

    def get_quantity_mp(self, symbol: str) -> int:
        return self.map[symbol]["quantity_multiplier"]

    def get_price_mp(self, symbol: str) -> float:
        return self.map[symbol]["price_multiplier"]

    def get_map(self) -> Dict:
        return self.map

    def get_symbols(self) -> List:
        return list(self.map.keys())


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies in a financial trading system.

    This class outlines the necessary structure and functions that all derived trading strategies must implement.\n
    These include initializing strategy-specific parameters, preparing any necessary pre-trade configurations, generating trading signals, and managing asset allocations based on the market data.

    Attributes:
    - historical_data: Storage for historical market data, initially set to None.
    """

    def __init__(self, symbols_map: SymbolMap):
        """
        Initializes the BaseStrategy with components for managing historical data.
        """

        self.symbols_map = symbols_map

    @abstractmethod
    def prepare(self, data: pd.DataFrame) -> str:
        """
        Prepares the trading environment or strategy parameters before the trading starts.

        This method should include any initialization or setup procedures that need to run before the strategy can start processing market data.
        It should return an integer representing the index in the dataset on which the backtest should start.

        Parameters:
        - data: The historical market data that the strategy will use for preparation.

        Returns:
        - int: The index in the dataset where the backtest should begin, i.e., last used data point + 1.
        """
        for symbol in self.symbols_map.get_symbols():
            self.data[f"{symbol}_signal"] = np.nan

    @abstractmethod
    def generate_signals(self) -> None:
        """
        Generates trading signals for a vectorized backtest environment.

        This method should implement the logic to produce entry and exit signals based on vectorized market data.
        """
        pass
