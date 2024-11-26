import numpy as np
import pandas as pd
from enum import Enum, auto
from quantAnalytics.backtest.base_strategy import BaseStrategy
from quantAnalytics.report.report import DivBuilder, Header
from quantAnalytics.backtest.base_strategy import SymbolMap


class Signal(Enum):
    """Long and short are treated as entry actions and short/cover are treated as exit actions."""

    Overvalued = auto()
    Undervalued = auto()
    Exit_Overvalued = auto()
    Exit_Undervalued = auto()
    NoSignal = auto()


class Cointegrationzscore(BaseStrategy):
    def __init__(self, symbols: SymbolMap):
        super().__init__(symbols)
        # parameters
        self.symbols = symbols.get_symbols()
        self.start_index = 100
        self.zscore_lookback = 30
        self.entry_threshold = 2
        self.exit_threshold = 1
        self.weights = {"HE.n.0": 2, "ZC.n.0": -3}  # update from research

        # data
        self.last_signal = Signal.NoSignal
        self.data: pd.DataFrame

    def prepare(self, data: pd.DataFrame) -> str:
        html_div = DivBuilder()
        html_div.add_header("Cointegration Z-score", Header.H2)

        # Log Transform
        self.data = self._log_transform(data)

        # Spread
        self.data["spread"] = self.update_spread(self.data)

        # Z-score
        self.data["zscore"] = self.update_zscore()

        super().prepare(data)

        return html_div.build()

    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Log-transform the prices of the input data.
        """
        for symbol in self.symbols:
            if symbol in data.columns:
                log_column_name = f"{symbol}_log"
                data[log_column_name] = np.log(data[symbol])
        return data

    def update_spread(self, data: pd.DataFrame) -> list:
        """
        Calculate the historical spread using the hedge ratios (weights).
        The spread is a weighted combination of the log prices of the instruments.
        """
        spread_series = sum(
            self.weights[symbol] * data[f"{symbol}_log"]
            for symbol in self.weights
        )  # Weighted sum of log prices
        return spread_series.tolist()

    def update_zscore(self) -> np.ndarray:
        # Convert historical spread to a pandas Series for convenience
        spread_series = pd.Series(self.data["spread"])

        if self.zscore_lookback:
            # Use a rolling window if lookback_period is specified
            mean = spread_series.rolling(window=self.zscore_lookback).mean()
            std = spread_series.rolling(window=self.zscore_lookback).std()
        else:
            # Use an expanding window if lookback_period is None, considering all data up to each point
            mean = spread_series.expanding().mean()
            std = spread_series.expanding().std()

        # Calculate z-score
        historical_zscore = ((spread_series - mean) / std).to_numpy()

        return historical_zscore

    # -- Strategy logic --
    def _entry_signal(self, z_score: float) -> bool:
        """
        Entry logic.

        Parameters:
        - z_score (float): Current value of the z-score.
        - entry_threshold (float): Absolute value of z-score that triggers an entry signal.

        Returns:
        - bool : True if an entry signal else False.
        """
        if (
            self.last_signal == Signal.NoSignal
            and z_score >= self.entry_threshold
        ):
            self.last_signal = Signal.Overvalued
            return True
        elif (
            self.last_signal == Signal.NoSignal
            and z_score <= -self.entry_threshold
        ):
            self.last_signal = Signal.Undervalued
            return True
        else:
            return False

    def _exit_signal(self, z_score: float) -> bool:
        """
        Exit logic.

        Parameters:
        - z_score (float): Current value of the z-score.
        - exit_threshold (float): Absolute value of z-score that triggers an exit signal.

        Returns:
        - bool : True if an exit signal else False.
        """
        if (
            self.last_signal == Signal.Undervalued
            and z_score >= -self.exit_threshold
        ):
            self.last_signal = Signal.Exit_Undervalued
            return True
        elif (
            self.last_signal == Signal.Overvalued
            and z_score <= self.exit_threshold
        ):
            self.last_signal = Signal.Exit_Overvalued
            return True
        else:
            return False

    def generate_signals(self) -> None:
        """
        Generate signals in backtest. LONG = 1, SHORT = -1, NO SIGNAL = 0

        Parameters:
        - entry_threshold: The threshold to trigger a trade entry.
        - exit_threshold: The threshold to trigger a trade exit.
        - lag (int): The number of periods the entry/exit of a position will be lagged after a signal.

        Returns:
        - pandas.DataFrame : Contains the results of the backtests, including original data plus signals and positions.
        """

        # Iterate through DataFrame rows
        for i in range(self.start_index, len(self.data)):
            current_zscore = self.data.iloc[i]["zscore"]

            # Continue to next iteration if current Z-score is NaN
            if np.isnan(current_zscore):
                continue

            # Check for entry signals
            if self.last_signal == Signal.NoSignal:
                if self._entry_signal(current_zscore):
                    for ticker, weight in self.weights.items():
                        if self.last_signal == Signal.Undervalued:
                            self.data.at[
                                self.data.index[i], f"{ticker}_signal"
                            ] = weight
                        elif self.last_signal == Signal.Overvalued:
                            self.data.at[
                                self.data.index[i], f"{ticker}_signal"
                            ] = -weight
            # Check for exit signals
            else:
                if self._exit_signal(current_zscore):
                    for ticker in self.weights.keys():
                        self.data.loc[
                            self.data.index[i], f"{ticker}_signal"
                        ] = 0
                    self.last_signal = Signal.NoSignal  # reset to no position
