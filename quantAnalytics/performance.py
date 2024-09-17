import numpy as np


class PerformanceStatistics:
    @staticmethod
    def simple_returns(prices: np.ndarray, decimals: int = 6) -> np.ndarray:
        """
        Calculate simple returns from an array of prices.

        Parameters:
        - prices (np.ndarray): A 1D array of prices.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - np.ndarray: A 1D array of simple returns.
        """
        if not isinstance(prices, np.ndarray):
            raise TypeError(
                f"'prices' must be of type np.ndarray. Recieved type : {type(prices)}"
            )
        try:
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
            return np.around(returns, decimals=decimals)
        except Exception as e:
            raise Exception(f"Error calculating simple returns {e}")

    @staticmethod
    def log_returns(prices: np.ndarray, decimals: int = 6) -> np.ndarray:
        """
        Calculate logarithmic returns from an array of prices.

        Parameters:
        - prices (np.ndarray): A 1D array of prices.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - np.ndarray: A 1D array of logarithmic returns.
        """
        if not isinstance(prices, np.ndarray):
            raise TypeError(
                f"'prices' must be of type np.ndarray. Recieved type : {type(prices)}"
            )

        try:
            returns = np.log(prices[1:] / prices[:-1])
            return np.around(returns, decimals=decimals)
        except Exception as e:
            raise Exception(f"Error calculating log returns {e}")

    @staticmethod
    def cumulative_returns(
        equity_curve: np.ndarray, decimals: int = 6
    ) -> np.ndarray:
        """
        Calculate cumulative returns from an equity curve.

        Parameters:
        - equity_curve (np.ndarray): A 1D array of equity values.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - np.ndarray: A 1D array of cumulative returns.
        """
        if not isinstance(equity_curve, np.ndarray):
            raise TypeError("equity_curve must be a numpy array")

        if len(equity_curve) == 0:
            return np.array([0])

        try:
            period_returns = (
                equity_curve[1:] - equity_curve[:-1]
            ) / equity_curve[:-1]
            cumulative_returns = np.cumprod(1 + period_returns) - 1
            return np.around(cumulative_returns, decimals=decimals)
        except Exception as e:
            raise Exception(f"Error calculating cumulative returns: {e}")

    @staticmethod
    def total_return(equity_curve: np.ndarray, decimals: int = 6) -> float:
        """
        Calculate the total return from an equity curve.

        Parameters:
        - equity_curve (np.ndarray): A 1D array of equity values.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The total return as a decimal.
        """

        if not isinstance(equity_curve, np.ndarray):
            raise TypeError("equity_curve must be a numpy array")

        if len(equity_curve) == 0:
            return np.array([0])
        try:
            return (
                PerformanceStatistics.cumulative_returns(
                    equity_curve, decimals
                )[-1]
                if len(equity_curve) > 0
                else 0.0
            )
        except Exception as e:
            raise Exception(f"Error calculating total return: {e}")

    @staticmethod
    def annualize_returns(
        returns: np.ndarray, periods_per_year: int = 252, decimals: int = 6
    ) -> float:
        """
        Annualize returns.

        Parameters:
        - returns (np.ndarray): A 1D array of returns.
        - periods_per_year (int): The number of periods per year. Default is 252.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The annualized return.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("'returns' must be a numpy.ndarray")

        try:
            compounded_growth = (1 + returns).prod()
            n_periods = returns.shape[0]
            return round(
                compounded_growth ** (periods_per_year / n_periods) - 1,
                decimals,
            )
        except Exception as e:
            raise Exception(f"Error calculating annualized returns {e}")

    @staticmethod
    def net_profit(equity_curve: np.ndarray, decimals: int = 6) -> float:
        """
        Calculate the net profit from an equity curve NumPy array.

        This method calculates the net profit by taking the difference between the
        first and the last element of the equity curve array.

        Parameters:
        - equity_curve (np.ndarray): The equity curve array. It should contain the equity values over time.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The net profit, rounded to four decimal places.
        """

        # Ensure the equity curve is not empty
        if equity_curve.size == 0:
            return 0.0

        # Calculate the difference between the last and first item
        net_profit = equity_curve[-1] - equity_curve[0]

        return round(net_profit, decimals)
