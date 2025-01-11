import numpy as np
import pandas as pd
from quant_analytics.result import Result


class AnnualizedVolZScore(Result):
    def __init__(self, name: str, data: dict):
        super().__init__("Annualized Volatility Z-Score", name, data)
        self.footer = "** Note: Z-scores provide a statistical measure of the volatility's deviation from its mean, with larger absolute values indicating more significant deviations."

    def _to_dataframe(self) -> pd.DataFrame:
        # Convert validation results to DataFrame
        data = []
        row = {
            "Name": self.timeseries_name,
            "Annualized Volatility": self.data["Annualized Volatility"],
            "Annualized Mean Return": round(
                self.data["Annualized Mean Return"], 6
            ),
        }

        i = 1
        for key, value in self.data["Z-Scores (Annualized)"].items():
            row.update({f"Z-score for {i} SD (annualized)": value})
            i += 1
        data.append(row)

        return pd.DataFrame(data)


class Metrics:
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
        equity_curve: np.ndarray,
        decimals: int = 6,
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
                Metrics.cumulative_returns(equity_curve, decimals)[-1]
                if len(equity_curve) > 0
                else 0.0
            )
        except Exception as e:
            raise Exception(f"Error calculating total return: {e}")

    @staticmethod
    def annualize_returns(
        returns: np.ndarray,
        periods_per_year: int = 252,
        decimals: int = 6,
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

    @staticmethod
    def drawdown(returns: np.ndarray, decimals: int = 6) -> np.ndarray:
        """
        Calculate the drawdown of a series of returns.

        This method calculates the drawdown, which is the decline from a historical peak in
        cumulative returns, for each point in the returns series. The drawdown values are in
        decimal format.

        Parameters:
        - returns (np.ndarray): A numpy array of returns.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - np.ndarray: An array of drawdown values, rounded to four decimal places.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return np.array([0])

        try:
            cumulative_returns = np.cumprod(
                1 + returns
            )  # Calculate cumulative returns
            rolling_max = np.maximum.accumulate(
                cumulative_returns
            )  # Calculate the rolling maximum
            drawdowns = (
                cumulative_returns - rolling_max
            ) / rolling_max  # Calculate drawdowns in decimal format
            return np.around(drawdowns, decimals=decimals)
        except Exception as e:
            raise Exception(f"Error calculating drawdown : {e}")

    @staticmethod
    def max_drawdown(returns: np.ndarray, decimals: int = 6) -> np.ndarray:
        """
        Calculate the maximum drawdown of a series of returns.

        This method calculates the maximum drawdown, which is the largest decline from a peak
        to a trough in the returns series. The drawdown values are in decimal format.

        Parameters:
        - returns (np.ndarray): A numpy array of returns.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The maximum drawdown value.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return np.array([0])

        try:
            drawdowns = Metrics.drawdown(returns, decimals)
            max_drawdown = np.min(drawdowns)  # Find the maximum drawdown
            return max_drawdown
        except Exception as e:
            raise Exception(f"Error calculating max drawdown : {e}")

    @staticmethod
    def standard_deviation(returns: np.ndarray, decimals: int = 6) -> float:
        """
        Calculate the standard deviation of given returns.

        Parameters:
        - returns (np.ndarray): A numpy array of returns.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The standard deviation, rounded to default 6 decimal places.
        """

        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return np.array([0])

        try:
            std_dev = np.std(returns, ddof=1)
            return np.around(std_dev, decimals=decimals)
        except Exception as e:
            raise Exception(
                f"Error calculating annualized standard deviation : {e}"
            )

    @staticmethod
    def annual_standard_deviation(
        returns: np.ndarray,
        decimals: int = 6,
    ) -> float:
        """
        Calculate the annualized standard deviation of returns.

        This method calculates the annualized standard deviation of returns from a numpy array
        of daily returns. It assumes 252 trading days in a year.

        Parameters:
        - returns (np.ndarray): A numpy array of daily returns.
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The annualized standard deviation, rounded to four decimal places.
        """

        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return np.array([0])

        try:
            # Calculate daily standard deviation
            daily_std_dev = np.std(returns, ddof=1)

            # Assuming 252 trading days in a year (annualize)
            annual_std_dev = daily_std_dev * np.sqrt(252)
            return np.around(annual_std_dev, decimals=decimals)
        except Exception as e:
            raise Exception(
                f"Error calculating annualized standard deviation : {e}"
            )

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.04,
        annualize_period: int = 252,
        decimals: int = 6,
    ) -> float:
        """
        Calculate the Sharpe ratio of the strategy.

        The Sharpe ratio measures the performance of an investment compared to a risk-free asset,
        after adjusting for its risk. The ratio is the average return earned in excess of the risk-free
        rate per unit of volatility or total risk.

        Parameters:
        - returns (np.ndarray): A 1D array of returns.
        - risk_free_rate (float): The risk-free rate. Default is 0.04 (4% annually).
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The Sharpe ratio, rounded to four decimal places.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return np.array([0])

        try:
            # Annualized return
            annualized_return = returns.mean() * annualize_period

            # Annual Standard Deviation returns
            std_return = returns.std(ddof=1) * np.sqrt(annualize_period)

            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / std_return

            return np.around(sharpe_ratio, decimals=decimals)
        except Exception as e:
            raise Exception(f"Error calculating sharpe ratio : {e}")

    # @staticmethod
    # def sharpe_ratio(
    #     returns: np.ndarray,
    #     risk_free_rate: float = 0.04,
    #     decimals: int = 6,
    # ) -> float:
    #     """
    #     Calculate the Sharpe ratio of the strategy.
    #
    #     The Sharpe ratio measures the performance of an investment compared to a risk-free asset,
    #     after adjusting for its risk. The ratio is the average return earned in excess of the risk-free
    #     rate per unit of volatility or total risk.
    #
    #     Parameters:
    #     - returns (np.ndarray): A 1D array of returns.
    #     - risk_free_rate (float): The risk-free rate. Default is 0.04 (4% annually).
    #     - decimals (int): Decimal rounding on return values. Defaults to 6.
    #
    #     Returns:
    #     - float: The Sharpe ratio, rounded to four decimal places.
    #     """
    #     if not isinstance(returns, np.ndarray):
    #         raise TypeError("returns must be a numpy array")
    #
    #     if len(returns) == 0:
    #         return np.array([0])
    #
    #     try:
    #         daily_risk_free_rate = risk_free_rate / 252
    #         excess_returns = returns - daily_risk_free_rate
    #
    #         # Annualized calculations
    #         annualized_avg_excess_return = excess_returns.mean() * 252
    #         annualized_std_excess_return = excess_returns.std(
    #             ddof=1
    #         ) * np.sqrt(252)
    #
    #         # Sharpe
    #         sharpe_ratio = (
    #             annualized_avg_excess_return / annualized_std_excess_return
    #         )
    #
    #         return (
    #             np.around(sharpe_ratio, decimals=decimals)
    #             if excess_returns.std(ddof=1) != 0
    #             else 0
    #         )
    #     except Exception as e:
    #         raise Exception(f"Error calculating sharpe ratio : {e}")
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.04,
        annualize_period: int = 252,
        decimals: int = 6,
    ) -> float:
        """
        Calculate the Sortino Ratio for a given returns array.

        The Sortino ratio differentiates harmful volatility from total overall volatility
        by using the asset's standard deviation of negative returns, called downside deviation.
        It measures the risk-adjusted return of an investment asset, portfolio, or strategy.

        Parameters:
        - returns (np.ndarray): A 1D array of returns.
        - risk_free_rate (float): The risk-free rate. Default is 0.04 (4% annually).
        - decimals (int): Decimal rounding on return values. Defaults to 6.

        Returns:
        - float: The Sortino ratio, rounded to four decimal places.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return 0

        try:
            # Annualized return
            annualized_return = returns.mean() * annualize_period

            # Downside
            downside_ret = returns[returns < 0]
            if len(downside_ret) == 0:
                return 0.0

            # Annual Standard Deviation downside returns
            std_return = downside_ret.std(ddof=1) * np.sqrt(annualize_period)

            excess_return = annualized_return - risk_free_rate
            sortino_ratio = excess_return / std_return

            return np.around(sortino_ratio, decimals=decimals)
        except Exception as e:
            raise Exception(f"Error calculating sortino ratio : {e}")

    # @staticmethod
    # def sortino_ratio(
    #     returns: np.ndarray,
    #     risk_free_rate: float = 0.04,
    #     decimals: int = 6,
    # ) -> float:
    #     """
    #     Calculate the Sortino Ratio for a given returns array.
    #
    #     The Sortino ratio differentiates harmful volatility from total overall volatility
    #     by using the asset's standard deviation of negative returns, called downside deviation.
    #     It measures the risk-adjusted return of an investment asset, portfolio, or strategy.
    #
    #     Parameters:
    #     - returns (np.ndarray): A 1D array of returns.
    #     - risk_free_rate (float): The risk-free rate. Default is 0.04 (4% annually).
    #     - decimals (int): Decimal rounding on return values. Defaults to 6.
    #
    #     Returns:
    #     - float: The Sortino ratio, rounded to four decimal places.
    #     """
    #     if not isinstance(returns, np.ndarray):
    #         raise TypeError("returns must be a numpy array")
    #
    #     if len(returns) == 0:
    #         return 0
    #
    #     try:
    #         daily_risk_free_rate = risk_free_rate / 252
    #         excess_returns = returns - daily_risk_free_rate
    #         downside_returns = excess_returns[excess_returns < 0]
    #
    #         avg_excess_return = np.mean(excess_returns)
    #         downside_deviation = np.std(downside_returns, ddof=1)
    #
    #         annualized_avg_excess_return = avg_excess_return * 252
    #         annualized_downside_deviation = downside_deviation * np.sqrt(252)
    #
    #         sortino_ratio = (
    #             annualized_avg_excess_return / annualized_downside_deviation
    #         )
    #         return (
    #             np.around(sortino_ratio, decimals=decimals)
    #             if annualized_downside_deviation != 0
    #             else 0
    #         )
    #     except Exception as e:
    #         raise Exception(f"Error calculating sortino ratio : {e}")

    @staticmethod
    def value_at_risk(
        returns: np.ndarray,
        confidence_level: float = 0.05,
    ) -> float:
        """
        Calculate the Value at Risk (VaR) at a specified confidence level using historical returns.

        VaR is a statistical technique used to measure the risk of loss on a specific portfolio of
        financial assets. It estimates how much a set of investments might lose, given normal market
        conditions, in a set time period such as a day.

        Parameters:
        - returns (np.ndarray): An array of returns.
        - confidence_level (float): The confidence level for VaR (e.g., 0.05 for 95% confidence).

        Returns:
        - float: The VaR value.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return np.nan
        return np.percentile(returns, confidence_level * 100)

    @staticmethod
    def conditional_value_at_risk(
        returns: np.ndarray,
        confidence_level: float = 0.05,
    ) -> float:
        """
        Calculate the Conditional Value at Risk (CVaR) at a specified confidence level using historical returns.

        CVaR, also known as Expected Shortfall (ES), measures the average loss that occurs beyond the VaR point,
        providing a more complete picture of tail risk.

        Parameters:
        - returns (np.ndarray): An array of returns.
        - confidence_level (float): The confidence level for CVaR (e.g., 0.05 for 95% confidence).

        Returns:
        - float: The CVaR value.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")
        if len(returns) == 0:
            return np.nan

        var = Metrics.value_at_risk(returns, confidence_level)
        tail_losses = returns[returns <= var]
        cvar = tail_losses.mean()
        return cvar

    @staticmethod
    def calculate_volatility_and_zscore_annualized(
        returns: np.ndarray,
    ) -> Result:
        """
        Calculate the strategy's annualized volatility and z-scores for 1, 2, and 3 standard deviation moves.

        This method calculates the annualized volatility and mean return from daily returns and provides
        z-scores adjusted for annualized values.

        Parameters:
        - returns (np.ndarray): A 1D array of daily returns.

        Returns:
        - dict: A dictionary containing annualized volatility, annualized mean return, and z-scores
                for 1, 2, and 3 standard deviation moves.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")

        if len(returns) == 0:
            return {
                "Annualized Volatility": 0,
                "Annualized Mean Return": 0,
                "Z-Scores (Annualized)": {},
            }

        try:
            daily_volatility = returns.std()
            daily_mean_return = returns.mean()

            # Annualizing the daily volatility and mean return
            annualized_volatility = daily_volatility * np.sqrt(252)
            annualized_mean_return = daily_mean_return * 252

            # Adjusting the calculation of z-scores for annualized values
            z_scores_annualized = {
                f"Z-score for {x} SD move (annualized)": (
                    annualized_mean_return - x * annualized_volatility
                )
                / annualized_volatility
                for x in range(1, 4)
            }
            output = {
                "Annualized Volatility": annualized_volatility,
                "Annualized Mean Return": annualized_mean_return,
                "Z-Scores (Annualized)": z_scores_annualized,
            }

            return AnnualizedVolZScore("", output)
        except Exception as e:
            raise Exception(
                f"Error calculating annualized volatility and z-scores : {e}"
            )
