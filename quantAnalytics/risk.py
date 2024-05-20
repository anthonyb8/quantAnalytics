import numpy as np
import pandas as pd

class RiskAnalysis:
    @staticmethod
    def drawdown(returns:np.ndarray) -> np.ndarray:
        """
        Calculate the drawdown of a series of returns.

        This method calculates the drawdown, which is the decline from a historical peak in 
        cumulative returns, for each point in the returns series. The drawdown values are in 
        decimal format.

        Parameters:
        - returns (np.ndarray): A numpy array of returns.

        Returns:
        - np.ndarray: An array of drawdown values, rounded to four decimal places.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")
        
        if len(returns) == 0:
            return np.array([0])
        
        try:
            cumulative_returns = np.cumprod(1 + returns)  # Calculate cumulative returns
            rolling_max = np.maximum.accumulate(cumulative_returns)  # Calculate the rolling maximum
            drawdowns = (cumulative_returns - rolling_max) / rolling_max  # Calculate drawdowns in decimal format
            return np.around(drawdowns, decimals=4)
        except Exception as e:
            raise Exception(f"Error calculating drawdown : {e}")
        
    @staticmethod
    def max_drawdown(returns:np.ndarray) -> np.ndarray:
        """
        Calculate the maximum drawdown of a series of returns.

        This method calculates the maximum drawdown, which is the largest decline from a peak 
        to a trough in the returns series. The drawdown values are in decimal format.

        Parameters:
        - returns (np.ndarray): A numpy array of returns.

        Returns:
        - float: The maximum drawdown value.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")
        
        if len(returns) == 0:
            return np.array([0])
        
        try:
            drawdowns = RiskAnalysis.drawdown(returns)
            max_drawdown = np.min(drawdowns)  # Find the maximum drawdown
            return max_drawdown
        except Exception as e:
            raise Exception(f"Error calculating max drawdown : {e}")

    @staticmethod
    def annual_standard_deviation(returns:np.ndarray) -> float:
        """
        Calculate the annualized standard deviation of returns.

        This method calculates the annualized standard deviation of returns from a numpy array 
        of daily returns. It assumes 252 trading days in a year.

        Parameters:
        - returns (np.ndarray): A numpy array of daily returns.

        Returns:
        - float: The annualized standard deviation, rounded to four decimal places.
        """

        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")
        
        if len(returns) == 0:
            return np.array([0])
        
        try:
            daily_std_dev = np.std(returns, ddof=1)  # Calculate daily standard deviation
            annual_std_dev = round(daily_std_dev * np.sqrt(252), 4)  # Assuming 252 trading days in a year
            return np.around(annual_std_dev, decimals=4)
        except Exception as e:
            raise Exception(f"Error calculating annualized standard deviation : {e}")
        
    @staticmethod
    def sharpe_ratio(returns:np.ndarray, risk_free_rate:float=0.04) -> float:
        """
        Calculate the Sharpe ratio of the strategy.

        The Sharpe ratio measures the performance of an investment compared to a risk-free asset, 
        after adjusting for its risk. The ratio is the average return earned in excess of the risk-free 
        rate per unit of volatility or total risk.

        Parameters:
        - returns (np.ndarray): A 1D array of returns.
        - risk_free_rate (float): The risk-free rate. Default is 0.04 (4% annually).

        Returns:
        - float: The Sharpe ratio, rounded to four decimal places.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")
        
        if len(returns) == 0:
            return np.array([0])
            
        try:
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1) * np.sqrt(252)
            return np.around(sharpe_ratio, decimals=4) if excess_returns.std(ddof=1) != 0 else 0
        except Exception as e:
            raise Exception(f"Error calculating sharpe ratio : {e}")
        
    @staticmethod
    def sortino_ratio(returns:np.ndarray, target_return:float=0) -> float:
        """
        Calculate the Sortino Ratio for a given returns array.

        The Sortino ratio differentiates harmful volatility from total overall volatility 
        by using the asset's standard deviation of negative returns, called downside deviation. 
        It measures the risk-adjusted return of an investment asset, portfolio, or strategy.

        Parameters:
        - returns (np.ndarray): A 1D array of returns.
        - target_return (float): The target return. Default is 0.

        Returns:
        - float: The Sortino ratio, rounded to four decimal places.
        """
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be a numpy array")
        try:
            negative_returns = returns[returns < target_return]
            expected_return = returns.mean() - target_return
            downside_deviation = negative_returns.std(ddof=1)
            
            if downside_deviation > 0:
                return np.around(expected_return / downside_deviation, decimals=4)
            return 0.0
        except Exception as e:
            raise Exception(f"Error calculating sortino ratio : {e}")

    @staticmethod
    def value_at_risk(returns:np.ndarray, confidence_level:float=0.05) -> float:
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
    def conditional_value_at_risk(returns:np.ndarray, confidence_level:float=0.05) -> float:
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

        var = RiskAnalysis.value_at_risk(returns, confidence_level)
        tail_losses = returns[returns <= var]
        cvar = tail_losses.mean()
        return cvar

    @staticmethod
    def calculate_volatility_and_zscore_annualized(returns:np.ndarray) -> dict:
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
                "Z-Scores (Annualized)": {}
            }

        try:
            daily_volatility = returns.std()
            daily_mean_return = returns.mean()
            
            # Annualizing the daily volatility and mean return
            annualized_volatility = daily_volatility * np.sqrt(252)
            annualized_mean_return = daily_mean_return * 252
            
            # Adjusting the calculation of z-scores for annualized values
            z_scores_annualized = {f"Z-score for {x} SD move (annualized)": (annualized_mean_return - x * annualized_volatility) / annualized_volatility for x in range(1, 4)}
            return {
                "Annualized Volatility": annualized_volatility,
                "Annualized Mean Return": annualized_mean_return,
                "Z-Scores (Annualized)": z_scores_annualized
            }
        except Exception as e:
            raise Exception(f"Error calculating annualized volatility and z-scores : {e}")
    