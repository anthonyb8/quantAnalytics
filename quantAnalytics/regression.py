
import numpy as np
import pandas as pd
from typing import List
from decimal import Decimal
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

class RegressionAnalysis:
    def __init__(self, strategy_values:pd.DataFrame, benchmark_values:pd.DataFrame, risk_free_rate:float=0.01):
        """
        Initialize the RegressionAnalysis with strategy and benchmark returns.

        Parameters:
        - strategy_values (pd.DataFrame): DataFrame containing strategy equity values with a 'timestamp' and 'equity_value' columns.
        - benchmark_values (pd.DataFrame): DataFrame containing benchmark equity values with a 'timestamp' and 'close' columns.
        - risk_free_rate (float, optional): The risk-free rate used in calculations, default is 0.01.
        """

        self.model = None
        self.risk_free_rate = risk_free_rate
        self.equity_curve = strategy_values['equity_value']
        self.strategy_returns, self.benchmark_returns = self._prepare_and_align_data(strategy_values, benchmark_values)

    # Data pre-processing
    def _standardize_to_daily_values(self, data:pd.DataFrame, value_column:str) -> pd.DataFrame:
        """
        Convert input DataFrame to daily frequency based on the 'timestamp' column and calculate daily returns of the 'value_column'.

        Parameters:
        - data (pd.DataFrame): DataFrame containing equity values with a 'timestamp' column.
        - value_column (str): The column name containing the values to be resampled and converted to returns.

        Returns:
        - pd.DataFrame: Series of daily returns.
        """
        # Ensure 'timestamp' column is datetime type and set as index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Resample to daily frequency, taking the last value of the day
        daily_data = data.resample('D').last()

        # Drop rows with any NaN values that might have resulted from resampling
        daily_data.dropna(inplace=True)

        # Calculate daily returns
        daily_returns = daily_data[value_column].pct_change().dropna()

        return daily_returns
    
    def _prepare_and_align_data(self, strategy_curve:pd.DataFrame, benchmark_curve:pd.DataFrame) -> tuple:
        """
        Align strategy and benchmark data on a common date index and calculate returns.

        Assumes 'equity_value' column in strategy_curve and 'close' column in benchmark_curve.

        Parameters:
        - strategy_curve (pd.DataFrame): DataFrame containing strategy equity values with a 'timestamp' and 'equity_value' columns.
        - benchmark_curve (pd.DataFrame): DataFrame containing benchmark equity values with a 'timestamp' and 'close' columns.

        Returns:
        - tuple: Two Series containing aligned daily returns for strategy and benchmark.
        """
        strategy_returns = self._standardize_to_daily_values(strategy_curve, 'equity_value')
        benchmark_returns = self._standardize_to_daily_values(benchmark_curve, 'close')

        # Align the two datasets, keeping only the dates that exist in both
        aligned_returns = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner').dropna()

        return aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1]

    # Regression Model
    def perform_regression_analysis(self) -> RegressionResultsWrapper:
        """
        Perform a simple linear regression between strategy returns and benchmark returns.

        This method fits a linear regression model where the strategy returns are regressed against the benchmark returns.
        It returns the summary of the regression analysis.

        Returns:
        - statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted regression model summary.
        """
        X = sm.add_constant(self.benchmark_returns)
        model = sm.OLS(self.strategy_returns, X).fit()
        self.model = model
        return model.summary()
    
    def validate_model(self, r_squared_threshold:float=0.02, p_value_threshold:float=0.05) -> dict:
        """
        Validate the regression model based on R-squared and p-values significance.

        This method checks if the R-squared value is above a given threshold and if the 
        p-values of the model coefficients are below a given threshold. It returns a 
        dictionary with the validation results.

        Parameters:
        - r_squared_threshold (float, optional): The threshold for R-squared value. Default is 0.02.
        - p_value_threshold (float, optional): The threshold for p-values. Default is 0.05.

        Returns:
        - dict: A dictionary containing the R-squared value, p-values, and validation checks.
              Returns None if the model is not fitted.
        """
        if not self.model:
            print("Regression analysis not performed yet.")
            return None

        # Validation criteria checks
        is_valid_r_squared = self.model.rsquared > r_squared_threshold
        is_valid_p_values = all(p < p_value_threshold for p in self.model.pvalues[1:])

        validation_results = {
            "R-squared": self.model.rsquared,
            "P-values": self.model.pvalues.to_dict(),
            "Validation Checks": {
                "R-squared above threshold": is_valid_r_squared,
                "P-values significant": is_valid_p_values,
                "Model is valid": is_valid_r_squared and is_valid_p_values
            }
        }

        return validation_results

    @staticmethod
    def display_regression_validation_results(validation_results:dict, print_output:bool=True, to_html:bool=False, indent:int=0) -> str:
        """
        Display the validation results of a regression model.

        This function formats regression validation results into a human-readable format, either as a plain text table or an HTML table.
        It includes an interpretation of R-squared and p-values to aid in understanding the model's validity.

        Parameters:
        - validation_results (dict): A dictionary containing the validation results of a regression model.
        - print_output (bool): If True, the results are printed to the console. If False, the results are returned as a string.
        - to_html (bool): If True, the results are formatted as an HTML string. If False, the results are formatted as plain text.
        - indent (int): The number of indentation levels to apply to the HTML output (useful for nested HTML structures).

        Returns:
        - str: The formatted regression validation results as a string (plain text or HTML).
        """
        # Convert validation results to DataFrame
        data = []
        row = {'R-squared': validation_results['R-squared'], 'p-value (const)': validation_results['P-values']['const'], 'p-value (close)': validation_results['P-values']['close']}
        row.update({'R-squared above threshold': validation_results['Validation Checks']['R-squared above threshold']})
        row.update({'P-values significant': validation_results['Validation Checks']['P-values significant']})
        row.update({'Model is valid': validation_results['Validation Checks']['Model is valid']})
        data.append(row)
        df = pd.DataFrame(data)

        title = "Regression Validation Results"
        footer = "** R-squared should be above the threshold and p-values should be below the threshold for model validity."
        
        if to_html:
            # Define the base indentation as a string of spaces
            base_indent = "    " * indent
            next_indent = "    " * (indent + 1)
            
            # Convert DataFrame to HTML table and add explanation
            html_table = df.to_html(index=False, border=1)
            html_table_indented = "\n".join(next_indent + line for line in html_table.split("\n"))

            html_title = f"{next_indent}<h4>{title}</h4>\n"
            html_footer = f"{next_indent}<p class='footnote'>{footer}</p>\n"
            html_output = f"{base_indent}<div class='regression_validation'>\n{html_title}{html_table}\n{html_footer}{base_indent}</div>"
            
            return html_output
        
        else:
            output = (
                f"\n{title}\n"
                f"{'=' * len(title)}\n"
                f"{df.to_string(index=False)}\n"
                f"{footer}"
            )
            if print_output:
                print(output)
            else:
                return output

    # Regression Analysis
    def beta(self) -> float:
        """
        Calculate the beta of the strategy based on aligned strategy and benchmark returns.

        This method calculates the beta of the strategy, which is the coefficient of the 
        benchmark returns in the regression model.

        Returns:
        - float: The beta value, rounded to four decimal places.
        """
        if self.model is None:
            raise ValueError("Regression model not fitted. Call perform_regression_analysis first.")

        # The beta is the coefficient of the benchmark returns in the regression model.
        beta_value = self.model.params.iloc[1]  # Assuming 'params[1]' is the beta coefficient
        return round(beta_value, 4)

    def alpha(self) -> float:
        """
        Calculate the alpha of the strategy based on aligned strategy and benchmark returns.

        This method calculates the alpha of the strategy, which is the excess return of the 
        strategy over the expected return based on the benchmark's performance.

        Returns:
        - float: The alpha value, rounded to four decimal places.
        """
        if self.model is None:
            raise ValueError("Regression model not fitted. Call perform_regression_analysis first.")
        
        # Annualizing the returns
        annualized_strategy_return = np.mean(self.strategy_returns) * 252
        annualized_benchmark_return = np.mean(self.benchmark_returns) * 252
        
        beta_value = self.beta()
        
        # The alpha can be calculated as the intercept in the regression model.
        alpha_value = annualized_strategy_return - (self.risk_free_rate + beta_value * (annualized_benchmark_return - self.risk_free_rate))
        
        return round(alpha_value, 4)
    
    def analyze_alpha(self, p_value_threshold:float=0.05) -> float:
        """
        Analyze the significance of alpha (intercept) in the regression model.

        This method checks if the alpha (intercept) of the regression model is statistically 
        significant based on its p-value and confidence interval.

        Parameters:
        - p_value_threshold (float, optional): The threshold for p-values. Default is 0.05.

        Returns:
        - dict: A dictionary containing the alpha value, its p-value, confidence interval, 
              and significance checks. Returns None if the model is not fitted.
        """
        if not self.model:
            print("Regression analysis not performed yet.")
            return None

        # Extract alpha (intercept) p-value and confidence interval
        p_value_alpha = self.model.pvalues['const']
        conf_interval_alpha = self.model.conf_int().loc['const'].values

        # Assess significance
        is_alpha_significant = p_value_alpha < p_value_threshold
        does_alpha_span_zero = conf_interval_alpha[0] < 0 < conf_interval_alpha[1]

        alpha_analysis_results = {
            "Alpha (Intercept)": self.model.params['const'],
            "P-value": p_value_alpha,
            "Confidence Interval": conf_interval_alpha.tolist(),
            "Alpha is significant": is_alpha_significant,
            "Confidence Interval spans zero": does_alpha_span_zero
        }

        return alpha_analysis_results
    
    @staticmethod
    def display_alpha_analysis_results(alpha_analysis_results:dict, print_output:bool=True, to_html:bool=False, indent:int=0) -> str:
        """
        Display the alpha analysis results of a regression model.

        This function formats the alpha analysis results into a human-readable format, either as a plain text table or an HTML table.
        It provides insights on the alpha (intercept) significance, its confidence interval, and a brief interpretation of model validity based on p-values.

        Parameters:
        - alpha_analysis_results (dict): A dictionary containing the alpha analysis results of a regression model.
        - print_output (bool): If True, the results are printed to the console. If False, the results are returned as a string.
        - to_html (bool): If True, the results are formatted as an HTML string. If False, they are formatted as plain text.
        - indent (int): The number of indentation levels to apply to the HTML output (useful for nested HTML structures).

        Returns:
        - str: The formatted alpha analysis results as a string (plain text or HTML).
        """
        # Convert validation results to DataFrame
        data = []
        row = {'Alpha (Intercept)': alpha_analysis_results['Alpha (Intercept)'], 'p-value': round(alpha_analysis_results['P-value'],6)}
        row.update({'Confidence Interval Lower Bound(2.5%)':alpha_analysis_results["Confidence Interval"][0]})
        row.update({'Confidence Interval Upper Bound(97.5%)':alpha_analysis_results["Confidence Interval"][1]})
        row.update({'Alpha is significant': alpha_analysis_results['Alpha is significant']})
        data.append(row)

        df = pd.DataFrame(data)

        title = "Alpha Analysis Results"
        footer = "** Note: For model validity, alpha should be significant (p-value < 0.05), and confidence intervals should not include zero."
        if to_html:
            # Define the base indentation as a string of spaces
            base_indent = "    " * indent
            next_indent = "    " * (indent + 1)
            
            # Convert DataFrame to HTML table and add explanation
            html_table = df.to_html(index=False, border=1)
            html_table_indented = "\n".join(next_indent + line for line in html_table.split("\n"))

            html_title = f"{next_indent}<h4>{title}</h4>\n"
            html_footer = f"{next_indent}<p class='footnote'>{footer}</p>\n"
            html_output = f"{base_indent}<div class='alpha_analysis'>\n{html_title}{html_table}\n{html_footer}{base_indent}</div>"
            
            return html_output
        
        else:
            output = (
                f"\n{title}\n"
                f"{'=' * len(title)}\n"
                f"{df.to_string(index=False)}\n"
                f"{footer}"
            )
            if print_output:
                print(output)
            else:
                return output

    def analyze_beta(self, p_value_threshold:float=0.05) -> float:
        """
        Analyze the significance of beta (slope) in the regression model.

        This method checks if the beta (slope) of the regression model is statistically 
        significant based on its p-value and confidence interval.

        Parameters:
        - p_value_threshold (float, optional): The threshold for p-values. Default is 0.05.

        Returns:
        - dict: A dictionary containing the beta value, its p-value, confidence interval, 
              and significance checks. Returns None if the model is not fitted.
        """
        if not self.model:
            print("Regression analysis not performed yet.")
            return None

        # Assuming beta is the first variable after the constant in the regression model
        beta = self.model.params.iloc[1]
        p_value_beta = self.model.pvalues.iloc[1]
        conf_interval_beta = self.model.conf_int().iloc[1].values

        # Assess significance
        is_beta_significant = p_value_beta < p_value_threshold
        does_beta_span_one = conf_interval_beta[0] < 1 < conf_interval_beta[1]

        beta_analysis_results = {
            "Beta (Slope)": beta,
            "P-value": p_value_beta,
            "Confidence Interval": conf_interval_beta.tolist(),
            "Beta is significant": is_beta_significant,
            "Confidence Interval spans one": does_beta_span_one
        }

        return beta_analysis_results
    
    @staticmethod
    def display_beta_analysis_results(beta_analysis_results:dict, print_output:bool=True, to_html:bool=False, indent:int=0) -> str:
        """
        Display the beta analysis results of a regression model.

        This function formats the beta analysis results into a human-readable format, either as a plain text table or an HTML table.
        It provides insights on the beta (slope) significance, its confidence intervals, and a brief interpretation of model validity based on p-values.

        Parameters:
        - beta_analysis_results (dict): A dictionary containing the beta analysis results of a regression model.
        - print_output (bool): If True, the results are printed to the console. If False, the results are returned as a string.
        - to_html (bool): If True, the results are formatted as an HTML string. If False, they are formatted as plain text.
        - indent (int): The number of indentation levels to apply to the HTML output (useful for nested HTML structures).

        Returns:
        - str: The formatted beta analysis results as a string (plain text or HTML).
        """
        # Convert validation results to DataFrame
        data = []
        row = {'Beta (Slope)': beta_analysis_results['Beta (Slope)'], 'p-value': round(beta_analysis_results['P-value'],6)}
        row.update({'Confidence Interval Lower Bound(2.5%)':beta_analysis_results["Confidence Interval"][0]})
        row.update({'Confidence Interval Upper Bound(97.5%)':beta_analysis_results["Confidence Interval"][1]})
        row.update({'Beta is significant': beta_analysis_results['Beta is significant']})
        data.append(row)

        df = pd.DataFrame(data)

        title = "Beta Analysis Results"
        footer = "** Note: For model validity, beta should be significant (p-value < 0.05), and confidence intervals should not include zero."
        
        if to_html:
            # Define the base indentation as a string of spaces
            base_indent = "    " * indent
            next_indent = "    " * (indent + 1)
            
            # Convert DataFrame to HTML table and add explanation
            html_table = df.to_html(index=False, border=1)
            html_table_indented = "\n".join(next_indent + line for line in html_table.split("\n"))

            html_title = f"{next_indent}<h4>{title}</h4>\n"
            html_footer = f"{next_indent}<p class='footnote'>{footer}</p>\n"
            html_output = f"{base_indent}<div class='beta_analysis'>\n{html_title}{html_table}\n{html_footer}{base_indent}</div>"
            
            return html_output
        
        else:
            output = (
                f"\n{title}\n"
                f"{'=' * len(title)}\n"
                f"{df.to_string(index=False)}\n"
                f"{footer}"
            )
            if print_output:
                print(output)
            else:
                return output

    # Measure Risk
    def risk_decomposition(self) -> dict:
        """
        Decompose risk into idiosyncratic, market, and total based on regression analysis.

        This method calculates and returns the market volatility, idiosyncratic volatility,
        and total volatility of the strategy based on the regression model.

        Returns:
        - dict: A dictionary containing market volatility, idiosyncratic volatility, and total volatility.
        """
        if self.model is None:
            self.perform_regression_analysis()
        
        beta = self.model.params.iloc[1]
        market_volatility = self.benchmark_returns.std() * beta
        idiosyncratic_volatility = self.model.resid.std()
        total_volatility = np.sqrt(market_volatility**2 + idiosyncratic_volatility**2)
        
        return {
            "Market Volatility": market_volatility,
            "Idiosyncratic Volatility": idiosyncratic_volatility,
            "Total Volatility": total_volatility
        }

    def performance_attribution(self) -> dict:
        """
        Attribute performance into idiosyncratic, market, and total.

        This method calculates and returns the contributions of market and idiosyncratic factors
        to the overall performance of the strategy.

        Returns:
        - dict: A dictionary containing market contribution, idiosyncratic contribution, and total contribution.
        """
        if self.model is None:
            self.perform_regression_analysis()
        
        alpha = self.model.params.iloc[0]
        beta = self.model.params.iloc[1]
        market_contrib = beta * self.benchmark_returns.mean()
        idiosyncratic_contrib = alpha
        total_contrib = market_contrib + idiosyncratic_contrib
        
        return {
            "Market Contribution": market_contrib,
            "Idiosyncratic Contribution": idiosyncratic_contrib,
            "Total Contribution": total_contrib
        }

    def hedge_analysis(self) -> dict:
        """
        Calculate portfolio dollar beta, market hedge NMV (Net Market Value), and other portfolio metrics.

        This method performs a hedge analysis using the stored equity curve and returns the portfolio
        dollar beta, market hedge NMV, and the portfolio's beta relative to the market.

        Returns:
        - dict: A dictionary containing portfolio dollar beta, market hedge NMV, and beta.
        """
        if not self.model:
            print("Regression analysis not performed yet.")
            return None

        # Using the last value of the equity curve to represent the current portfolio value
        portfolio_value = self.equity_curve.iloc[-1]
        
        # Assuming self.model.params['beta'] exists and represents the portfolio's beta relative to the market
        beta = self.model.params.iloc[1]
        
        # Calculate portfolio dollar beta
        portfolio_dollar_beta = portfolio_value * beta
        
        # Market Hedge NMV calculation
        market_hedge_nmv = -portfolio_dollar_beta

        return {
            "Portfolio Dollar Beta": portfolio_dollar_beta,
            "Market Hedge NMV": market_hedge_nmv,
            "Beta": beta,
        }

    # Results
    def compile_results(self) -> dict:
        """
        Compile the regression analysis results into a dictionary format suitable for input into a Django model.

        This method compiles various metrics from the regression analysis, including R-squared, p-values,
        alpha, beta, volatility, and contributions, and returns them in a dictionary format.

        Returns:
        - dict: A dictionary containing the compiled results of the regression analysis.
        """
        if self.model is None:
            raise ValueError("Regression model not fitted. Call perform_regression_analysis first.")

        # Assuming other methods have been called as needed to populate these attributes
        results_dict = {
            "r_squared": str(round(self.model.rsquared, 4)),
            "p_value_alpha" :str(round(self.model.pvalues['const'], 4)),
            "p_value_beta": str(round(self.model.pvalues.iloc[1], 4)),
            "risk_free_rate": str(round(self.risk_free_rate,4)),
            "alpha": str(round(self.alpha(), 4)),
            "beta": str(round(self.beta(),4)),
            # "annualized_return": str(round(self.calculate_sharpe_ratio(self.risk_free_rate)["annualized_return"], 4)),
            "market_contribution": str(round(self.performance_attribution()["Market Contribution"], 4)),
            "idiosyncratic_contribution": str(round(self.performance_attribution()["Idiosyncratic Contribution"],4)),
            "total_contribution": str(round(self.performance_attribution()["Total Contribution"], 4)),
            # "annualized_volatility": str(round(self.calculate_sharpe_ratio(self.risk_free_rate)["annualized_volatility"], 4)),
            "market_volatility": str(round(self.risk_decomposition()["Market Volatility"], 4)),
            "idiosyncratic_volatility": str(round(self.risk_decomposition()["Idiosyncratic Volatility"], 4)),
            "total_volatility": str(round(self.risk_decomposition()["Total Volatility"], 4)),
            "portfolio_dollar_beta": str(round(self.hedge_analysis()["Portfolio Dollar Beta"], 4)),
            "market_hedge_nmv": str(round(self.hedge_analysis()["Market Hedge NMV"], 4))
        }

        return results_dict
