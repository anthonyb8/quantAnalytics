import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

from quantAnalytics.statistics import TimeseriesTests

class RegressionAnalysis:
    def __init__(self, y:pd.Series, X:pd.DataFrame, risk_free_rate:float=0.01):
        """
        Initialize the RegressionAnalysis with strategy and benchmark returns.

        Parameters:
        - y (pd.DataFrame): DataFrame containing strategy equity values with a 'timestamp' and 'equity_value' columns.
        - X (pd.DataFrame): DataFrame containing benchmark equity values with a 'timestamp' and 'close' columns.
        - risk_free_rate (float, optional): The risk-free rate used in calculations, default is 0.01.
        """
        # Ensure the inputs are DataFrames
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError("y must be a pandas Series or DataFrame")
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        if len(y) != len(X):
            raise ValueError(f"Independent(X) and dependent(y) variables must be the same length.")

        self.X = X
        self.y = y
        self.X_train, self.X_test = TimeseriesTests.split_data(X, train_ratio=0.7)
        self.y_train, self.y_test = TimeseriesTests.split_data(y, train_ratio=0.7)
        self.model : sm.OLS = None

    # Regression Model
    def fit(self) -> RegressionResultsWrapper:
        """
        Perform a linear regression between a single dependent and independent variable(s).
        It returns the summary of the regression analysis.

        Returns:
        - statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted regression model summary.
        """
        try:
            self.X_train = sm.add_constant(self.X_train)  # Add the intercept term
            self.model = sm.OLS(self.y_train, self.X_train).fit()
            return self.model.summary()
        except Exception as e:
            raise Exception(f"Error fitting OLS model : {e}")
        
    def predict(self, X_new:pd.DataFrame) -> pd.Series:
        if not isinstance(X_new, (pd.DataFrame, pd.Series)):
            raise TypeError("X must be a pandas DataFrame")
        
        try:
            X_new =sm.add_constant(X_new)
            X_new_pred = self.model.predict(X_new)
            return X_new_pred
        except Exception as e:
            raise Exception(f"Error occured while making predictions : {e}")
    
    def evaluate(self, r_squared_threshold: float = 0.02, p_value_threshold: float = 0.05):
        """
        Evaluate the model on the test data and validate it based on R-squared and p-values significance.

        This method uses the fitted regression model to predict the dependent variable values for the test set and
        calculates performance metrics. It also validates the model based on R-squared and p-values thresholds.

        Parameters:
        - r_squared_threshold (float, optional): The threshold for R-squared value. Default is 0.02.
        - p_value_threshold (float, optional): The threshold for p-values. Default is 0.05.

        Returns:
        - dict: A dictionary containing the R-squared, RMSE, MAE of the model on the test set, 
                and validation checks for R-squared and p-values.
        """
        if not self.model:
            raise ValueError("Regression analysis not performed yet. Please fit the model first.")

        # Predictions on test set
        X_test_with_const = sm.add_constant(self.X_test)
        y_pred_test = self.model.predict(X_test_with_const)
        
        # Calculate performance metrics
        r_squared = self.model.rsquared
        adj_r_squared =self.model.rsquared_adj
        rmse = mean_squared_error(self.y_test, y_pred_test, squared=False)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        
        # Diagnostic checks
        residuals = self.y_test - y_pred_test
        dw_stat = sm.stats.durbin_watson(residuals)
        jb_stat, jb_pvalue, _, _= sm.stats.jarque_bera(residuals)
        condition_number = np.linalg.cond(X_test_with_const)
        
        # Collinearity checks
        collinearity_check = self.check_collinearity()

        # Significance checks
        p_values = self.model.pvalues
        significant_alpha = p_values['const'] < 0.05
        significant_betas = all(p_values.drop('const') < 0.05)

        # Coefficients
        coefficients = self.model.params
        alpha = coefficients['const']
        betas = coefficients.drop('const').to_dict()

        # Summary report
        report = {
            'Model Performance': {
                'R-squared': r_squared,
                'Adjusted R-squared': adj_r_squared,
                'RMSE': rmse,
                'MAE': mae
            },
            'Diagnostic Checks': {
                'Durbin-Watson': dw_stat,
                'Jarque-Bera': jb_stat,
                'Jarque-Bera p-value': jb_pvalue,
                'Condition Number': condition_number,
                'Collinearity Check': collinearity_check
            },
            'Significance': {
                'Alpha significant': significant_alpha,
                'Betas significant': significant_betas
            },
            'Coefficients': {
                'Alpha': alpha,
                'Beta' : betas
            },
            'Model Validity': significant_alpha and significant_betas and (r_squared > 0.3)
        }
        
        self._display_evaluate_results(report)
        return report
        
    def _display_evaluate_results(self, validation_results: dict, print_output: bool = True, to_html: bool = False, indent: int = 0) -> str:
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
        # Convert validation results to a dictionary for DataFrame
        data = {
            'R-squared': validation_results['Model Performance']['R-squared'],
            'Adjusted R-squared': validation_results['Model Performance']['Adjusted R-squared'],
            'RMSE': validation_results['Model Performance']['RMSE'],
            'MAE': validation_results['Model Performance']['MAE'],
            'Durbin-Watson': validation_results['Diagnostic Checks']['Durbin-Watson'],
            'Jarque-Bera': validation_results['Diagnostic Checks']['Jarque-Bera'],
            'Jarque-Bera p-value': validation_results['Diagnostic Checks']['Jarque-Bera p-value'],
            'Condition Number': validation_results['Diagnostic Checks']['Condition Number'],
        }

        # Ensure collinearity check results are added properly
        collinearity_check = validation_results['Diagnostic Checks']['Collinearity Check']
        for key, value in collinearity_check.items():
            data[f'VIF ({key})'] = value
        
        # Coefficients
        data['Alpha'] = validation_results['Coefficients']['Alpha']
        data['Alpha significant']= validation_results['Significance']['Alpha significant']

        betas = validation_results['Coefficients']['Beta']
        for key, value in betas.items():
            data[f'Beta ({key})'] = value
        data['Betas significant']= validation_results['Significance']['Betas significant']
        
        # Model Validity
        data['Model Validity'] =validation_results['Model Validity']

        # Convert the dictionary to a DataFrame and transpose it
        df = pd.DataFrame(data, index=[0]).T
        df.columns = ['Value']

        # Convert DataFrame to string
        report = df.to_string(header=True, index=True)

        # Add title and footer
        title = "Regression Validation Results"
        footer = "** R-squared should be above the threshold and p-values should be below the threshold for model validity."
        full_report = f"{title}\n{'=' * len(title)}\n{report}\n{footer}"

        # Print or return the report based on the parameters
        if print_output:
            print(full_report)
        return full_report
    
    # Checks
    def check_collinearity(self):
        """
        Calculate the Variance Inflation Factor (VIF) for each predictor variable in the regression model.

        This method helps to identify collinearity among predictor variables. High VIF values suggest that the model variables
        are highly correlated with each other, potentially distorting regression coefficients and p-values.

        Returns:
        - pd.DataFrame: A DataFrame with predictor variables and their corresponding VIF values.
        """
        if not self.model:
            raise ValueError("Model not fitted yet. Please fit the model before checking for collinearity.")
        try:
            X_with_const = sm.add_constant(self.X_test) # Adding a constant for the intercept
            # Calculating VIF for each feature
            vif_data = {X_with_const.columns[i]: variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])}
            return vif_data
        except Exception as e:
            raise  Exception(f"Error calculating collinearity : {e}")
        
    # Plots
    def plot_residuals(self):
        """
        Plot residuals against fitted values to diagnose the regression model.

        This method generates a scatter plot of residuals versus fitted values and includes a horizontal line at zero.
        It is used to check for non-random patterns in residuals which could indicate problems with the model such as 
        non-linearity, outliers, or heteroscedasticity.
        """
        if not self.model:
            raise ValueError("Model not fitted yet. Please fit the model before plotting residuals.")
        
        # Calculate residuals
        residuals = self.y_train - self.model.fittedvalues
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(self.model.fittedvalues, residuals, alpha=0.5, color='blue', edgecolor='k')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted Values')
        ax.grid(True)
        
        # Adjust layout to minimize white space
        plt.tight_layout()

        # Return the figure object
        return fig

    def plot_qq(self):
        """
        Generate a Q-Q plot to analyze the normality of residuals in the regression model.

        This method creates a Q-Q plot comparing the quantiles of the residuals to the quantiles of a normal distribution.
        This helps in diagnosing deviations from normality such as skewness and kurtosis.
        """
        if not self.model:
            raise ValueError("Model not fitted yet. Please fit the model before plotting Q-Q plot.")

        # Calculate residuals
        residuals = self.y_train- self.model.fittedvalues

        # Generate Q-Q plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        sm.qqplot(residuals, line='45', ax=ax, fit=True)
        ax.set_title('Q-Q Plot of Residuals')
        ax.grid(True)

        # Adjust layout to minimize white space
        plt.tight_layout()

        # Return the figure object
        return fig

    def plot_influence_measures(self):
        """
        Plot influence measures such as Cook's distance to identify influential cases in the regression model.

        This method plots the Cook's distance for each observation to help identify influential points that might
        affect the robustness of the regression model.
        """
        if not self.model:
            raise ValueError("Model not fitted yet. Please fit the model before assessing influence.")

        influence = self.model.get_influence()
        cooks_d = influence.cooks_distance[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
        ax.set_title('Cook\'s Distance Plot')
        ax.set_xlabel('Observation Index')
        ax.set_ylabel('Cook\'s Distance')
        
        # Adjust layout to minimize white space
        plt.tight_layout()

        # Return the figure object
        return fig

    # Measure Risk
    def risk_decomposition(self, data='all') -> dict:
        """
        Decompose risk into idiosyncratic, market, and total based on regression analysis.

        This method calculates and returns the market volatility, idiosyncratic volatility,
        and total volatility of the strategy based on the regression model.

        Parameters:
        - data (str): The dataset to use for calculation. Options are 'train', 'test', or 'all'.

        Returns:
        - dict: A dictionary containing market volatility, idiosyncratic volatility, and total volatility.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Please fit the model before performing risk decomposition.")

        if data == 'train':
            y = self.y_train
            X = self.X_train
        elif data == 'test':
            y = self.y_test
            X = self.X_test
        elif data == 'all':
            y = self.y
            X = self.X
        else:
            raise ValueError("Invalid data option. Choose from 'train', 'test', or 'all'.")

        # Calculate total variance of the dependent variable (y)
        total_variance = y.var()

        # Calculate systematic variance based on the number of predictors
        if len(X.columns) > 1:  # Multi-factor model (R^2 * VAR(Y))
            systematic_variance = self.model.rsquared * total_variance
        else:  # Single-factor model (Beta^2 * Var(X))
            beta = self.model.params.iloc[1]
            systematic_variance = beta**2 * X.iloc[:, 0].var()

        idiosyncratic_variance = total_variance - systematic_variance

        # Ensure idiosyncratic variance is non-negative
        if idiosyncratic_variance < 0:
            idiosyncratic_variance = 0
            
        return {
            "Total Volatility": np.sqrt(total_variance),
            "Systematic Volatility": np.sqrt(systematic_variance),
            "Idiosyncratic Volatility": np.sqrt(idiosyncratic_variance)
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
            raise ValueError("Model not fitted yet. Please fit the model before performing performance decomposition.")

        # Calculate total return of the dependent variable (y)
        total_return = self.y.mean()

        # Calculate alpha (intercept)
        alpha = self.model.params['const']

        # Calculate systematic return based on the number of predictors
        if len(self.X.columns) > 1:  # Multi-factor model
            systematic_return = sum(self.model.params.iloc[i+1] * self.X.iloc[:, i].mean() for i in range(len(self.X.columns)))
        else:  # Single-factor model
            beta = self.model.params.iloc[1]
            systematic_return = beta * self.X.iloc[:, 0].mean()

        # Calculate idiosyncratic return as total return minus systematic return
        idiosyncratic_return = total_return - systematic_return
        
        return {
            "Total Contribution": total_return,
            "Systematic Contribution": systematic_return,
            "Idiosyncratic Contribution": idiosyncratic_return,
            "Alpha Contribution": alpha,
            "Randomness": idiosyncratic_return - alpha
        }

    # def hedge_analysis(self) -> dict:
    #     """
    #     Calculate portfolio dollar beta, market hedge NMV (Net Market Value), and other portfolio metrics.

    #     This method performs a hedge analysis using the stored equity curve and returns the portfolio
    #     dollar beta, market hedge NMV, and the portfolio's beta relative to the market.

    #     Returns:
    #     - dict: A dictionary containing portfolio dollar beta, market hedge NMV, and beta.
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not fitted yet. Please fit the model before performing performance decomposition.")
        
    #     # Using the last value of the equity curve to represent the current portfolio value
    #     portfolio_value = self.equity_curve.iloc[-1]
        
    #     # Assuming self.model.params['beta'] exists and represents the portfolio's beta relative to the market
    #     beta = self.model.params.iloc[1]
        
    #     # Calculate portfolio dollar beta
    #     portfolio_dollar_beta = portfolio_value * beta
        
    #     # Market Hedge NMV calculation
    #     market_hedge_nmv = -portfolio_dollar_beta

    #     return {
    #         "Portfolio Dollar Beta": portfolio_dollar_beta,
    #         "Market Hedge NMV": market_hedge_nmv,
    #         "Beta": beta,
    #     }





# import numpy as np
# import pandas as pd
# from typing import List
# from decimal import Decimal
# import statsmodels.api as sm
# from statsmodels.regression.linear_model import RegressionResultsWrapper

# class RegressionAnalysis:
#     def __init__(self, strategy_values:pd.DataFrame, benchmark_values:pd.DataFrame, risk_free_rate:float=0.01):
#         """
#         Initialize the RegressionAnalysis with strategy and benchmark returns.

#         Parameters:
#         - strategy_values (pd.DataFrame): DataFrame containing strategy equity values with a 'timestamp' and 'equity_value' columns.
#         - benchmark_values (pd.DataFrame): DataFrame containing benchmark equity values with a 'timestamp' and 'close' columns.
#         - risk_free_rate (float, optional): The risk-free rate used in calculations, default is 0.01.
#         """

#         self.model = None
#         self.risk_free_rate = risk_free_rate
#         self.equity_curve = strategy_values['equity_value']
#         self.strategy_returns, self.benchmark_returns = self._prepare_and_align_data(strategy_values, benchmark_values)

#     # Data pre-processing
#     def _standardize_to_daily_values(self, data:pd.DataFrame, value_column:str) -> pd.DataFrame:
#         """
#         Convert input DataFrame to daily frequency based on the 'timestamp' column and calculate daily returns of the 'value_column'.

#         Parameters:
#         - data (pd.DataFrame): DataFrame containing equity values with a 'timestamp' column.
#         - value_column (str): The column name containing the values to be resampled and converted to returns.

#         Returns:
#         - pd.DataFrame: Series of daily returns.
#         """
#         # Ensure 'timestamp' column is datetime type and set as index
#         data['timestamp'] = pd.to_datetime(data['timestamp'])
#         data.set_index('timestamp', inplace=True)
        
#         # Resample to daily frequency, taking the last value of the day
#         daily_data = data.resample('D').last()

#         # Drop rows with any NaN values that might have resulted from resampling
#         daily_data.dropna(inplace=True)

#         # Calculate daily returns
#         daily_returns = daily_data[value_column].pct_change().dropna()

#         return daily_returns
    
#     def _prepare_and_align_data(self, strategy_curve:pd.DataFrame, benchmark_curve:pd.DataFrame) -> tuple:
#         """
#         Align strategy and benchmark data on a common date index and calculate returns.

#         Assumes 'equity_value' column in strategy_curve and 'close' column in benchmark_curve.

#         Parameters:
#         - strategy_curve (pd.DataFrame): DataFrame containing strategy equity values with a 'timestamp' and 'equity_value' columns.
#         - benchmark_curve (pd.DataFrame): DataFrame containing benchmark equity values with a 'timestamp' and 'close' columns.

#         Returns:
#         - tuple: Two Series containing aligned daily returns for strategy and benchmark.
#         """
#         strategy_returns = self._standardize_to_daily_values(strategy_curve, 'equity_value')
#         benchmark_returns = self._standardize_to_daily_values(benchmark_curve, 'close')

#         # Align the two datasets, keeping only the dates that exist in both
#         aligned_returns = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner').dropna()

#         return aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1]

#     # Regression Model
#     def perform_regression_analysis(self) -> RegressionResultsWrapper:
#         """
#         Perform a simple linear regression between strategy returns and benchmark returns.

#         This method fits a linear regression model where the strategy returns are regressed against the benchmark returns.
#         It returns the summary of the regression analysis.

#         Returns:
#         - statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted regression model summary.
#         """
#         X = sm.add_constant(self.benchmark_returns)
#         model = sm.OLS(self.strategy_returns, X).fit()
#         self.model = model
#         return model.summary()
    
#     def validate_model(self, r_squared_threshold:float=0.02, p_value_threshold:float=0.05) -> dict:
#         """
#         Validate the regression model based on R-squared and p-values significance.

#         This method checks if the R-squared value is above a given threshold and if the 
#         p-values of the model coefficients are below a given threshold. It returns a 
#         dictionary with the validation results.

#         Parameters:
#         - r_squared_threshold (float, optional): The threshold for R-squared value. Default is 0.02.
#         - p_value_threshold (float, optional): The threshold for p-values. Default is 0.05.

#         Returns:
#         - dict: A dictionary containing the R-squared value, p-values, and validation checks.
#               Returns None if the model is not fitted.
#         """
#         if not self.model:
#             print("Regression analysis not performed yet.")
#             return None

#         # Validation criteria checks
#         is_valid_r_squared = self.model.rsquared > r_squared_threshold
#         is_valid_p_values = all(p < p_value_threshold for p in self.model.pvalues[1:])

#         validation_results = {
#             "R-squared": self.model.rsquared,
#             "P-values": self.model.pvalues.to_dict(),
#             "Validation Checks": {
#                 "R-squared above threshold": is_valid_r_squared,
#                 "P-values significant": is_valid_p_values,
#                 "Model is valid": is_valid_r_squared and is_valid_p_values
#             }
#         }

#         return validation_results

#     @staticmethod
#     def display_regression_validation_results(validation_results:dict, print_output:bool=True, to_html:bool=False, indent:int=0) -> str:
#         """
#         Display the validation results of a regression model.

#         This function formats regression validation results into a human-readable format, either as a plain text table or an HTML table.
#         It includes an interpretation of R-squared and p-values to aid in understanding the model's validity.

#         Parameters:
#         - validation_results (dict): A dictionary containing the validation results of a regression model.
#         - print_output (bool): If True, the results are printed to the console. If False, the results are returned as a string.
#         - to_html (bool): If True, the results are formatted as an HTML string. If False, the results are formatted as plain text.
#         - indent (int): The number of indentation levels to apply to the HTML output (useful for nested HTML structures).

#         Returns:
#         - str: The formatted regression validation results as a string (plain text or HTML).
#         """
#         # Convert validation results to DataFrame
#         data = []
#         row = {'R-squared': validation_results['R-squared'], 'p-value (const)': validation_results['P-values']['const'], 'p-value (close)': validation_results['P-values']['close']}
#         row.update({'R-squared above threshold': validation_results['Validation Checks']['R-squared above threshold']})
#         row.update({'P-values significant': validation_results['Validation Checks']['P-values significant']})
#         row.update({'Model is valid': validation_results['Validation Checks']['Model is valid']})
#         data.append(row)
#         df = pd.DataFrame(data)

#         title = "Regression Validation Results"
#         footer = "** R-squared should be above the threshold and p-values should be below the threshold for model validity."
        
#         if to_html:
#             # Define the base indentation as a string of spaces
#             base_indent = "    " * indent
#             next_indent = "    " * (indent + 1)
            
#             # Convert DataFrame to HTML table and add explanation
#             html_table = df.to_html(index=False, border=1)
#             html_table_indented = "\n".join(next_indent + line for line in html_table.split("\n"))

#             html_title = f"{next_indent}<h4>{title}</h4>\n"
#             html_footer = f"{next_indent}<p class='footnote'>{footer}</p>\n"
#             html_output = f"{base_indent}<div class='regression_validation'>\n{html_title}{html_table}\n{html_footer}{base_indent}</div>"
            
#             return html_output
        
#         else:
#             output = (
#                 f"\n{title}\n"
#                 f"{'=' * len(title)}\n"
#                 f"{df.to_string(index=False)}\n"
#                 f"{footer}"
#             )
#             if print_output:
#                 print(output)
#             else:
#                 return output

#     # Regression Analysis
#     def beta(self) -> float:
#         """
#         Calculate the beta of the strategy based on aligned strategy and benchmark returns.

#         This method calculates the beta of the strategy, which is the coefficient of the 
#         benchmark returns in the regression model.

#         Returns:
#         - float: The beta value, rounded to four decimal places.
#         """
#         if self.model is None:
#             raise ValueError("Regression model not fitted. Call perform_regression_analysis first.")

#         # The beta is the coefficient of the benchmark returns in the regression model.
#         beta_value = self.model.params.iloc[1]  # Assuming 'params[1]' is the beta coefficient
#         return round(beta_value, 4)

#     def alpha(self) -> float:
#         """
#         Calculate the alpha of the strategy based on aligned strategy and benchmark returns.

#         This method calculates the alpha of the strategy, which is the excess return of the 
#         strategy over the expected return based on the benchmark's performance.

#         Returns:
#         - float: The alpha value, rounded to four decimal places.
#         """
#         if self.model is None:
#             raise ValueError("Regression model not fitted. Call perform_regression_analysis first.")
        
#         # Annualizing the returns
#         annualized_strategy_return = np.mean(self.strategy_returns) * 252
#         annualized_benchmark_return = np.mean(self.benchmark_returns) * 252
        
#         beta_value = self.beta()
        
#         # The alpha can be calculated as the intercept in the regression model.
#         alpha_value = annualized_strategy_return - (self.risk_free_rate + beta_value * (annualized_benchmark_return - self.risk_free_rate))
        
#         return round(alpha_value, 4)
    
#     def analyze_alpha(self, p_value_threshold:float=0.05) -> float:
#         """
#         Analyze the significance of alpha (intercept) in the regression model.

#         This method checks if the alpha (intercept) of the regression model is statistically 
#         significant based on its p-value and confidence interval.

#         Parameters:
#         - p_value_threshold (float, optional): The threshold for p-values. Default is 0.05.

#         Returns:
#         - dict: A dictionary containing the alpha value, its p-value, confidence interval, 
#               and significance checks. Returns None if the model is not fitted.
#         """
#         if not self.model:
#             print("Regression analysis not performed yet.")
#             return None

#         # Extract alpha (intercept) p-value and confidence interval
#         p_value_alpha = self.model.pvalues['const']
#         conf_interval_alpha = self.model.conf_int().loc['const'].values

#         # Assess significance
#         is_alpha_significant = p_value_alpha < p_value_threshold
#         does_alpha_span_zero = conf_interval_alpha[0] < 0 < conf_interval_alpha[1]

#         alpha_analysis_results = {
#             "Alpha (Intercept)": self.model.params['const'],
#             "P-value": p_value_alpha,
#             "Confidence Interval": conf_interval_alpha.tolist(),
#             "Alpha is significant": is_alpha_significant,
#             "Confidence Interval spans zero": does_alpha_span_zero
#         }

#         return alpha_analysis_results
    
#     @staticmethod
#     def display_alpha_analysis_results(alpha_analysis_results:dict, print_output:bool=True, to_html:bool=False, indent:int=0) -> str:
#         """
#         Display the alpha analysis results of a regression model.

#         This function formats the alpha analysis results into a human-readable format, either as a plain text table or an HTML table.
#         It provides insights on the alpha (intercept) significance, its confidence interval, and a brief interpretation of model validity based on p-values.

#         Parameters:
#         - alpha_analysis_results (dict): A dictionary containing the alpha analysis results of a regression model.
#         - print_output (bool): If True, the results are printed to the console. If False, the results are returned as a string.
#         - to_html (bool): If True, the results are formatted as an HTML string. If False, they are formatted as plain text.
#         - indent (int): The number of indentation levels to apply to the HTML output (useful for nested HTML structures).

#         Returns:
#         - str: The formatted alpha analysis results as a string (plain text or HTML).
#         """
#         # Convert validation results to DataFrame
#         data = []
#         row = {'Alpha (Intercept)': alpha_analysis_results['Alpha (Intercept)'], 'p-value': round(alpha_analysis_results['P-value'],6)}
#         row.update({'Confidence Interval Lower Bound(2.5%)':alpha_analysis_results["Confidence Interval"][0]})
#         row.update({'Confidence Interval Upper Bound(97.5%)':alpha_analysis_results["Confidence Interval"][1]})
#         row.update({'Alpha is significant': alpha_analysis_results['Alpha is significant']})
#         data.append(row)

#         df = pd.DataFrame(data)

#         title = "Alpha Analysis Results"
#         footer = "** Note: For model validity, alpha should be significant (p-value < 0.05), and confidence intervals should not include zero."
#         if to_html:
#             # Define the base indentation as a string of spaces
#             base_indent = "    " * indent
#             next_indent = "    " * (indent + 1)
            
#             # Convert DataFrame to HTML table and add explanation
#             html_table = df.to_html(index=False, border=1)
#             html_table_indented = "\n".join(next_indent + line for line in html_table.split("\n"))

#             html_title = f"{next_indent}<h4>{title}</h4>\n"
#             html_footer = f"{next_indent}<p class='footnote'>{footer}</p>\n"
#             html_output = f"{base_indent}<div class='alpha_analysis'>\n{html_title}{html_table}\n{html_footer}{base_indent}</div>"
            
#             return html_output
        
#         else:
#             output = (
#                 f"\n{title}\n"
#                 f"{'=' * len(title)}\n"
#                 f"{df.to_string(index=False)}\n"
#                 f"{footer}"
#             )
#             if print_output:
#                 print(output)
#             else:
#                 return output

#     def analyze_beta(self, p_value_threshold:float=0.05) -> float:
#         """
#         Analyze the significance of beta (slope) in the regression model.

#         This method checks if the beta (slope) of the regression model is statistically 
#         significant based on its p-value and confidence interval.

#         Parameters:
#         - p_value_threshold (float, optional): The threshold for p-values. Default is 0.05.

#         Returns:
#         - dict: A dictionary containing the beta value, its p-value, confidence interval, 
#               and significance checks. Returns None if the model is not fitted.
#         """
#         if not self.model:
#             print("Regression analysis not performed yet.")
#             return None

#         # Assuming beta is the first variable after the constant in the regression model
#         beta = self.model.params.iloc[1]
#         p_value_beta = self.model.pvalues.iloc[1]
#         conf_interval_beta = self.model.conf_int().iloc[1].values

#         # Assess significance
#         is_beta_significant = p_value_beta < p_value_threshold
#         does_beta_span_one = conf_interval_beta[0] < 1 < conf_interval_beta[1]

#         beta_analysis_results = {
#             "Beta (Slope)": beta,
#             "P-value": p_value_beta,
#             "Confidence Interval": conf_interval_beta.tolist(),
#             "Beta is significant": is_beta_significant,
#             "Confidence Interval spans one": does_beta_span_one
#         }

#         return beta_analysis_results
    
#     @staticmethod
#     def display_beta_analysis_results(beta_analysis_results:dict, print_output:bool=True, to_html:bool=False, indent:int=0) -> str:
#         """
#         Display the beta analysis results of a regression model.

#         This function formats the beta analysis results into a human-readable format, either as a plain text table or an HTML table.
#         It provides insights on the beta (slope) significance, its confidence intervals, and a brief interpretation of model validity based on p-values.

#         Parameters:
#         - beta_analysis_results (dict): A dictionary containing the beta analysis results of a regression model.
#         - print_output (bool): If True, the results are printed to the console. If False, the results are returned as a string.
#         - to_html (bool): If True, the results are formatted as an HTML string. If False, they are formatted as plain text.
#         - indent (int): The number of indentation levels to apply to the HTML output (useful for nested HTML structures).

#         Returns:
#         - str: The formatted beta analysis results as a string (plain text or HTML).
#         """
#         # Convert validation results to DataFrame
#         data = []
#         row = {'Beta (Slope)': beta_analysis_results['Beta (Slope)'], 'p-value': round(beta_analysis_results['P-value'],6)}
#         row.update({'Confidence Interval Lower Bound(2.5%)':beta_analysis_results["Confidence Interval"][0]})
#         row.update({'Confidence Interval Upper Bound(97.5%)':beta_analysis_results["Confidence Interval"][1]})
#         row.update({'Beta is significant': beta_analysis_results['Beta is significant']})
#         data.append(row)

#         df = pd.DataFrame(data)

#         title = "Beta Analysis Results"
#         footer = "** Note: For model validity, beta should be significant (p-value < 0.05), and confidence intervals should not include zero."
        
#         if to_html:
#             # Define the base indentation as a string of spaces
#             base_indent = "    " * indent
#             next_indent = "    " * (indent + 1)
            
#             # Convert DataFrame to HTML table and add explanation
#             html_table = df.to_html(index=False, border=1)
#             html_table_indented = "\n".join(next_indent + line for line in html_table.split("\n"))

#             html_title = f"{next_indent}<h4>{title}</h4>\n"
#             html_footer = f"{next_indent}<p class='footnote'>{footer}</p>\n"
#             html_output = f"{base_indent}<div class='beta_analysis'>\n{html_title}{html_table}\n{html_footer}{base_indent}</div>"
            
#             return html_output
        
#         else:
#             output = (
#                 f"\n{title}\n"
#                 f"{'=' * len(title)}\n"
#                 f"{df.to_string(index=False)}\n"
#                 f"{footer}"
#             )
#             if print_output:
#                 print(output)
#             else:
#                 return output

#     # Measure Risk
#     def risk_decomposition(self) -> dict:
#         """
#         Decompose risk into idiosyncratic, market, and total based on regression analysis.

#         This method calculates and returns the market volatility, idiosyncratic volatility,
#         and total volatility of the strategy based on the regression model.

#         Returns:
#         - dict: A dictionary containing market volatility, idiosyncratic volatility, and total volatility.
#         """
#         if self.model is None:
#             self.perform_regression_analysis()
        
#         beta = self.model.params.iloc[1]
#         market_volatility = self.benchmark_returns.std() * beta
#         idiosyncratic_volatility = self.model.resid.std()
#         total_volatility = np.sqrt(market_volatility**2 + idiosyncratic_volatility**2)
        
#         return {
#             "Market Volatility": market_volatility,
#             "Idiosyncratic Volatility": idiosyncratic_volatility,
#             "Total Volatility": total_volatility
#         }

#     def performance_attribution(self) -> dict:
#         """
#         Attribute performance into idiosyncratic, market, and total.

#         This method calculates and returns the contributions of market and idiosyncratic factors
#         to the overall performance of the strategy.

#         Returns:
#         - dict: A dictionary containing market contribution, idiosyncratic contribution, and total contribution.
#         """
#         if self.model is None:
#             self.perform_regression_analysis()
        
#         alpha = self.model.params.iloc[0]
#         beta = self.model.params.iloc[1]
#         market_contrib = beta * self.benchmark_returns.mean()
#         idiosyncratic_contrib = alpha
#         total_contrib = market_contrib + idiosyncratic_contrib
        
#         return {
#             "Market Contribution": market_contrib,
#             "Idiosyncratic Contribution": idiosyncratic_contrib,
#             "Total Contribution": total_contrib
#         }

#     def hedge_analysis(self) -> dict:
#         """
#         Calculate portfolio dollar beta, market hedge NMV (Net Market Value), and other portfolio metrics.

#         This method performs a hedge analysis using the stored equity curve and returns the portfolio
#         dollar beta, market hedge NMV, and the portfolio's beta relative to the market.

#         Returns:
#         - dict: A dictionary containing portfolio dollar beta, market hedge NMV, and beta.
#         """
#         if not self.model:
#             print("Regression analysis not performed yet.")
#             return None

#         # Using the last value of the equity curve to represent the current portfolio value
#         portfolio_value = self.equity_curve.iloc[-1]
        
#         # Assuming self.model.params['beta'] exists and represents the portfolio's beta relative to the market
#         beta = self.model.params.iloc[1]
        
#         # Calculate portfolio dollar beta
#         portfolio_dollar_beta = portfolio_value * beta
        
#         # Market Hedge NMV calculation
#         market_hedge_nmv = -portfolio_dollar_beta

#         return {
#             "Portfolio Dollar Beta": portfolio_dollar_beta,
#             "Market Hedge NMV": market_hedge_nmv,
#             "Beta": beta,
#         }

#     # Results
#     def compile_results(self) -> dict:
#         """
#         Compile the regression analysis results into a dictionary format suitable for input into a Django model.

#         This method compiles various metrics from the regression analysis, including R-squared, p-values,
#         alpha, beta, volatility, and contributions, and returns them in a dictionary format.

#         Returns:
#         - dict: A dictionary containing the compiled results of the regression analysis.
#         """
#         if self.model is None:
#             raise ValueError("Regression model not fitted. Call perform_regression_analysis first.")

#         # Assuming other methods have been called as needed to populate these attributes
#         results_dict = {
#             "r_squared": str(round(self.model.rsquared, 4)),
#             "p_value_alpha" :str(round(self.model.pvalues['const'], 4)),
#             "p_value_beta": str(round(self.model.pvalues.iloc[1], 4)),
#             "risk_free_rate": str(round(self.risk_free_rate,4)),
#             "alpha": str(round(self.alpha(), 4)),
#             "beta": str(round(self.beta(),4)),
#             # "annualized_return": str(round(self.calculate_sharpe_ratio(self.risk_free_rate)["annualized_return"], 4)),
#             "market_contribution": str(round(self.performance_attribution()["Market Contribution"], 4)),
#             "idiosyncratic_contribution": str(round(self.performance_attribution()["Idiosyncratic Contribution"],4)),
#             "total_contribution": str(round(self.performance_attribution()["Total Contribution"], 4)),
#             # "annualized_volatility": str(round(self.calculate_sharpe_ratio(self.risk_free_rate)["annualized_volatility"], 4)),
#             "market_volatility": str(round(self.risk_decomposition()["Market Volatility"], 4)),
#             "idiosyncratic_volatility": str(round(self.risk_decomposition()["Idiosyncratic Volatility"], 4)),
#             "total_volatility": str(round(self.risk_decomposition()["Total Volatility"], 4)),
#             "portfolio_dollar_beta": str(round(self.hedge_analysis()["Portfolio Dollar Beta"], 4)),
#             "market_hedge_nmv": str(round(self.hedge_analysis()["Market Hedge NMV"], 4))
#         }

#         return results_dict
