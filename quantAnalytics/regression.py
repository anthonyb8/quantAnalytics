import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

from quantAnalytics.statistics import TimeseriesTests, Result


class RegressionResult(Result):
    def __init__(self, data: dict):
        super().__init__("Regression Analysis", "", data)
        self.footer = "** R-squared should be above the threshold and p-values should be below the threshold for model validity."

    def _to_dataframe(self) -> pd.DataFrame:
        # Flatten results
        flattened_data = {}
        for section, metrics in self.data.items():
            for metric, values in metrics.items():
                flattened_data[f"{metric}"] = values

        # Convert the flattened dictionary to a DataFrame
        df = pd.DataFrame(flattened_data).T
        df = df.reset_index()
        df.columns = ["Field", "Value", "Significant"]
        return df


class RegressionAnalysis:
    def __init__(
        self, y: pd.Series, X: pd.DataFrame, risk_free_rate: float = 0.01
    ):
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
            raise ValueError(
                f"Independent(X) and dependent(y) variables must be the same length."
            )

        self.X = X
        self.y = y
        self.X_train, self.X_test = TimeseriesTests.split_data(
            X, train_ratio=0.7
        )
        self.y_train, self.y_test = TimeseriesTests.split_data(
            y, train_ratio=0.7
        )
        self.model: sm.OLS = None

    # Regression Model
    def fit(self) -> RegressionResultsWrapper:
        """
        Perform a linear regression between a single dependent and independent variable(s).
        It returns the summary of the regression analysis.

        Returns:
        - statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted regression model summary.
        """
        try:
            self.X_train = sm.add_constant(
                self.X_train
            )  # Add the intercept term
            self.model = sm.OLS(self.y_train, self.X_train).fit()
            return self.model.summary()
        except Exception as e:
            raise Exception(f"Error fitting OLS model : {e}")

    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        if not isinstance(X_new, (pd.DataFrame, pd.Series)):
            raise TypeError("X must be a pandas DataFrame")

        try:
            X_new = sm.add_constant(X_new)
            X_new_pred = self.model.predict(X_new)
            return X_new_pred
        except Exception as e:
            raise Exception(f"Error occured while making predictions : {e}")

    def evaluate(
        self, r_squared_threshold: float = 0.3, p_value_threshold: float = 0.05
    ) -> Result:
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
            raise ValueError(
                "Regression analysis not performed yet. Please fit the model first."
            )

        # Predictions on test set
        X_test_with_const = sm.add_constant(self.X_test)
        y_pred_test = self.model.predict(X_test_with_const)
        residuals = self.y_test - y_pred_test

        # Calculate performance metrics
        r_squared = self.model.rsquared
        adj_r_squared = self.model.rsquared_adj
        rmse = mean_squared_error(self.y_test, y_pred_test, squared=False)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        f_statistic = self.model.fvalue
        f_pvalue = self.model.f_pvalue

        model_performance = {
            "R-squared": {
                "value": r_squared,
                "significant": r_squared > r_squared_threshold,
            },
            "Adjusted R-squared": {
                "value": adj_r_squared,
                "significant": adj_r_squared > r_squared_threshold,
            },
            "RMSE": {
                "value": rmse,
                "significant": True,
            },  # No threshold for RMSE, always include value
            "MAE": {
                "value": mae,
                "significant": True,
            },  # No threshold for MAE, always include value
            "F-statistic": {
                "value": f_statistic,
                "significant": f_pvalue < p_value_threshold,
            },
            "F-statistic p-value": {
                "value": f_pvalue,
                "significant": f_pvalue < p_value_threshold,
            },
        }

        # Diagnostic checks
        dw_stat = sm.stats.durbin_watson(residuals)
        jb_stat, jb_pvalue, _, _ = sm.stats.jarque_bera(residuals)
        condition_number = np.linalg.cond(X_test_with_const)
        vif = self.check_collinearity()

        diagnostic_check = {
            "Durbin-Watson": {
                "value": dw_stat,
                "significant": 1.5 < dw_stat < 2.5,
            },
            "Jarque-Bera": {
                "value": jb_stat,
                "significant": jb_pvalue > p_value_threshold,
            },
            "Jarque-Bera p-value": {
                "value": jb_pvalue,
                "significant": jb_pvalue > p_value_threshold,
            },
            "Condition Number": {
                "value": condition_number,
                "significant": condition_number < 30,
            },
        }

        for key, value in vif.items():
            diagnostic_check[f"VIF ({key})"] = {
                "value": value,
                "significant": value < 10,
            }

        # Coefficients
        coefficients = self.model.params
        p_values = self.model.pvalues
        alpha = coefficients["const"]
        alpha_p_value = p_values["const"]
        beta = coefficients.drop("const").to_dict()
        beta_p_value = p_values.drop("const").to_dict()

        coefficients = {
            "Alpha": {
                "value": alpha,
                "significant": alpha_p_value < p_value_threshold,
            },
            "Alpha p-value": {
                "value": alpha_p_value,
                "significant": alpha_p_value < p_value_threshold,
            },
        }
        for key, value in beta.items():
            coefficients[f"Beta ({key})"] = {
                "value": value,
                "significant": beta_p_value[key] < p_value_threshold,
            }
            coefficients[f"Beta ({key}) p-value"] = {
                "value": beta_p_value[key],
                "significant": beta_p_value[key] < p_value_threshold,
            }

        # Validation
        model_validity = (
            p_values["const"] < p_value_threshold
            and all(p_values.drop("const") < p_value_threshold)
            and f_pvalue < p_value_threshold
            and r_squared > r_squared_threshold
            and 1.5 < dw_stat < 2.5
            and jb_pvalue > p_value_threshold
            and condition_number < 30
        )

        # Validation summary
        report = {
            "Model Performance": model_performance,
            "Diagnostic Checks": diagnostic_check,
            "Coefficients": coefficients,
            "Model Validity": {
                "Model Validity": {
                    "value": model_validity,
                    "significant": model_validity,
                }
            },
        }

        return RegressionResult(report)

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
            raise ValueError(
                "Model not fitted yet. Please fit the model before checking for collinearity."
            )
        try:
            X_with_const = sm.add_constant(
                self.X_test
            )  # Adding a constant for the intercept
            # Calculating VIF for each feature
            vif_data = {
                X_with_const.columns[i]: variance_inflation_factor(
                    X_with_const.values, i
                )
                for i in range(X_with_const.shape[1])
            }
            return vif_data
        except Exception as e:
            raise Exception(f"Error calculating collinearity : {e}")

    # Plots
    def plot_residuals(self):
        """
        Plot residuals against fitted values to diagnose the regression model.

        This method generates a scatter plot of residuals versus fitted values and includes a horizontal line at zero.
        It is used to check for non-random patterns in residuals which could indicate problems with the model such as
        non-linearity, outliers, or heteroscedasticity.
        """
        if not self.model:
            raise ValueError(
                "Model not fitted yet. Please fit the model before plotting residuals."
            )

        # Calculate residuals
        residuals = self.y_train - self.model.fittedvalues

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            self.model.fittedvalues,
            residuals,
            alpha=0.5,
            color="blue",
            edgecolor="k",
        )
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted Values")
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
            raise ValueError(
                "Model not fitted yet. Please fit the model before plotting Q-Q plot."
            )

        # Calculate residuals
        residuals = self.y_train - self.model.fittedvalues

        # Generate Q-Q plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        sm.qqplot(residuals, line="45", ax=ax, fit=True)
        ax.set_title("Q-Q Plot of Residuals")
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
            raise ValueError(
                "Model not fitted yet. Please fit the model before assessing influence."
            )

        influence = self.model.get_influence()
        cooks_d = influence.cooks_distance[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
        ax.set_title("Cook's Distance Plot")
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Cook's Distance")

        # Adjust layout to minimize white space
        plt.tight_layout()

        # Return the figure object
        return fig

    # Measure Risk
    def risk_decomposition(self, data="all") -> dict:
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
            raise ValueError(
                "Model not fitted yet. Please fit the model before performing risk decomposition."
            )

        if data == "train":
            y = self.y_train
            X = self.X_train
        elif data == "test":
            y = self.y_test
            X = self.X_test
        elif data == "all":
            y = self.y
            X = self.X
        else:
            raise ValueError(
                "Invalid data option. Choose from 'train', 'test', or 'all'."
            )

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
            "Idiosyncratic Volatility": np.sqrt(idiosyncratic_variance),
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
            raise ValueError(
                "Model not fitted yet. Please fit the model before performing performance decomposition."
            )

        # Calculate total return of the dependent variable (y)
        total_return = self.y.mean()

        # Calculate alpha (intercept)
        alpha = self.model.params["const"]

        # Calculate systematic return based on the number of predictors
        if len(self.X.columns) > 1:  # Multi-factor model
            systematic_return = sum(
                self.model.params.iloc[i + 1] * self.X.iloc[:, i].mean()
                for i in range(len(self.X.columns))
            )
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
            "Randomness": idiosyncratic_return - alpha,
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
