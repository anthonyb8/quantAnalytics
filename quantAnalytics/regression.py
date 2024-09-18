import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from quantAnalytics.statistics import Result
from quantAnalytics.report import ReportBuilder, Header
from quantAnalytics.visualization import Visualization
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


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
        self,
        data: pd.DataFrame,
        dependent_var: str,
        risk_free_rate: float = 0.01,
        file_name: str = "regression.html",
        output_directory: str = "report",
        css_path: str = "",
    ):
        """
        Initialize the RegressionAnalysis with strategy and benchmark returns.

        Parameters:
        - data (pd.DataFrame): DataFrame containing both independent and dependent variables.
        - dependent_var (str): Name of the column representing the dependent variable (Y).
        - risk_free_rate (float, optional): The risk-free rate used in calculations, default is 0.01.
        - file_name (str, optional): Name of the output file for the regression report.
        - output_directory (str, optional): Directory to save the regression report.
        - css_path (str, optional): Path to a custom CSS file for styling the report.
        """

        # Ensure the inputs are valid
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if dependent_var not in data.columns:
            raise ValueError(
                f"Dependent variable '{dependent_var}' not found in the data"
            )

        # Store variables for later use
        self.output_dir = output_directory
        self.data = data
        self.dependent_var = dependent_var
        self.independent_vars = [
            col for col in data.columns if col != dependent_var
        ]
        self.risk_free_rate = risk_free_rate

        # Set up report builder (assuming you have a ReportBuilder class)
        self.report = ReportBuilder(file_name, output_directory, css_path)
        self.model: sm.OLS = None

    def run(self, cv_splits=5, report_file="regression_report.html"):
        """
        Perform cross-validation, fit models, evaluate them, and generate a report.

        Parameters:
        - cv_splits (int): Number of splits for cross-validation.
        - report_file (str): Path to save the final report.
        """
        self.report.add_header("Regression Analysis", Header.H3)

        kf = KFold(n_splits=cv_splits)
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.data), 1):
            print(f"Running fold {fold}/{cv_splits}...")

            # Split the data into training and test sets
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            # Train the model on the training set
            summary = self.fit(train_data)

            # Evaluate the model on the test set
            eval_report = self.evaluate(test_data)

            # Perform risk decomposition and performance attribution
            risk_report = self.risk_decomposition(test_data)
            performance_report = self.performance_attribution(test_data)

            # Plot data
            residuals = (
                train_data[self.dependent_var] - self.model.fittedvalues
            )

            influence = self.model.get_influence()
            cooks_dist = influence.cooks_distance[0]

            # Plots
            fitted_plot_path = (
                f"{self.output_dir}/residuals_v_fitted_{fold}.png"
            )
            Visualization.plot_residuals_vs_fitted(
                residuals=residuals,
                fittedvalues=self.model.fittedvalues,
                save_path=fitted_plot_path,
            )
            qq_plot = f"{self.output_dir}/qq_plot_{fold}.png"
            Visualization.qq_plot(residuals, save_path=qq_plot)

            influence_plot = f"{self.output_dir}/influence_plot_{fold}.png"
            Visualization.plot_influence_measures(
                cooks_dist, save_path=influence_plot
            )

            # Combine all reports
            self.report.add_header(f"Fold: {fold}", Header.H3)
            self.report.add_html_block(summary)
            self.report.add_html_block(eval_report.to_html())
            self.report.add_header("Risk & Performance", Header.H4)
            self.report.add_unorderedlist_dict(risk_report)
            self.report.add_unorderedlist_dict(performance_report)
            self.report.add_image(fitted_plot_path)
            self.report.add_image(qq_plot)
            self.report.add_image(influence_plot)

        self.report.build()

    def fit(self, train_data: pd.DataFrame) -> str:
        """
        Run the regression analysis with the dependent variable against all independent variables.
        """
        X = sm.add_constant(train_data[self.independent_vars])
        y = train_data[self.dependent_var]

        # Fit the OLS model
        model = sm.OLS(y, X).fit()

        # Save the model result
        self.model = model
        return model.summary().as_html()

    def evaluate(
        self,
        test_data: pd.DataFrame,
        r_squared_threshold: float = 0.3,
        p_value_threshold: float = 0.05,
    ) -> Result:
        """
        Evaluate the model on the test data and validate it based on R-squared and p-values significance.
        This method uses the fitted regression model to predict the dependent variable values for the test set and
        calculates performance metrics. It also validates the model based on R-squared and p-values thresholds.
        """
        if not self.model:
            raise ValueError(
                "Regression analysis not performed yet. Please fit the model first."
            )

        # Predictions on test set
        X_test, y_predict, y_actual, residuals = self.get_predictions(
            test_data
        )

        # Model performance metrics
        model_performance = self.get_model_performance(
            r_squared_threshold, p_value_threshold, y_predict, y_actual
        )

        # Diagnostic checks
        diagnostic_check = self.get_diagnostic_checks(
            residuals, X_test, p_value_threshold
        )

        # Coefficients (Alpha, Beta, and significance)
        coefficients = self.get_coefficients(p_value_threshold)

        # Final validation of the model
        model_validity = self.validate_model(
            p_value_threshold,
            model_performance["R-squared"]["value"],
            diagnostic_check,
        )

        # Build the evaluation report
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

    def get_predictions(self, test_data: pd.DataFrame):
        """Get predictions from the test set and calculate residuals."""
        X = sm.add_constant(test_data[self.independent_vars])
        y_actual = test_data[self.dependent_var]

        y_predict = self.model.predict(X)
        residuals = y_actual - y_predict
        return X, y_predict, y_actual, residuals

    def get_model_performance(
        self, r_squared_threshold, p_value_threshold, y_predict, y_actual
    ):
        """Calculate the R-squared, RMSE, MAE, and other performance metrics."""
        r_squared = self.model.rsquared
        adj_r_squared = self.model.rsquared_adj
        rmse = root_mean_squared_error(y_actual, y_predict)
        mae = mean_absolute_error(y_actual, y_predict)
        f_statistic = self.model.fvalue
        f_pvalue = self.model.f_pvalue

        return {
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
            },  # Always include RMSE
            "MAE": {"value": mae, "significant": True},  # Always include MAE
            "F-statistic": {
                "value": f_statistic,
                "significant": f_pvalue < p_value_threshold,
            },
            "F-statistic p-value": {
                "value": f_pvalue,
                "significant": f_pvalue < p_value_threshold,
            },
        }

    def check_collinearity(self, X_test: pd.DataFrame):
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
            # Calculating VIF for each feature
            vif_data = {
                X_test.columns[i]: variance_inflation_factor(X_test.values, i)
                for i in range(X_test.shape[1])
            }
            return vif_data
        except Exception as e:
            raise Exception(f"Error calculating collinearity : {e}")

    def get_diagnostic_checks(self, residuals, X_test, p_value_threshold):
        """Run diagnostic checks such as Durbin-Watson, Jarque-Bera, and VIF."""
        dw_stat = sm.stats.durbin_watson(residuals)
        jb_stat, jb_pvalue, _, _ = sm.stats.jarque_bera(residuals)
        condition_number = np.linalg.cond(X_test)
        vif = self.check_collinearity(X_test)

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

        return diagnostic_check

    def get_coefficients(self, p_value_threshold):
        """Extract and evaluate the coefficients (Alpha, Beta, and p-values)."""
        coefficients = self.model.params
        p_values = self.model.pvalues
        alpha = coefficients["const"]
        alpha_p_value = p_values["const"]
        beta = coefficients.drop("const").to_dict()
        beta_p_value = p_values.drop("const").to_dict()

        coef = {
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
            coef[f"Beta ({key})"] = {
                "value": value,
                "significant": beta_p_value[key] < p_value_threshold,
            }
            coef[f"Beta ({key}) p-value"] = {
                "value": beta_p_value[key],
                "significant": beta_p_value[key] < p_value_threshold,
            }

        return coef

    def validate_model(self, p_value_threshold, r_squared, diagnostic_check):
        """Validate the model based on significance of coefficients, p-values, and diagnostic checks."""
        return (
            self.model.pvalues["const"] < p_value_threshold
            and all(self.model.pvalues.drop("const") < p_value_threshold)
            and self.model.f_pvalue < p_value_threshold
            and r_squared
            > 0.3  # You can use r_squared_threshold here if necessary
            and diagnostic_check["Durbin-Watson"]["significant"]
            and diagnostic_check["Jarque-Bera"]["significant"]
            and diagnostic_check["Condition Number"]["significant"]
        )

    # Measure Risk
    def risk_decomposition(self, data: pd.DataFrame) -> dict:
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
            raise ValueError("Model not fitted yet. Fit the model first.")

        X = sm.add_constant(data[self.independent_vars])
        y = data[self.dependent_var]

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

    def performance_attribution(self, data: pd.DataFrame) -> dict:
        """
        Attribute performance into idiosyncratic, market, and total.

        This method calculates and returns the contributions of market and idiosyncratic factors
        to the overall performance of the strategy.

        Returns:
        - dict: A dictionary containing market contribution, idiosyncratic contribution, and total contribution.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Fit the model first.")

        X = sm.add_constant(data[self.independent_vars])
        y = data[self.dependent_var]

        # Calculate total return of the dependent variable (y)
        total_return = y.mean()

        # Calculate alpha (intercept)
        alpha = self.model.params["const"]

        # Calculate systematic return based on the number of predictors
        if len(X.columns) > 1:  # Multi-factor model
            systematic_return = sum(
                self.model.params.iloc[i] * X.iloc[:, i].mean()
                for i in range(len(X.columns))
            )
        else:  # Single-factor model
            beta = self.model.params.iloc[1]
            systematic_return = beta * X.iloc[:, 0].mean()

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
