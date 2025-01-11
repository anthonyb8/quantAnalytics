# ignore E501
import warnings
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from arch.unitroot import PhillipsPerron
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.tsa.vector_ar.vecm import (
    coint_johansen,
    VECM,
    select_coint_rank,
)
from scipy.stats import shapiro
from quant_analytics.result import Result

from .results import *

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", 100)
pd.set_option(
    "display.width", 1000
)  # Adjust the width of the display in characters
pd.set_option("display.max_rows", None)


class TimeseriesTests:
    """
    Easy use and display of the results from statsmodels tests, for timerseries analysis.
    """

    # -- Stationarity --
    @staticmethod
    def adf_test(
        series: pd.Series,
        series_name: str,
        trend: str = "c",
        confidence_interval: str = "5%",
        significance_level: float = 0.05,
    ) -> Result:
        """
        Perform the Augmented Dickey-Fuller (ADF) test to determine the stationarity of a time series.

        The null hypothesis (H0) of the ADF test posits that the time series has a unit root, indicating it is non-stationary.
        The alternative hypothesis (H1) suggests that the series is stationary.

        The test rejects the null hypothesis (and thus indicates stationarity) under two conditions:
        1. The ADF statistic is less than the critical value at the specified confidence interval.
        2. The p-value is less than the specified significance level, suggesting the result is statistically significant.

        Parameters:
        - series (pd.Series): The time series to be tested.
        - trend (str): Type of regression applied in the test. Options include:
            - 'c': Only a constant (default). Use if the series is expected to be stationary around a mean.
            - 'ct': Constant and trend. Use if the series is suspected to have a trend.
        - confidence_interval (str): Confidence interval for the critical values. Common options are '1%', '5%', and '10%'.
        - significance_level (float): The significance level for determining statistical significance. Default is 0.05.

        Returns:
        - dict: A dictionary with the ADF test statistic, p-value, critical values, and an indication of stationarity
        ('Stationary' or 'Non-Stationary') based on the test results.

        **Note: A low p-value (less than the specified significance level) combined with an ADF statistic lower than the
        critical value at the specified confidence interval suggests rejecting the null hypothesis in favor of stationarity.
        Conversely, a high p-value suggests failing to reject the null hypothesis, indicating non-stationarity.
        """

        result = adfuller(x=series, regression=trend)
        adf_statistic = result[0]
        p_value = result[1]
        lags = result[2]
        # num_observations = result[3]
        critical_values = result[4]

        # Prepare the results in a clean format
        output = {
            "Lags": lags,
            "ADF Statistic": adf_statistic,
            "p-value": p_value,
            "Critical Values": critical_values,
            "Stationarity": "",
        }

        # Determine stationarity based on the specified confidence interval and p-value
        stationarity = (
            adf_statistic < critical_values[confidence_interval]
            and p_value < significance_level
        )

        output["Stationarity"] = (
            "Stationary" if stationarity else "Non-Stationary"
        )
        return ADFResult(series_name, output)

    @staticmethod
    def rolling_adf(
        series: pd.Series, window: int, trend: str = "c"
    ) -> pd.Series:
        """
        Calculate rolling ADF statistics over a specified window.

        Paremeters:
        - series (pd.Series): The time series to be tested.
        - window (int): The rolling window size.
        - trend (str): Type of regression applied in the test. Options include:
            - 'c': Only a constant (default).
            - 'ct': Constant and trend.

        Returns:
        - pd.Series: Rolling ADF statistics.
        """
        # Convert numpy array to pandas Series if necessary
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        rolling_adf_results = series.rolling(window=window).apply(
            lambda x: (
                adfuller(x, regression=trend)[0]
                if len(x) == window
                else np.nan
            )
        )

        return rolling_adf_results

    @staticmethod
    def display_rolling_adf_results(
        series: pd.Series,
        rolling_adf_results: pd.Series,
        confidence_interval: str = "5%",
    ) -> plt.Figure:
        """
        Creates the plot of a rolling adf statistic verse the critical value for the original timeseries at the confidence interval passed.

        Parameters:
        - series (pd.Series): The original series.
        - rolling_adf_results (pd.Series): The results of the rolling_adf.
        - confidence_interval (str): Confidence interval for the critical values. Common options are '1%', '5%', and '10%'.

        Returns:
        -  plt.Figure: Figure representing the plot.

        Example:
        >>> TimeseriesTests.display_rolling_adf_results(pd.Series(series), rolling_adf_results)
        >>> plt.show()
        """
        fig, ax = plt.subplots()
        ax.plot(rolling_adf_results, label="Rolling ADF Statistic")
        ax.axhline(
            y=adfuller(series)[4][confidence_interval],
            color="r",
            linestyle="--",
            label=f"Critical Value @ {confidence_interval}",
        )
        ax.set_title("Rolling ADF Statistic")
        ax.set_xlabel("Time")
        ax.set_ylabel("ADF Statistic")
        ax.legend()
        return fig

    @staticmethod
    def kpss_test(
        series: pd.Series,
        series_name: str,
        trend: str = "c",
        confidence_interval: str = "5%",
    ) -> Result:
        """
        Perform the KPSS test to determine the stationarity of a time series.

        Null Hypothesis (H0): The series is stationary around a deterministic trend (level or trend stationarity).
        Alternative Hypothesis (H1): The series has a unit root (is non-stationary).

        Unlike other tests, a high p-value in the KPSS test suggests failure to reject the null hypothesis, indicating
        stationarity. Conversely, a low p-value suggests rejecting the null hypothesis in favor of the alternative,
        indicating non-stationarity.

        Parameters:
        - series (pd.Series): The time series to be tested.
        - trend (str): Type of regression applied in the test. Options include:
            - 'c': Only a constant. Use if the series is expected to be stationary around a mean.
            - 'ct': Constant and trend. Use if the series is suspected to have a trend and is stationary around a trend.
        - significance_level (float): The significance level for determining statistical significance. This is used to
        - adjust the interpretation of the test result but note that KPSS uses critical values directly for decision making.

        Returns:
        - dict: A dictionary with the KPSS test statistic, p-value, critical values, and an indication of stationarity
        ('Stationary' or 'Non-Stationary') based on the test results. Stationarity is determined by comparing the test
        statistic to critical values, not directly by p-value.

        **Note: KPSS test results are interpreted differently from tests like ADF. Here, stationarity is suggested by not
        rejecting the null hypothesis (high p-value), while evidence of non-stationarity comes from rejecting H0 (low p-value).
        """
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                kpss_statistic, p_value, n_lags, critical_values = kpss(
                    series, regression=trend
                )

                # Check for specific KPSS warning about the p-value being outside the lookup table range
                for warning in caught_warnings:
                    if "p-value is smaller than the p-value returned" in str(
                        warning.message
                    ):
                        p_value = 0.0  # Adjust p-value to indicate strong evidence of non-stationarity

        except ValueError as e:
            raise ValueError(f"KPSS test encountered an error: {e}")

        # Prepare the results in a structured format
        output = {
            "KPSS Statistic": kpss_statistic,
            "p-value": p_value,
            "Critical Values": critical_values,
            "Stationarity": "",
        }

        # Determine stationarity based on the 1% confidence interval
        stationarity = kpss_statistic < critical_values[confidence_interval]

        output["Stationarity"] = (
            "Stationary" if stationarity else "Non-Stationary"
        )
        return KPSSResult(series_name, output)

    @staticmethod
    def phillips_perron_test(
        series: pd.Series,
        series_name: str,
        trend: str = "c",
        confidence_interval: str = "5%",
        significance_level: float = 0.05,
    ) -> Result:
        """
        Perform the Phillips-Perron test to determine the stationarity of a time series.

        The null hypothesis (H0) of the Phillips-Perron (PP) test posits that the time series has a unit root,
        indicating it is non-stationary. The alternative hypothesis (H1) suggests that the series is stationary.

        The test rejects the null hypothesis (and thus indicates stationarity) under two conditions:
        1. The PP statistic is less than the critical value at the specified confidence interval.
        2. The p-value is less than the specified significance level, suggesting the result is statistically significant.

        Parameters:
        - series (pd.Series): The time series to be tested.
        - trend (str): Type of regression applied in the test. Options include:
            - 'c': Only a constant (default). Use if the series is expected to be stationary around a mean.
            - 'ct': Constant and trend. Use if the series is suspected to have a trend.
        - confidence_interval (str): Confidence interval for the critical values. Common options are '1%', '5%', and '10%'.
        - significance_level (float): The significance level for determining statistical significance. Default is 0.05.

        Returns:
        - dict: A dictionary with the PP test statistic, p-value, critical values, and an indication of stationarity
        ('Stationary' or 'Non-Stationary') based on the test results.

        **Note: A low p-value (less than the specified significance level) combined with a PP statistic lower than the
        critical value at the specified confidence interval suggests rejecting the null hypothesis in favor of stationarity.
        Conversely, a high p-value suggests failing to reject the null hypothesis, indicating non-stationarity.
        """
        pp_test = PhillipsPerron(series, trend=trend)
        pp_statistic = pp_test.stat
        p_value = pp_test.pvalue
        critical_values = pp_test.critical_values

        # Prepare the results in a clean format
        output = {
            "PP Statistic": pp_statistic,
            "p-value": p_value,
            "Critical Values": critical_values,
            "Stationarity": "",
        }

        # Determine stationarity based on the specified confidence interval and p-value
        stationarity = (
            pp_statistic < critical_values[confidence_interval]
            and p_value < significance_level
        )

        output["Stationarity"] = (
            "Stationary" if stationarity else "Non-Stationary"
        )

        return PPResult(series_name, output)

    @staticmethod
    def seasonal_adf_test(
        series: pd.Series,
        series_name: str,
        maxlag: int = None,
        regression: str = "c",
        seasonal_periods: int = 12,
        confidence_interval: str = "5%",
        significance_level: float = 0.05,
    ) -> Result:
        """
        Perform Seasonal Augmented Dickey-Fuller (ADF) test to assess the stationarity of a seasonal time series.

        The test decomposes the series into seasonal, trend, and residual components, and then performs the ADF test
        on the detrended series. This method helps in identifying whether the time series is stationary, taking into
        account its seasonality.

        Parameters:
        - series (pd.Series): Time series data.
        - maxlag (int, optional): Maximum number of lags to include in the ADF test. Defaults to None, letting the test choose the lag based on AIC.
        - regression (str, optional): Type of regression ('c' constant, 'ct' constant and trend, 'ctt' constant, trend, and trend squared, 'nc' no constant). Defaults to 'c'.
        - seasonal_periods (int, optional): Number of periods in a season. Defaults to 12.
        - confidence_interval (str, optional): The confidence interval used to determine stationarity. Defaults to '5%'.

        Returns:
        - dict: A dictionary with the test statistic ('ADF Statistic'), the p-value ('p-value'), the critical values
                ('Critical Values'), and an indication of stationarity ('Stationarity') based on the specified
                confidence interval.
        """
        if series.empty or len(series) < seasonal_periods:
            raise ValueError(
                "The time series must contain more observations than the number of seasonal periods."
            )

        # Seasonal decomposition
        dftest = sm.tsa.seasonal_decompose(
            series, model="additive", period=seasonal_periods
        )
        detrended = series - dftest.trend  # Remove the trend component

        # Perform the ADF test on the detrended series
        result = sm.tsa.adfuller(
            detrended.dropna(),
            maxlag=maxlag,
            regression=regression,
            autolag="AIC",
        )

        adf_statistic, p_value, _, _, critical_values, _ = result

        # Determine stationarity
        stationarity = (
            "Stationary"
            if adf_statistic < critical_values[confidence_interval]
            and p_value < significance_level
            else "Non-Stationary"
        )

        output = {
            "ADF Statistic": adf_statistic,
            "p-value": p_value,
            "Critical Values": critical_values,
            "Stationarity": stationarity,
        }

        return ADFResult(series_name, output)

    @staticmethod
    def monte_carlo_simulation(
        n_simulations: int,
        series_length: int,
        mean: float,
        std_dev: float,
        trend: str = "c",
        confidence_interval: str = "5%",
        significance_level: float = 0.05,
    ) -> Tuple[list, list]:
        """
        Perform Monte Carlo Simulation, generating a number of random timeseries and performing the augmented dickey fuller on them.

        Parameters:
        - n_simulations (int): Number of time series generated.
        - series_length (int): Length of each time series generated.
        - mean (float): Mean of each time series generated.
        - std_dev (float): Standard deviation of each time series generated.
        - trend (str): Type of regression applied in the test. Options include:
            - 'c': Only a constant (default). Use if the series is expected to be stationary around a mean.
            - 'ct': Constant and trend. Use if the series is suspected to have a trend.
        - confidence_interval (str): Confidence interval for the critical values. Common options are '1%', '5%', and '10%'.
        - significance_level (float): The significance level for determining statistical significance. Default is 0.05.

        """
        adf_stats = []
        p_values = []
        for _ in range(n_simulations):
            simulated_series = np.random.normal(mean, std_dev, series_length)
            adf_result = TimeseriesTests.adf_test(
                pd.Series(simulated_series),
                "",
                trend,
                confidence_interval,
                significance_level,
            )
            adf_stats.append(adf_result.data["ADF Statistic"])
            p_values.append(adf_result.data["p-value"])
        return adf_stats, p_values

    @staticmethod
    def display_monte_carlo_simulation(
        adf_stats: list,
        original_series: pd.Series,
        confidence_interval: str = "5%",
    ) -> plt.Figure:
        """
        Display the histogram of ADF statistics from Monte Carlo simulations and the ADF statistic of the original series.

        Parameters:
        - adf_stats (list): ADF statistics from Monte Carlo simulations.
        - original_series (pd.Series): The original time series to compare.
        - confidence_interval (str): Confidence interval for critical values.

        Returns:
        -  plt.Figure: Figure representing the plot.

        Example:
        >>> TimeseriesTests.display_monte_carlo_simulation(adf_stats, original_series)
        >>> plt.show()
        """
        adf_statistic_original = adfuller(original_series)[0]

        fig, ax = plt.subplots()
        ax.hist(adf_stats, bins=30, edgecolor="k", alpha=0.7)
        ax.axvline(
            x=adf_statistic_original,
            color="r",
            linestyle="--",
            label="ADF Statistic of Original Series",
        )
        ax.set_title("Distribution of ADF Statistics from Simulations")
        ax.set_xlabel("ADF Statistic")
        ax.set_ylabel("Frequency")
        ax.legend()
        return fig

    @staticmethod
    def fit_arima(
        series: pd.Series, order: tuple = (1, 0, 0)
    ) -> Tuple[ARIMAResultsWrapper, pd.Series]:
        """
        Fit an ARIMA model to the series.

        Parameters:
        - series (pd.Series): The time series data.
        - order (tuple): The (p, d, q) order of the ARIMA model.

        Returns:
        - model_fit (ARIMAResultsWrapper): The fitted ARIMA model.
        - residuals (pd.Series): Residuals of the fitted model.
        """
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        residuals = model_fit.resid
        return model_fit, residuals

    @staticmethod
    def plot_acf_pacf(residuals: pd.Series, lags: int = 40) -> plt.Figure:
        """
        Plot the ACF and PACF of residuals.

        Parameters:
        - residuals (pd.Series): Residuals of a time series model.
        - lags (int): Number of lags for ACF and PACF plots.

        Returns:
          -  plt.Figure: Figure representing the plot.

        Example:
        >>> TimeseriesTests.plot_acf_pacf(residuals, lags)
        >>> plt.show()
        """

        fig, axs = plt.subplots(1, 2, figsize=[15, 5])

        # Plot ACF
        sm.graphics.tsa.plot_acf(residuals, lags=lags, ax=axs[0])
        axs[0].set_title("ACF of Residuals")

        # Plot PACF
        sm.graphics.tsa.plot_pacf(residuals, lags=lags, ax=axs[1])
        axs[1].set_title("PACF of Residuals")

        # Adjust layout
        fig.tight_layout()

        # Return the figure object
        return fig

    # -- Cointegration --
    @staticmethod
    def johansen_test(
        data: pd.DataFrame, det_order: int = -1, k_ar_diff: int = 1
    ) -> Result:
        """
        Perform Johansen cointegration test to assess the presence of cointegration relationships
        among several time series.

        The Johansen test helps to determine the number of cointegrating relationships in a system
        of multiple time series.

        Parameters:
        - data (pd.DataFrame): A pandas DataFrame where each column represents a time series.
        - det_order (int): Specifies the deterministic trend in the data. The default value is -1,
                            which includes a constant term but no trend. 0 includes no constant or trend,
                            and 1 includes a constant and a linear trend.
        - k_ar_diff (int): The number of lagged differences used in the test. The default is 1.

        Returns:
        - dict: A dictionary containing the test's eigenvalues, critical values, statistics, and
                the cointegrating vector. It also includes analysis of trace and max eigenvalue statistics.

        Raises:
        - ValueError: If `data` is empty or not a pandas DataFrame.
        """
        if data.empty or not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Input data must be a non-empty pandas DataFrame."
            )

        # Perform the Johansen cointegration test
        result = coint_johansen(data, det_order, k_ar_diff)

        # Structured results
        output = {
            "Eigenvalues": result.eig,
            "Critical Values for Trace Statistic": result.cvt[:, 0],
            "Critical Values for Max Eigenvalue Statistic": result.cvt[:, 1],
            "Trace Statistics": result.lr1,
            "Max Eigenvalue Statistics": result.lr2,
            "Cointegrating Vector": result.evec.T.tolist(),  # Transpose to list for readability
        }
        # Analysis of Trace and Max Eigenvalue Statistics
        num_cointegrations = 0
        trace_analysis = []
        max_eig_analysis = []

        for idx, (trace_stat, max_eig_stat) in enumerate(
            zip(
                output["Trace Statistics"], output["Max Eigenvalue Statistics"]
            )
        ):
            trace_crit_value = output["Critical Values for Trace Statistic"][
                idx
            ]
            max_eig_crit_value = output[
                "Critical Values for Max Eigenvalue Statistic"
            ][idx]

            trace_decision = trace_stat > trace_crit_value
            max_eig_decision = max_eig_stat > max_eig_crit_value

            trace_analysis.append(
                f'Hypothesis {idx}: {"Reject" if trace_decision else "Fail to Reject"} the null hypothesis of no cointegration at this level.'
            )
            max_eig_analysis.append(
                f'Hypothesis {idx}: {"Reject" if max_eig_decision else "Fail to Reject"} the null hypothesis of no cointegration at this level.'
            )

            if trace_decision and max_eig_decision:
                num_cointegrations += 1

        return JohansenResult(output, num_cointegrations, k_ar_diff)

    @staticmethod
    def select_lag_length(
        data: pd.DataFrame, maxlags: int = 10, criterion: str = "bic"
    ) -> int:
        """
        Selects the optimal lag length for a time series dataset based on a specified information criterion.

        This method fits Vector AutoRegression (VAR) models with different lags up to a specified maximum.
        It evaluates each model based on the chosen information criterion (AIC, BIC, FPE, or HQIC)
        and selects the lag length that minimizes the criterion value.
        **Note: BIC will be more conservative and less overfitted than AIC

        Parameters:
        - data (pd.DataFrame): A pandas DataFrame containing the time series data.
        - maxlags (int, optional): The maximum number of lags to test. Defaults to 10.
        - criterion (str, optional): The information criterion to use for selecting the optimal lag.
                                    Options are 'aic' (default), 'bic', 'fpe', and 'hqic'.

        Returns:
        - int: The optimal number of lags according to the specified information criterion.

        Raises:
        - ValueError: If the input data is not a pandas DataFrame or is empty.
                        If the specified criterion is not supported.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Data must be a non-empty pandas DataFrame.")

        if criterion not in ["aic", "bic", "fpe", "hqic"]:
            raise ValueError(
                f"Unsupported criterion '{criterion}'. Choose from 'aic', 'bic', 'fpe', or 'hqic'."
            )

        # Initialize variables to store the best lag and its corresponding criterion value
        best_lag = 0
        best_criterion = float("inf")

        # Iterate over possible lag values
        for lag in range(1, maxlags + 1):
            model = VAR(data)
            result = model.fit(lag)

            # Retrieve the criterion value based on the user's choice
            criterion_value = getattr(result, criterion)

            # Update the best lag if this criterion value is the best so far
            if criterion_value < best_criterion:
                best_criterion = criterion_value
                best_lag = lag

        return best_lag

    @staticmethod
    def select_coint_rank(
        data: pd.DataFrame,
        k_ar_diff: int,
        method: str = "trace",
        signif: float = 0.05,
        det_order: int = -1,
    ) -> dict:
        """
        Selects the cointegration rank for a dataset using the Johansen cointegration test.

        Parameters:
        - data (pd.DataFrame): A pandas DataFrame containing the time series data.
        - k_ar_diff (int): The number of lags minus one to be used in the Johansen test.
        - method (str, optional): The test statistic to use ('trace' or 'maxeig'). Defaults to 'trace'.
        - signif (float, optional): Significance level for rejecting the null hypothesis. Defaults to 0.05.
        - det_order (int, optional): The order of the deterministic trend to include in the test.
                                    -1 for no deterministic term, 0 for constant term only, and 1 for constant with linear trend.
                                    Defaults to -1.

        Returns:
        - dict : A summary of the Johansen cointegration test results, including the selected cointegration rank based on the specified significance level.

        Raises:
        - ValueError: If `data` is not a pandas DataFrame or if other input parameters do not meet the required conditions.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if method not in ["trace", "maxeig"]:
            raise ValueError("Method must be either 'trace' or 'maxeig'.")
        if not (0 < signif < 1):
            raise ValueError("Signif must be between 0 and 1.")
        if det_order not in [-1, 0, 1]:
            raise ValueError("det_order must be -1, 0, or 1.")

        result = select_coint_rank(
            data,
            det_order=det_order,
            k_ar_diff=k_ar_diff,
            method=method,
            signif=signif,
        )

        # Format the results for clearer interpretation
        summary = {
            "Cointegration Rank": result.rank,
            "Test Statistic": result.test_stats,
            "Critical Value": result.crit_vals,
            "Significance Level": signif,
            "Method": method,
        }

        return summary

    # -- Autocorrelation --
    @staticmethod
    def durbin_watson(residuals: pd.DataFrame, ts_name: str) -> Result:
        """
        Perform the Durbin-Watson test to assess autocorrelation in the residuals of a regression model.

        The Durbin-Watson statistic ranges from 0 to 4, where:
        - A value of approximately 2 indicates no autocorrelation.
        - Values less than 2 suggest positive autocorrelation.
        - Values greater than 2 indicate negative autocorrelation.

        The closer the statistic is to 0, the stronger the evidence for positive serial correlation. Conversely,
        the closer to 4, the stronger the evidence for negative serial correlation.

        Parameters:
        - residuals (pd.DataFrame): A pandas DataFrame of residuals from a regression model, where each column represents
                                a different set of residuals (e.g., from multiple models or dependent variables).

        Returns:
        - dict: A dictionary with column names as keys and values as another dictionary containing the Durbin-Watson statistic
            and an interpretation of autocorrelation ('Positive', 'Negative', or 'Absent').
        """
        dw_stats = {}
        for col in residuals.columns:
            dw_stat = durbin_watson(residuals[col])
            # Interpret the Durbin-Watson statistic
            autocorrelation = "Absent"
            if dw_stat < 1.5:
                autocorrelation = "Positive"
            elif dw_stat > 2.5:
                autocorrelation = "Negative"

            dw_stats[col] = {
                "Durbin-Watson Statistic": dw_stat,
                "Autocorrelation": autocorrelation,
            }
        return DurbinWatsonResult(ts_name, dw_stats)

    @staticmethod
    def ljung_box(
        residuals: pd.DataFrame,
        ts_name: str,
        lags: int,
        significance_level: float = 0.05,
    ) -> Result:
        """
        Perform the Ljung-Box test to assess the presence of autocorrelation at multiple lag levels
        in the residuals of a regression model.

        This test is useful for checking randomness in a time series. The null hypothesis suggests
        that autocorrelations of the residual time series are absent, i.e., the data are independently distributed.
        Rejection of the null hypothesis indicates the presence of autocorrelation.

        Parameters:
        - residuals (pd.DataFrame): A pandas DataFrame of residuals from a regression model,
                                where each column represents a different set of residuals
                                (e.g., from multiple models or dependent variables).
        - lags (int): The number of lags to include in the test. Can also be an array of integers specifying the lags.
        - significance_level (float): The significance level for determining the presence of autocorrelation.
                                    Default is 0.05.

        Returns:
        - dict: A dictionary with column names as keys and values as dictionaries containing the Ljung-Box
            test statistics, p-values, a boolean array indicating which lags are significantly autocorrelated,
            and an overall indication of autocorrelation ('Present' or 'Absent') based on the test results.
        """
        lb_results = {}
        for col in residuals.columns:
            lb_test = acorr_ljungbox(residuals[col], lags=lags, return_df=True)
            is_autocorrelated = any(lb_test["lb_pvalue"] < significance_level)
            lb_results[col] = {
                "test_statistic": lb_test["lb_stat"].to_list(),
                "p_value": lb_test["lb_pvalue"].to_list(),
                "significance": (
                    lb_test["lb_pvalue"] < significance_level
                ).to_list(),
                "Autocorrelation": (
                    "Present" if is_autocorrelated else "Absent"
                ),
            }

        return LjungBoxResult(ts_name, lb_results)

    # -- Normality --
    @staticmethod
    def shapiro_wilk(
        data: pd.Series, ts_name: str, significance_level: float = 0.05
    ) -> dict:
        """
        Perform the Shapiro-Wilk test to assess the normality of a dataset.

        The Shapiro-Wilk test evaluates the null hypothesis that the data was drawn from a normal distribution.

        Parameters:
        - data (pd.Series or array-like): The dataset to test for normality. Should be one-dimensional.
        - significance_level (float): The significance level at which to test the null hypothesis. Default is 0.05.

        Returns:
        - dict: A dictionary containing the test statistic ('Statistic'), the p-value ('p-value'), and an indication
                of whether the data is considered 'Normal' or 'Not Normal' based on the significance level.

        Raises:
        - ValueError: If the input data contains fewer than 3 elements, as the test cannot be applied in such cases.
        """
        if len(data) < 3:
            raise ValueError(
                "Data must contain at least 3 elements to perform the Shapiro-Wilk test."
            )

        stat, p_value = shapiro(data)

        # Determine normality based on the p-value
        normality = "Normal" if p_value >= significance_level else "Not Normal"

        output = {
            "Statistic": stat,
            "p-value": p_value,
            "Normality": normality,
        }
        return ShapiroWilkResult(ts_name, output)

    # -- Heteroscedasticity --
    @staticmethod
    def breusch_pagan(
        x: np.array,
        y: np.array,
        ts_name: str,
        significance_level: float = 0.05,
    ) -> Result:
        """
        Perform the Breusch-Pagan test to assess the presence of heteroscedasticity in a linear regression model.

        Heteroscedasticity occurs when the variance of the residuals is not constant across all levels of the independent
        variables, potentially violating an assumption of linear regression models and affecting inference.

        Parameters:
        - x (np.array): The independent variables (explanatory variables) of the regression model. Should be 2D.
        - y (np.array): The dependent variable (response variable) of the regression model. Should be 1D.
        - significance_level (float): The significance level at which to test for heteroscedasticity. Default is 0.05.

        Returns:
        - dict: A dictionary containing the Breusch-Pagan test statistic ('Breusch-Pagan Test Statistic'),
                the p-value ('p-value'), and an indication of heteroscedasticity ('Heteroscedasticity': 'Present'
                if detected, otherwise 'Absent').

        Raises:
        - ValueError: If `x` or `y` are empty, or if their dimensions are incompatible.
        """
        # Add a constant to the independent variables matrix
        X = sm.add_constant(x)

        # Fit the regression model
        model = sm.OLS(y, X).fit()

        # Get the residuals
        residuals = model.resid

        # Perform the Breusch-Pagan test
        bp_test = het_breuschpagan(residuals, model.model.exog)

        # Extract the test statistic and p-value
        bp_test_statistic = bp_test[0]
        bp_test_pvalue = bp_test[1]

        # Prepare the results in a clean format
        output = {
            "Breusch-Pagan Test Statistic": bp_test_statistic,
            "p-value": bp_test_pvalue,
            "Heteroscedasticity": "",
        }

        # Determine heteroscedasticity based on the significance level and p-value
        heteroscedasticity = bp_test_pvalue < significance_level

        output["Heteroscedasticity"] = (
            "Present" if heteroscedasticity else "Absent"
        )

        return BreuschPaganResult(ts_name, output)

    @staticmethod
    def white_test(
        x: np.array,
        y: np.array,
        ts_name: str,
        significance_level: float = 0.05,
    ) -> Result:
        """
        Perform White's test for heteroscedasticity in a linear regression model.

        White's test assesses the null hypothesis that the variance of the residuals in the regression model is homoscedastic
        (constant across levels of the independent variables).

        Parameters:
        - x (np.array): The independent variables of the regression model, excluding the intercept.
                        Should be a 2D array where each column is a variable.
        - y (np.array): The dependent variable of the regression model. Should be a 1D array.
        - significance_level (float): The significance level for testing heteroscedasticity. Defaults to 0.05.

        Returns:
        - dict: A dictionary containing the White test statistic ('White Test Statistic'), the p-value ('p-value'), and
                an indication of heteroscedasticity ('Heteroscedasticity': 'Present' if detected, otherwise 'Absent').

        Raises:
        - ValueError: If `x` or `y` are empty, if their dimensions are incompatible, or if `x` is not 2-dimensional.
        """
        if x.size == 0 or y.size == 0:
            raise ValueError("Input arrays x and y must not be empty.")
        if x.ndim != 2:
            raise ValueError("Input array x must be 2-dimensional.")
        if y.ndim != 1:
            raise ValueError("Input array y must be 1-dimensional.")

        # Add a constant to the independent variables matrix
        X = sm.add_constant(x)

        # Fit the regression model
        model = sm.OLS(y, X).fit()

        # Perform White's test
        test_statistic, p_value, _, _ = het_white(
            model.resid, model.model.exog
        )

        # Determine heteroscedasticity based on the p-value and significance level
        heteroscedasticity = (
            "Present" if p_value < significance_level else "Absent"
        )

        output = {
            "White Test Statistic": test_statistic,
            "p-value": p_value,
            "Heteroscedasticity": heteroscedasticity,
        }

        return WhiteResult(ts_name, output)

    # -- Granger Causality --
    @staticmethod
    def granger_causality(
        data: pd.DataFrame,
        ts_name: str,
        max_lag: int,
        significance_level: float = 0.05,
    ) -> Result:
        """
        Perform Granger Causality tests to determine if one time series can forecast another.

        This method tests for each pair of variables in the provided DataFrame to see if the past values
        of one variable help to predict the future values of another, indicating a Granger causal relationship.

        Parameters:
        - data (pd.DataFrame): A pandas DataFrame where each column represents a time series variable.
        - max_lag (int): The maximum number of lags to test for Granger causality.
        - significance_level (float): The significance level for determining statistical significance.

        Returns:
        - dict: A dictionary with keys as tuples (cause_variable, effect_variable) and values as dictionaries containing
                the minimum p-value across all tested lags, a boolean indicating Granger causality based on the
                significance level, and the used significance level.

        Raises:
        - ValueError: If `data` contains fewer than two columns, as Granger causality requires pairwise comparison.
        - ValueError: If `max_lag` is less than 1, as at least one lag is necessary for testing causality.
        """
        if data.shape[1] < 2:
            raise ValueError(
                "DataFrame must contain at least two time series for pairwise Granger causality tests."
            )
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1.")

        granger_results = {}

        for var1 in data.columns:
            for var2 in data.columns:
                if var1 != var2:
                    test_result = grangercausalitytests(
                        data[[var1, var2]],
                        maxlag=max_lag,
                        verbose=False,
                    )
                    p_values = [
                        round(test_result[lag][0]["ssr_chi2test"][1], 4)
                        for lag in range(1, max_lag + 1)
                    ]
                    min_p_value = min(p_values)
                    causality = (
                        "Causality"
                        if min_p_value < significance_level
                        else "Non-Causality"
                    )
                    granger_results[(var1, var2)] = {
                        "Min P-Value": min_p_value,
                        "Granger Causality": causality,
                        "Significance Level": significance_level,
                    }

        return GrangerCausalityResult(ts_name, granger_results)

    # -- Historcal Nature --
    @staticmethod
    def half_life(
        Y: pd.Series, include_constant: bool = True
    ) -> Tuple[float, pd.Series]:
        """
        Calculate the expected half-life of a mean-reverting time series using an AR(1) model.

        The half-life is the period over which the deviation from the mean is expected to be halved,
        based on the autoregressive coefficient from the AR(1) model.

        Parameters:
        - Y (pd.Series): The time series for which the half-life is to be calculated.
        - include_constant (bool, optional): Whether to include a constant term in the regression model. Defaults to True.

        Returns:
        - float: The half-life of mean reversion, representing how quickly the series reverts to its mean.
        - pd.Series: Residuals from the regression model, indicating the error term for each observation.

        Raises:
        - ValueError: If the calculated autoregressive coefficient (phi) is outside the expected range (0, 1),
                    indicating that the time series may not exhibit mean-reverting behavior.
        """
        # Create the lagged series
        Y_lagged = Y.shift(1).dropna()
        Y = Y.dropna()

        # Align series by dropping the first item of Y
        Y = Y[1:]

        # Ensure alignment
        if len(Y) != len(Y_lagged):
            raise ValueError(
                "The lengths of Y and Y_lagged must match after dropping NA values."
            )

        # Prepare the lagged series with or without a constant
        if include_constant:
            Y_lagged = sm.add_constant(Y_lagged)

        # Fit the AR(1) model
        model = sm.OLS(Y, Y_lagged).fit()

        # Extract the coefficient 'phi'
        phi = model.params[0]  # Slope coefficient

        # Ensure phi is in the expected range for mean reversion
        if not (0 < phi < 1):
            raise ValueError(
                "Phi is outside the expected range (0, 1). The time series may not be mean-reverting."
            )

        # Calculate the half-life of mean reversion
        half_life = -np.log(2) / np.log(phi)

        # Extract the residuals
        residuals = model.resid

        return half_life, residuals

        # -- VECM Model --

    # -- Forescast Metrics --
    @staticmethod
    def evaluate_forecast(
        actual: pd.DataFrame, forecast: pd.DataFrame, print_output: bool = True
    ) -> dict:
        """
        Evaluate the accuracy of forecasted values against actual observations using various metrics.

        Parameters:
        - actual (pd.DataFrame or pd.Series): The actual observed values.
        - forecast (pd.DataFrame or pd.Series): The forecasted values.
        - print_output (bool, optional): Flag indicating whether to print the output. Defaults to True.

        Returns:
        - dict: A dictionary of forecast accuracy metrics for each series, including MAE, MSE, RMSE, and MAPE.

        Raises:
        - ValueError: If `actual` or `forecast` are not pandas Series or DataFrame.
                        If `actual` and `forecast` have different lengths or indexes.
        """
        if not isinstance(actual, (pd.Series, pd.DataFrame)) or not isinstance(
            forecast, (pd.Series, pd.DataFrame)
        ):
            raise ValueError(
                "Actual and forecast data must be pandas Series or DataFrame."
            )
        if actual.shape != forecast.shape:
            raise ValueError("Actual and forecast must have the same shape.")
        if not actual.index.equals(forecast.index):
            raise ValueError("Actual and forecast must have the same index.")

        results = {}
        columns = (
            actual.columns
            if isinstance(actual, pd.DataFrame)
            else [actual.name]
        )
        for col in columns:
            ac = actual[col] if isinstance(actual, pd.DataFrame) else actual
            fc = (
                forecast[col]
                if isinstance(forecast, pd.DataFrame)
                else forecast
            )
            mae = mean_absolute_error(ac, fc)
            mse = mean_squared_error(ac, fc)
            rmse = np.sqrt(mse)
            mape = (
                np.mean(
                    np.abs((ac - fc) / ac).replace([np.inf, -np.inf], np.nan)
                )
                * 100
            )

            metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
            results[col] = metrics

            if print_output:
                print(f"Metrics for {col}:\n{pd.DataFrame([metrics])}\n")

        return results if not print_output else None

    # !!! Untested!!!
    @staticmethod
    def vecm_model(data: pd.DataFrame, coint_rank: int, k_ar_diff: int):
        # Estimate the VECM model
        model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank)
        fitted_model = model.fit()
        return fitted_model

    # --- TODO: Hurst Exponent --
    @staticmethod
    def rescaled_range_analysis(
        time_series: np.array, min_chunk_size: int = 8
    ) -> np.ndarray:
        """
        Perform rescaled range analysis on a given time series.

        Parameters:
        - time_series (np.array): The time series data as a NumPy array.
        - min_chunk_size (int): The minimum size of the chunks to start the analysis.

        Returns:
        - np.array: An array of average R/S values for each chunk size.
        """
        N = len(time_series)
        chunk_sizes = [
            size for size in range(min_chunk_size, N // 2 + 1, min_chunk_size)
        ]
        rs_values = []

        for size in chunk_sizes:
            # Creating chunks
            chunks = [
                time_series[i : i + size]
                for i in range(0, N, size)
                if len(time_series[i : i + size]) == size
            ]

            rs_values_for_size = []
            for chunk in chunks:
                # Mean Adjusting
                mean_adjusted_chunk = chunk - np.mean(chunk)

                # Cumulative Deviation
                cumulative_deviation = np.cumsum(mean_adjusted_chunk)

                # Range and Standard Deviation
                R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                S = np.std(chunk)

                # Calculating R/S and adding to the list for the current chunk size
                if S != 0:
                    rs_values_for_size.append(R / S)
                else:
                    rs_values_for_size.append(0)

            # Averaging R/S values for the current chunk size if any values exist
            if rs_values_for_size:
                avg_rs = np.mean(rs_values_for_size)
                rs_values.append(avg_rs)
            else:
                rs_values.append(0)

        return np.array(rs_values)

    @staticmethod
    def hurst_exponent(time_series, min_chunk_size=8):
        """
        (Adjusted documentation to reflect the use of actual chunk sizes)
        """
        rs_values = TimeseriesTests.rescaled_range_analysis(
            time_series, min_chunk_size=min_chunk_size
        )

        N = len(time_series)
        chunk_sizes = np.array(
            [
                size
                for size in range(min_chunk_size, N // 2 + 1, min_chunk_size)
            ]
        )
        log_chunk_sizes = np.log(chunk_sizes)
        log_rs_values = np.log(rs_values)

        log_chunk_sizes_with_constant = sm.add_constant(log_chunk_sizes)

        model = sm.OLS(log_rs_values, log_chunk_sizes_with_constant)
        results = model.fit()

        return results.params[1]  # Hurst exponent
