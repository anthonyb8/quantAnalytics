import unittest
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from quant_analytics.statistics.statistics import TimeseriesTests
from quant_analytics.data.generator import DataGenerator


def display_adf_results(
    adf_results: dict,
    print_output: bool = True,
    to_html: bool = False,
    indent: int = 0,
) -> str:
    print(adf_results)
    # Convert ADF results to DataFrame
    adf_data = []
    for ticker, values in adf_results.items():
        print(ticker, values)
        row = {
            "Ticker": ticker,
            "ADF Statistic": values["ADF Statistic"],
            "p-value": round(values["p-value"], 6),
        }
        row.update(
            {
                f"Critical Value ({key})": val
                for key, val in values["Critical Values"].items()
            }
        )
        row.update({"Stationarity": values["Stationarity"]})
        adf_data.append(row)

    title = "ADF Test Results"
    adf_df = pd.DataFrame(adf_data)
    footer = "** IF p-value < 0.05 and/or statistic < statistic @ confidence interval, then REJECT the Null that the time series posses a unit root (non-stationary)."

    if to_html:
        # Define the base indentation as a string of spaces
        base_indent = "    " * indent
        next_indent = "    " * (indent + 1)

        # Convert DataFrame to HTML table and add explanation
        html_table = adf_df.to_html(index=False, border=1)
        html_table_indented = "\n".join(
            next_indent + line for line in html_table.split("\n")
        )

        html_title = f"{next_indent}<h4>{title}</h4>\n"
        html_footer = f"{next_indent}<p class='footnote'>{footer}</p>\n"
        html_output = f"{base_indent}<div class='adf_test'>\n{html_title}{html_table_indented}\n{html_footer}{base_indent}</div>"

        return html_output

    else:
        output = (
            f"\n{title}\n"
            f"{'=' * len(title)}\n"
            f"{adf_df.to_string(index=False)}\n"
            f"{footer}"
        )
        if print_output:
            print(output)
        else:
            return output


def generate_residuals(autocorr_type="none", size=100):
    """
    Helper function to generate simulated residuals with specific autocorrelation properties.
    """
    np.random.seed(42)  # For reproducibility
    if autocorr_type == "positive":
        # Generate positively autocorrelated residuals
        residuals = [np.random.randn()]
        for _ in range(1, size):
            residuals.append(residuals[-1] * 0.9 + np.random.randn())
    elif autocorr_type == "negative":
        # Alternate signs for negative autocorrelation
        residuals = [(-1) ** i * abs(np.random.randn()) for i in range(size)]
    else:
        # Generate random residuals (no autocorrelation)
        residuals = np.random.randn(size)
    return pd.DataFrame(residuals, columns=["Residuals"])


def generate_causal_data():
    np.random.seed(42)
    x = np.random.randn(100)
    # Increase the coefficient to make the causality stronger
    y = x + np.random.normal(scale=5, size=100)  # Increased noise
    data = pd.DataFrame({"x": x, "y": y})
    return data


def generate_mean_reverting_series_known_half_life(
    half_life: float, size: int = 1000
):
    """
    Generate a synthetic mean-reverting time series based on an AR(1) process where
    the half-life of mean reversion is known.

    Parameters:
    half_life (float): The half-life of the mean-reverting series.
    size (int): The size of the time series to generate.

    Returns:
    pd.Series: Generated mean-reverting time series.
    """
    # Calculate the phi coefficient from the half-life
    phi = np.exp(-np.log(2) / half_life)

    # Generate AR(1) series
    series = [np.random.normal()]
    for _ in range(1, size):
        series.append(phi * series[-1] + np.random.normal())

    return pd.Series(series)


def generate_seasonal_stationary_series(length=1000, seasonal_periods=12):
    """Generate a seasonal stationary series with a specified length and seasonality."""
    np.random.seed(42)
    # Generate a seasonal pattern
    seasonal_component = np.sin(np.linspace(0, 2 * np.pi, seasonal_periods))
    # Repeat the pattern but ensure the final series length matches the desired length
    repeats = length // seasonal_periods + 1  # Ensure enough repeats
    series = np.tile(seasonal_component, repeats)[
        :length
    ]  # Trim to exact length
    noise = np.random.normal(0, 0.5, length)
    return pd.Series(
        series + noise
    )  # No need to slice noise as it's already correct length


def generate_seasonal_non_stationary_series(length=1000, start=0, slope=1):
    """Generate a non-stationary series with a specified length."""
    np.random.seed(42)
    # Create an array of increasing values based on the specified slope
    series = np.arange(start, length * slope + start, slope)
    return pd.Series(series)


def generate_cointegrated_series(n=100):
    """
    Generate a set of cointegrated time series for testing.
    """
    np.random.seed(42)
    # Generate a random walk series
    x0 = np.random.normal(0, 1, n).cumsum()
    # Generate another series which is a linear combination of x0 plus some noise
    x1 = x0 + np.random.normal(0, 1, n)
    return pd.DataFrame({"x0": x0, "x1": x1})


def generate_non_cointegrated_series(n=100):
    """
    Generate a set of non-cointegrated time series for testing.
    """
    np.random.seed(42)
    # Generate two independent random walk series
    x0 = np.random.normal(0, 1, n).cumsum()
    x1 = np.random.normal(0, 1, n).cumsum()
    return pd.DataFrame({"x0": x0, "x1": x1})


def generate_random_time_series(length=100, lag=2):
    """
    Generate a simple time series dataset with a known lagged relationship for testing,
    where the first 'lag' values of the lagged series are filled with random numbers.

    Parameters:
    - length (int): Length of the time series.
    - lag (int): The lag introduced between x0 and x1 to establish a known relationship.

    Returns:
    - pd.DataFrame: A DataFrame containing the time series with a known lagged relationship.
    """
    np.random.seed(42)  # Ensure reproducibility

    # Generate the base series x0
    x0 = np.random.normal(0, 1, length).cumsum()

    # Initialize x1 with the same size as x0
    x1 = np.zeros(length)

    # Fill the first 'lag' elements of x1 with random noise
    x1[:lag] = np.random.normal(0, 1, lag)

    # Fill in the rest of x1 with values from x0 lagged by 'lag' steps
    # Here, adding some noise to maintain a similar variability in x1 as in x0
    x1[lag:] = x0[:-lag] + np.random.normal(
        0, 0.1, length - lag
    )  # Adjust noise level as needed

    data = pd.DataFrame({"x0": x0, "x1": x1})
    return data


class TestTimeseriesTests(unittest.TestCase):
    def setUp(self):
        # Create sample time series data
        np.random.seed(0)
        self.sample_series = pd.Series(np.random.randn(100))
        self.sample_dataframe = pd.DataFrame(
            {"series1": np.random.randn(100), "series2": np.random.randn(100)}
        )

        self.mean_reverting_series = (
            DataGenerator.generate_mean_reverting_series()
        )
        self.trending_series = DataGenerator.generate_trending_series()
        self.random_walk_series = DataGenerator.generate_random_walk_series(
            n=2000, start_value=0, step_std=1
        )

    # # Basic Validation
    # def test_generate_mean_reverting_series_basic(self):
    #     n = 2000
    #     mu = 0
    #     theta = 0.1  # Adjust if needed
    #     sigma = 0.2  # Adjust if needed
    #     series = DataGenerator.generate_mean_reverting_series(
    #         n=n, mu=mu, theta=theta, sigma=sigma, start_value=1
    #     )
    #     self.assertEqual(len(series), n)
    #     self.assertEqual(series[0], 1)
    #     self.assertTrue(isinstance(series, np.ndarray))
    #
    #     series_mean = np.mean(series)
    #     self.assertTrue(
    #         np.abs(series_mean - mu) < 0.05
    #     )  # threshold for difference between actual mean and long-term mean
    #
    # def test_generate_trending_series_basic(self):
    #     n = 2000
    #     series = TimeseriesTests.generate_trending_series(
    #         n=n, start_value=0, trend=0.1, step_std=1
    #     )
    #     self.assertEqual(len(series), n)
    #     self.assertEqual(series[0], 0)
    #     self.assertTrue(isinstance(series, np.ndarray))
    #
    #     # Perform linear regression
    #     X = np.arange(len(series)).reshape(
    #         -1, 1
    #     )  # Time steps as independent variable
    #     y = series  # Series values as dependent variable
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(
    #         X.ravel(), y
    #     )
    #
    #     # Verify the trend direction and significance
    #     self.assertGreater(
    #         slope, 0
    #     )  # Check if slope is significantly greater than 0
    #     self.assertLess(
    #         p_value, 0.05
    #     )  # Check if the slope is statistically significant
    #
    # def test_generate_random_walk_series_basic(self):
    #     n = 2000
    #     series = TimeseriesTests.generate_random_walk_series(
    #         n=n, start_value=0, step_std=1
    #     )
    #     self.assertEqual(len(series), n)
    #     self.assertEqual(series[0], 0)
    #     self.assertTrue(isinstance(series, np.ndarray))
    #
    #     # Test for no clear trend using Augmented Dickey-Fuller
    #     adf_result = adfuller(series)
    #     self.assertTrue(
    #         adf_result[1] > 0.05
    #     )  # P-value should be low to reject null hypothesis of a unit root
    #
    #     # Test for independence using autocorrelation at lag 1
    #     autocorrelation = pd.Series(series).autocorr(lag=1)
    #
    #     # Expect autocorrelation for a random walk to be significant but not perfect
    #     self.assertTrue(0.9 < autocorrelation <= 1)

    # def test_split_data_default_ratio(self):
    #     train, test = TimeseriesTests.split_data(self.sample_dataframe)
    #     # Default split ratio is 0.8
    #     expected_train_length = int(len(self.sample_dataframe) * 0.8)
    #     expected_test_length = (
    #         len(self.sample_dataframe) - expected_train_length
    #     )
    #     self.assertEqual(len(train), expected_train_length)
    #     self.assertEqual(len(test), expected_test_length)
    #
    # def test_lag_series_default(self):
    #     # Test with the default lag of 1
    #     lagged_series = TimeseriesTests.lag_series(
    #         self.sample_series
    #     ).reset_index(drop=True)
    #     expected_series = self.sample_series[:-1].reset_index(drop=True)
    #     self.assertTrue((lagged_series.values == expected_series.values).all())
    #
    # def test_lag_series_custom_lag(self):
    #     # Test with a custom lag of 5
    #     lag = 5
    #     lagged_series = TimeseriesTests.lag_series(self.sample_series, lag=lag)
    #     self.assertEqual(len(lagged_series), len(self.sample_series) - lag)
    #     self.assertTrue(
    #         (lagged_series.values == self.sample_series[:-lag].values).all()
    #     )
    #
    # def test_split_data_custom_ratio(self):
    #     custom_ratio = 0.7
    #     train, test = TimeseriesTests.split_data(
    #         self.sample_dataframe, train_ratio=custom_ratio
    #     )
    #     expected_train_length = int(len(self.sample_dataframe) * custom_ratio)
    #     expected_test_length = (
    #         len(self.sample_dataframe) - expected_train_length
    #     )
    #     self.assertEqual(len(train), expected_train_length)
    #     self.assertEqual(len(test), expected_test_length)
    #
    # def test_split_data_data_integrity(self):
    #     train, test = TimeseriesTests.split_data(self.sample_dataframe)
    #     # Check if concatenated train and test sets equal the original data
    #     pd.testing.assert_frame_equal(
    #         pd.concat([train, test]).reset_index(drop=True),
    #         self.sample_dataframe,
    #     )

    def test_adf_mean_reverting(self):
        result = TimeseriesTests.adf_test(
            self.mean_reverting_series, "testing"
        )

        # Validate
        # Expect mean-reverting series to be potentially stationary
        self.assertEqual(result.data["Stationarity"], "Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_adf_trending(self):
        result = TimeseriesTests.adf_test(self.trending_series, "testing")

        # Validate
        # Expect trending series to be non-stationary
        self.assertEqual(result.data["Stationarity"], "Non-Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_adf_random_walk(self):
        n = 2000
        series = DataGenerator.generate_random_walk_series(
            n=n, start_value=0, step_std=1
        )
        result = TimeseriesTests.adf_test(series, "testing")

        # Validate
        # Expect random walk series to be non-stationary
        self.assertEqual(result.data["Stationarity"], "Non-Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_rolling_adf(self):
        # Parameters
        window = 20
        trend = "c"
        series = self.mean_reverting_series

        # test
        rolling_adf_stats = TimeseriesTests.rolling_adf(series, window, trend)

        # validate
        self.assertIsInstance(
            rolling_adf_stats, pd.Series, "Output is not a pandas Series"
        )
        self.assertEqual(
            len(rolling_adf_stats),
            len(series),
            "Output Series length mismatch",
        )
        self.assertTrue(
            rolling_adf_stats[window - 1 :].isnull().sum() == 0,
            "NaNs present in fully populated rolling window outputs",
        )
        self.assertTrue(
            rolling_adf_stats[: window - 1].isnull().all(),
            "Non-NaN values present in initial positions where window is not fully populated",
        )

    def test_display_rolling_adf_results(self):
        # Parameters
        window = 20
        trend = "c"
        series = self.mean_reverting_series
        rolling_adf_results = TimeseriesTests.rolling_adf(
            series, window, trend
        )

        # test
        fig = TimeseriesTests.display_rolling_adf_results(
            pd.Series(series), rolling_adf_results
        )

        # validate
        self.assertIsInstance(fig, plt.Figure)

    def test_monte_carlo_simulation(self):
        # Define parameters for the simulation
        n_simulations = 10
        series_length = 100
        mean = 0
        std_dev = 1
        trend = "c"
        confidence_interval = "5%"
        significance_level = 0.05

        # Call the monte_carlo_simulation method
        adf_stats, p_values = TimeseriesTests.monte_carlo_simulation(
            n_simulations,
            series_length,
            mean,
            std_dev,
            trend,
            confidence_interval,
            significance_level,
        )

        # Validation checks
        self.assertIsInstance(
            adf_stats, list, "ADF stats output is not a list"
        )
        self.assertIsInstance(p_values, list, "P-values output is not a list")
        self.assertEqual(
            len(adf_stats),
            n_simulations,
            "Length of ADF stats list does not match number of simulations",
        )
        self.assertEqual(
            len(p_values),
            n_simulations,
            "Length of p-values list does not match number of simulations",
        )
        self.assertTrue(
            all(isinstance(stat, float) for stat in adf_stats),
            "Not all ADF stats are floats",
        )
        self.assertTrue(
            all(isinstance(p, float) for p in p_values),
            "Not all p-values are floats",
        )
        self.assertTrue(
            all(-np.inf < stat < np.inf for stat in adf_stats),
            "ADF stats contain invalid values",
        )
        self.assertTrue(
            all(0 <= p <= 1 for p in p_values),
            "P-values are out of the expected range [0, 1]",
        )

    def test_display_monte_carlo_simulation(self):
        # Define parameters for the simulation
        n_simulations = 10
        series_length = 100
        mean = 0
        std_dev = 1
        trend = "c"
        confidence_interval = "5%"
        significance_level = 0.05
        series = self.mean_reverting_series

        # Call the monte_carlo_simulation method
        adf_stats, p_values = TimeseriesTests.monte_carlo_simulation(
            n_simulations,
            series_length,
            mean,
            std_dev,
            trend,
            confidence_interval,
            significance_level,
        )

        # test
        fig = TimeseriesTests.display_monte_carlo_simulation(adf_stats, series)

        # validate
        self.assertIsInstance(fig, plt.Figure)

    def test_fit_arima(self):
        # Create a synthetic time series
        np.random.seed(42)
        data = np.random.randn(100)
        series = pd.Series(data)

        # Define ARIMA order
        order = (1, 0, 0)

        # Call the fit_arima method
        model_fit, residuals = TimeseriesTests.fit_arima(series, order)

        # Validation checks
        self.assertIsInstance(
            residuals, pd.Series, "Residuals output is not a pandas Series"
        )
        self.assertEqual(
            len(residuals),
            len(series),
            "Residuals length does not match input series length",
        )
        self.assertFalse(
            residuals.isnull().any(), "Residuals contain NaN values"
        )

    def test_plot_acf_pacf(self):
        # Create a synthetic time series
        np.random.seed(42)
        data = np.random.randn(100)
        series = pd.Series(data)

        # Define ARIMA order
        order = (1, 0, 0)
        model_fit, residuals = TimeseriesTests.fit_arima(series, order)

        # test
        fig = TimeseriesTests.plot_acf_pacf(residuals, lags=40)

        # validate
        self.assertIsInstance(fig, plt.Figure)

    def test_kpss_mean_reverting(self):
        result = TimeseriesTests.kpss_test(
            self.mean_reverting_series, "test series"
        )

        # Validate
        # Expect mean-reverting series to be potentially stationary
        self.assertEqual(result.data["Stationarity"], "Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_kpss_trending(self):
        result = TimeseriesTests.kpss_test(self.trending_series, "testSeries")

        # Validate
        # Expect trending series to be non-stationary
        self.assertEqual(result.data["Stationarity"], "Non-Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_kpss_random_walk(self):
        n = 2000
        series = DataGenerator.generate_random_walk_series(
            n=n, start_value=0, step_std=1
        )
        result = TimeseriesTests.kpss_test(series, "testSeries")

        # Validate
        # Expect random walk series to be non-stationary
        self.assertEqual(result.data["Stationarity"], "Non-Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_phillips_perron_mean_reverting(self):
        result = TimeseriesTests.phillips_perron_test(
            self.mean_reverting_series, "series_test"
        )

        # Validate
        # Expect mean-reverting series to be potentially stationary
        self.assertEqual(result.data["Stationarity"], "Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_phillips_perron_trending(self):
        result = TimeseriesTests.phillips_perron_test(
            self.trending_series, "test_name", trend="ct"
        )

        # Validate
        # Expect trending series to be non-stationary
        self.assertEqual(result.data["Stationarity"], "Non-Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_phillips_perron_random_walk(self):
        series = pd.Series(
            DataGenerator.generate_random_walk_series(
                n=2000, start_value=0, step_std=1
            )
        )
        result = TimeseriesTests.phillips_perron_test(series, "test_name")

        # Validate
        # Expect random walk series to be non-stationary
        self.assertEqual(result.data["Stationarity"], "Non-Stationary")
        self.assertIsInstance(result.to_html(), str)

    def test_seasonal_stationary(self):
        """Test that the method identifies a seasonal stationary series as stationary."""
        series = generate_seasonal_stationary_series()
        result = TimeseriesTests.seasonal_adf_test(
            series, "series_name", seasonal_periods=12
        )

        # Validate
        self.assertEqual(
            result.data["Stationarity"],
            "Stationary",
            "Failed to identify a seasonal stationary series as stationary",
        )

    def test_non_seasonal_stationary(self):
        """Test that the method identifies a non-stationary series as non-stationary."""
        series = generate_seasonal_non_stationary_series()
        result = TimeseriesTests.seasonal_adf_test(
            series, "series_name", seasonal_periods=12
        )

        # Validate
        self.assertEqual(
            result.data["Stationarity"],
            "Non-Stationary",
            "Failed to identify a non-stationary series as non-stationary",
        )

    def test_johansen_test_cointegration(self):
        """Test the Johansen test can detect cointegration in a synthetic dataset."""
        data = generate_cointegrated_series()

        # Test
        result = TimeseriesTests.johansen_test(data)

        # Validation
        self.assertGreater(result.num_cointegrations, 0)
        self.assertIsInstance(result.to_html(), str)

    def test_johansen_test_no_cointegration(self):
        """Test the Johansen test does not falsely detect cointegration."""
        data = generate_non_cointegrated_series()

        # Test
        result = TimeseriesTests.johansen_test(data)

        # Validation
        self.assertEqual(result.num_cointegrations, 0)
        self.assertIsInstance(result.to_html(), str)

    def test_select_lag_known_optimal_lag(self):
        """Test that the select_lag_length function identifies the expected optimal lag."""
        data = generate_random_time_series(lag=4)
        expected_lag = 4

        # Test
        selected_lag = TimeseriesTests.select_lag_length(
            data, maxlags=10, criterion="bic"
        )

        # Validate
        self.assertEqual(
            selected_lag,
            expected_lag,
            f"Expected optimal lag of {expected_lag}, but got {selected_lag}.",
        )

    def test_lag_selection_criteria(self):
        """Test the select_lag_length function with different information criteria."""
        data = generate_random_time_series()
        for criterion in ["aic", "bic", "hqic", "fpe"]:
            with self.subTest(criterion=criterion):
                selected_lag = TimeseriesTests.select_lag_length(
                    data, maxlags=10, criterion=criterion
                )
                self.assertIsInstance(
                    selected_lag,
                    int,
                    f"Selected lag should be an integer for criterion {criterion}.",
                )
                self.assertTrue(
                    1 <= selected_lag <= 10,
                    f"Selected lag should be within the specified range for criterion {criterion}.",
                )

    def test_select_coint_rank_no_cointegration(self):
        """Test cointegration rank selection on non-cointegrated data."""
        data = generate_non_cointegrated_series()
        result = TimeseriesTests.select_coint_rank(data, k_ar_diff=1)
        self.assertEqual(
            result["Cointegration Rank"],
            0,
            "Failed to correctly identify no cointegration.",
        )

    def test_select_coint_rank_potential_cointegration(self):
        """Test cointegration rank selection on potentially cointegrated data."""
        data = generate_cointegrated_series()
        result = TimeseriesTests.select_coint_rank(data, k_ar_diff=1)
        # Here, we expect some level of cointegration due to the common trend
        self.assertGreaterEqual(
            result["Cointegration Rank"],
            1,
            "Failed to identify potential cointegration.",
        )

    def test_durbin_watson_no_autocorrelation(self):
        residuals = generate_residuals("none")

        # Test
        result = TimeseriesTests.durbin_watson(residuals, "residuals")

        # Validate
        self.assertIn("Absent", result.data["Residuals"]["Autocorrelation"])
        self.assertIsInstance(result.to_html(), str)

    def test_durbin_watson_positive_autocorrelation(self):
        residuals = generate_residuals("positive")

        # Test
        result = TimeseriesTests.durbin_watson(residuals, "ts_name")

        # Validate
        self.assertIn("Positive", result.data["Residuals"]["Autocorrelation"])
        self.assertIsInstance(result.to_html(), str)

    def test_durbin_watson_negative_autocorrelation(self):
        residuals = generate_residuals("negative")

        # Test
        result = TimeseriesTests.durbin_watson(residuals, "test_name")

        # Validate
        self.assertIn("Negative", result.data["Residuals"]["Autocorrelation"])
        self.assertIsInstance(result.to_html(), str)

    def test_ljung_box_no_autocorrelation(self):
        residuals = generate_residuals("none")

        # Test
        result = TimeseriesTests.ljung_box(residuals, "test_name", lags=10)

        # Valdiate
        self.assertEqual(
            result.data["Residuals"]["Autocorrelation"],
            "Absent",
            "Failed to correctly identify absence of autocorrelation",
        )

    def test_ljung_box_positive_autocorrelation(self):
        residuals = generate_residuals("positive")

        # Test
        result = TimeseriesTests.ljung_box(residuals, "test_name", lags=10)

        # Validate
        self.assertEqual(
            result.data["Residuals"]["Autocorrelation"],
            "Present",
            "Failed to correctly identify presence of autocorrelation",
        )

    def test_shapiro_wilk_normal_distribution(self):
        # Generate a normally distributed dataset
        data = pd.Series(np.random.normal(loc=0, scale=1, size=100))

        # Test
        result = TimeseriesTests.shapiro_wilk(data, "test_name")

        # Validate
        self.assertEqual(
            result.data["Normality"],
            "Normal",
            "Failed to correctly identify a normal distribution",
        )

    def test_shapiro_wilk_non_normal_distribution(self):
        # Generate a uniformly distributed dataset
        data = pd.Series(np.random.uniform(low=-1, high=1, size=100))

        # Test
        result = TimeseriesTests.shapiro_wilk(data, "test_name")

        # Validate
        self.assertEqual(
            result.data["Normality"],
            "Not Normal",
            "Failed to correctly identify a non-normal distribution",
        )

    def test_breusch_pagan_no_heteroscedasticity(self):
        # Simulate data with no heteroscedasticity
        np.random.seed(42)
        x = np.random.normal(size=(100, 2))
        y = 2 + 3 * x[:, 0] + 4 * x[:, 1] + np.random.normal(size=100)

        # Test
        result = TimeseriesTests.breusch_pagan(x, y, "test_name")

        # Validate
        self.assertEqual(
            result.data["Heteroscedasticity"],
            "Absent",
            "Failed to correctly identify absence of heteroscedasticity",
        )

    def test_breusch_pagan_heteroscedasticity(self):
        # Simulate data with heteroscedasticity
        np.random.seed(42)
        x = np.random.normal(size=(100, 2))

        # Heteroscedasticity introduced more strongly
        y = (
            2
            + 3 * x[:, 0]
            + 4 * x[:, 1]
            + np.random.normal(size=100) * (1 + x[:, 0])
        )

        # Test
        result = TimeseriesTests.breusch_pagan(x, y, "test_name")

        # Validate
        self.assertEqual(
            result.data["Heteroscedasticity"],
            "Present",
            "Failed to correctly identify presence of heteroscedasticity",
        )

    def test_white_homoscedasticity(self):
        # Simulate data with homoscedastic residuals
        np.random.seed(42)
        x = np.random.normal(size=(100, 2))
        y = 2 + 3 * x[:, 0] + 4 * x[:, 1] + np.random.normal(size=100)

        # Test
        result = TimeseriesTests.white_test(x, y, "test_name")

        # Validate
        self.assertEqual(
            result.data["Heteroscedasticity"],
            "Absent",
            "Failed to correctly identify homoscedastic residuals",
        )

    def test_white_heteroscedasticity(self):
        # Simulate data with heteroscedastic residuals
        np.random.seed(42)
        x = np.random.normal(size=(100, 2))
        y = (
            2
            + 3 * x[:, 0]
            + 4 * x[:, 1]
            + np.random.normal(size=100) * (5 * abs(x[:, 0]))
        )

        # Test
        result = TimeseriesTests.white_test(x, y, "test_name")

        # Validate
        self.assertEqual(
            result.data["Heteroscedasticity"],
            "Present",
            "Failed to correctly identify heteroscedastic residuals",
        )

    def test_granger_causality_x_to_y(self):
        data = generate_causal_data()

        # Test
        result = TimeseriesTests.granger_causality(
            data, "name", max_lag=10, significance_level=0.05
        )

        # Validate
        self.assertEqual(
            result.data[("x", "y")]["Granger Causality"], "Causality"
        )

    def test_granger_causality_y_to_x(self):
        data = generate_causal_data()

        # Test
        result = TimeseriesTests.granger_causality(
            data, "name", max_lag=10, significance_level=0.05
        )

        # Validate
        self.assertEqual(
            result.data[("y", "x")]["Granger Causality"], "Non-Causality"
        )

    def test_half_life_valid(self):
        """
        Test the half_life function with a known half-life.
        """
        known_half_life = 10  # Known half-life of the series
        series = generate_mean_reverting_series_known_half_life(
            half_life=known_half_life
        )

        # Create a lagged version of the series
        # series_lagged = series.shift(1).bfill()

        # Calculate the half-life using the function under test
        calculated_half_life, _ = TimeseriesTests.half_life(
            series, include_constant=False
        )

        # Assert the calculated half-life is close to the known value
        self.assertAlmostEqual(
            known_half_life,
            calculated_half_life,
            delta=1,
            msg="Calculated half-life deviates from the known value",
        )

    def test_evaluate_forecast_dataframe(self):
        """Test the evaluate_forecast function with pd.DataFrame input."""
        # For DataFrame testing
        self.actual = pd.Series(
            np.random.normal(100, 10, 100), name="TestSeries"
        )
        self.forecast = self.actual + np.random.normal(
            0, 5, 100
        )  # Adding noise to create a forecast

        self.actual_df = pd.DataFrame(
            {
                "Series1": self.actual,
                "Series2": self.actual
                + np.random.normal(
                    0, 3, 100
                ),  # Another series with less noise
            }
        )
        self.forecast_df = pd.DataFrame(
            {
                "Series1": self.forecast,
                "Series2": self.actual_df["Series2"]
                + np.random.normal(0, 2, 100),  # Forecast with different noise
            }
        )
        result = TimeseriesTests.evaluate_forecast(
            self.actual_df, self.forecast_df, print_output=False
        )
        self.assertIsInstance(
            result, dict, "The result should be a dictionary."
        )
        for col in self.actual_df.columns:
            self.assertIn(
                col,
                result,
                f"Result dictionary should include metrics for {col}.",
            )
            self.assertIn(
                "MAE", result[col], f"Result for {col} should include MAE."
            )
            self.assertIn(
                "MSE", result[col], f"Result for {col} should include MSE."
            )
            self.assertIn(
                "RMSE", result[col], f"Result for {col} should include RMSE."
            )
            self.assertIn(
                "MAPE", result[col], f"Result for {col} should include MAPE."
            )

    def test_evaluate_forecast_series(self):
        """Test the evaluate_forecast function with pd.Series input."""
        self.actual = pd.Series(
            np.random.normal(100, 10, 100), name="TestSeries"
        )
        self.forecast = self.actual + np.random.normal(
            0, 5, 100
        )  # Adding noise to create a forecast

        result = TimeseriesTests.evaluate_forecast(
            self.actual, self.forecast, print_output=False
        )

        self.assertIsInstance(
            result, dict, "The result should be a dictionary."
        )
        self.assertIn(
            "MAE",
            result["TestSeries"],
            "Result dictionary should include MAE.",
        )
        self.assertIn(
            "MSE",
            result["TestSeries"],
            "Result dictionary should include MSE.",
        )
        self.assertIn(
            "RMSE",
            result["TestSeries"],
            "Result dictionary should include RMSE.",
        )
        self.assertIn(
            "MAPE",
            result["TestSeries"],
            "Result dictionary should include MAPE.",
        )

    # def test_hurst_exp_mean_reverting_series(self):
    #     hurst = TimeseriesTests.hurst_exponent(self.mean_reverting_series)
    #     self.assertLess(hurst, 0.5)

    # def test_hurst_exp_random_walk(self):
    #     series = TimeseriesTests.generate_random_walk_series()
    #     hurst = TimeseriesTests.hurst_exponent(series)
    #     self.assertAlmostEqual(hurst, 0.5, delta=0.05)

    # def test_hurst_exp_trending_series(self):
    #     self.trending_series = TimeseriesTests.generate_trending_series()

    #     hurst = TimeseriesTests.hurst_exponent(self.trending_series)
    #     self.assertGreater(hurst, 0.5)


if __name__ == "__main__":
    unittest.main()
