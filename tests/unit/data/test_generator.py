import unittest
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from quant_analytics.data.generator import DataGenerator


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

    # Basic Validation
    def test_generate_mean_reverting_series_basic(self):
        n = 2000
        mu = 0
        theta = 0.1  # Adjust if needed
        sigma = 0.2  # Adjust if needed
        series = DataGenerator.generate_mean_reverting_series(
            n=n, mu=mu, theta=theta, sigma=sigma, start_value=1
        )
        self.assertEqual(len(series), n)
        self.assertEqual(series[0], 1)
        self.assertTrue(isinstance(series, np.ndarray))

        series_mean = np.mean(series)
        self.assertTrue(
            np.abs(series_mean - mu) < 0.05
        )  # threshold for difference between actual mean and long-term mean

    def test_generate_trending_series_basic(self):
        n = 2000
        series = DataGenerator.generate_trending_series(
            n=n, start_value=0, trend=0.1, step_std=1
        )
        self.assertEqual(len(series), n)
        self.assertEqual(series[0], 0)
        self.assertTrue(isinstance(series, np.ndarray))

        # Perform linear regression
        X = np.arange(len(series)).reshape(
            -1, 1
        )  # Time steps as independent variable
        y = series  # Series values as dependent variable
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            X.ravel(), y
        )

        # Verify the trend direction and significance
        self.assertGreater(
            slope, 0
        )  # Check if slope is significantly greater than 0
        self.assertLess(
            p_value, 0.05
        )  # Check if the slope is statistically significant

    def test_generate_random_walk_series_basic(self):
        n = 2000
        series = DataGenerator.generate_random_walk_series(
            n=n, start_value=0, step_std=1
        )
        self.assertEqual(len(series), n)
        self.assertEqual(series[0], 0)
        self.assertTrue(isinstance(series, np.ndarray))

        # Test for no clear trend using Augmented Dickey-Fuller
        adf_result = adfuller(series)
        self.assertTrue(
            adf_result[1] > 0.05
        )  # P-value should be low to reject null hypothesis of a unit root

        # Test for independence using autocorrelation at lag 1
        autocorrelation = pd.Series(series).autocorr(lag=1)

        # Expect autocorrelation for a random walk to be significant but not perfect
        self.assertTrue(0.9 < autocorrelation <= 1)


if __name__ == "__main__":
    unittest.main()
