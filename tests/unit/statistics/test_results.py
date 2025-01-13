import unittest
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from quant_analytics.statistics.statistics import TimeseriesTests
from quant_analytics.data.generator import DataGenerator
from quant_analytics.statistics.results import GrangerCausalityResult


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

    def test_grangercausalityreuslt(self):
        granger_results = {
            ("x", "y"): {
                "Min P-Value": 0.0255,
                "Granger Causality": "Causality",
                "Significance Level": 0.05,
            },
            ("y", "x"): {
                "Min P-Value": 0.4608,
                "Granger Causality": "Non-Causality",
                "Significance Level": 0.05,
            },
        }

        # Test
        result_obj = GrangerCausalityResult("test_name", granger_results)
        result_df = result_obj._to_dataframe()

        # Validate
        self.assertEqual(result_df.loc[0, "Granger Causality"], "Yes")
        self.assertEqual(result_df.loc[1, "Granger Causality"], "No")


if __name__ == "__main__":
    unittest.main()
