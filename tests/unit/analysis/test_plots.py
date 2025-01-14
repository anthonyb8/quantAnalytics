import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from quant_analytics.analysis.plots import Plot


class TestPlot(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {
                "symbol1": [100, 101, 102, 103, 104],
                "symbol2": [200, 198, 197, 199, 202],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        self.signals = [
            {"timestamp": "2023-01-01", "price": 100, "direction": 1},
            {"timestamp": "2023-01-01", "price": 200, "direction": 1},
            {"timestamp": "2023-01-03", "price": 102, "direction": -1},
            {"timestamp": "2023-01-05", "price": 104, "direction": 1},
        ]

    def test_plot_line(self):
        # Test
        fig = Plot.plot_line(
            data=self.data["symbol1"],
            title="Test Line Plot",
            xlabel="Index",
            ylabel="Random Value",
        )

        # Show
        # plt.show()

        # Validate
        self.assertIsInstance(fig, Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)

    def test_plot_multiline(self):
        # Plotting
        fig = Plot.plot_multiline(
            data=self.data,
            title="Custom Multi-Line Plot",
            xlabel="Time",
            ylabel="Value",
        )

        # Show
        # plt.show()

        # Validate
        self.assertIsInstance(fig, Figure)

    def test_line_plot_with_markers(self):
        # Sample data with general x and y fields
        data = pd.DataFrame(
            {"symbol1": [100, 102, 105, 103], "symbol2": [50, 48, 49, 51]},
            index=pd.date_range("2023-01-01", periods=4),
        )

        # Example markers with custom field names
        markers = [
            {
                "time": "2023-01-01",
                "value": 101,
                "signal": 1,
            },  # Positive marker
            {
                "time": "2023-01-02",
                "value": 49,
                "signal": -1,
            },  # Negative marker
            {
                "time": "2023-01-03",
                "value": 103,
                "signal": 0.5,
            },  # Another positive marker
            {
                "time": "2023-01-04",
                "value": 50,
                "signal": -0.7,
            },  # Another negative marker
        ]

        # Plot the data with generalized markers
        fig = Plot.line_plot_with_markers(
            data=data,
            markers=markers,
            x_field="time",
            y_field="value",
            marker_field="signal",
            title="Generalized Line Plot with Markers",
            x_label="Date",
            y_label="Price",
        )

        # Show
        # plt.show()

        # Validate
        self.assertIsInstance(fig, Figure)

    def test_line_plot_dual_axis(self):
        # Sample data with index as x-axis
        index_values = pd.Index(range(50))  # Non-timestamp index
        price_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(150, 5, 50),
                "MSFT": np.random.normal(250, 5, 50),
            },
            index=index_values,
        )

        # Sample secondary data
        secondary_data = pd.Series(
            np.random.normal(0, 1, 50), index=index_values
        )

        # Example standard deviation bands
        std_1 = secondary_data.rolling(window=20).std()
        std_2 = 2 * secondary_data.rolling(window=20).std()

        # Plot with non-timestamp index
        fig = Plot.line_plot_dual_axis(
            primary_data=price_data,
            secondary_data=secondary_data,
            primary_label="Stock Prices",
            secondary_label="Spread",
            primary_y_label="Price",
            secondary_y_label="Spread",
            show_std=True,
            std_1=std_1,
            std_2=std_2,
            split_index=25,  # Example of splitting at index 25
            title="Stock Prices and Spread with Standard Deviations (Non-Timestamp Index)",
        )

        # Show
        # plt.show()

        # Validate
        self.assertIsInstance(fig, Figure)

    def test_line_plot_with_std(self):
        # Sample series data
        index_values = pd.date_range("2023-01-01", periods=100)
        series_data = pd.Series(
            np.random.normal(0, 1, 100), index=index_values
        )

        # Plot the series with rolling statistics
        fig = Plot.line_plot_with_std(
            series=series_data,
            window=20,
            primary_label="Z-Score",
            secondary_label="Mean and Std Dev",
            x_label="Date",
            y_label_primary="Z-Score",
            y_label_secondary="Mean and Std Dev",
            title="Z-Score with Rolling Mean and Std Dev",
        )

        # Show
        # plt.show()

        # Validate
        self.assertIsInstance(fig, Figure)

    def test_line_plot_dual_axis_with_markers(self):
        # Sample price data (primary data)
        index_values = pd.date_range("2023-01-01", periods=50)
        price_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(150, 5, 50),
                "MSFT": np.random.normal(250, 5, 50),
            },
            index=index_values,
        )

        # Sample spread or secondary data
        spread_data = pd.Series(np.random.normal(0, 1, 50), index=index_values)

        # Example markers with generalized fields and value-based colors
        markers = [
            {"timestamp": "2023-01-05", "price": 148, "value": 1.5},
            {"timestamp": "2023-01-10", "price": 252, "value": 0.8},
            {"timestamp": "2023-01-20", "price": 250, "value": 1.0},
        ]

        # Plot the data with generalized marker fields
        fig = Plot.line_plot_dual_axis_with_markers(
            primary_data=price_data,
            secondary_data=spread_data,
            markers=markers,
            x_field="timestamp",
            y_field="price",
            marker_field="value",
            primary_label="Stock Prices",
            secondary_label="Spread",
            x_label="Date",
            y_label_primary="Price",
            y_label_secondary="Spread",
            title="Price Data, Spread, and Markers Based on Values",
        )

        # Show
        # plt.show()

        # Validate
        self.assertIsInstance(fig, Figure)

    def test_histogram_ndc(self):
        # Create sample data
        data = np.random.normal(loc=0, scale=1, size=1000)

        # test
        fig = Plot.histogram_ndc(
            data, bins="auto", title="Test Histogram with NDC"
        )

        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) >= 1)
        self.assertTrue(len(fig.axes[0].patches) > 0)

    def test_histogram_kde(self):
        # Create sample data
        data = np.random.normal(loc=0, scale=1, size=1000)

        # test
        fig = Plot.histogram_kde(
            data, bins="auto", title="Test Histogram with KDE"
        )

        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)
        self.assertTrue(len(fig.axes[0].patches) > 0)

    def test_qq_plot(self):
        # Create sample data
        data = np.random.normal(loc=0, scale=1, size=1000)

        # test
        fig = Plot.qq_plot(data, title="Test Q-Q Plot")

        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)

    def test_residuals_v_fitted(self):
        # Create sample data
        residuals = np.random.normal(loc=0, scale=1, size=1000)
        fittedvalues = np.random.normal(loc=0, scale=1, size=1000)

        # test
        fig = Plot.plot_residuals_vs_fitted(residuals, fittedvalues)

        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)

    def test_plot_influence_measures(self):
        # Create sample data
        data = np.random.normal(loc=0, scale=1, size=1000)

        # test
        fig = Plot.plot_influence_measures(data)

        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
