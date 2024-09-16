import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from quantAnalytics.visualization import Visualization


class TestVisualization(unittest.TestCase):
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

    def test_line_plot(self):
        # Create sample data
        x = pd.Series(np.arange(0, 10))
        y = pd.Series(np.random.random(10))

        # Test
        fig = Visualization.line_plot(
            x, y, title="Test Line Plot", x_label="Index", y_label="Random Value"
        )

        # Show
        # plt.show()

        # Validate
        self.assertIsInstance(fig, Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)

    def test_multi_line_plot(self):
        # Creating dummy data
        data = {
            "Series 1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=[1, 2, 3]),
            "Series 2": pd.DataFrame({"A": [1, 1, 2], "B": [6, 5, 4]}, index=[1, 2, 3]),
        }

        # Custom styles
        line_styles = {"Series 1": "--", "Series 2": ":"}

        colors = {"Series 1": "blue", "Series 2": "red"}

        # Plotting
        fig = Visualization.multi_line_plot(
            data=data,
            title="Custom Multi-Line Plot",
            x_label="Time",
            y_label="Value",
            line_styles=line_styles,
            colors=colors,
            rotate_xticks=True,
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
            {"time": "2023-01-01", "value": 101, "signal": 1},  # Positive marker
            {"time": "2023-01-02", "value": 49, "signal": -1},  # Negative marker
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
        fig = Visualization.line_plot_with_markers(
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
        secondary_data = pd.Series(np.random.normal(0, 1, 50), index=index_values)

        # Example standard deviation bands
        std_1 = secondary_data.rolling(window=20).std()
        std_2 = 2 * secondary_data.rolling(window=20).std()

        # Plot with non-timestamp index
        fig = Visualization.line_plot_dual_axis(
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
        series_data = pd.Series(np.random.normal(0, 1, 100), index=index_values)

        # Plot the series with rolling statistics
        fig = Visualization.line_plot_with_std(
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
        fig = Visualization.line_plot_dual_axis_with_markers(
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

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()

