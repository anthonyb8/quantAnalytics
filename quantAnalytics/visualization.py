import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Optional
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm


class Visualization:
    @staticmethod
    def line_plot(
        x: pd.Series,
        y: pd.Series,
        title: str = "Line Plot",
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        line_style: str = "-",
        marker: str = "o",
        color: str = "b",
        grid: bool = True,
        layout_tight: bool = True,
    ) -> Figure:
        """
        Create a line plot for the given x and y data.

        Parameters:
        - x (pd.Series): The x-axis data.
        - y (pd.Series): The y-axis data.
        - title (str): Title of the plot.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.
        - line_style (str): Style of the plot line.
        - marker (str): Marker type for plot points.
        - color (str): Color of the plot line.
        - grid (bool): Whether to display grid lines.
        - layout_tight (bool): Whether to use tight layout.

        Returns:
        - plt.Figure: The line plot.

        Example:
        >>> Visualizations.line_plot(x, y, title='Test Line Plot', x_label='Index', y_label='Value', color='red')
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, linestyle=line_style, marker=marker, color=color)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if grid:
            ax.grid(True)
        if layout_tight:
            plt.tight_layout()
        return fig

    @staticmethod
    def multi_line_plot(
        data: Dict[str, pd.DataFrame],
        title: str = "Multi-Line Plot",
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        line_styles: Optional[Dict[str, str]] = None,
        colors: Optional[Dict[str, str]] = None,
        grid: bool = True,
        rotate_xticks: bool = False,
        layout_tight: bool = True,
    ) -> Figure:
        """
        Plot multiple data series from a dictionary of DataFrames.

        Parameters:
        - data (dict): Dictionary where keys are series labels and values are DataFrames with data to plot.
        - title (str): Title of the plot.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.
        - line_styles (dict): Optional. Dictionary mapping series labels to line styles.
        - colors (dict): Optional. Dictionary mapping series labels to colors.
        - grid (bool): Whether to display grid lines.
        - rotate_xticks (bool): Whether to rotate x-axis labels for better readability.
        - layout_tight (bool): Whether to use tight layout.

        Returns:
        - plt.Figure: The figure object containing the plots.

        Example:
        >>> data = {
                "Series 1": pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}),
                "Series 2": pd.DataFrame({"A": [1, 1, 1], "B": [3, 2, 1]})
            }
        # Custom styles
        >>> line_styles = {
                "Series 1": "--",
                "Series 2": ":"
            }

        >>> colors = {
                "Series 1": "blue",
                "Series 2": "red"
            }
        >>> Visualizations.multi_line_plot(data, title="Example Plot")
        >>> plt.show()
        """
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each series
        for label, df in data.items():
            if line_styles and label in line_styles:
                line_style = line_styles[label]
            else:
                line_style = "-"  # Default to solid line

            if colors and label in colors:
                color = colors[label]
            else:
                color = None  # Let matplotlib auto-assign colors

            for column in df.columns:
                ax.plot(
                    df.index,
                    df[column],
                    label=f"{label} {column}",
                    linestyle=line_style,
                    color=color,
                )

        # Add labels, title, and grid
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if grid:
            ax.grid(True)

        # Add legend
        ax.legend()

        # Rotate x-axis labels if necessary
        if rotate_xticks:
            plt.xticks(rotation=45)

        # Adjust layout to minimize white space
        if layout_tight:
            plt.tight_layout()

        return fig

    @staticmethod
    def line_plot_with_markers(
        data: pd.DataFrame,
        markers: list,
        x_field: str = "timestamp",
        y_field: str = "price",
        marker_field: str = "direction",
        title: str = "Line Plot with Markers",
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        line_styles: Optional[Dict[str, str]] = None,
        colors: Optional[Dict[str, str]] = None,
        grid: bool = True,
        layout_tight: bool = True,
    ) -> Figure:
        """
        Create a line plot with markers, where the marker behavior is based on a general 'marker_field' (positive or negative).

        Parameters:
        - data (pd.DataFrame): The data with symbols as columns and index representing x-values.
        - markers (list): A list of dictionaries with custom fields for x, y, and markers.
        - x_field (str): Field name for x-axis values (e.g., timestamps).
        - y_field (str): Field name for y-axis values (e.g., prices).
        - marker_field (str): Field name for marker values (e.g., direction or signal strength).
        - title (str): Title of the plot.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.
        - line_styles (dict): Optional. Line styles for each symbol.
        - colors (dict): Optional. Colors for each symbol's line.
        - grid (bool): Whether to display grid lines.
        - layout_tight (bool): Whether to use tight layout to reduce white space.

        Returns:
        - plt.Figure: The figure object containing the plot.

        Example:
        >>> data = pd.DataFrame({'symbol1': [1, 2, 3], 'symbol2': [4, 5, 6]}, index=pd.date_range('2023-01-01', periods=3))
        >>> markers = [{'time': '2023-01-01', 'value': 2, 'signal': 1}, {'time': '2023-01-02', 'value': 5, 'signal': -1}]
        >>> fig = Visualizations.line_plot_with_markers(data, markers, x_field="time", y_field="value", marker_field="signal")
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot the data lines
        for symbol in data.columns:
            line_style = (
                line_styles[symbol]
                if line_styles and symbol in line_styles
                else "-"
            )
            color = colors[symbol] if colors and symbol in colors else None
            ax.plot(
                data.index,
                data[symbol],
                label=symbol,
                linestyle=line_style,
                color=color,
                marker="o",
                zorder=1,
            )

        # Plot the markers based on the general field names
        for marker in markers:
            x_value = pd.to_datetime(marker[x_field])
            marker_value = marker[marker_field]
            marker_color = "green" if marker_value > 0 else "red"
            marker_shape = "o" if marker_value > 0 else "x"
            ax.scatter(
                x_value,
                marker[y_field],
                color=marker_color,
                marker=marker_shape,
                label=f"{'Positive' if marker_value > 0 else 'Negative'} Marker",
                zorder=2,
            )

        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Add grid
        if grid:
            ax.grid(True)

        # Add legend
        ax.legend()

        # Adjust layout to minimize white space
        if layout_tight:
            plt.tight_layout()

        return fig

    @staticmethod
    def line_plot_dual_axis(
        primary_data: pd.DataFrame,
        secondary_data: pd.Series,
        primary_label: str = "Primary Data",
        secondary_label: str = "Secondary Data",
        x_label: str = "X-axis",
        primary_y_label: str = "Primary Y-axis",
        secondary_y_label: str = "Secondary Y-axis",
        show_std: bool = False,
        std_1: pd.Series = None,
        std_2: pd.Series = None,
        split_index=None,
        title: str = "Dual Axis Plot",
    ) -> Figure:
        """
        Create a dual-axis plot where the left y-axis plots the primary data (multiple tickers) and
        the right y-axis plots the secondary data (e.g., spread) with optional mean and standard deviations.

        Parameters:
        - primary_data (pd.DataFrame): DataFrame containing the primary data with index as x-axis.
        - secondary_data (pd.Series): Series containing the secondary data (e.g., spread or other data for the right y-axis).
        - primary_label (str): Label for the primary data series.
        - secondary_label (str): Label for the secondary data series.
        - x_label (str): Label for the x-axis.
        - primary_y_label (str): Label for the left y-axis (primary).
        - secondary_y_label (str): Label for the right y-axis (secondary).
        - show_std (bool): Whether to plot standard deviation bands around the secondary data.
        - std_1 (pd.Series): Optional. 1 standard deviation band around the secondary data.
        - std_2 (pd.Series): Optional. 2 standard deviation band around the secondary data.
        - split_index (str or None): Optional. Index value to split the plot with a vertical line.
        - title (str): Title of the plot.
        - show_plot (bool): Whether to display the plot immediately.

        Returns:
        - plt.Figure: The figure object containing the dual-axis plot.

        Example:
        >>> timestamps = pd.date_range('2023-01-01', periods=50)
        >>> price_data = pd.DataFrame({
                'AAPL': np.random.normal(150, 5, 50),
                'MSFT': np.random.normal(250, 5, 50),
            }, index=timestamps)

        >>> spread_data = pd.Series(np.random.normal(0, 1, 50), index=timestamps)
        >>> std_1 = spread_data.rolling(window=20).std()
        >>> std_2 = 2 * spread_data.rolling(window=20).std()
        """
        # Use the index as the x-axis
        x_values = primary_data.index

        # Create a figure and primary axis for primary data (left y-axis)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot each ticker on the left y-axis
        colors = [
            "blue",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "orange",
        ]  # Extend this list as needed
        for i, ticker in enumerate(primary_data.columns):
            color = colors[i % len(colors)]  # Cycle through colors
            ax1.plot(
                x_values,
                primary_data[ticker],
                label=f"{primary_label}: {ticker}",
                color=color,
                linewidth=2,
            )

        ax1.set_ylabel(primary_y_label)
        ax1.legend(loc="upper left")

        # Create a secondary axis for secondary data (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(
            x_values,
            secondary_data,
            label=secondary_label,
            color="purple",
            linewidth=2,
        )
        ax2.set_ylabel(secondary_y_label)

        # Plot standard deviation bands if provided
        if show_std:
            if std_1 is not None:
                ax2.fill_between(
                    x_values,
                    secondary_data - std_1,
                    secondary_data + std_1,
                    color="gray",
                    alpha=0.2,
                    label="1 Std Dev",
                )
            if std_2 is not None:
                ax2.fill_between(
                    x_values,
                    secondary_data - std_2,
                    secondary_data + std_2,
                    color="gray",
                    alpha=0.4,
                    label="2 Std Dev",
                )
        ax2.legend(loc="upper right")

        # Draw a dashed vertical line to separate test and training data if a split index is provided
        if split_index is not None:
            ax1.axvline(
                x=split_index, color="black", linestyle="--", linewidth=1
            )

        # Add grid lines and format x-axis labels for better readability
        ax1.grid(True)
        plt.xticks(rotation=45)
        plt.xlabel(x_label)

        # Title
        plt.title(title)

        # Adjust layout to minimize white space
        plt.tight_layout()

        # Show the plot if requested
        # if show_plot:
        #     plt.show()

        return fig

    @staticmethod
    def line_plot_with_std(
        series: pd.Series,
        window: int = 20,
        primary_label: str = "Series",
        secondary_label: str = "Statistics",
        x_label: str = "Index",
        y_label_primary: str = "Value",
        y_label_secondary: str = "Mean and Std Dev",
        title: str = "Series with Rolling Statistics",
    ) -> Figure:
        """
        Plot a time series along with its mean and standard deviations (1 and 2) on the right y-axis.

        Parameters:
            series (pd.Series): Series containing the data to be plotted.
            window (int): Rolling window size for calculating mean and standard deviations (default is 20).
            primary_label (str): Label for the primary series on the left y-axis.
            secondary_label (str): Label for the mean and standard deviations on the right y-axis.
            x_label (str): Label for the x-axis (default is 'Index').
            y_label_primary (str): Label for the left y-axis (default is 'Value').
            y_label_secondary (str): Label for the right y-axis (default is 'Mean and Std Dev').
            title (str): Title of the plot.
        """
        # Create a figure and primary axis for the series (left y-axis)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the series on the left y-axis
        ax1.plot(
            series.index,
            series,
            label=primary_label,
            color="blue",
            linewidth=2,
        )

        ax1.set_ylabel(y_label_primary)
        ax1.legend(loc="upper left")

        # Calculate rolling mean and standard deviations for the series
        series_mean = series.rolling(window=window).mean()
        series_std_1 = series.rolling(
            window=window
        ).std()  # 1 standard deviation
        series_std_2 = (
            2 * series.rolling(window=window).std()
        )  # 2 standard deviations

        # Create a secondary axis for mean and standard deviations (right y-axis)
        ax2 = ax1.twinx()

        # Plot mean and standard deviations on the right y-axis
        ax2.plot(
            series.index,
            series_mean,
            label="Mean",
            color="orange",
            linestyle="--",
        )
        ax2.fill_between(
            series.index,
            series_mean - series_std_1,
            series_mean + series_std_1,
            color="gray",
            alpha=0.2,
            label="1 Std Dev",
        )
        ax2.fill_between(
            series.index,
            series_mean - series_std_2,
            series_mean + series_std_2,
            color="gray",
            alpha=0.4,
            label="2 Std Dev",
        )

        ax2.set_ylabel(y_label_secondary)
        ax2.legend(loc="upper right")

        # Add grid lines
        ax1.grid(True)

        # Format x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.xlabel(x_label)

        # Title
        plt.title(title)

        # Adjust layout to minimize white space
        plt.tight_layout()

        return fig

    @staticmethod
    def line_plot_dual_axis_with_markers(
        primary_data: pd.DataFrame,
        secondary_data: pd.Series,
        markers: list,
        x_field: str = "timestamp",
        y_field: str = "price",
        marker_field: str = "value",
        primary_label: str = "Primary Data",
        secondary_label: str = "Secondary Data",
        x_label: str = "Index",
        y_label_primary: str = "Primary Y-axis",
        y_label_secondary: str = "Secondary Y-axis",
        show_std: bool = False,
        std_1: pd.Series = None,
        std_2: pd.Series = None,
        split_index=None,
        title: str = "Dual Axis Plot with Markers",
    ) -> Figure:
        """
        Create a dual-axis plot where the left y-axis plots the primary data (multiple tickers),
        the right y-axis plots the secondary data (e.g., spread), and markers are plotted based on customizable fields.

        Parameters:
        - primary_data (pd.DataFrame): DataFrame containing the primary data with index as x-axis.
        - secondary_data (pd.Series): Series containing the secondary data (e.g., spread or other data for the right y-axis).
        - markers (list): List of dictionaries containing marker data with customizable fields for x, y, and marker.
        - x_field (str): Field name in the markers for the x-axis values (default is 'timestamp').
        - y_field (str): Field name in the markers for the y-axis values (default is 'price').
        - marker_field (str): Field name in the markers for identifying marker values (default is 'value').
        - primary_label (str): Label for the primary data series.
        - secondary_label (str): Label for the secondary data series.
        - x_label (str): Label for the x-axis.
        - y_label_primary (str): Label for the left y-axis (primary).
        - y_label_secondary (str): Label for the right y-axis (secondary).
        - show_std (bool): Whether to plot standard deviation bands around the secondary data.
        - std_1 (pd.Series): Optional. 1 standard deviation band around the secondary data.
        - std_2 (pd.Series): Optional. 2 standard deviation band around the secondary data.
        - split_index (str or None): Optional. Index value to split the plot with a vertical line.
        - title (str): Title of the plot.

        Returns:
        - plt.Figure: The figure object containing the dual-axis plot with markers.
        """
        # Use the index as the x-axis
        x_values = primary_data.index

        # Create a figure and primary axis for primary data (left y-axis)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot each ticker on the left y-axis
        colors = [
            "blue",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "orange",
        ]  # Extend this list as needed
        for i, ticker in enumerate(primary_data.columns):
            color = colors[i % len(colors)]  # Cycle through colors
            ax1.plot(
                x_values,
                primary_data[ticker],
                label=f"{primary_label}: {ticker}",
                color=color,
                linewidth=2,
            )

        ax1.set_ylabel(y_label_primary)
        ax1.legend(loc="upper left")

        # Create a secondary axis for secondary data (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(
            x_values,
            secondary_data,
            label=secondary_label,
            color="purple",
            linewidth=2,
        )
        ax2.set_ylabel(y_label_secondary)

        # Plot standard deviation bands if provided
        if show_std:
            if std_1 is not None:
                ax2.fill_between(
                    x_values,
                    secondary_data - std_1,
                    secondary_data + std_1,
                    color="gray",
                    alpha=0.2,
                    label="1 Std Dev",
                )
            if std_2 is not None:
                ax2.fill_between(
                    x_values,
                    secondary_data - std_2,
                    secondary_data + std_2,
                    color="gray",
                    alpha=0.4,
                    label="2 Std Dev",
                )
        ax2.legend(loc="upper right")

        # Plot markers based on general fields and marker values
        for marker in markers:
            x_value = pd.to_datetime(marker[x_field])
            y_value = marker[y_field]
            marker_value = marker[marker_field]

            # Color and marker logic based on marker values
            if marker_value > 1:
                marker_shape = "^"
                color = "green"
            elif marker_value < 1:
                marker_shape = "v"
                color = "red"
            else:
                marker_shape = "o"
                color = "gray"

            ax1.scatter(
                x_value, y_value, marker=marker_shape, color=color, s=100
            )

        # Draw a dashed vertical line to separate test and training data if a split index is provided
        if split_index is not None:
            ax1.axvline(
                x=split_index, color="black", linestyle="--", linewidth=1
            )

        # Add grid lines and format x-axis labels for better readability
        ax1.grid(True)
        plt.xticks(rotation=45)
        plt.xlabel(x_label)

        # Title
        plt.title(title)

        # Adjust layout to minimize white space
        plt.tight_layout()

        return fig

    @staticmethod
    def histogram_ndc(
        data: pd.Series,
        bins: str = "auto",
        title: str = "Histogram with Normal Distribution Curve",
    ) -> plt.Figure:
        """
        Create a histogram for the given data and overlay a normal distribution fit.

        Parameters:
        - data (array-like): The dataset for which the histogram is to be created.
        - bins (int or sequence or str): Specification of bin sizes. Default is 'auto'.
        - title (str): Title of the plot.

        Returns:
        - plt.Figure: A histogram with a normal distribution fit.

        Example:
        >>> TimeseriesTests.histogram_ndc(data, bins='auto', title='Test Histogram with NDC')
        >>> plt.show()

        """
        # Convert data to a numpy array if it's not already
        data = np.asarray(data)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate histogram
        sns.histplot(
            data, bins=bins, kde=False, color="blue", stat="density", ax=ax
        )

        # Fit and overlay a normal distribution
        mean, std = norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std)
        ax.plot(x, p, "k", linewidth=2)

        title += f"\n Fit Results: Mean = {mean:.2f},  Std. Dev = {std:.2f}"
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        # Return the figure object
        return fig

    @staticmethod
    def histogram_kde(
        data: pd.Series,
        bins: str = "auto",
        title: str = "Histogram with Kernel Density Estimate (KDE)",
    ) -> plt.Figure:
        """
        Create a histogram for the given data to visually check for normal distribution.

        Parameters:
        - data (array-like): The dataset for which the histogram is to be created.
        - bins (int or sequence or str): Specification of bin sizes. Default is 'auto'.
        - title (str): Title of the plot.

        Returns:
        - plt.Figure: A histogram for assessing normality.

        Example
        >>> TimeseriesTests.histogram_kde(data, bins='auto', title='Test Histogram with KDE')
        >>> plt.show()
        """
        # Convert data to a numpy array if it's not already
        data = np.asarray(data)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate histogram with KDE
        sns.histplot(data, bins=bins, kde=True, color="blue", ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

        # Return the figure object
        return fig

    @staticmethod
    def qq_plot(data: pd.Series, title: str = "Q-Q Plot") -> plt.Figure:
        """
        Create a Q-Q plot for the given data comparing it against a normal distribution.

        Parameters:
        - data (array-like): The dataset for which the Q-Q plot is to be created.
        - title (str): Title of the plot.

        Returns:
        - plt.Figure: A Q-Q plot.

        Example:
        >>> TimeseriesTests.qq_plot(data, title='Test Q-Q Plot')
        >>> plt.show()
        """
        # Convert data to a numpy array if it's not already
        data = np.asarray(data)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))

        # Generate Q-Q plot
        stats.probplot(data, dist="norm", plot=ax)

        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")

        # Return the figure object
        return fig

    # -- Old --
    # class Visualizations:
    # @staticmethod
    # def line_plot(
    #     x: pd.Series,
    #     y: pd.Series,
    #     title: str = "Line Plot",
    #     x_label: str = "X-axis",
    #     y_label: str = "y-axis",
    # ) -> Figure:
    #     """
    #     Create a line plot for the given x and y data.

    #     Parameters:
    #     - x (pd.Series): The x-axis data.
    #     - y (pd.Series): The y-axis data.
    #     - title (str): Title of the plot.
    #     - x_label (str): Label for the x-axis.
    #     - y_label (str): Label for the y-axis.

    #     Returns:
    #     - plt.Figure: The line plot.

    #     Example:
    #     >>> TimeseriesTests.line_plot(x, y, title='Test Line Plot', x_label='Index', y_label='Random Value')
    #     >>> plt.show()
    #     """
    #     # Create figure and axis
    #     fig, ax = plt.subplots(figsize=(10, 6))

    #     # Plot the line
    #     ax.plot(x, y, marker="o")

    #     # Add title and labels
    #     ax.set_title(title)
    #     ax.set_xlabel(x_label)
    #     ax.set_ylabel(y_label)

    #     # Add grid
    #     ax.grid(True)

    #     # Adjust layout to minimize white space
    #     plt.tight_layout()

    #     # Return the figure object
    #     return fig

    # @staticmethod
    # def double_line_plot(
    #     data1: pd.DataFrame,
    #     data2: pd.DataFrame,
    #     label1: str = "Series 1",
    #     label2: str = "Series 2",
    #     title: str = "Double Line Plot",
    # ) -> Union[Figure, None]:
    #     """
    #     Plot two data series for each column in the DataFrame.

    #     Parameters:
    #     - data1 (pd.DataFrame): First DataFrame of values.
    #     - data2 (pd.DataFrame): Second DataFrame of values.
    #     - label1 (str): Label for the first series.
    #     - label2 (str): Label for the second series.
    #     - title (str): Title of the plot.

    #     Returns:
    #     - plt.Figure: The figure object containing the plots.

    #     Example:
    #     >>> TimeseriesTests.double_line_plot(data1, data2, label1='Data 1', label2='Data 2', title='Test Double Line Plot')
    #     >>> plt.show()
    #     """
    #     # Ensure data1 and data2 columns match
    #     if set(data1.columns) != set(data2.columns):
    #         raise ValueError("Columns of data1 and data2 do not match")

    #     # Iterate over each column to create separate plots
    #     for column in data1.columns:
    #         fig, ax = plt.subplots(figsize=(12, 6))

    #         # Plot the first series
    #         ax.plot(data1[column], label=f"{label1} {column}", color="blue")

    #         # Plot the second series
    #         if column in data2.columns:
    #             ax.plot(data2[column], label=f"{label2} {column}", color="red")

    #         # Add labels, legend, and grid
    #         ax.grid(True)
    #         ax.set_xlabel("Time")
    #         ax.set_ylabel("Value")
    #         ax.set_title(f"{title} for {column}")
    #         ax.legend()

    #         # Customize x-axis labels for readability
    #         plt.xticks(rotation=45)

    #         # Adjust layout to minimize white space
    #         plt.tight_layout()

    #         # Return the figure object
    #         return fig

    # @staticmethod
    # def plot_data_with_signals(
    #     data: pd.DataFrame,
    #     signals: list,
    #     title: str = "Price Data with Trade Signals",
    #     x_label: str = "Timestamp",
    #     y_label: str = "Price",
    # ) -> Figure:
    #     """
    #     Create a line plot with trade signals for the given data.

    #     Parameters:
    #     - data (pd.DataFrame): The price data with symbols as columns and timestamps as the index.
    #     - signals (list): A list of dictionaries containing 'timestamp', 'price', and 'direction' (1 for buy, -1 for sell).
    #     - title (str): Title of the plot.
    #     - x_label (str): Label for the x-axis.
    #     - y_label (str): Label for the y-axis.

    #     Returns:
    #     - plt.Figure: The line plot with signals.

    #     Example:
    #     >>> data = pd.DataFrame({'symbol1': [1, 2, 3], 'symbol2': [4, 5, 6]}, index=pd.date_range('2023-01-01', periods=3))
    #     >>> signals = [{'timestamp': '2023-01-01', 'price': 2, 'direction': 1}, {'timestamp': '2023-01-02', 'price': 5, 'direction': -1}]
    #     >>> Plotting.plot_data_with_signals(data, signals)
    #     >>> plt.show()
    #     """
    #     fig, ax = plt.subplots(figsize=(15, 7))

    #     # Plot the price data
    #     for symbol in data.columns:
    #         ax.plot(data.index, data[symbol], label=symbol, marker="o", zorder=1)

    #     # Plot the signals
    #     for signal in signals:
    #         signal_timestamp = pd.to_datetime(signal["timestamp"])
    #         color = "green" if signal["direction"] == 1 else "red"
    #         marker = "o" if signal["direction"] == 1 else "x"
    #         ax.scatter(
    #             signal_timestamp, signal["price"], color=color, marker=marker, zorder=2
    #         )

    #     # Add title and labels
    #     ax.set_title(title)
    #     ax.set_xlabel(x_label)
    #     ax.set_ylabel(y_label)

    #     # Add legend
    #     ax.legend()

    #     # Add grid
    #     ax.grid(True)

    #     # Adjust layout to minimize white space
    #     plt.tight_layout()

    #     return fig

    # # --- UNTESTED --
    # @staticmethod
    # def plot_price_and_spread(
    #     price_data: pd.DataFrame,
    #     spread: list,
    #     signals: list,
    #     split_date=None,
    #     show_plot=True,
    # ):
    #     """
    #     Plot multiple ticker data on the left y-axis and spread with mean and standard deviations on the right y-axis.

    #     Parameters:
    #         price_data (pd.DataFrame): DataFrame containing the data with timestamps as index and multiple ticker columns.
    #         spread (pd.Series): Series containing the spread data.
    #     """
    #     # Extract data from the DataFrame
    #     timestamps = price_data.index
    #     spread = pd.Series(spread, index=timestamps)

    #     # Create a figure and primary axis for price data (left y-axis)
    #     fig, ax1 = plt.subplots(figsize=(12, 6))

    #     # Plot each ticker on the left y-axis
    #     colors = [
    #         "blue",
    #         "green",
    #         "red",
    #         "cyan",
    #         "magenta",
    #         "yellow",
    #         "black",
    #         "orange",
    #     ]  # Extend this list as needed
    #     for i, ticker in enumerate(price_data.columns):
    #         color = colors[i % len(colors)]  # Cycle through colors
    #         ax1.plot(
    #             timestamps, price_data[ticker], label=ticker, color=color, linewidth=2
    #         )

    #     ax1.set_yscale("linear")
    #     ax1.set_ylabel("Price")
    #     ax1.legend(loc="upper left")

    #     # Calculate mean and standard deviations for spread
    #     spread_mean = spread.rolling(
    #         window=20
    #     ).mean()  # Adjust the window size as needed
    #     spread_std_1 = spread.rolling(window=20).std()  # 1 standard deviation
    #     spread_std_2 = 2 * spread.rolling(window=20).std()  # 2 standard deviations

    #     # Create a secondary axis for the spread with mean and standard deviations (right y-axis)
    #     ax2 = ax1.twinx()

    #     # Plot Spread on the right y-axis
    #     ax2.plot(timestamps, spread, label="Spread", color="purple", linewidth=2)
    #     ax2.plot(timestamps, spread_mean, label="Mean", color="orange", linestyle="--")
    #     ax2.fill_between(
    #         timestamps,
    #         spread_mean - spread_std_1,
    #         spread_mean + spread_std_1,
    #         color="gray",
    #         alpha=0.2,
    #         label="1 Std Dev",
    #     )
    #     ax2.fill_between(
    #         timestamps,
    #         spread_mean - spread_std_2,
    #         spread_mean + spread_std_2,
    #         color="gray",
    #         alpha=0.4,
    #         label="2 Std Dev",
    #     )
    #     ax2.set_yscale("linear")
    #     ax2.set_ylabel("Spread and Statistics")
    #     ax2.legend(loc="upper right")

    #     # Plot signals
    #     for signal in signals:
    #         ts = pd.to_datetime(signal["timestamp"])
    #         price = signal["price"]
    #         action = signal["action"]
    #         if action in ["LONG", "COVER"]:
    #             marker = "^"
    #             color = "lime"
    #         elif action in ["SHORT", "SELL"]:
    #             marker = "v"
    #             color = "red"
    #         else:
    #             # Default marker for undefined actions
    #             marker = "o"
    #             color = "gray"
    #         ax1.scatter(ts, price, marker=marker, color=color, s=100)

    #     # Draw a dashed vertical line to separate test and training data
    #     if split_date is not None:
    #         split_date = pd.to_datetime(split_date)
    #         ax1.axvline(x=split_date, color="black", linestyle="--", linewidth=1)

    #     # Add grid lines
    #     ax1.grid(True)

    #     # Format x-axis labels for better readability
    #     plt.xticks(rotation=45)
    #     plt.xlabel("Timestamp")

    #     # Title
    #     plt.title("Price Data, Spread, and Statistics Over Time")

    #     # Show the plot
    #     plt.tight_layout()

    #     return fig

    # @staticmethod
    # def plot_price_and_spread(
    #     price_data: pd.DataFrame, spread: pd.Series, show_plot=True
    # ):
    #     """
    #     Plot multiple ticker data on the left y-axis and spread with mean and standard deviations on the right y-axis.

    #     Parameters:
    #         price_data (pd.DataFrame): DataFrame containing the data with timestamps as index and multiple ticker columns.
    #         spread (pd.Series): Series containing the spread data.
    #     """
    #     # Extract data from the DataFrame
    #     timestamps = price_data.index

    #     # Create a figure and primary axis for price data (left y-axis)
    #     fig, ax1 = plt.subplots(figsize=(12, 6))

    #     # Plot each ticker on the left y-axis
    #     colors = [
    #         "blue",
    #         "green",
    #         "red",
    #         "cyan",
    #         "magenta",
    #         "yellow",
    #         "black",
    #         "orange",
    #     ]  # Extend this list as needed
    #     for i, ticker in enumerate(price_data.columns):
    #         color = colors[i % len(colors)]  # Cycle through colors
    #         ax1.plot(
    #             timestamps, price_data[ticker], label=ticker, color=color, linewidth=2
    #         )

    #     ax1.set_yscale("linear")
    #     ax1.set_ylabel("Price")
    #     ax1.legend(loc="upper left")

    #     # Calculate mean and standard deviations for spread
    #     spread_mean = spread.rolling(
    #         window=20
    #     ).mean()  # Adjust the window size as needed
    #     spread_std_1 = spread.rolling(window=20).std()  # 1 standard deviation
    #     spread_std_2 = 2 * spread.rolling(window=20).std()  # 2 standard deviations

    #     # Create a secondary axis for the spread with mean and standard deviations (right y-axis)
    #     ax2 = ax1.twinx()

    #     # Plot Spread on the right y-axis
    #     ax2.plot(timestamps, spread, label="Spread", color="purple", linewidth=2)
    #     ax2.plot(timestamps, spread_mean, label="Mean", color="orange", linestyle="--")
    #     ax2.fill_between(
    #         timestamps,
    #         spread_mean - spread_std_1,
    #         spread_mean + spread_std_1,
    #         color="gray",
    #         alpha=0.2,
    #         label="1 Std Dev",
    #     )
    #     ax2.fill_between(
    #         timestamps,
    #         spread_mean - spread_std_2,
    #         spread_mean + spread_std_2,
    #         color="gray",
    #         alpha=0.4,
    #         label="2 Std Dev",
    #     )
    #     ax2.set_yscale("linear")
    #     ax2.set_ylabel("Spread and Statistics")
    #     ax2.legend(loc="upper right")

    #     # Add grid lines
    #     ax1.grid(True)

    #     # Format x-axis labels for better readability
    #     plt.xticks(rotation=45)
    #     plt.xlabel("Timestamp")

    #     # Title
    #     plt.title("Price Data, Spread, and Statistics Over Time")

    #     # Show the plot
    #     plt.tight_layout()

    #     if show_plot:
    #         plt.show()

    # @staticmethod
    # def plot_zscore(zscore_series: pd.Series, window=20):
    #     """
    #     Plot Z-score along with its mean and standard deviations (1 and 2) on the right y-axis.

    #     Parameters:
    #         zscore_series (pd.Series): Series containing the Z-score data.
    #         window (int): Rolling window size for calculating mean and standard deviations (default is 20).
    #     """
    #     # Create a figure and primary axis for Z-score (left y-axis)
    #     fig, ax1 = plt.subplots(figsize=(12, 6))

    #     # Plot Z-score on the left y-axis
    #     ax1.plot(
    #         zscore_series.index,
    #         zscore_series,
    #         label="Z-Score",
    #         color="blue",
    #         linewidth=2,
    #     )

    #     ax1.set_yscale("linear")
    #     ax1.set_ylabel("Z-Score")
    #     ax1.legend(loc="upper left")

    #     # Calculate mean and standard deviations for Z-score
    #     zscore_mean = zscore_series.rolling(window=window).mean()
    #     zscore_std_1 = zscore_series.rolling(
    #         window=window
    #     ).std()  # 1 standard deviation
    #     zscore_std_2 = (
    #         2 * zscore_series.rolling(window=window).std()
    #     )  # 2 standard deviations

    #     # Create a secondary axis for mean and standard deviations (right y-axis)
    #     ax2 = ax1.twinx()

    #     # Plot mean and standard deviations on the right y-axis
    #     ax2.plot(
    #         zscore_series.index,
    #         zscore_mean,
    #         label="Mean",
    #         color="orange",
    #         linestyle="--",
    #     )
    #     ax2.fill_between(
    #         zscore_series.index,
    #         zscore_mean - zscore_std_1,
    #         zscore_mean + zscore_std_1,
    #         color="gray",
    #         alpha=0.2,
    #         label="1 Std Dev",
    #     )
    #     ax2.fill_between(
    #         zscore_series.index,
    #         zscore_mean - zscore_std_2,
    #         zscore_mean + zscore_std_2,
    #         color="gray",
    #         alpha=0.4,
    #         label="2 Std Dev",
    #     )
    #     ax2.set_yscale("linear")
    #     ax2.set_ylabel("Statistics")
    #     ax2.legend(loc="upper right")

    #     # Add grid lines
    #     ax1.grid(True)

    #     # Format x-axis labels for better readability
    #     plt.xticks(rotation=45)
    #     plt.xlabel("Timestamp")

    #     # Title
    #     plt.title("Z-Score and Statistics Over Time")

    #     # Show the plot
    #     plt.tight_layout()
    #     plt.show()

    # @staticmethod
    # def plot_price_and_spread_w_signals(
    #     price_data: pd.DataFrame,
    #     spread: pd.Series,
    #     signals: list,
    #     split_date=None,
    #     show_plot=True,
    # ):
    #     """
    #     Plot multiple ticker data on the left y-axis, spread with mean and standard deviations on the right y-axis,
    #     and trading signals as icons.

    #     Parameters:
    #         price_data (pd.DataFrame): DataFrame containing the data with timestamps as index and multiple ticker columns.
    #         spread (pd.Series): Series containing the spread data.
    #         signals (pd.DataFrame): DataFrame containing signal data with timestamps as index and 'signal' column indicating 'long' or 'short'.
    #     """
    #     # Extract data from the DataFrame
    #     timestamps = price_data.index

    #     # Create a figure and primary axis for price data (left y-axis)
    #     fig, ax1 = plt.subplots(figsize=(12, 6))

    #     # Plot each ticker on the left y-axis
    #     colors = [
    #         "blue",
    #         "green",
    #         "red",
    #         "cyan",
    #         "magenta",
    #         "yellow",
    #         "black",
    #         "orange",
    #     ]  # Extend this list as needed
    #     for i, ticker in enumerate(price_data.columns):
    #         color = colors[i % len(colors)]  # Cycle through colors
    #         ax1.plot(
    #             timestamps, price_data[ticker], label=ticker, color=color, linewidth=2
    #         )

    #     ax1.set_yscale("linear")
    #     ax1.set_ylabel("Price")
    #     ax1.legend(loc="upper left")

    #     # Calculate mean and standard deviations for spread
    #     spread_mean = spread.rolling(
    #         window=20
    #     ).mean()  # Adjust the window size as needed
    #     spread_std_1 = spread.rolling(window=20).std()  # 1 standard deviation
    #     spread_std_2 = 2 * spread.rolling(window=20).std()  # 2 standard deviations

    #     # Create a secondary axis for the spread with mean and standard deviations (right y-axis)
    #     ax2 = ax1.twinx()

    #     # Plot Spread on the right y-axis
    #     ax2.plot(timestamps, spread, label="Spread", color="purple", linewidth=2)
    #     ax2.plot(timestamps, spread_mean, label="Mean", color="orange", linestyle="--")
    #     ax2.fill_between(
    #         timestamps,
    #         spread_mean - spread_std_1,
    #         spread_mean + spread_std_1,
    #         color="gray",
    #         alpha=0.2,
    #         label="1 Std Dev",
    #     )
    #     ax2.fill_between(
    #         timestamps,
    #         spread_mean - spread_std_2,
    #         spread_mean + spread_std_2,
    #         color="gray",
    #         alpha=0.4,
    #         label="2 Std Dev",
    #     )
    #     ax2.set_yscale("linear")
    #     ax2.set_ylabel("Spread and Statistics")
    #     ax2.legend(loc="upper right")

    #     # Plot signals
    #     for signal in signals:
    #         ts = pd.to_datetime(signal["timestamp"])
    #         price = signal["price"]
    #         action = signal["action"]
    #         if action == "long":
    #             marker = "^"
    #             color = "lime"
    #         elif action == "short":
    #             marker = "v"
    #             color = "red"
    #         else:
    #             # Default marker for undefined actions
    #             marker = "o"
    #             color = "gray"
    #         ax1.scatter(ts, price, marker=marker, color=color, s=100)

    #     # Draw a dashed vertical line to separate test and training data
    #     if split_date is not None:
    #         split_date = pd.to_datetime(split_date)
    #         ax1.axvline(x=split_date, color="black", linestyle="--", linewidth=1)

    #     # Add grid lines
    #     ax1.grid(True)

    #     # Format x-axis labels for better readability
    #     plt.xticks(rotation=45)
    #     plt.xlabel("Timestamp")

    #     # Title
    #     plt.title("Price Data, Spread, Statistics, and Trading Signals Over Time")

    #     # Show the plot
    #     plt.tight_layout()

    #     if show_plot:
    #         plt.show()
