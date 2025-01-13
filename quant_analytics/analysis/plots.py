import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Optional, Union
import numpy as np
import seaborn as sns
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.dates as mdates
from typing import Callable


class Plot:
    @staticmethod
    def plot_line(
        data: pd.Series,
        title="Line Plot",
        xlabel="Date",
        ylabel="Value",
        xtick_interval=None,
        hline=False,
        height: int = 12,
        width: int = 6,
        save_path=None,
    ) -> plt.Figure:
        """
        Plots a line graph from a pandas Series.

        Parameters:
            data (pd.Series): The data to plot. The index is used as the x-axis, and values as the y-axis.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            xtick_interval (int): Interval for x-axis ticks (optional).
            save_path (str): Path to save the plot. If None, the plot is returned.

        Returns:
            plt.Figure: The created matplotlib figure.
        """
        # Check if input is a pandas Series
        if not isinstance(data, pd.Series):
            raise ValueError("The data parameter must be a pandas Series.")

        # Extract x (index) and y (values) from the Series
        x = data.index
        y = data.values

        # Create the plot
        fig, ax = plt.subplots(figsize=(height, width))
        ax.plot(
            x, y, alpha=0.8, label=data.name or "Data"
        )  # Use the Series name if available
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

        # Customize x-axis ticks for datetime data
        if isinstance(x, pd.DatetimeIndex):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()  # Rotate and format x-axis labels

        # Add legend
        ax.legend()

        if hline:
            ax.axhline(y=0, color="red", linestyle="--")

        # Save or return the figure
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def plot_multiline(
        data: pd.DataFrame,
        title: str = "Multiline Plot",
        xlabel: str = "Date",
        ylabel: str = "Value",
        xtick_interval: int = 300,
        grid: bool = True,
        layout_tight: bool = True,
        height: int = 12,
        width: int = 6,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plots a multiline graph where each line represents a column in the DataFrame.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the data to plot. Columns are plotted as separate lines.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            xtick_interval (int): Interval between x-axis ticks.
            grid (bool): Whether to include a grid on the plot.
            layout_tight (bool): Whether to use tight layout to prevent overlap.
            save_path (str): Path to save the plot. If None, the plot will be returned.

        Returns:
            plt.Figure: The created matplotlib figure.
        """
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(height, width))

        # Plot each column as a line
        for column in data.columns:
            ax.plot(data.index, data[column], label=column, alpha=0.8)

        # Add titles and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title="Columns")

        # Enable grid if specified
        if grid:
            ax.grid(alpha=0.3)

        # Customize x-axis ticks for datetime data
        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()  # Rotate and format x-axis labels

        # Use tight layout if specified
        if layout_tight:
            plt.tight_layout()

        # Save or return the plot
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def plot_scatter(
        x: Union[np.ndarray, pd.Series, list],
        y: Union[np.ndarray, pd.Series, list],
        title: str = "Scatter Plot",
        xlabel: str = "X-axis",
        ylabel: str = "Y-axis",
        alpha: float = 0.5,
        color: str = "blue",
        marker: str = "o",
        grid: bool = True,
        hline: float = 0,  # Add a horizontal line at this value if not None
        hline_color: str = "red",
        hline_style: str = "--",
        height: int = 12,
        width: int = 6,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Creates a scatter plot with optional customization.

        Parameters:
            x (array-like): Values for the x-axis.
            y (array-like): Values for the y-axis.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            alpha (float): Transparency level for scatter points.
            color (str): Color of the scatter points.
            marker (str): Marker style for the scatter points.
            grid (bool): Whether to display a grid.
            hline (float): Optional horizontal line value. If None, no line is drawn.
            hline_color (str): Color of the horizontal line.
            hline_style (str): Style of the horizontal line.
            save_path (str): Path to save the plot. If None, the figure is returned.

        Returns:
            plt.Figure: The created matplotlib figure.
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=(height, width))
        ax.scatter(x, y, alpha=alpha, color=color, marker=marker)

        # Add horizontal line if specified
        if hline is not None:
            ax.axhline(y=hline, color=hline_color, linestyle=hline_style)

        # Add titles and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Add grid if specified
        if grid:
            ax.grid(alpha=0.3)

        # Save or return the plot
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def correlation_heatmap(
        df: pd.DataFrame,
        title: str,
        height: int = 12,
        width: int = 6,
        save_path: str = None,
    ) -> plt.Figure:

        correlation_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(height, width))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        plt.title(title)

        # Adjust layout to minimize white space
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def plot_information_criteria(
        model_fit_function: Callable[[int], sm.tsa.VAR],
        maxlags: int = 12,
        title: str = "Lag Order Selection",
        height: int = 12,
        width: int = 6,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plots the AIC, BIC, and HQIC information criteria for selecting the optimal lag order.

        Parameters:
            model_fit_function (Callable[[int], VAR]): A function that takes lag order as input and returns a fitted VAR model.
            maxlags (int): Maximum number of lags to evaluate.
            title (str): Title of the plot.
            save_path (str): Path to save the plot. If None, the figure is returned.

        Returns:
            plt.Figure: The created matplotlib figure.
        """
        # Initialize lists to store the information criteria values
        lags = range(1, maxlags + 1)
        aic, bic, hqic = [], [], []

        # Calculate the information criteria for each lag order
        for lag in lags:
            model = model_fit_function(lag)
            aic.append(model.aic)
            bic.append(model.bic)
            hqic.append(model.hqic)

        # Create the plot
        fig, ax = plt.subplots(figsize=(height, width))
        ax.plot(lags, aic, marker="o", label="AIC")
        ax.plot(lags, bic, marker="o", label="BIC")
        ax.plot(lags, hqic, marker="o", label="HQIC")

        # Add titles and labels
        ax.set_title(title)
        ax.set_xlabel("Lag Order")
        ax.set_ylabel("Criterion Value")
        ax.legend()
        ax.grid(alpha=0.3)

        # Save or return the figure
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid rendering
        else:
            return fig

    @staticmethod
    def plot_acf(residuals: pd.Series, lags: int = 12) -> plt.Figure:
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
        fig, ax = plt.subplots()

        # Plot ACF
        sm.graphics.tsa.plot_acf(residuals, lags=lags, ax=ax)
        ax.set_title("ACF of Residuals")

        # Adjust layout
        fig.tight_layout()

        # Return the figure object
        return fig

    @staticmethod
    def plot_dual_axis(
        primary_data: pd.Series,
        secondary_data: pd.Series,
        title: str = "Dual Axis Plot",
        primary_label: str = "Primary",
        secondary_label: str = "Secondary",
        primary_color: str = "blue",
        secondary_color: str = "red",
        xlabel: str = "Time",
        height: int = 12,
        width: int = 6,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plots a dual-axis chart with two different datasets on separate y-axes.

        Parameters:
            primary_data (pd.Series): Data for the primary y-axis.
            secondary_data (pd.Series): Data for the secondary y-axis.
            title (str): Title of the plot.
            primary_label (str): Label for the primary y-axis data.
            secondary_label (str): Label for the secondary y-axis data.
            primary_color (str): Color for the primary y-axis line and labels.
            secondary_color (str): Color for the secondary y-axis line and labels.
            xlabel (str): Label for the x-axis.
            save_path (str): Path to save the plot. If None, the figure is returned.

        Returns:
            plt.Figure: The created matplotlib figure.
        """
        if not isinstance(primary_data, pd.Series) or not isinstance(
            secondary_data, pd.Series
        ):
            raise ValueError(
                "Both primary_data and secondary_data must be pandas Series."
            )

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(height, width))

        # Plot the primary data
        ax1.plot(
            primary_data.index,
            primary_data,
            label=primary_label,
            color=primary_color,
            alpha=0.7,
        )
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel(primary_label, fontsize=12, color=primary_color)
        ax1.tick_params(axis="y", labelcolor=primary_color)
        ax1.set_title(title, fontsize=14)

        # Plot the secondary data on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(
            secondary_data.index,
            secondary_data,
            label=secondary_label,
            color=secondary_color,
            alpha=0.7,
        )
        ax2.set_ylabel(secondary_label, fontsize=12, color=secondary_color)
        ax2.tick_params(axis="y", labelcolor=secondary_color)

        # Add legend
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=10)

        # Save or return the plot
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid rendering
        else:
            return fig

    @staticmethod
    def qq_plot(
        residuals: pd.Series,
        title: str = "Q-Q Plot",
        height: int = 12,
        width: int = 6,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Generate a Q-Q plot to analyze the normality of residuals in the regression model.

        This method creates a Q-Q plot comparing the quantiles of the residuals to the quantiles of a normal distribution.
        This helps in diagnosing deviations from normality such as skewness and kurtosis.
        """

        # Generate Q-Q plot
        fig = plt.figure(figsize=(height, width))
        ax = fig.add_subplot(111)
        sm.qqplot(residuals, line="45", ax=ax, fit=True)
        ax.set_title(title)
        ax.grid(True)

        # Adjust layout to minimize white space
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def histogram_ndc(
        data: pd.Series,
        bins: str = "auto",
        title: str = "Histogram with Normal Distribution Curve",
        height: int = 12,
        width: int = 6,
        save_path: str = None,
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
        fig, ax = plt.subplots(figsize=(height, width))

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

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def histogram_kde(
        data: pd.Series,
        bins: str = "auto",
        title: str = "Histogram with Kernel Density Estimate (KDE)",
        height: int = 12,
        width: int = 6,
        save_path: str = None,
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
        fig, ax = plt.subplots(figsize=(height, width))

        # Generate histogram with KDE
        sns.histplot(data, bins=bins, kde=True, color="blue", ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    # == old ==

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
        save_path: str = None,
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
        >>> fig = Plots.line_plot_with_markers(data, markers, x_field="time", y_field="value", marker_field="signal")
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

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
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
        save_path: str = None,
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

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
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
        save_path: str = None,
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

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
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
        save_path: str = None,
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

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def plot_residuals_vs_fitted(
        residuals: pd.Series,
        fittedvalues: pd.Series,
        title: str = "Residuals vs Fitted Values",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot residuals against fitted values to diagnose the regression model.

        This method generates a scatter plot of residuals versus fitted values and includes a horizontal line at zero.
        It is used to check for non-random patterns in residuals which could indicate problems with the model such as
        non-linearity, outliers, or heteroscedasticity.

        Parameters:
        - residuals (pd.Series): The residuals from the regression model.
        - fittedvalues (pd.Series): The fitted values from the regression model.
        - title (str): The title of the plot (optional).

        Returns:
        - plt.Figure: The matplotlib figure object for further customization or saving.
        """

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot of residuals vs fitted values
        ax.scatter(
            fittedvalues,
            residuals,
            alpha=0.5,
            color="blue",
            edgecolor="k",
        )

        # Add a horizontal line at 0
        ax.axhline(0, color="red", linestyle="--")

        # Add labels and title
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title(title)

        # Add grid for better readability
        ax.grid(True)

        # Adjust layout to avoid unnecessary whitespace
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig

    @staticmethod
    def plot_influence_measures(
        cooks_d,
        title: str = "Cook's Distance Plot",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot influence measures such as Cook's distance to identify influential cases in the regression model.

        This method plots the Cook's distance for each observation to help identify influential points that might
        affect the robustness of the regression model.
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
        ax.set_title("Cook's Distance Plot")
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Cook's Distance")

        # Adjust layout to minimize white space
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)  # Close the figure to avoid display
        else:
            return fig
