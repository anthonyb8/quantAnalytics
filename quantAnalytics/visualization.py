import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualizations:
    @staticmethod
    def line_plot(x: pd.Series, y:pd.Series, title:str="Time Series Plot", x_label:str="Time", y_label:str="Value") -> plt.Figure:
        """
        Create a line plot for the given x and y data.

        Parameters:
        - x (pd.Series): The x-axis data.
        - y (pd.Series): The y-axis data.
        - title (str): Title of the plot.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.

        Returns:
        - plt.Figure: The line plot.

        Example:
        >>> TimeseriesTests.line_plot(x, y, title='Test Line Plot', x_label='Index', y_label='Random Value')
        >>> plt.show()
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the line
        ax.plot(x, y, marker='o')

        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Add grid
        ax.grid(True)

        # Adjust layout to minimize white space
        plt.tight_layout()

        # Return the figure object
        return fig

    @staticmethod
    def double_line_plot(data1:pd.DataFrame, data2:pd.DataFrame, label1:str="Series 1", label2:str="Series 2", title:str="Double Line Plot") -> plt.Figure:
        """
        Plot two data series for each column in the DataFrame.

        Parameters:
        - data1 (pd.DataFrame): First DataFrame of values.
        - data2 (pd.DataFrame): Second DataFrame of values.
        - label1 (str): Label for the first series.
        - label2 (str): Label for the second series.
        - title (str): Title of the plot.

        Returns:
        - plt.Figure: The figure object containing the plots.

        Example:
        >>> TimeseriesTests.double_line_plot(data1, data2, label1='Data 1', label2='Data 2', title='Test Double Line Plot')
        >>> plt.show()
        """
        # Ensure data1 and data2 columns match
        if set(data1.columns) != set(data2.columns):
            raise ValueError("Columns of data1 and data2 do not match")

        # Iterate over each column to create separate plots
        for column in data1.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot the first series
            ax.plot(data1[column], label=f"{label1} {column}", color='blue')

            # Plot the second series
            if column in data2.columns:
                ax.plot(data2[column], label=f"{label2} {column}", color='red')

            # Add labels, legend, and grid
            ax.grid(True)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(f'{title} for {column}')
            ax.legend()

            # Customize x-axis labels for readability
            plt.xticks(rotation=45)

            # Adjust layout to minimize white space
            plt.tight_layout()

            # Return the figure object
            return fig

    @staticmethod
    def plot_data_with_signals(data:pd.DataFrame, signals:list, title:str="Price Data with Trade Signals", x_label:str="Timestamp", y_label:str="Price") -> plt.Figure:
        """
        Create a line plot with trade signals for the given data.

        Parameters:
        - data (pd.DataFrame): The price data with symbols as columns and timestamps as the index.
        - signals (list): A list of dictionaries containing 'timestamp', 'price', and 'direction' (1 for buy, -1 for sell).
        - title (str): Title of the plot.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.

        Returns:
        - plt.Figure: The line plot with signals.

        Example:
        >>> data = pd.DataFrame({'symbol1': [1, 2, 3], 'symbol2': [4, 5, 6]}, index=pd.date_range('2023-01-01', periods=3))
        >>> signals = [{'timestamp': '2023-01-01', 'price': 2, 'direction': 1}, {'timestamp': '2023-01-02', 'price': 5, 'direction': -1}]
        >>> Plotting.plot_data_with_signals(data, signals)
        >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot the price data
        for symbol in data.columns:
            ax.plot(data.index, data[symbol], label=symbol, marker='o', zorder=1)

        # Plot the signals
        for signal in signals:
            signal_timestamp = pd.to_datetime(signal['timestamp'])
            color = 'green' if signal['direction'] == 1 else 'red'
            marker = 'o' if signal['direction'] == 1 else 'x'
            ax.scatter(signal_timestamp, signal['price'], color=color, marker=marker, zorder=2)

        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True)

        # Adjust layout to minimize white space
        plt.tight_layout()

        return fig

    
    # # --- UNTESTED --
    @staticmethod
    def plot_price_and_spread(price_data:pd.DataFrame, spread:list, signals: list, split_date=None, show_plot=True):
        """
        Plot multiple ticker data on the left y-axis and spread with mean and standard deviations on the right y-axis.
        
        Parameters:
            price_data (pd.DataFrame): DataFrame containing the data with timestamps as index and multiple ticker columns.
            spread (pd.Series): Series containing the spread data.
        """
        # Extract data from the DataFrame
        timestamps = price_data.index
        spread = pd.Series(spread, index=timestamps) 

        # Create a figure and primary axis for price data (left y-axis)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot each ticker on the left y-axis
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']  # Extend this list as needed
        for i, ticker in enumerate(price_data.columns):
            color = colors[i % len(colors)]  # Cycle through colors
            ax1.plot(timestamps, price_data[ticker], label=ticker, color=color, linewidth=2)

        ax1.set_yscale('linear')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')

        # Calculate mean and standard deviations for spread
        spread_mean = spread.rolling(window=20).mean()  # Adjust the window size as needed
        spread_std_1 = spread.rolling(window=20).std()  # 1 standard deviation
        spread_std_2 = 2 * spread.rolling(window=20).std()  # 2 standard deviations

        # Create a secondary axis for the spread with mean and standard deviations (right y-axis)
        ax2 = ax1.twinx()

        # Plot Spread on the right y-axis
        ax2.plot(timestamps, spread, label='Spread', color='purple', linewidth=2)
        ax2.plot(timestamps, spread_mean, label='Mean', color='orange', linestyle='--')
        ax2.fill_between(timestamps, spread_mean - spread_std_1, spread_mean + spread_std_1, color='gray', alpha=0.2, label='1 Std Dev')
        ax2.fill_between(timestamps, spread_mean - spread_std_2, spread_mean + spread_std_2, color='gray', alpha=0.4, label='2 Std Dev')
        ax2.set_yscale('linear')
        ax2.set_ylabel('Spread and Statistics')
        ax2.legend(loc='upper right')


        # Plot signals
        for signal in signals:
            ts = pd.to_datetime(signal['timestamp'])
            price = signal['price']
            action = signal['action']
            if action in ['LONG', 'COVER']:
                marker = '^'
                color = 'lime'
            elif action in ['SHORT', 'SELL']:
                marker = 'v'
                color = 'red'
            else:
                # Default marker for undefined actions
                marker = 'o'
                color = 'gray'
            ax1.scatter(ts, price, marker=marker, color=color, s=100)

        # Draw a dashed vertical line to separate test and training data
        if split_date is not None:
            split_date = pd.to_datetime(split_date)
            ax1.axvline(x=split_date, color='black', linestyle='--', linewidth=1)

        # Add grid lines
        ax1.grid(True)

        # Format x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.xlabel('Timestamp')

        # Title
        plt.title('Price Data, Spread, and Statistics Over Time')

        # Show the plot
        plt.tight_layout()

        return fig 
   
    @staticmethod
    def plot_price_and_spread(price_data:pd.DataFrame, spread:pd.Series, show_plot=True):
        """
        Plot multiple ticker data on the left y-axis and spread with mean and standard deviations on the right y-axis.
        
        Parameters:
            price_data (pd.DataFrame): DataFrame containing the data with timestamps as index and multiple ticker columns.
            spread (pd.Series): Series containing the spread data.
        """
        # Extract data from the DataFrame
        timestamps = price_data.index

        # Create a figure and primary axis for price data (left y-axis)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot each ticker on the left y-axis
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']  # Extend this list as needed
        for i, ticker in enumerate(price_data.columns):
            color = colors[i % len(colors)]  # Cycle through colors
            ax1.plot(timestamps, price_data[ticker], label=ticker, color=color, linewidth=2)

        ax1.set_yscale('linear')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')

        # Calculate mean and standard deviations for spread
        spread_mean = spread.rolling(window=20).mean()  # Adjust the window size as needed
        spread_std_1 = spread.rolling(window=20).std()  # 1 standard deviation
        spread_std_2 = 2 * spread.rolling(window=20).std()  # 2 standard deviations

        # Create a secondary axis for the spread with mean and standard deviations (right y-axis)
        ax2 = ax1.twinx()

        # Plot Spread on the right y-axis
        ax2.plot(timestamps, spread, label='Spread', color='purple', linewidth=2)
        ax2.plot(timestamps, spread_mean, label='Mean', color='orange', linestyle='--')
        ax2.fill_between(timestamps, spread_mean - spread_std_1, spread_mean + spread_std_1, color='gray', alpha=0.2, label='1 Std Dev')
        ax2.fill_between(timestamps, spread_mean - spread_std_2, spread_mean + spread_std_2, color='gray', alpha=0.4, label='2 Std Dev')
        ax2.set_yscale('linear')
        ax2.set_ylabel('Spread and Statistics')
        ax2.legend(loc='upper right')

        # Add grid lines
        ax1.grid(True)

        # Format x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.xlabel('Timestamp')

        # Title
        plt.title('Price Data, Spread, and Statistics Over Time')

        # Show the plot
        plt.tight_layout()
        
        if show_plot:
            plt.show()

    @staticmethod
    def plot_zscore(zscore_series:pd.Series, window=20):
        """
        Plot Z-score along with its mean and standard deviations (1 and 2) on the right y-axis.
        
        Parameters:
            zscore_series (pd.Series): Series containing the Z-score data.
            window (int): Rolling window size for calculating mean and standard deviations (default is 20).
        """
        # Create a figure and primary axis for Z-score (left y-axis)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Z-score on the left y-axis
        ax1.plot(zscore_series.index, zscore_series, label='Z-Score', color='blue', linewidth=2)

        ax1.set_yscale('linear')
        ax1.set_ylabel('Z-Score')
        ax1.legend(loc='upper left')

        # Calculate mean and standard deviations for Z-score
        zscore_mean = zscore_series.rolling(window=window).mean()
        zscore_std_1 = zscore_series.rolling(window=window).std()  # 1 standard deviation
        zscore_std_2 = 2 * zscore_series.rolling(window=window).std()  # 2 standard deviations

        # Create a secondary axis for mean and standard deviations (right y-axis)
        ax2 = ax1.twinx()

        # Plot mean and standard deviations on the right y-axis
        ax2.plot(zscore_series.index, zscore_mean, label='Mean', color='orange', linestyle='--')
        ax2.fill_between(zscore_series.index, zscore_mean - zscore_std_1, zscore_mean + zscore_std_1, color='gray', alpha=0.2, label='1 Std Dev')
        ax2.fill_between(zscore_series.index, zscore_mean - zscore_std_2, zscore_mean + zscore_std_2, color='gray', alpha=0.4, label='2 Std Dev')
        ax2.set_yscale('linear')
        ax2.set_ylabel('Statistics')
        ax2.legend(loc='upper right')

        # Add grid lines
        ax1.grid(True)

        # Format x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.xlabel('Timestamp')

        # Title
        plt.title('Z-Score and Statistics Over Time')

        # Show the plot
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_price_and_spread_w_signals(price_data: pd.DataFrame, spread: pd.Series, signals: list, split_date=None, show_plot=True):
        """
        Plot multiple ticker data on the left y-axis, spread with mean and standard deviations on the right y-axis,
        and trading signals as icons.
        
        Parameters:
            price_data (pd.DataFrame): DataFrame containing the data with timestamps as index and multiple ticker columns.
            spread (pd.Series): Series containing the spread data.
            signals (pd.DataFrame): DataFrame containing signal data with timestamps as index and 'signal' column indicating 'long' or 'short'.
        """
        # Extract data from the DataFrame
        timestamps = price_data.index

        # Create a figure and primary axis for price data (left y-axis)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot each ticker on the left y-axis
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']  # Extend this list as needed
        for i, ticker in enumerate(price_data.columns):
            color = colors[i % len(colors)]  # Cycle through colors
            ax1.plot(timestamps, price_data[ticker], label=ticker, color=color, linewidth=2)

        ax1.set_yscale('linear')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')

        # Calculate mean and standard deviations for spread
        spread_mean = spread.rolling(window=20).mean()  # Adjust the window size as needed
        spread_std_1 = spread.rolling(window=20).std()  # 1 standard deviation
        spread_std_2 = 2 * spread.rolling(window=20).std()  # 2 standard deviations

        # Create a secondary axis for the spread with mean and standard deviations (right y-axis)
        ax2 = ax1.twinx()

        # Plot Spread on the right y-axis
        ax2.plot(timestamps, spread, label='Spread', color='purple', linewidth=2)
        ax2.plot(timestamps, spread_mean, label='Mean', color='orange', linestyle='--')
        ax2.fill_between(timestamps, spread_mean - spread_std_1, spread_mean + spread_std_1, color='gray', alpha=0.2, label='1 Std Dev')
        ax2.fill_between(timestamps, spread_mean - spread_std_2, spread_mean + spread_std_2, color='gray', alpha=0.4, label='2 Std Dev')
        ax2.set_yscale('linear')
        ax2.set_ylabel('Spread and Statistics')
        ax2.legend(loc='upper right')

        # Plot signals
        for signal in signals:
            ts = pd.to_datetime(signal['timestamp'])
            price = signal['price']
            action = signal['action']
            if action == 'long':
                marker = '^'
                color = 'lime'
            elif action == 'short':
                marker = 'v'
                color = 'red'
            else:
                # Default marker for undefined actions
                marker = 'o'
                color = 'gray'
            ax1.scatter(ts, price, marker=marker, color=color, s=100)

        # Draw a dashed vertical line to separate test and training data
        if split_date is not None:
            split_date = pd.to_datetime(split_date)
            ax1.axvline(x=split_date, color='black', linestyle='--', linewidth=1)

        # Add grid lines
        ax1.grid(True)

        # Format x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.xlabel('Timestamp')

        # Title
        plt.title('Price Data, Spread, Statistics, and Trading Signals Over Time')

        # Show the plot
        plt.tight_layout()
        
        if show_plot:
            plt.show()

 