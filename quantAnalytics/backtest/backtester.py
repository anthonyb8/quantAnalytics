import numpy as np
import pandas as pd
from quantAnalytics.report.report import (
    ReportBuilder,
    DivBuilder,
    Header,
)
from quantAnalytics.utils import resample_daily
from quantAnalytics.backtest.base_strategy import BaseStrategy
from quantAnalytics.backtest.metrics import Metrics
from quantAnalytics.analysis.plots import Plot


class VectorizedBacktest(Metrics):
    """
    A class for conducting vectorized backtesting of trading strategies on historical data.

    This class provides a structured approach to evaluate the performance of trading strategies by simulating trades based on historical price data and calculating key performance metrics. It is designed to be efficient by utilizing vectorized operations, which typically provide better performance over iterative methods for large datasets.

    Attributes:
    - full_data (pd.DataFrame): The complete dataset containing historical price data for one or more securities.
    - strategy (BaseStrategy): An instance of a strategy derived from the BaseStrategy class, which defines the logic for trade signal generation.
    - initial_capital (float): The starting capital for the backtest. Defaults to $10,000.
    - symbols (list): A list of column names from `full_data` representing different securities.
    - equity_curve (pd.Series): A pandas Series representing the equity value over time, updated during the backtest.
    - backtest_data (pd.DataFrame): A DataFrame to store the results of the backtest, including trading signals and performance metrics.

    Methods:
    - setup(): Optional preparation for the backtesting environment, such as loading additional data or configuring parameters.
    - run_backtest(entry_threshold, exit_threshold): Executes the backtest using specified entry and exit thresholds for trading signals.
    - _calculate_equity_curve(): Calculates the equity curve based on trading signals and updates the `equity_curve` attribute.
    - _calculate_metrics(risk_free_rate=0.04): Computes various performance metrics such as the Sharpe Ratio and maximum drawdown.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbols_map: dict,
        initial_capital: float = 10000,
        file_name: str = "backtest",
        output_directory: str = "report",
        css_path: str = "",
    ):
        """
        Initializes the VectorizedBacktest object with a dataset, a trading strategy, and initial capital.

        Parameters:
        - data (pandas.DataFrame): A DataFrame containing the full dataset to be used in the backtest.
        - strategy (BaseStrategy): An instance of BaseStrategy that defines the trading logic.
        - initial_capital (float): The initial capital amount in dollars. Defaults to 10,000.
        """
        self.report = ReportBuilder(file_name, output_directory, css_path)
        self.strategy = strategy
        self.data = data
        self.symbols_map = symbols_map
        self.initial_capital = initial_capital
        self.output_dir = output_directory
        self.daily_data: pd.DataFrame
        self.stats_df: pd.DataFrame

        self.setup()

    def _initialize_signal_columns(self):
        """
        Pre-create signal columns for each symbol in the data with all zeros.
        """
        for symbol in self.symbols_map.keys():
            self.data[f"{symbol}_signal"] = np.nan

    def setup(self):
        """
        Dynamic setup based on the strategy’s preparation needs.
        After the preparation, the backtest will start where the preparation ended.
        """
        print("Setting up backtest...")

        try:
            # Pre-create signal columns with zeros in self.data
            self._initialize_signal_columns()

            preparation_html = self.strategy.prepare(self.data)

            if preparation_html:
                self.report.add_html_block(preparation_html)

            print("Preparation complete.")
        except Exception as e:
            raise Exception(f"Error during strategy preparation: {e}")

    def run(self, position_lag: int = 1):
        """
        Executes the backtest by generating trading signals and calculating positions, P&L, and equity curve.
        """
        print("Running backtest...")

        try:
            # Generate signals from the strategy
            self.strategy.generate_signals()

            # Calculate positions based on signals
            self._calculate_positions(position_lag)

            # Calculate profit and loss (PnL)
            self._calculate_positions_pnl()

            # Calculate equity curve
            self._calculate_equity_curve()

            print("Backtest run completed successfully.")
        except Exception as e:
            raise Exception(f"Backtest run failed: {e}")

    def summary(self):
        """
        Generates performance summary metrics and builds a report.
        """
        print("Generating summary and report...")
        self._calculate_metrics()

        self.data["datetime"] = pd.to_datetime(self.data.index, unit="ns")
        # self.daily_data["datetime"] = pd.to_datetime(
        #     self.daily_data.index, unit="ns"
        # )
        #
        # Build performance report
        self.build_report()

        # Dump dataframes to excel
        self.dump_to_excel()

        # Dump to parquet for easy use in Future
        self.dump_to_parquet()

        print("Summary and report generation completed.")

    def _calculate_positions(self, lag: int) -> None:
        """
        Calculates the positions held over the course of the backtest.

        Parameters:
        - signals (pd.DataFrame): DataFrame containing trading signals for each ticker.
        - lag (int): The number of periods the entry/exit of a position will be lagged after a signal.
        """
        for symbol in self.symbols_map.keys():
            # Position signals are filled forward until changed
            position_column = f"{symbol}_position"
            self.data[position_column] = (
                self.data[f"{symbol}_signal"].ffill().shift(lag).fillna(0)
            )

            # Retrieve multipliers and weights
            hedge_ratio = abs(self.strategy.weights[symbol])
            quantity_multiplier = self.symbols_map[symbol][
                "quantity_multiplier"
            ]
            price_multiplier = self.symbols_map[symbol]["price_multiplier"]

            # Calculate the dollar value of each position
            position_value_column = f"{symbol}_position_value"
            self.data[position_value_column] = (
                self.data[position_column]
                * self.data[symbol]
                * hedge_ratio
                * quantity_multiplier
                * price_multiplier
            )

    def _calculate_positions_pnl(self) -> None:
        pnl_columns = []

        for column in self.data.filter(like="_position_value").columns:
            profit_loss_column = column.replace(
                "position_value", "position_pnl"
            )
            self.data[profit_loss_column] = self.data[column].diff().fillna(0)

            # Identify the start of a new position (assuming a new position starts when the previous value is 0)
            # Set the P&L to zero at the start of new positions
            new_position_starts = (self.data[column] != 0) & (
                self.data[column].shift(1) == 0
            )
            self.data.loc[new_position_starts, profit_loss_column] = 0

            # Identify the end of a position (position value goes to zero)
            # Set the P&L to zero at the end of positions
            position_ends = (self.data[column] == 0) & (
                self.data[column].shift(1) != 0
            )
            self.data.loc[position_ends, profit_loss_column] = 0

            # Add the P&L column name to the list
            pnl_columns.append(profit_loss_column)

        # Sum all P&L columns to create a single 'portfolio_pnl' column
        self.data["portfolio_pnl"] = self.data[pnl_columns].sum(axis=1)

    def _calculate_equity_curve(self) -> None:
        # Initialize the equity_curve column with the initial capital
        self.data["equity_value"] = self.initial_capital

        # Calculate the equity_curve by cumulatively summing the portfolio_pnl with the initial capital
        self.data["equity_value"] = (
            self.data["equity_value"].shift(1).fillna(self.initial_capital)
            + self.data["portfolio_pnl"].cumsum()
        )

    def _calculate_metrics(self, risk_free_rate: float = 0.04) -> None:
        """
        Calculates performance metrics for the backtest including period return, cumulative return, drawdown, and Sharpe ratio.
        ** Note: All Summary statistics are calculated on an annualized basis.

        Parameters:
        - risk_free_rate (float): The annual risk-free rate used for calculating the Sharpe ratio. Default is 0.04 (4%).
        """
        # Ensure that equity values are numeric and NaNs are handled
        self.daily_data = resample_daily(self.data.copy(), "EST")

        daily_equity_values = pd.to_numeric(
            self.daily_data["equity_value"], errors="coerce"
        ).fillna(0)
        equity_values = pd.to_numeric(
            self.data["equity_value"], errors="coerce"
        ).fillna(0)

        # Compute simple and cumulative returns
        period_returns = self.simple_returns(equity_values.values)
        period_returns_adjusted = np.insert(
            period_returns, 0, 0
        )  # Adjust for initial zero return
        cumulative_returns = self.cumulative_returns(equity_values.values)
        cumulative_returns_adjusted = np.insert(cumulative_returns, 0, 0)

        # Update test DataFrame
        self.data["period_return"] = period_returns_adjusted
        self.data["cumulative_return"] = cumulative_returns_adjusted
        self.data["drawdown"] = Metrics.drawdown(period_returns_adjusted)

        # Compute simple and cumulative returns daily data
        daily_returns = self.simple_returns(daily_equity_values.values)
        daily_returns_adjusted = np.insert(
            daily_returns, 0, 0
        )  # Adjust for initial zero return
        daily_cumulative_returns = self.cumulative_returns(
            daily_equity_values.values
        )
        daily_cumulative_returns_adjusted = np.insert(
            daily_cumulative_returns, 0, 0
        )

        # Update daily dataframe
        self.daily_data["period_return"] = daily_returns_adjusted
        self.daily_data["cumulative_return"] = (
            daily_cumulative_returns_adjusted
        )
        self.daily_data["drawdown"] = Metrics.drawdown(daily_returns_adjusted)

        # Calculate summary statistics
        self.summary_stats = {
            "annual_standard_deviation": Metrics.annual_standard_deviation(
                daily_returns_adjusted
            ),
            "sharpe_ratio": Metrics.sharpe_ratio(
                daily_returns_adjusted, risk_free_rate
            ),
            "sortino_ratio": Metrics.sortino_ratio(daily_returns_adjusted),
            "max_drawdown": Metrics.max_drawdown(
                period_returns_adjusted
            ),  # standardized
            "ending_equity": equity_values.values[-1],  # raw
        }

        # Summary Stats table
        self.stats_df = pd.DataFrame.from_dict(
            self.summary_stats, orient="index", columns=["Value"]
        )

    def dump_to_excel(self) -> None:
        """
        Dumps the raw data and daily data into an Excel document with separate sheets.

        Parameters:
        - file_path (str): The path where the Excel document will be saved.
        """
        with pd.ExcelWriter(
            f"{self.output_dir}/data.xlsx", engine="xlsxwriter"
        ) as writer:
            # Write raw data to the first sheet
            self.data.to_excel(writer, sheet_name="Raw Data")

            # Write daily data to the second sheet
            self.daily_data.to_excel(writer, sheet_name="Daily Data")

            # Write summary stats to excel thir sheet
            self.stats_df.to_excel(writer, sheet_name="Summary Stats")

    def dump_to_parquet(self) -> None:
        """
        Dumps the raw data and daily data to Parquet for regression analysis.
        """
        # Store daily data in Parquet
        self.daily_data.to_parquet(f"{self.output_dir}/daily_data.parquet")

        # Store raw data in Parquet
        self.data.to_parquet(f"{self.output_dir}/raw_data.parquet")

        print(
            f"Data successfully written to {self.output_dir} in Parquet format."
        )

    def build_report(self):
        # self.data["datetime"] = pd.to_datetime(self.data.index, unit="ns")
        # self.data.index.name = "datetime"

        # Plots
        equity_plot_path = f"{self.output_dir}/equity_plot.png"
        Plot.plot_line(
            data=self.data.set_index("datetime")["equity_value"],
            title="Equity Curve",
            xlabel="Time",
            ylabel="Equity Value",
            save_path=equity_plot_path,
        )

        cum_return_path = f"{self.output_dir}/return_plot.png"
        Plot.plot_line(
            data=self.data.set_index("datetime")["cumulative_return"],
            title="Cumulative Return",
            xlabel="Time",
            ylabel="Return Value",
            save_path=cum_return_path,
        )

        drawdown_path = f"{self.output_dir}/drawdown_plot.png"
        Plot.plot_line(
            data=self.data.set_index("datetime")["drawdown"],
            title="Drawdown Curve",
            xlabel="Time",
            ylabel="Drawdown Value",
            save_path=drawdown_path,
        )

        performance_div = DivBuilder("performance")
        performance_div.add_header("Performance Metrics", Header.H3)
        performance_div.add_table(
            self.stats_df,
            header="Summary Stats",
            header_size=Header.H3,
            index=True,
        )
        performance_div.add_image(equity_plot_path)
        performance_div.add_image(cum_return_path)
        performance_div.add_image(drawdown_path)

        self.report.add_div(performance_div.build())
        self.report.build()
