import numpy as np
import pandas as pd
from quantAnalytics.regression.regression import RegressionAnalysis
from quantAnalytics.backtest.metrics import Metrics
import yfinance as yf
from quantAnalytics.utils import unix_to_iso


def main():
    # Strategy Data
    # From file
    strategy_df = pd.read_parquet(
        "/Users/anthony/projects/midas/quantAnalytics/tests/integration/outputs/backtest/daily_data.parquet"
    )
    # Step 2: Reset index to make 'timestamp' a column
    strategy_df.reset_index(inplace=True)

    # Step 3: Apply 'unix_to_iso' to convert 'timestamp' to ISO format
    strategy_df["date"] = strategy_df["timestamp"].apply(unix_to_iso)

    # Step 4: Convert 'date' to a proper datetime format
    strategy_df["date"] = pd.to_datetime(strategy_df["date"])
    strategy_df["date"] = strategy_df["date"].dt.date
    # Step 5: Set the 'date' column as the new index
    strategy_df.set_index("date", inplace=True)

    # Optional: Drop the 'timestamp' column if it's no longer needed
    strategy_df.drop(columns=["timestamp"], inplace=True)
    strategy_df.drop(
        strategy_df.filter(regex="signal|position").columns,
        axis=1,
        inplace=True,
    )

    # MARKET DATA
    # Get the first and last date
    benchmark = ["^SPGSCI"]
    first_date = strategy_df.index[0]  # Extract the date part
    last_date = strategy_df.index[-1]  # Extract the date part
    benchmark_df = yf.download(benchmark, first_date, last_date)

    # Returns
    benchmark_close = pd.to_numeric(benchmark_df["Close"])
    daily_returns = Metrics.simple_returns(benchmark_close.values)
    benchmark_df["bm_period_return"] = np.insert(daily_returns, 0, 0)

    # Align Data
    data = pd.merge(
        strategy_df,
        benchmark_df[["bm_period_return"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    data.rename(columns={"period_return": "Y"}, inplace=True)

    data = data.iloc[:-1]  # bm one daty short dro last row

    # Check nulls DataFrame
    nan_rows = data[data.isna().any(axis=1)]
    print(f"Rows with NaN values: {len(nan_rows)}")

    inf_rows = data[(data == np.inf) | (data == -np.inf)].any(axis=1)
    print(f"Rows with Infinity or -Infinity values: {len(inf_rows)}")

    # REGRESSION ANALYSIS
    analysis = RegressionAnalysis(
        data=data,
        dependent_var="Y",
        risk_free_rate=0.05,
        file_name="regression.html",
        output_directory="/Users/anthony/projects/midas/quantAnalytics/tests/integration/outputs/regression",
    )
    analysis.run()


if __name__ == "__main__":
    main()

    # # From db
    # client = midasClient.DatabaseClient("http://127.0.0.1:8080")
    # backtest = client.get_backtest(168)["data"]
    # strategy_daily_return = pd.DataFrame.from_records(
    #     backtest["daily_timeseries_stats"], index="ts_event"
    # )
    # strategy_period_return = pd.DataFrame.from_records(
    #     backtest["period_timeseries_stats"], index="ts_event"
    # )
