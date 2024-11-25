from .logic import Cointegrationzscore
from quantAnalytics.data.handler import DataHandler
from quantAnalytics.backtest.backtester import VectorizedBacktest
from mbn import BufferStore


def main():
    # Load data
    buffer_obj = BufferStore.from_file("tests/integration/test_data.bin")
    metadata = buffer_obj.metadata
    print(metadata)
    df = buffer_obj.decode_to_df(pretty_ts=False, pretty_px=True)

    # Process Data
    df.drop(columns=["length", "rtype", "instrument_id"], inplace=True)
    print(DataHandler.check_duplicates(df))
    print(DataHandler.check_null(df))
    df = DataHandler.align_timestamps(df, "drop")
    df = df.pivot(index="ts_event", columns="symbol", values="close")
    df.dropna(inplace=True)
    # df.reset_index(inplace=True)

    # Parameters
    initial_capital = 10000
    contract_details = {
        "HE.n.0": {"quantity_multiplier": 40000, "price_multiplier": 0.01},
        "ZC.n.0": {"quantity_multiplier": 5000, "price_multiplier": 0.01},
    }
    tickers = list(contract_details.keys())

    # Backtest
    backtest = VectorizedBacktest(
        Cointegrationzscore(tickers),
        df,
        contract_details,
        initial_capital,
        "backtest.html",
        "/Users/anthony/projects/midas/quantAnalytics/tests/integration/outputs/backtest",
    )
    backtest.run(position_lag=1)
    backtest.summary()


if __name__ == "__main__":
    main()
