from .logic import Cointegrationzscore
from quant_analytics.data.handler import DataHandler
from quant_analytics.backtest.backtester import VectorizedBacktest
from mbn import BufferStore
from quant_analytics.backtest.base_strategy import SymbolMap
import pandas as pd


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

    df.insert(
        loc=0,
        column="datetime",
        value=pd.to_datetime(df.index, unit="ns"),
    )

    # Parameters
    initial_capital = 10000
    rf_rate = 0.04
    symbol_map = SymbolMap()
    symbol_map.append_symbol("HE.n.0", 40000, 0.01)
    symbol_map.append_symbol("ZC.n.0", 5000, 0.01)

    # Backtest
    backtest = VectorizedBacktest(
        Cointegrationzscore(symbol_map),
        df,
        symbol_map,
        initial_capital,
        rf_rate,
        "backtest.html",
        "/Users/anthony/projects/midas/quant_analytics/tests/integration/outputs/backtest",
    )
    backtest.run(position_lag=1)
    backtest.summary()


if __name__ == "__main__":
    main()
