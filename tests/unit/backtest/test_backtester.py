import unittest
import pandas as pd
from unittest.mock import MagicMock
from quantAnalytics.backtest.base_strategy import BaseStrategy
from quantAnalytics.backtest.backtester import VectorizedBacktest


class TestVectorizedBacktest(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame(
            {
                "ts_event": [
                    1700593200000000000,
                    1700852400000000000,
                    1700938800000000000,
                    1701025200000000000,
                    1701111600000000000,
                ],
                "A": [74.425, 75.400, 75.075, 75.600, 75.500],
                "B": [485.5, 491.5, 493.0, 493.0, 492.5],
            }
        )
        # Parameters
        self.initial_capital = 10000
        self.contract_details = {
            "A": {"quantity_multiplier": 40000, "price_multiplier": 0.01},
            "B": {"quantity_multiplier": 5000, "price_multiplier": 0.01},
        }
        # tickers = list(contract_details.keys())
        self.mock_strategy = MagicMock(spec=BaseStrategy)
        self.mock_strategy.prepare.return_value = None
        self.backtest = VectorizedBacktest(
            self.mock_strategy,
            self.test_data,
            self.contract_details,
            self.initial_capital,
            "backtest.html",
            "/Users/anthony/projects/midas/quantAnalytics/tests/integration/backtest_unit_output",
        )

    def test_setup(self):
        # Test
        self.mock_strategy.prepare.return_value = None
        backtest = VectorizedBacktest(
            self.mock_strategy,
            self.test_data,
            self.contract_details,
            self.initial_capital,
            "backtest.html",
            "/Users/anthony/projects/midas/quantAnalytics/tests/integration/backtest_unit_output",
        )

        # Validate
        self.assertIn("A_signal", backtest.data.columns)
        self.assertIn("B_signal", backtest.data.columns)

    def test_run(self):
        lag = 1
        self.mock_strategy.prepare.return_value = None
        self.mock_strategy.weights = {"A": 1, "B": -1}
        self.backtest.data = pd.DataFrame(
            {
                "ts_event": [
                    1700593200000000000,
                    1700852400000000000,
                    1700938800000000000,
                    1701025200000000000,
                    1701111600000000000,
                ],
                "A": [74.425, 75.400, 75.075, 75.600, 75.500],
                "B": [485.5, 491.5, 493.0, 493.0, 492.5],
                "A_signal": [None, None, -1.0, None, 0.0],
                "B_signal": [None, None, 1.0, None, 0.0],
            }
        )

        # Test
        self.backtest.run(lag)

        expected_df = pd.DataFrame(
            {
                "ts_event": [
                    1700593200000000000,
                    1700852400000000000,
                    1700938800000000000,
                    1701025200000000000,
                    1701111600000000000,
                ],
                "A": [74.425, 75.400, 75.075, 75.600, 75.500],
                "B": [485.5, 491.5, 493.0, 493.0, 492.5],
                "A_signal": [None, None, -1.0, None, 0.0],
                "B_signal": [None, None, 1.0, None, 0.0],
                "A_position": [0.0, 0.0, 0.0, -1.0, -1.0],
                "A_position_value": [0.0, 0.0, 0.0, -30240.0, -30200.0],
                "B_position": [0.0, 0.0, 0.0, 1.0, 1.0],
                "B_position_value": [0.0, 0.0, 0.0, 24650.0, 24625.0],
                "A_position_pnl": [0.0, 0.0, 0.0, 0.0, 40.0],
                "B_position_pnl": [0.0, 0.0, 0.0, 0.0, -25.0],
                "portfolio_pnl": [0.0, 0.0, 0.0, 0.0, 15.0],
                "equity_value": [10000.0, 10000.0, 10000.0, 10000.0, 10015.0],
                # "period_return": [0.0, 0.0, 0.0, 0.0, 0.0015],
                # "cumulative_return": [0.0, 0.0, 0.0, 0.0, 0.0015],
                # "drawdown": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

        # Validate
        pd.testing.assert_frame_equal(
            self.backtest.data,
            expected_df,
        )

    def test_summary(self):
        self.backtest.data = pd.DataFrame(
            {
                "ts_event": [
                    1700593200000000000,
                    1700852400000000000,
                    1700938800000000000,
                    1701025200000000000,
                    1701111600000000000,
                ],
                "A": [74.425, 75.400, 75.075, 75.600, 75.500],
                "B": [485.5, 491.5, 493.0, 493.0, 492.5],
                "A_signal": [None, None, -1.0, None, 0.0],
                "B_signal": [None, None, 1.0, None, 0.0],
                "A_position": [0.0, 0.0, 0.0, -1.0, -1.0],
                "A_position_value": [0.0, 0.0, 0.0, -30240.0, -30200.0],
                "B_position": [0.0, 0.0, 0.0, 1.0, 1.0],
                "B_position_value": [0.0, 0.0, 0.0, 24650.0, 24625.0],
                "A_position_pnl": [0.0, 0.0, 0.0, 0.0, 40.0],
                "B_position_pnl": [0.0, 0.0, 0.0, 0.0, -25.0],
                "portfolio_pnl": [0.0, 0.0, 0.0, 0.0, 15.0],
                "equity_value": [10000.0, 10000.0, 10000.0, 10000.0, 10015.0],
            }
        )
        self.backtest.data.set_index("ts_event", inplace=True)

        # self.backtest.data.index = self.backtest.data["ts_event"]

        # Test
        self.backtest.summary()

        # Expected
        expected_df = pd.DataFrame(
            {
                "ts_event": [
                    1700593200000000000,
                    1700852400000000000,
                    1700938800000000000,
                    1701025200000000000,
                    1701111600000000000,
                ],
                "A": [74.425, 75.400, 75.075, 75.600, 75.500],
                "B": [485.5, 491.5, 493.0, 493.0, 492.5],
                "A_signal": [None, None, -1.0, None, 0.0],
                "B_signal": [None, None, 1.0, None, 0.0],
                "A_position": [0.0, 0.0, 0.0, -1.0, -1.0],
                "A_position_value": [0.0, 0.0, 0.0, -30240.0, -30200.0],
                "B_position": [0.0, 0.0, 0.0, 1.0, 1.0],
                "B_position_value": [0.0, 0.0, 0.0, 24650.0, 24625.0],
                "A_position_pnl": [0.0, 0.0, 0.0, 0.0, 40.0],
                "B_position_pnl": [0.0, 0.0, 0.0, 0.0, -25.0],
                "portfolio_pnl": [0.0, 0.0, 0.0, 0.0, 15.0],
                "equity_value": [10000.0, 10000.0, 10000.0, 10000.0, 10015.0],
                "period_return": [0.0, 0.0, 0.0, 0.0, 0.0015],
                "cumulative_return": [0.0, 0.0, 0.0, 0.0, 0.0015],
                "drawdown": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

        # Decide which column to use as the index
        expected_df.set_index("ts_event", inplace=True)
        expected_df["datetime"] = pd.to_datetime(expected_df.index, unit="ns")

        # expected_df.index = pd.to_datetime(expected_df["ts_event"], unit="ns")
        # expected_df.index.name = "datetime"

        expected_summary_keys = [
            "annual_standard_deviation",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "ending_equity",
        ]
        # Validate
        pd.testing.assert_frame_equal(expected_df, self.backtest.data)
        for key in expected_summary_keys:
            self.assertIn(key, self.backtest.summary_stats.keys())

    # Errors
    def test_run_backtest_with_exception_in_generate_signals(self):
        # Configure the mock to raise an exception
        self.mock_strategy.generate_signals.side_effect = Exception(
            "Mock error"
        )
        with self.assertRaises(Exception):
            self.backtest.run_backtest()

    def test_setup_with_error(self):
        # Configure the mock strategy's prepare method to raise an exception
        self.mock_strategy.prepare.side_effect = Exception(
            "Mock preparation error"
        )

        # Verify that setup raises an Exception when strategy.prepare fails
        with self.assertRaises(Exception) as context:
            self.backtest.setup()

        # validate
        self.assertTrue("Mock preparation error" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
