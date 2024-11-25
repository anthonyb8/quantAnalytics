import unittest
import numpy as np
import pandas as pd
from quantAnalytics.backtest.metrics import (
    Metrics,
    # Metrics,
    AnnualizedVolZScore,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.equity_curve = np.array([100, 105, 103, 110, 108])
        self.daily_returns = np.array([0.001, -0.002, 0.003, 0.002, -0.001])
        self.returns = np.array([0.05, -0.01904762, 0.06796117, -0.01818182])
        self.benchmark_equity_curve = np.array([100, 103, 102, 106, 108])

    # Basic Validation
    def test_simple_returns(self):
        # expected
        expected_returns = np.array([0.05, -0.019047, 0.067961, -0.018181])

        # test
        actual = Metrics.simple_returns(self.equity_curve)

        # validate
        np.testing.assert_allclose(actual, expected_returns, atol=1e-5)

    def test_log_returns(self):
        # expected
        expected_returns = np.array(
            [0.04879016, -0.01923136, 0.06575138, -0.01834914]
        )

        # test
        actual = Metrics.log_returns(self.equity_curve)

        # validate
        np.testing.assert_allclose(actual, expected_returns, atol=1e-5)

    def test_cumulative_return(self):
        # expected
        expected_cumulative_returns = np.array([0.05, 0.03, 0.10, 0.08])

        # test
        cumulative_returns = Metrics.cumulative_returns(self.equity_curve)

        # validate
        np.testing.assert_array_almost_equal(
            cumulative_returns, expected_cumulative_returns, decimal=4
        )

    def test_total_return(self):
        # expected
        expected_total_return = 0.08

        # test
        total_return = Metrics.total_return(self.equity_curve)

        # validate
        self.assertEqual(total_return, expected_total_return)

    def test_annualize_returns(self):
        # expected
        compounded_growth = (1 + self.daily_returns).prod()
        n_periods = self.daily_returns.shape[0]
        expected_annualized_return = compounded_growth ** (252 / n_periods) - 1

        # test
        actual = Metrics.annualize_returns(self.daily_returns)

        # validate
        self.assertAlmostEqual(actual, expected_annualized_return)

    def test_net_profit(self):
        # expected
        expected_net_profit = self.equity_curve[-1] - self.equity_curve[0]

        # test
        actual = Metrics.net_profit(self.equity_curve)

        # validate
        self.assertAlmostEqual(actual, expected_net_profit)

    # Type Constraint
    def test_simple_returns_type_error(self):
        with self.assertRaises(TypeError):
            Metrics.simple_returns([1, 2, 3, 4, 5])

    def test_log_returns_type_error(self):
        with self.assertRaises(TypeError):
            Metrics.log_returns([1, 2, 3, 4, 5])

    def test_cumulative_return_type_error(self):
        with self.assertRaises(TypeError):
            Metrics.cumulative_returns([10, -5, 15])

    def test_total_return_type_error(self):
        with self.assertRaises(TypeError):
            Metrics.total_return([10, -5, 15])

    def test_annualize_returns_type_error(self):
        with self.assertRaises(TypeError):
            Metrics.annualize_returns([1, 2, 3, 4, 5])

    # Error Handling
    def test_simple_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Metrics.simple_returns(np.array([1, 2, 3, "4"]))

    def test_log_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Metrics.log_returns(np.array([1, 2, 3, "4"]))

    def test_cumulative_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Metrics.cumulative_returns(np.array([1, 2, 3, "4"]))

    def test_total_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Metrics.total_return(np.array([1, 2, 3, "4"]))

    def test_annualize_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Metrics.annualize_returns(np.array([1, 2, 3, "4"]))

    # Edge Case
    # Basic Validation
    def test_drawdown(self):
        # expected
        expected_drawdowns = np.array([0, -0.01914, 0, -0.01818])

        # test
        drawdowns = Metrics.drawdown(self.returns)

        # validate
        np.testing.assert_array_almost_equal(
            drawdowns, expected_drawdowns, decimal=4
        )

    def test_max_drawdown(self):
        # expected
        expected_max_drawdown = -0.019048

        # test
        max_drawdown = Metrics.max_drawdown(self.returns)

        # validate
        self.assertEqual(max_drawdown, expected_max_drawdown)

    def test_standard_deviation(self):
        # expected
        expected_annual_std_dev = np.std(self.returns, ddof=1)

        # test
        annual_std_dev = Metrics.standard_deviation(self.returns)

        # validate
        self.assertAlmostEqual(
            annual_std_dev, expected_annual_std_dev, places=4
        )

    def test_annual_standard_deviation(self):
        # expected
        expected_annual_std_dev = np.std(self.returns, ddof=1) * np.sqrt(252)

        # test
        annual_std_dev = Metrics.annual_standard_deviation(self.returns)

        # validate
        self.assertAlmostEqual(
            annual_std_dev, expected_annual_std_dev, places=4
        )

    def test_sharpe_ratio(self):
        daily_returns = (
            np.diff(self.equity_curve) / self.equity_curve[:-1]
        )  # Calculate daily returns
        excess_returns = daily_returns - (
            0.04 / 252
        )  # Excess returns calculation
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std(ddof=1) * np.sqrt(252)
        )
        expected_sharpe_ratio = (
            np.around(sharpe_ratio, decimals=4)
            if excess_returns.std(ddof=1) != 0
            else 0
        )

        # test
        sharpe_ratio = Metrics.sharpe_ratio(self.returns)

        # validate
        self.assertAlmostEqual(sharpe_ratio, expected_sharpe_ratio, places=3)

    def test_sortino_ratio(self):
        # expected
        target_return = 0.04 / 252
        negative_returns = self.returns[self.returns < target_return]
        expected_return = self.returns.mean() - target_return
        downside_deviation = negative_returns.std(ddof=1)

        if downside_deviation > 0:
            expected_sortino_ratio = (
                expected_return / downside_deviation * np.sqrt(252)
            )
        else:
            expected_sortino_ratio = 0.0

        # test
        sortino_ratio = Metrics.sortino_ratio(self.returns)

        # validate
        self.assertAlmostEqual(sortino_ratio, expected_sortino_ratio, places=4)

    def test_value_at_risk(self):
        expected = np.percentile(self.returns, 0.05 * 100)

        # test
        result = Metrics.value_at_risk(self.returns)

        # validate
        self.assertEqual(result, expected)

    def test_conditional_value_at_risk(self):
        var = np.percentile(self.returns, 0.05 * 100)
        tail_losses = self.returns[self.returns <= var]
        expected = tail_losses.mean()

        # test
        result = Metrics.conditional_value_at_risk(self.returns)

        # validate
        self.assertEqual(result, expected)

    def test_calculate_volatility_and_zscore(self):
        # expected
        annualized_volatility = self.returns.std() * np.sqrt(252)
        annualized_mean_return = self.returns.mean() * 252

        # test
        volatility_results = (
            Metrics.calculate_volatility_and_zscore_annualized(self.returns)
        )

        # validate
        z_score_dict = volatility_results.data["Z-Scores (Annualized)"]
        self.assertEqual(
            volatility_results.data["Annualized Volatility"],
            annualized_volatility,
        )
        self.assertEqual(
            volatility_results.data["Annualized Mean Return"],
            annualized_mean_return,
        )
        self.assertGreater(
            z_score_dict["Z-score for 1 SD move (annualized)"], 0
        )
        self.assertGreater(
            z_score_dict["Z-score for 2 SD move (annualized)"], 0
        )
        self.assertGreater(
            z_score_dict["Z-score for 3 SD move (annualized)"], 0
        )

    def test_display_volatility_zscore_results(self):
        volatility_zscore_results = {
            "Annualized Volatility": 0.21862629936907577,
            "Annualized Mean Return": 0.12302713066962165,
            "Z-Scores (Annualized)": {
                "Z-score for 1 SD move (annualized)": -0.43727204355258104,
                "Z-score for 2 SD move (annualized)": -1.4372720435525812,
                "Z-score for 3 SD move (annualized)": -2.437272043552581,
            },
        }

        # Test
        result = AnnualizedVolZScore("test", volatility_zscore_results)

        # Validate
        self.assertTrue(len(result.to_html()) > 0)

    # Type Constraints
    def test_drawdown_type_check(self):
        with self.assertRaises(TypeError):
            Metrics.drawdown([10, -5, 15])

    def test_max_drawdown_type_check(self):
        with self.assertRaises(TypeError):
            Metrics.max_drawdown([10, -5, 15])

    def test_annual_standard_deviation_type_check(self):
        with self.assertRaises(TypeError):
            Metrics.annual_standard_deviation([10, -5, 15])

    def test_sharpe_ratio_type_check(self):
        with self.assertRaises(TypeError):
            Metrics.sharpe_ratio([10, -5, 15])

    def test_sortino_ratio_type_check(self):
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            Metrics.sortino_ratio([10, -5, 15])

        # Test with missing column
        with self.assertRaises(TypeError):
            Metrics.sortino_ratio(pd.DataFrame())

    def test_value_at_risk_type_check(self):
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            Metrics.value_at_risk([10, -5, 15])

    def test_conditional_value_at_risk_type_check(self):
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            Metrics.conditional_value_at_risk([10, -5, 15])

    # Edge Cases
    def test_drawdown_null_handling(self):
        list = []
        equity_curve = np.array(list)

        # test
        result = Metrics.drawdown(equity_curve)

        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(
            len(result), 1
        )  # Expecting an array with a single zero

    def test_max_drawdown_null_handling(self):
        list = []
        equity_curve = np.array(list)

        # test
        result = Metrics.drawdown(equity_curve)

        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result, 0)  # Expecting an array with a single zero

    def test_annual_standard_deviation_null_handling(self):
        list = []
        equity_curve = np.array(list)

        # test
        result = Metrics.annual_standard_deviation(equity_curve)

        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result, 0)  # Expecting an array with a single zero

    def test_sharpe_ratio_null_handling(self):
        # Test with empty input
        list = []
        equity_curve = np.array(list)

        # test
        result = Metrics.sharpe_ratio(equity_curve)

        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result, 0)  # Expecting an array with a single zero

    def test_sortino_ratio_null_handling(self):
        # test
        result = Metrics.sortino_ratio(np.array([]))

        # validate
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
