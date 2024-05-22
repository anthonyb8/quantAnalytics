import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from quantAnalytics.risk import RiskAnalysis

# TODO: edge cases
class TestRiskAnalysis(unittest.TestCase):    
    def setUp(self):
        # Sample equity curve and trade log for testing
        self.equity_curve = np.array([100, 105, 103, 110, 108])
        self.returns = np.array([0.05, -0.01904762, 0.06796117, -0.01818182])
        self.benchmark_equity_curve = np.array([100, 103, 102, 106, 108])

    # Basic Validation
    def test_drawdown(self):
        # expected
        expected_drawdowns = np.array([ 0, -0.01914, 0, -0.01818])

        # test
        drawdowns = RiskAnalysis.drawdown(self.returns)

        # validate
        np.testing.assert_array_almost_equal(drawdowns, expected_drawdowns, decimal=4)

    def test_max_drawdown(self):
        # expected
        expected_max_drawdown = -0.019  

        # test
        max_drawdown = RiskAnalysis.max_drawdown(self.returns)

        # validate
        self.assertEqual(max_drawdown, expected_max_drawdown)

    def test_annual_standard_deviation(self):
        # expected
        expected_annual_std_dev = np.std(self.returns, ddof=1) * np.sqrt(252)

        # test
        annual_std_dev = RiskAnalysis.annual_standard_deviation(self.returns)

        # validate
        self.assertAlmostEqual(annual_std_dev, expected_annual_std_dev, places=4)

    def test_sharpe_ratio(self):
        daily_returns = np.diff(self.equity_curve) / self.equity_curve[:-1] # Calculate daily returns
        excess_returns = daily_returns - (0.04 / 252) # Excess returns calculation
        sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1) * np.sqrt(252)
        expected_sharpe_ratio = np.around(sharpe_ratio, decimals=4) if excess_returns.std(ddof=1) != 0 else 0

        # test
        sharpe_ratio = RiskAnalysis.sharpe_ratio(self.returns)
        
        # validate
        self.assertAlmostEqual(sharpe_ratio, expected_sharpe_ratio, places=3)

    def test_sortino_ratio(self):
        # expected
        target_return = 0
        negative_returns = self.returns[self.returns < target_return]
        expected_return = self.returns.mean() - target_return
        downside_deviation = negative_returns.std(ddof=1)

        if downside_deviation > 0:
            expected_sortino_ratio = expected_return / downside_deviation
        else:
            expected_sortino_ratio = 0.0

        # test
        sortino_ratio = RiskAnalysis.sortino_ratio(self.returns)
        
        # validate
        self.assertAlmostEqual(sortino_ratio, expected_sortino_ratio, places=4)

    def test_value_at_risk(self):
        expected = np.percentile(self.returns, 0.05 * 100)

        # test
        result = RiskAnalysis.value_at_risk(self.returns)

        # validate
        self.assertEqual(result, expected)

    def test_conditional_value_at_risk(self):
        var = np.percentile(self.returns, 0.05 * 100)
        tail_losses = self.returns[self.returns <= var]
        expected = tail_losses.mean()

        # test
        result = RiskAnalysis.conditional_value_at_risk(self.returns)

        # validate
        self.assertEqual(result, expected)

    def test_calculate_volatility_and_zscore(self):
        # expected
        annualized_volatility = self.returns.std() * np.sqrt(252)
        annualized_mean_return = self.returns.mean() * 252
        
        # test
        volatility_results = RiskAnalysis.calculate_volatility_and_zscore_annualized(self.returns)

        # validate
        z_score_dict  = volatility_results["Z-Scores (Annualized)"]
        self.assertEqual(volatility_results['Annualized Volatility'], annualized_volatility)
        self.assertEqual(volatility_results["Annualized Mean Return"], annualized_mean_return)
        self.assertGreater(z_score_dict['Z-score for 1 SD move (annualized)'], 0)
        self.assertGreater(z_score_dict['Z-score for 2 SD move (annualized)'], 0)
        self.assertGreater(z_score_dict['Z-score for 3 SD move (annualized)'], 0)


    def test_display_volatility_zscore_results(self):
        volatility_zscore_results={'Annualized Volatility': 0.21862629936907577, 'Annualized Mean Return': 0.12302713066962165, 'Z-Scores (Annualized)': {'Z-score for 1 SD move (annualized)': -0.43727204355258104, 'Z-score for 2 SD move (annualized)': -1.4372720435525812, 'Z-score for 3 SD move (annualized)': -2.437272043552581}}

        # test
        result = RiskAnalysis.display_volatility_zscore_results(volatility_zscore_results, False, False)

        # expected
        expected = (f"zscore volatility Results"
                    f"========================="
                    f"Annualized Volatility  Annualized Mean Return  Z-score for 1 SD (annualized)  Z-score for 2 SD (annualized)  Z-score for 3 SD (annualized)"
                    f"0.218626                0.123027                      -0.437272                      -1.437272                      -2.437272"
                    f"** R-squared should be above the threshold and p-values should be below the threshold for model validity."
        )

        # validate
        self.assertTrue(len(result) > 0)

    # Type Constraints
    def test_drawdown_type_check(self):
        with self.assertRaises(TypeError):
            RiskAnalysis.drawdown([10, -5, 15])

    def test_max_drawdown_type_check(self):
        with self.assertRaises(TypeError):
            RiskAnalysis.max_drawdown([10, -5, 15])
    
    def test_annual_standard_deviation_type_check(self):
        with self.assertRaises(TypeError):
            RiskAnalysis.annual_standard_deviation([10, -5, 15])

    def test_sharpe_ratio_type_check(self):
        with self.assertRaises(TypeError):
            RiskAnalysis.sharpe_ratio([10, -5, 15])

    def test_sortino_ratio_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            RiskAnalysis.sortino_ratio([10, -5, 15])

        # Test with missing column
        with self.assertRaises(TypeError):
            RiskAnalysis.sortino_ratio(pd.DataFrame())

    def test_value_at_risk_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            RiskAnalysis.value_at_risk([10, -5, 15])

    def test_conditional_value_at_risk_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            RiskAnalysis.conditional_value_at_risk([10, -5, 15])

    # Edge Cases
    def test_drawdown_null_handling(self):
        list = []
        equity_curve = np.array(list)

        # test
        result = RiskAnalysis.drawdown(equity_curve)

        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 1)  # Expecting an array with a single zero

    def test_max_drawdown_null_handling(self):
        list = []
        equity_curve = np.array(list)

        # test
        result = RiskAnalysis.drawdown(equity_curve)

        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result, 0)  # Expecting an array with a single zero

    def test_annual_standard_deviation_null_handling(self):
        list = []
        equity_curve = np.array(list)

        # test
        result = RiskAnalysis.annual_standard_deviation(equity_curve)

        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result, 0)  # Expecting an array with a single zero

    def test_sharpe_ratio_null_handling(self):
        # Test with empty input
        list = []
        equity_curve = np.array(list)
        
        # test
        result = RiskAnalysis.sharpe_ratio(equity_curve)
        
        # validate
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result, 0)  # Expecting an array with a single zero

    def test_sortino_ratio_null_handling(self):
        # test
        returns = RiskAnalysis.sharpe_ratio(np.array([]))

        # validate
        self.assertEqual(RiskAnalysis.sortino_ratio(returns), 0)


if __name__ == "__main__":
    unittest.main()