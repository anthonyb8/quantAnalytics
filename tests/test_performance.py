import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from quantAnalytics.performance import PerformanceStatistics

# TODO: edge cases

class TestReturns(unittest.TestCase):    
    def setUp(self):
        self.equity_curve = np.array([100, 105, 103, 110, 108])
        self.daily_returns = np.array([0.001, -0.002, 0.003, 0.002, -0.001])
    
    # Basic Validation
    def test_simple_returns(self):
        # expected
        expected_returns = np.array([0.05, -0.019047, 0.067961, -0.018181])
        
        # test
        actual =PerformanceStatistics.simple_returns(self.equity_curve)

        # validate
        np.testing.assert_allclose(actual, expected_returns, atol=1e-5)
    
    def test_log_returns(self):
        # expected 
        expected_returns = np.array([ 0.04879016 ,-0.01923136  ,0.06575138 ,-0.01834914])

        # test
        actual =PerformanceStatistics.log_returns(self.equity_curve)

        # validate
        np.testing.assert_allclose(actual, expected_returns, atol=1e-5)

    def test_cumulative_return(self):
        # expected
        expected_cumulative_returns = np.array([0.05, 0.03, 0.10, 0.08])
        
        # test
        cumulative_returns =PerformanceStatistics.cumulative_returns(self.equity_curve)
        
        # validate
        np.testing.assert_array_almost_equal(cumulative_returns, expected_cumulative_returns, decimal=4)
    
    def test_total_return(self):
        # expected
        expected_total_return = 0.08 
        
        # test
        total_return =PerformanceStatistics.total_return(self.equity_curve)
        
        # validate
        self.assertEqual(total_return, expected_total_return)

    def test_annualize_returns(self):
        # expected
        compounded_growth = (1+self.daily_returns).prod()
        n_periods = self.daily_returns.shape[0]
        expected_annualized_return = compounded_growth ** (252 / n_periods) - 1

        # test
        actual =PerformanceStatistics.annualize_returns(self.daily_returns)

        # validate
        self.assertAlmostEqual(actual, expected_annualized_return)

    def test_net_profit(self):
        # expected
        expected_net_profit = self.equity_curve[-1] - self.equity_curve[0]

        # test
        actual =PerformanceStatistics.net_profit(self.equity_curve)

        # validate
        self.assertAlmostEqual(actual, expected_net_profit)

    # Type Constraint
    def test_simple_returns_type_error(self):
        with self.assertRaises(TypeError):
           PerformanceStatistics.simple_returns([1,2,3,4,5])

    def test_log_returns_type_error(self):
        with self.assertRaises(TypeError):
           PerformanceStatistics.log_returns([1,2,3,4,5])
    
    def test_cumulative_return_type_error(self):
        with self.assertRaises(TypeError):
           PerformanceStatistics.cumulative_returns([10, -5, 15])
    
    def test_total_return_type_error(self):
        with self.assertRaises(TypeError):
           PerformanceStatistics.total_return([10, -5, 15])
    
    def test_annualize_returns_type_error(self):
        with self.assertRaises(TypeError):
           PerformanceStatistics.annualize_returns([1,2,3,4,5])
    
    # Error Handling
    def test_simple_returns_calculation_error(self):
        with self.assertRaises(Exception):
           PerformanceStatistics.simple_returns(np.array([1,2,3,'4']))
    
    def test_log_returns_calculation_error(self):
        with self.assertRaises(Exception):
           PerformanceStatistics.log_returns(np.array([1,2,3,'4']))

    def test_cumulative_returns_calculation_error(self):
        with self.assertRaises(Exception):
           PerformanceStatistics.cumulative_returns(np.array([1,2,3,'4']))

    def test_total_returns_calculation_error(self):
        with self.assertRaises(Exception):
           PerformanceStatistics.total_return(np.array([1,2,3,'4']))
    
    def test_annualize_returns_calculation_error(self):
        with self.assertRaises(Exception):
           PerformanceStatistics.annualize_returns(np.array([1,2,3,'4']))
        
    # Edge Case

if __name__ == "__main__":
    unittest.main()