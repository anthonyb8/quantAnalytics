import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from quantAnalytics.returns import Returns

# TODO: edge cases

class TestReturns(unittest.TestCase):    
    def setUp(self):
        self.equity_curve = np.array([100, 105, 103, 110, 108])
        self.daily_returns = np.array([0.001, -0.002, 0.003, 0.002, -0.001])
    
    # Basic Validation
    def test_simple_returns(self):
        # expected
        expected_returns = np.array([0.05, -0.01904762, 0.06796117, -0.01818182])
        
        # test
        actual = Returns.simple_returns(self.equity_curve)

        # validate
        np.testing.assert_allclose(actual, expected_returns, rtol=1e-7, atol=0)
    
    def test_log_returns(self):
        # expected 
        expected_returns = np.array([ 0.04879016 ,-0.01923136  ,0.06575138 ,-0.01834914])

        # test
        actual = Returns.log_returns(self.equity_curve)

        # validate
        np.testing.assert_allclose(actual, expected_returns, rtol=1e-5, atol=0)

    def test_cumulative_return(self):
        # expected
        expected_cumulative_returns = np.array([0.05, 0.03, 0.10, 0.08])
        
        # test
        cumulative_returns = Returns.cumulative_returns(self.equity_curve)
        
        # validate
        np.testing.assert_array_almost_equal(cumulative_returns, expected_cumulative_returns, decimal=4)
    
    def test_total_return(self):
        # expected
        expected_total_return = 0.08 
        
        # test
        total_return = Returns.total_return(self.equity_curve)
        
        # validate
        self.assertEqual(total_return, expected_total_return)

    def test_annualize_returns(self):
        # expected
        compounded_growth = (1+self.daily_returns).prod()
        n_periods = self.daily_returns.shape[0]
        expected_annualized_return = compounded_growth ** (252 / n_periods) - 1

        # test
        actual = Returns.annualize_returns(self.daily_returns)

        # validate
        self.assertAlmostEqual(actual, expected_annualized_return)

    # Type Constraint
    def test_simple_returns_type_error(self):
        with self.assertRaises(TypeError):
            Returns.simple_returns([1,2,3,4,5])

    def test_log_returns_type_error(self):
        with self.assertRaises(TypeError):
            Returns.log_returns([1,2,3,4,5])
    
    def test_cumulative_return_type_error(self):
        with self.assertRaises(TypeError):
            Returns.cumulative_returns([10, -5, 15])
    
    def test_total_return_type_error(self):
        with self.assertRaises(TypeError):
            Returns.total_return([10, -5, 15])
    
    def test_annualize_returns_type_error(self):
        with self.assertRaises(TypeError):
            Returns.annualize_returns([1,2,3,4,5])
    
    # Error Handling
    def test_simple_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Returns.simple_returns(np.array([1,2,3,'4']))
    
    def test_log_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Returns.log_returns(np.array([1,2,3,'4']))

    def test_cumulative_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Returns.cumulative_returns(np.array([1,2,3,'4']))

    def test_total_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Returns.total_return(np.array([1,2,3,'4']))
    
    def test_annualize_returns_calculation_error(self):
        with self.assertRaises(Exception):
            Returns.annualize_returns(np.array([1,2,3,'4']))
        
    # Edge Case

if __name__ == "__main__":
    unittest.main()