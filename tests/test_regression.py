import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from quantAnalytics.regression import RegressionAnalysis

class RegressionAnalysisTests(unittest.TestCase):
    def setUp(self):
        # Sample strategy and benchmark data
        np.random.seed(42)  # For reproducibility

        dates = pd.date_range(start='2024-01-01', periods=1000, freq='D')
        equity_values = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))  # Simulated equity values with some noise
        benchmark_values = 50 + np.cumsum(np.random.normal(0, 1, size=len(dates)))  # Simulated benchmark values with some noise

        self.y = pd.DataFrame({
            'timestamp': dates,
            'equity_value': equity_values
        })
        self.y.set_index('timestamp', inplace=True)

        self.X = pd.DataFrame({
            'timestamp': dates,
            'close': benchmark_values
        })
        self.X.set_index('timestamp', inplace=True)

        # Instance
        self.analysis = RegressionAnalysis(X=self.X.copy(),y=self.y.copy())

    # Basic Validation 
    def test_fit(self):
        # Test
        summary = self.analysis.fit()
        
        # validation
        result_model = self.analysis.model
        self.assertIsNotNone(result_model, "Regression model is not fitted.")
        self.assertTrue(hasattr(result_model, 'params'), "Model has no parameters.")

    def test_predict(self):
        new_X= pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-10', periods=5, freq='D'),
            'close': [50, 51, 52, 51, 53]
        })
        new_X.set_index('timestamp', inplace=True)
        self.analysis.fit()

        # test
        results = self.analysis.predict(new_X)

        # validate
        self.assertTrue(len(results) == len(new_X))

    def test_evaluate(self):
        self.analysis.fit()

        # test
        result = self.analysis.evaluate()

        # validate
        for key, value in result["Model Performance"].items():
            self.assertIn(key,['R-squared', 'Adjusted R-squared', 'RMSE','MAE'])
            self.assertIsNotNone(value)

        for key, value in result["Diagnostic Checks"].items():
            self.assertIn(key,['Durbin-Watson', 'Jarque-Bera', 'Jarque-Bera p-value', 'Condition Number', 'Collinearity Check'])
            self.assertIsNotNone(value)

        for key, value in result["Significance"].items():
            self.assertIn(key,['Alpha significant', 'Betas significant'])
            self.assertIsNotNone(value)
    
        for key, value in result["Coefficients"].items():
            self.assertIn(key,['Alpha', 'Beta'])
            self.assertIsNotNone(value)

        self.assertIn(result["Model Validity"], [True, False])

    def test_display_evaluate_results(self):
        validation_results = {'Model Performance': {'R-squared': 0.007085115263561148, 'Adjusted R-squared': 0.005662601102047526, 'RMSE': 14.415406782802307, 'MAE': 11.690757704078676}, 'Diagnostic Checks': {'Durbin-Watson': 0.0044781869311308715, 'Jarque-Bera': 33.255782105490034, 'Jarque-Bera p-value': 6.006184517041462e-08, 'Condition Number': 3242.7411812936634, 'Collinearity Check': {'const': 796.763885488744, 'close': 0.9999999999999998}}, 'Significance': {'Alpha significant': True, 'Betas significant': True}, 'Coefficients': {'Alpha': 94.55454169409774, 'Beta': {'close': 0.03704162764916609}}, 'Model Validity': False}
        
        # test
        result = self.analysis._display_evaluate_results(validation_results, False, False)

        # expected
        expected = (f"Regression Validation Results"
                    f"============================="
                    f"                          Value"
                    f"R-squared               0.007085"
                    f"Adjusted R-squared      0.005663"
                    f"RMSE                   14.415407"
                    f"MAE                    11.690758"
                    f"Durbin-Watson           0.004478"
                    f"Jarque-Bera            33.255782"
                    f"Jarque-Bera p-value          0.0"
                    f"Condition Number     3242.741181"
                    f"VIF (const)           796.763885"
                    f"VIF (close)                  1.0"
                    f"Alpha                  94.554542"
                    f"Alpha significant           True"
                    f"Beta (close)            0.037042"
                    f"Betas significant           True"
                    f"Model Validity             False"
                    f"** R-squared should be above the threshold and p-values should be below the threshold for model validity."
        )

        # validate
        self.assertTrue(len(result) > 0)

    # def test_hedge_analysis(self):
    #     self.analysis.fit()


    #     # test
    #     hedge_results = self.analysis.hedge_analysis()

    #     # validation
    #     self.assertGreater(hedge_results["Portfolio Dollar Beta"], 0)
    #     self.assertGreater(hedge_results["Beta"], 0)
    #     self.assertIn("Market Hedge NMV", hedge_results)
    
    def test_check_collinearity(self):
        self.analysis.fit()

        # test
        results = self.analysis.check_collinearity()

        # validate
        self.assertEqual(type(results), dict)
        self.assertTrue(len(results.keys()) > 0)
        self.assertIn('close',results.keys())
        self.assertIn('const',results.keys())

    def test_plot_residuals(self):
        summary = self.analysis.fit()

        # test
        fig = self.analysis.plot_residuals()
        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)
    
    def test_plot_qq(self):
        summary = self.analysis.fit()

        # test
        fig = self.analysis.plot_qq()
        
        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)
    
    def test_plot_influence_measures(self):
        summary = self.analysis.fit()

        # test
        fig = self.analysis. plot_influence_measures()

        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)
    
    def test_risk_decomposition(self):
        self.analysis.fit()

        # test
        results = self.analysis.risk_decomposition()
        
        # validate
        self.assertTrue(results['Total Volatility'] > 0)
        self.assertTrue(results['Systematic Volatility'] > 0)
        self.assertTrue(results['Idiosyncratic Volatility'] > 0)

    def test_performance_atribution(self):
        self.analysis.fit()

        # test
        results = self.analysis.performance_attribution()

        # validate
        self.assertTrue(results['Total Contribution'] > 0)
        self.assertTrue(results['Systematic Contribution'] > 0)
        self.assertTrue(results['Idiosyncratic Contribution'] > 0)
        self.assertTrue(results['Alpha Contribution'] > 0)
        self.assertTrue(results['Randomness'] > 0)
    
    # Type Check
    def test_predict_type_check(self):
        self.analysis.fit()

        with self.assertRaises(TypeError):
            self.analysis.predict([1,2,3,4])

    # Value Error
    def test_check_collinearity_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.check_collinearity()

    def test_risk_decomposition_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.risk_decomposition()

    def test_performance_attribution_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.risk_decomposition()

    def test_plot_residuals_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.plot_residuals()

    def test_plot_qq_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.plot_residuals()

    def test_plot_influence_measures_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.plot_residuals()

    # Error Haandling
    def test_fit_error(self):
        self.analysis.X_train= None
        self.analysis.y_train= None

        # Test
        with self.assertRaises(Exception):
            self.analysis.fit()

    def test_predict_error(self):
        new_X= pd.DataFrame()

        # test
        with self.assertRaises(Exception):
            self.analysis.predict(new_X)

    def test_collinearity_error(self):
        self.analysis.X_test = None
        self.analysis.fit()

        # test
        with self.assertRaises(Exception):
            self.analysis.check_collinearity()


if __name__ == '__main__':
    unittest.main()

    # def test_beta(self):
    #     # Assuming a regression model has already been fitted in setup
    #     expected_beta = self.mock_model.params[1]  # Mock expected beta value
        
    #     # test
    #     calculated_beta = self.analysis.beta()
        
    #     # validate
    #     self.assertAlmostEqual(calculated_beta, expected_beta, places=4)

    # def test_alpha(self):
    #     # Expected alpha calculation
    #     annualized_strategy_return = np.mean(self.analysis.strategy_returns) * 252
    #     annualized_benchmark_return = np.mean(self.analysis.benchmark_returns) * 252
    #     expected_alpha = annualized_strategy_return - (self.analysis.risk_free_rate +  self.mock_model.params[1] * (annualized_benchmark_return - self.analysis.risk_free_rate))

    #     # test
    #     calculated_alpha = self.analysis.alpha()

    #     # validate
    #     self.assertAlmostEqual(calculated_alpha, expected_alpha, places=3, msg="Calculated alpha does not match expected alpha.")

    # def test_analyze_alpha(self):
    #     # test
    #     alpha_results = self.analysis.analyze_alpha()

    #     # validate
    #     self.assertEqual(alpha_results["Alpha (Intercept)"], 0.04)
    #     self.assertEqual(alpha_results["P-value"], 0.03)
    #     self.assertTrue(alpha_results['Alpha is significant'])
    #     self.assertTrue(alpha_results['Confidence Interval spans zero'])

    # def test_display_alpha_analysis_results(self):
    #     alpha_analysis_results={'Alpha (Intercept)': 0.0031231873179405336, 'P-value': 0.11052398250364748, 'Confidence Interval': [-0.0007163438974512504, 0.006962718533332317], 'Alpha is significant': False, 'Confidence Interval spans zero': True}


    #    # test
    #     result = RegressionAnalysis.display_alpha_analysis_results(alpha_analysis_results, False, False)

    #     # expected
    #     expected = (f"Alpha Analysis Results"
    #                 f"======================"
    #                 f"Alpha (Intercept)  p-value  Confidence Interval Lower Bound(2.5%)  Confidence Interval Upper Bound(97.5%)  Alpha is significant"
    #                 f"0.003123 0.110524                              -0.000716                                0.006963                 False"
    #                 f"** Note: For model validity, alpha should be significant (p-value < 0.05), and confidence intervals should not include zero."
    #     )

    #     # validate
    #     self.assertTrue(len(result) > 0)

    # def test_analyze_beta(self):
    #     beta_results = self.analysis.analyze_beta()

    #     self.assertEqual(beta_results["Beta (Slope)"], 0.076941)
    #     self.assertEqual(beta_results["P-value"], 0.08)
    #     self.assertFalse(beta_results['Beta is significant'])

    # def test_display_beta_analysis_results(self):
    #     beta_analysis_results={'Beta (Slope)': -0.14951528991152752, 'P-value': 0.5352449917450732, 'Confidence Interval': [-0.6233804345394574, 0.3243498547164023], 'Beta is significant': False, 'Confidence Interval spans one': False}

    #    # test
    #     result = RegressionAnalysis.display_beta_analysis_results(beta_analysis_results, False, False)

    #     # expected
    #     expected = (f"Beta Analysis Results"
    #                 f"====================="
    #                 f"Beta (Slope)  p-value  Confidence Interval Lower Bound(2.5%)  Confidence Interval Upper Bound(97.5%)  Beta is significant"
    #                 f"-0.149515 0.535245                               -0.62338                                 0.32435                False"
    #                 f"** Note: For model validity, beta should be significant (p-value < 0.05), and confidence intervals should not include zero."
    #     )

    #     # validate
    #     self.assertTrue(len(result) > 0)
