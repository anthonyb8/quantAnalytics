import sys
import unittest
import numpy as np
import pandas as pd
from io import StringIO
from quantAnalytics.regression.regression import (
    RegressionAnalysis,
    RegressionResult,
)


class RegressionResultsTests(unittest.TestCase):
    def setUp(self):
        validation_results = {
            "Model Performance": {
                "R-squared": {
                    "value": 0.07945784006432621,
                    "significant": False,
                },
                "Adjusted R-squared": {
                    "value": 0.05585419493777055,
                    "significant": False,
                },
                "RMSE": {"value": 0.06292741161674, "significant": True},
                "MAE": {"value": 0.051336598197091465, "significant": True},
                "F-statistic": {
                    "value": 3.366337683789805,
                    "significant": True,
                },
                "F-statistic p-value": {
                    "value": 0.039600772589498276,
                    "significant": True,
                },
            },
            "Diagnostic Checks": {
                "Durbin-Watson": {
                    "value": 2.042989055805527,
                    "significant": True,
                },
                "Jarque-Bera": {
                    "value": 0.2597410094407664,
                    "significant": True,
                },
                "Jarque-Bera p-value": {
                    "value": 0.8782091474966396,
                    "significant": True,
                },
                "Condition Number": {
                    "value": 92.46590426344575,
                    "significant": False,
                },
                "VIF (const)": {
                    "value": 1.036878946992657,
                    "significant": True,
                },
                "VIF (HE_futures)": {
                    "value": 1.0452111916769846,
                    "significant": True,
                },
                "VIF (ZC_futures)": {
                    "value": 1.0452111916769848,
                    "significant": True,
                },
            },
            "Coefficients": {
                "Alpha": {"value": 0.031923312262071624, "significant": False},
                "Alpha p-value": {
                    "value": 0.3228344222246903,
                    "significant": False,
                },
                "Beta (HE_futures)": {
                    "value": -2.3294492722303493,
                    "significant": True,
                },
                "Beta (HE_futures) p-value": {
                    "value": 0.01724398196089229,
                    "significant": True,
                },
                "Beta (ZC_futures)": {
                    "value": -2.5783556343954555,
                    "significant": False,
                },
                "Beta (ZC_futures) p-value": {
                    "value": 0.43378700254395897,
                    "significant": False,
                },
            },
            "Model Validity": {
                "Model Validity": {"value": False, "significant": False}
            },
        }
        self.regression_results = RegressionResult(validation_results)

        # expected print
        self.output = """\
Regression Analysis Results
===========================
                    Field      Value Significant
                R-squared   0.079458       False
       Adjusted R-squared   0.055854       False
                     RMSE   0.062927        True
                      MAE   0.051337        True
              F-statistic   3.366338        True
      F-statistic p-value   0.039601        True
            Durbin-Watson   2.042989        True
              Jarque-Bera   0.259741        True
      Jarque-Bera p-value   0.878209        True
         Condition Number  92.465904       False
              VIF (const)   1.036879        True
         VIF (HE_futures)   1.045211        True
         VIF (ZC_futures)   1.045211        True
                    Alpha   0.031923       False
            Alpha p-value   0.322834       False
        Beta (HE_futures)  -2.329449        True
Beta (HE_futures) p-value   0.017244        True
        Beta (ZC_futures)  -2.578356       False
Beta (ZC_futures) p-value   0.433787       False
           Model Validity      False       False
** R-squared should be above the threshold and p-values should be below the threshold for model validity."""

    def test_print(self):
        # Capture the printed output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Test
        print(self.regression_results)
        sys.stdout = sys.__stdout__
        print(captured_output.getvalue().strip())

        # Validate
        self.assertEqual(
            captured_output.getvalue().strip(), self.output.strip()
        )

    def test_to_html(self):
        # Test
        result = self.regression_results.to_html()

        # Validate
        self.assertIsInstance(result, str)


class RegressionAnalysisTests(unittest.TestCase):
    def setUp(self):
        # Sample strategy and benchmark data
        np.random.seed(42)  # For reproducibility

        dates = pd.date_range(start="2024-01-01", periods=1000, freq="D")
        strategy_values = 100 + np.cumsum(
            np.random.normal(0, 1, size=len(dates))
        )  # Simulated equity values with some noise
        benchmark_values = 50 + np.cumsum(
            np.random.normal(0, 1, size=len(dates))
        )  # Simulated benchmark values with some noise

        self.data = pd.DataFrame(
            {
                "timestamp": dates,
                "Y": strategy_values,
                "bm_values": benchmark_values,
            }
        )
        self.data.set_index("timestamp", inplace=True)

        # Instance
        self.analysis = RegressionAnalysis(
            data=self.data, dependent_var="Y", risk_free_rate=0.05
        )

    # Basic Validation
    def test_fit(self):
        train_data = self.data.iloc[:500]

        # Test
        summary = self.analysis.fit(train_data)

        # validation
        result_model = self.analysis.model
        self.assertIsNotNone(result_model, "Regression model is not fitted.")
        self.assertTrue(
            hasattr(result_model, "params"), "Model has no parameters."
        )
        self.assertIsInstance(summary, str)

    def test_get_predictions(self):
        train_data = self.data.iloc[:500]
        test_data = self.data.iloc[500:]
        _ = self.analysis.fit(train_data)

        # Test
        x, y_predict, y_actual, residuals = self.analysis.get_predictions(
            test_data
        )
        # Validate
        expected_residuals = y_actual - y_predict

        self.assertTrue(len(y_predict) == len(y_actual))
        pd.testing.assert_series_equal(residuals, expected_residuals)

    def test_evaluate(self):
        train_data = self.data.iloc[:500]
        test_data = self.data.iloc[500:]
        _ = self.analysis.fit(train_data)

        # test
        result = self.analysis.evaluate(test_data)

        # validate
        self.assertIsInstance(result, RegressionResult)

    def test_risk_decomposition(self):
        train_data = self.data.iloc[:500]
        test_data = self.data.iloc[500:]
        _ = self.analysis.fit(train_data)

        # test
        results = self.analysis.risk_decomposition(test_data)

        # validate
        self.assertTrue(results["Total Volatility"] != 0)
        self.assertTrue(results["Systematic Volatility"] != 0)
        self.assertTrue(results["Idiosyncratic Volatility"] != 0)

    def test_performance_atribution(self):
        train_data = self.data.iloc[:500]
        test_data = self.data.iloc[500:]
        _ = self.analysis.fit(train_data)

        # test
        results = self.analysis.performance_attribution(test_data)

        # validate
        self.assertTrue(results["Total Contribution"] != 0)
        self.assertTrue(results["Systematic Contribution"] != 0)
        self.assertTrue(results["Idiosyncratic Contribution"] != 0)
        self.assertTrue(results["Alpha Contribution"] != 0)
        self.assertTrue(results["Randomness"] != 0)

    # def test_hedge_analysis(self):
    #     self.analysis.fit()

    #     # test
    #     hedge_results = self.analysis.hedge_analysis()

    #     # validation
    #     self.assertGreater(hedge_results["Portfolio Dollar Beta"], 0)
    #     self.assertGreater(hedge_results["Beta"], 0)
    #     self.assertIn("Market Hedge NMV", hedge_results)

    # Type Check
    def test_risk_decomposition_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.risk_decomposition(None)

    def test_performance_attribution_no_model(self):
        with self.assertRaises(ValueError):
            self.analysis.risk_decomposition(None)

    # Error Haandling
    def test_fit_error(self):
        train_data = None

        # Test
        with self.assertRaises(Exception):
            self.analysis.fit(train_data)


if __name__ == "__main__":
    unittest.main()
