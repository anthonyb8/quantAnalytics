import unittest
import numpy as np
import pandas as pd

from quantAnalytics.performance import PerformanceStatistics

# TODO: edge cases
class TestPerformancStatistics(unittest.TestCase):    
    def setUp(self):
        # Sample equity curve and trade log for testing
        self.equity_curve = np.array([100, 105, 103, 110, 108])
        self.benchmark_equity_curve = np.array([100, 103, 102, 106, 108])
        
        self.trade_log = pd.DataFrame({
            'pnl': [20, -10, 15, -5, 30, -20],
            'gain/loss': [0.10, -0.05, 0.075, -0.25, 0.15, -0.010]
        })
    
    # Net Profit
    def test_net_profit(self):
        # Expected result based on the sample trade log
        expected_net_profit = 30  
        
        # test
        net_profit = PerformanceStatistics.net_profit(self.trade_log)
        
        # validate
        self.assertEqual(net_profit, expected_net_profit)

    def test_net_profit_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.net_profit([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.net_profit(pd.DataFrame())

    def test_net_profit_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        
        # validate
        self.assertEqual(PerformanceStatistics.net_profit(trade_log), 0)

    # Total Trades
    def test_total_trades(self):
        expected_total_trades = 6
        # test
        total_trades = PerformanceStatistics.total_trades(self.trade_log)
        # validate
        self.assertEqual(total_trades, expected_total_trades)

    # Total Winning Trades
    def test_total_winning_trades(self):
        expected_winning_trades = 3
        # test
        total_winning_trades = PerformanceStatistics.total_winning_trades(self.trade_log)
        # validate
        self.assertEqual(total_winning_trades, expected_winning_trades)

    def test_total_winning_trades_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.total_winning_trades([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.total_winning_trades(pd.DataFrame())

    def test_total_winning_trades_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        # validate
        self.assertEqual(PerformanceStatistics.total_winning_trades(trade_log), 0)
    
    # Total Losing Trades
    def test_total_losing_trades(self):
        expected_losing_trades = 3
        # test
        total_losing_trades = PerformanceStatistics.total_losing_trades(self.trade_log)
        # validate
        self.assertEqual(total_losing_trades, expected_losing_trades)

    def test_total_losing_trades_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.total_losing_trades([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.total_losing_trades(pd.DataFrame())

    def test_total_losing_trades_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        self.assertEqual(PerformanceStatistics.total_losing_trades(trade_log), 0)
    
    # Total Avg Win Percent
    def test_avg_win_return_rate(self):
        expected_avg_win = 0.108333  # Based on the provided gain/loss values
        # test
        avg_win_return_rate = PerformanceStatistics.avg_win_return_rate(self.trade_log)
        # validate
        self.assertAlmostEqual(avg_win_return_rate, expected_avg_win, places=4)

    def test_avg_win_return_rate_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.avg_win_return_rate([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.avg_win_return_rate(pd.DataFrame())

    def test_avg_win_return_rate_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        # validate
        self.assertEqual(PerformanceStatistics.avg_win_return_rate(trade_log), 0)
    
    # Total avg_loss_return_rate
    def test_avg_loss_return_rate(self):
        expected_avg_loss = -0.10333  # Based on the provided gain/loss values
        # test
        avg_loss_return_rate = PerformanceStatistics.avg_loss_return_rate(self.trade_log)
        # validate
        self.assertAlmostEqual(avg_loss_return_rate, expected_avg_loss, places=4)

    def test_avg_loss_return_rate_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.avg_loss_return_rate([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.avg_loss_return_rate(pd.DataFrame())

    def test_avg_loss_return_rate_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        # validate
        self.assertEqual(PerformanceStatistics.avg_loss_return_rate(trade_log), 0)

    # Percent Profitable
    def test_profitability_ratio(self):
        expected_profitability_ratio = 0.50  # 3 winning trades out of 6 total trades
        # test
        profitability_ratio = PerformanceStatistics.profitability_ratio(self.trade_log)
        # validate
        self.assertEqual(profitability_ratio, expected_profitability_ratio)

    def test_profitability_ratio_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.profitability_ratio([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.profitability_ratio(pd.DataFrame())

    def test_profitability_ratio_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        # valdiate
        self.assertEqual(PerformanceStatistics.profitability_ratio(trade_log), 0)

    # Avg Trade Profit
    def test_avg_trade_profit(self):
        expected_avg_trade_profit = 5  # (20-10+15-5+30-20) / 6
        # test
        average_trade_profit = PerformanceStatistics.avg_trade_profit(self.trade_log)
        # validate
        self.assertEqual(average_trade_profit, expected_avg_trade_profit)

    def test_avg_trade_profit_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.avg_trade_profit([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.avg_trade_profit(pd.DataFrame())

    def test_avg_trade_profit_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        # validate
        self.assertEqual(PerformanceStatistics.avg_trade_profit(trade_log), 0)

    # Profit Factor
    def test_profit_factor(self):
        expected_profit_factor = 1.8571 # (20+15+30) / abs(-10-5-20)
        # test
        profit_factor = PerformanceStatistics.profit_factor(self.trade_log)
        # validate
        self.assertEqual(profit_factor, expected_profit_factor)

    def test_profit_factor_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.profit_factor([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.profit_factor(pd.DataFrame())

    def test_profit_factor_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        # validate
        self.assertEqual(PerformanceStatistics.profit_factor(trade_log), 0)

    # Profit & Loss Ratio
    def test_profit_and_loss_ratio(self):
        expected_pnl_ratio = 1.8571  # abs(mean([20,15,30]) / mean([-10,-5,-20]))
        # test
        profit_and_loss_ratio = PerformanceStatistics.profit_and_loss_ratio(self.trade_log)
        # validate
        self.assertEqual(profit_and_loss_ratio, expected_pnl_ratio)

    def test_profit_and_loss_ratio_type_check(self):        
        # Test with incorrect type (should raise an error or handle gracefully)
        with self.assertRaises(TypeError):
            PerformanceStatistics.profit_and_loss_ratio([10, -5, 15])

        # Test with missing column
        with self.assertRaises(ValueError):
            PerformanceStatistics.profit_and_loss_ratio(pd.DataFrame())

    def test_profit_and_loss_ratio_null_handling(self):
        # Test with empty DataFrame
        trade_log = pd.DataFrame({'pnl': []})
        # validate
        self.assertEqual(PerformanceStatistics.profit_and_loss_ratio(trade_log), 0)


if __name__ == "__main__":
    unittest.main()