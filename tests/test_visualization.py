import unittest
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

from quantAnalytics.visualization import Visualizations

# TODO: edge cases
class TestPerformancStatistics(unittest.TestCase):    
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'symbol1': [100, 101, 102, 103, 104],
            'symbol2': [200, 198, 197, 199, 202]
        }, index=pd.date_range('2023-01-01', periods=5))

        self.signals = [
            {'timestamp': '2023-01-01', 'price': 100, 'direction': 1},
            {'timestamp': '2023-01-01', 'price': 200, 'direction': 1},
            {'timestamp': '2023-01-03', 'price': 102, 'direction': -1},
            {'timestamp': '2023-01-05', 'price': 104, 'direction': 1}
        ]

    def test_line_plot(self):
        # Create sample data
        x = pd.Series(np.arange(0, 10))
        y = pd.Series(np.random.random(10))

        # test
        fig = Visualizations.line_plot(x, y, title='Test Line Plot', x_label='Index', y_label='Random Value')
        plt.show()
        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) > 0)

    def test_double_line_plot(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        data1 = pd.DataFrame({
            'Value': np.random.random(10)
        }, index=dates)
        data2 = pd.DataFrame({
            'Value': np.random.random(10)
        }, index=dates)

        # test
        fig = Visualizations.double_line_plot(data1, data2, label1='Data 1', label2='Data 2', title='Test Double Line Plot')
        plt.show()

        # validate
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes[0].lines) >= 2)

    def test_plot_data_with_signals(self):
        fig = Visualizations.plot_data_with_signals(self.data, self.signals)
        plt.show()
        
        # Ensure the figure is created
        self.assertIsInstance(fig, plt.Figure)

    def tearDown(self):
        plt.close('all')

if __name__ == "__main__":
    unittest.main()