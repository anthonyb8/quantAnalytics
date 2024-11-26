import unittest
from tests.integration.backtest.main import main as bt_main
from tests.integration.regression.main import main as reg_main


class TestIntegration(unittest.TestCase):
    def test_backtest(self):
        bt_main()

    # def test_regression(self):
    #     reg_main()


if __name__ == "__main__":
    unittest.main()
