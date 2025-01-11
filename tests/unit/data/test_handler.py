import unittest
import pandas as pd
from typing import List, Dict
from pandas.testing import assert_frame_equal
from quant_analytics.data.handler import DataHandler
import numpy as np


def valid_process_data(db_response: List[Dict]):
    df = pd.DataFrame(db_response)
    df.drop(columns=["id"], inplace=True)

    # Convert OHLCV columns to floats
    ohlcv_columns = ["open", "high", "low", "close", "volume"]
    df[ohlcv_columns] = df[ohlcv_columns].astype(float)

    df = df.sort_values(by="ts_event", ascending=True).reset_index(drop=True)

    return df


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        # Create sample time series data
        np.random.seed(0)
        self.sample_series = pd.Series(np.random.randn(100))
        self.sample_dataframe = pd.DataFrame(
            {"series1": np.random.randn(100), "series2": np.random.randn(100)}
        )

    # Basic Validation
    def test_check_null_df_with_null(self):
        """Test check_missing with a dataframe containing missing values."""
        self.data_with_missing = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "values": [1, None, 3],
            }
        )
        # test
        response = DataHandler.check_null(self.data_with_missing)

        # validate
        self.assertTrue(response)

    def test_check_null_series_with_null(self):
        # test series
        self.data_with_missing = pd.Series([1, None, 3])

        # test
        response = DataHandler.check_null(self.data_with_missing)

        # validate
        self.assertTrue(response)

    def test_check_null_df_without_null(self):
        """Test check_missing with a dataframe without missing values."""
        self.data_without_missing = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "values": [1, 2, 3],
            }
        )

        # test
        response = DataHandler.check_null(self.data_without_missing)

        # validate
        self.assertFalse(response)

    def test_check_null_series_without_null(self):
        """Test check_missing with a dataframe without missing values."""
        self.data_without_missing = pd.Series([1, 2, 3])

        # test
        response = DataHandler.check_null(self.data_without_missing)

        # validate
        self.assertFalse(response)

    def test_handle_align_timestamps_fill_forward(self):
        response_missing_data = [
            {
                "id": 49252,
                "ts_event": "1651500000000000000",
                "symbol": "HE.n.0",
                "open": "104.0250",
                "close": "103.9250",
                "high": "104.2500",
                "low": "102.9500",
                "volume": 3553,
            },
            {
                "id": 49253,
                "ts_event": "1651500000000000000",
                "symbol": "ZC.n.0",
                "open": "802.0000",
                "close": "797.5000",
                "high": "804.0000",
                "low": "797.0000",
                "volume": 12195,
            },
            {
                "id": 49256,
                "ts_event": "1651503600000000000",
                "symbol": "ZC.n.0",
                "open": "797.5000",
                "close": "798.2500",
                "high": "800.5000",
                "low": "795.7500",
                "volume": 7173,
            },
            {
                "id": 49258,
                "ts_event": "1651507200000000000",
                "symbol": "HE.n.0",
                "open": "105.7750",
                "close": "104.7000",
                "high": "105.9500",
                "low": "104.2750",
                "volume": 2146,
            },
            {
                "id": 49259,
                "ts_event": "1651507200000000000",
                "symbol": "ZC.n.0",
                "open": "798.5000",
                "close": "794.2500",
                "high": "800.2500",
                "low": "794.0000",
                "volume": 9443,
            },
            {
                "id": 49262,
                "ts_event": "1651510800000000000",
                "symbol": "ZC.n.0",
                "open": "794.5000",
                "close": "801.5000",
                "high": "803.0000",
                "low": "794.2500",
                "volume": 8135,
            },
            {
                "id": 49263,
                "ts_event": "1651510800000000000",
                "symbol": "HE.n.0",
                "open": "104.7500",
                "close": "105.0500",
                "high": "105.2750",
                "low": "103.9500",
                "volume": 3057,
            },
        ]

        fill_forward_df = pd.DataFrame(
            [
                {
                    "ts_event": "1651500000000000000",
                    "symbol": "HE.n.0",
                    "open": "104.0250",
                    "close": "103.9250",
                    "high": "104.2500",
                    "low": "102.9500",
                    "volume": 3553,
                },
                {
                    "ts_event": "1651500000000000000",
                    "symbol": "ZC.n.0",
                    "open": "802.0000",
                    "close": "797.5000",
                    "high": "804.0000",
                    "low": "797.0000",
                    "volume": 12195,
                },
                {
                    "ts_event": "1651503600000000000",
                    "symbol": "ZC.n.0",
                    "open": "797.5000",
                    "close": "798.2500",
                    "high": "800.5000",
                    "low": "795.7500",
                    "volume": 7173,
                },
                {
                    "ts_event": "1651503600000000000",
                    "symbol": "HE.n.0",
                    "open": "104.0250",
                    "close": "103.9250",
                    "high": "104.2500",
                    "low": "102.9500",
                    "volume": 3553,
                },
                {
                    "ts_event": "1651507200000000000",
                    "symbol": "HE.n.0",
                    "open": "105.7750",
                    "close": "104.7000",
                    "high": "105.9500",
                    "low": "104.2750",
                    "volume": 2146,
                },
                {
                    "ts_event": "1651507200000000000",
                    "symbol": "ZC.n.0",
                    "open": "798.5000",
                    "close": "794.2500",
                    "high": "800.2500",
                    "low": "794.0000",
                    "volume": 9443,
                },
                {
                    "ts_event": "1651510800000000000",
                    "symbol": "ZC.n.0",
                    "open": "794.5000",
                    "close": "801.5000",
                    "high": "803.0000",
                    "low": "794.2500",
                    "volume": 8135,
                },
                {
                    "ts_event": "1651510800000000000",
                    "symbol": "HE.n.0",
                    "open": "104.7500",
                    "close": "105.0500",
                    "high": "105.2750",
                    "low": "103.9500",
                    "volume": 3057,
                },
            ]
        )

        # Expected df
        fill_forward_df["volume"] = fill_forward_df["volume"].astype("float64")
        fill_forward_df = fill_forward_df.sort_values(
            by=["ts_event", "symbol"]
        ).reset_index(drop=True)

        # Test
        df = pd.DataFrame(response_missing_data)
        df.drop(columns=["id"], inplace=True)
        result = DataHandler.align_timestamps(
            data=df, missing_values_strategy="fill_forward"
        )

        # Validation
        assert_frame_equal(result, fill_forward_df, check_dtype=True)

    def test_align_timestamps_df_drop(self):
        response_missing_data = [
            {
                "id": 49252,
                "ts_event": "1651500000000000000",
                "symbol": "HE.n.0",
                "open": "104.0250",
                "close": "103.9250",
                "high": "104.2500",
                "low": "102.9500",
                "volume": 3553,
            },
            {
                "id": 49253,
                "ts_event": "1651500000000000000",
                "symbol": "ZC.n.0",
                "open": "802.0000",
                "close": "797.5000",
                "high": "804.0000",
                "low": "797.0000",
                "volume": 12195,
            },
            {
                "id": 49256,
                "ts_event": "1651503600000000000",
                "symbol": "ZC.n.0",
                "open": "797.5000",
                "close": "798.2500",
                "high": "800.5000",
                "low": "795.7500",
                "volume": 7173,
            },
            {
                "id": 49258,
                "ts_event": "1651507200000000000",
                "symbol": "HE.n.0",
                "open": "105.7750",
                "close": "104.7000",
                "high": "105.9500",
                "low": "104.2750",
                "volume": 2146,
            },
            {
                "id": 49259,
                "ts_event": "1651507200000000000",
                "symbol": "ZC.n.0",
                "open": "798.5000",
                "close": "794.2500",
                "high": "800.2500",
                "low": "794.0000",
                "volume": 9443,
            },
            {
                "id": 49262,
                "ts_event": "1651510800000000000",
                "symbol": "ZC.n.0",
                "open": "794.5000",
                "close": "801.5000",
                "high": "803.0000",
                "low": "794.2500",
                "volume": 8135,
            },
            {
                "id": 49263,
                "ts_event": "1651510800000000000",
                "symbol": "HE.n.0",
                "open": "104.7500",
                "close": "105.0500",
                "high": "105.2750",
                "low": "103.9500",
                "volume": 3057,
            },
        ]

        drop_df = pd.DataFrame(
            [
                {
                    "ts_event": "1651500000000000000",
                    "symbol": "HE.n.0",
                    "open": "104.0250",
                    "close": "103.9250",
                    "high": "104.2500",
                    "low": "102.9500",
                    "volume": 3553,
                },
                {
                    "ts_event": "1651500000000000000",
                    "symbol": "ZC.n.0",
                    "open": "802.0000",
                    "close": "797.5000",
                    "high": "804.0000",
                    "low": "797.0000",
                    "volume": 12195,
                },
                {
                    "ts_event": "1651507200000000000",
                    "symbol": "HE.n.0",
                    "open": "105.7750",
                    "close": "104.7000",
                    "high": "105.9500",
                    "low": "104.2750",
                    "volume": 2146,
                },
                {
                    "ts_event": "1651507200000000000",
                    "symbol": "ZC.n.0",
                    "open": "798.5000",
                    "close": "794.2500",
                    "high": "800.2500",
                    "low": "794.0000",
                    "volume": 9443,
                },
                {
                    "ts_event": "1651510800000000000",
                    "symbol": "ZC.n.0",
                    "open": "794.5000",
                    "close": "801.5000",
                    "high": "803.0000",
                    "low": "794.2500",
                    "volume": 8135,
                },
                {
                    "ts_event": "1651510800000000000",
                    "symbol": "HE.n.0",
                    "open": "104.7500",
                    "close": "105.0500",
                    "high": "105.2750",
                    "low": "103.9500",
                    "volume": 3057,
                },
            ]
        )

        # Expected df
        drop_df["volume"] = drop_df["volume"].astype("float64")
        drop_df = drop_df.sort_values(by=["ts_event", "symbol"]).reset_index(
            drop=True
        )

        # Test
        df = pd.DataFrame(response_missing_data)
        df.drop(columns=["id"], inplace=True)
        result = DataHandler.align_timestamps(
            data=df, missing_values_strategy="drop"
        )

        # Validation
        assert_frame_equal(result, drop_df, check_dtype=True)

    def test_align_timestamps_fill_forward(self):
        data = pd.Series([1, 2, None, 3])

        # Expected
        expected = pd.Series([1, 2, 2, 3])

        # Test
        result = DataHandler.align_timestamps(
            data, missing_values_strategy="fill_forward"
        )

        # Validation
        self.assertEqual(list(result), list(expected))

    def test_align_timestamps_drop(self):
        data = pd.Series([1, 2, None, 3])

        # Expected
        expected = pd.Series([1, 2, 3])

        # Test
        result = DataHandler.align_timestamps(
            data, missing_values_strategy="drop"
        )

        # Validation
        self.assertEqual(list(result), list(expected))

    def test_check_duplicates_df_with_duplicates(self):
        self.data_with_duplicates = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-01", "2021-01-02"],
                "values": [1, 1, 3],
            }
        )

        # Test
        response = DataHandler.check_duplicates(self.data_with_duplicates)

        # validate
        self.assertTrue(response)

    def test_check_duplicates_df_with_duplicates_subset(self):
        self.data_with_duplicates = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-01", "2021-01-02"],
                "values": [1, 2, 3],
            }
        )

        # Test
        response = DataHandler.check_duplicates(
            self.data_with_duplicates, subset=["dates"]
        )

        # validate
        self.assertTrue(response)

    def test_check_duplicates_df_without_duplicates(self):
        self.data_without_duplicates = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "values": [1, 2, 3],
            }
        )

        # test
        response = DataHandler.check_duplicates(self.data_without_duplicates)

        # validate
        self.assertFalse(response)

    def test_check_duplicates_series_with_duplicates(self):
        self.data_with_duplicates = pd.Series([1, 2, 2, 3])

        # Test
        response = DataHandler.check_duplicates(self.data_with_duplicates)

        # validate
        self.assertTrue(response)

    def test_check_duplicates_series_without_duplicates(self):
        self.data_without_duplicates = pd.Series([1, 2, 3])

        # test
        response = DataHandler.check_duplicates(self.data_without_duplicates)

        # validate
        self.assertFalse(response)

    def test_handle_duplicates_df(self):
        data_with_duplicates = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-01", "2021-01-02"],
                "values": [1, 1, 3],
            }
        )

        # Expected
        expected = pd.DataFrame(
            {"dates": ["2021-01-01", "2021-01-02"], "values": [1, 3]}
        )

        # Test
        response = DataHandler.handle_duplicates(data_with_duplicates)

        # Validate
        assert_frame_equal(response, expected, check_dtype=True)

    def test_handle_duplicates_df_subset(self):
        data_with_duplicates = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-01-01", "2021-01-02"],
                "values": [1, 2, 3],
            }
        )

        # expected
        expected = pd.DataFrame(
            {"dates": ["2021-01-01", "2021-01-02"], "values": [1, 3]}
        )

        # test
        response = DataHandler.handle_duplicates(
            data_with_duplicates, subset=["dates"]
        )

        # validate
        assert_frame_equal(response, expected, check_dtype=True)

    def test_handle_duplicates_df_without(self):
        data_without_duplicates = pd.DataFrame(
            {
                "dates": ["2021-01-01", "2021-02-01", "2021-01-02"],
                "values": [1, 1, 3],
            }
        )

        # test
        response = DataHandler.handle_duplicates(data_without_duplicates)

        # validate
        assert_frame_equal(response, data_without_duplicates, check_dtype=True)

    def test_handle_duplicates_series(self):
        data = pd.Series([1, 2, 2, 3])

        # Expected
        expected = pd.Series([1, 2, 3])

        # test
        response = DataHandler.handle_duplicates(data)

        # Validation
        self.assertEqual(list(response), list(expected))

    def test_check_outliers_df(self):
        df = pd.DataFrame(
            {
                "A": [10, 12, 14, 1000, 16, 18, 20],
                "B": [5, 7, 9, 11, 13, 15, 1000],
            }
        )

        # expected
        expected = pd.DataFrame(
            {
                "A": [False, False, False, True, False, False, False],
                "B": [False, False, False, False, False, False, True],
            }
        )

        # test
        df_outliers_iqr = DataHandler.check_outliers(df, method="IQR")

        # validate
        assert_frame_equal(expected, df_outliers_iqr, check_dtype=True)

    def test_check_outliers_series(self):
        series = pd.Series([10, 12, 14, 1000, 16, 18, 20])

        # expected
        expected = pd.Series([False, False, False, True, False, False, False])

        # test
        series_outliers_zscore = DataHandler.check_outliers(
            series, method="Z-score", threshold=2
        )

        # validate
        self.assertEqual(list(expected), list(series_outliers_zscore))

    def test_normalize_column_wise_df(self):
        df = pd.DataFrame(
            {"A": [10, 12, 14, 16, 18, 20], "B": [5, 7, 9, 11, 13, 15]}
        )

        # expected
        expected = pd.DataFrame(
            {
                "A": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "B": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            }
        )

        # test
        df_normalized = DataHandler.normalize(df)

        # validate
        assert_frame_equal(expected, df_normalized)

    def test_normalize_global_df(self):
        df = pd.DataFrame(
            {"A": [10, 12, 14, 16, 18, 20], "B": [5, 7, 9, 11, 13, 15]}
        )

        # expected
        expected = pd.DataFrame(
            {
                "A": [
                    0.3333333333333333,
                    0.4666666666666667,
                    0.6,
                    0.7333333333333333,
                    0.8666666666666667,
                    1.0,
                ],
                "B": [
                    0.0,
                    0.13333333,
                    0.2666666666,
                    0.4000000000,
                    0.53333333,
                    0.66666667,
                ],
            }
        )

        # test
        df_normalized = DataHandler.normalize(df, "global")

        # validate
        assert_frame_equal(expected, df_normalized)

    def test_normalize_series(self):
        series = pd.Series([10, 12, 14, 16, 18, 20])

        # expected
        expected = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # test
        series_normalized = DataHandler.normalize(series)

        # validate
        self.assertEqual(list(series_normalized), list(expected))

    def test_standardize_column_wise_df(self):
        df = pd.DataFrame(
            {"A": [10, 12, 14, 16, 18, 20], "B": [5, 7, 9, 11, 13, 15]}
        )

        # expected
        expected = pd.DataFrame(
            {
                "A": [
                    -1.336306,
                    -0.801784,
                    -0.267261,
                    0.267261,
                    0.801784,
                    1.336306,
                ],
                "B": [
                    -1.336306,
                    -0.801784,
                    -0.267261,
                    0.267261,
                    0.801784,
                    1.336306,
                ],
            }
        )

        # test
        df_standardized = DataHandler.standardize(df)

        # validate
        assert_frame_equal(expected, df_standardized)

    def test_standardize_global_df(self):
        df = pd.DataFrame(
            {"A": [10, 12, 14, 16, 18, 20], "B": [5, 7, 9, 11, 13, 15]}
        )

        # expected
        expected = pd.DataFrame(
            {
                "A": [
                    -0.590624,
                    -0.118125,
                    0.354375,
                    0.826874,
                    1.299374,
                    1.771873,
                ],
                "B": [
                    -1.771873,
                    -1.299374,
                    -0.826874,
                    -0.354375,
                    0.118125,
                    0.590624,
                ],
            }
        )

        # test
        df_standardized = DataHandler.standardize(df, "global")

        # validate
        assert_frame_equal(expected, df_standardized)

    def test_standardize_series(self):
        series = pd.Series([10, 12, 14, 16, 18, 20])

        # expected
        expected = [
            -1.3363062095621219,
            -0.8017837257372732,
            -0.2672612419124244,
            0.2672612419124244,
            0.8017837257372732,
            1.3363062095621219,
        ]

        # test
        series_standardized = DataHandler.standardize(series)

        # validate
        self.assertEqual(list(series_standardized), list(expected))

    def test_sample_data_df(self):
        df = pd.DataFrame({"A": range(100), "B": range(100, 200)})
        # test
        df_sampled = DataHandler.sample_data(df, frac=0.1, random_state=1)

        # validate
        self.assertEqual(len(df_sampled), len(df) * 0.1)

    def test_sample_data_series(self):
        series = pd.Series(range(100))

        # test
        series_sampled = DataHandler.sample_data(
            series, frac=0.1, random_state=1
        )

        # validate
        self.assertEqual(len(series_sampled), len(series) * 0.1)

    def test_handle_outliers_df(self):
        df = pd.DataFrame(
            {
                "A": [10, 12, 14, 1000, 16, 18, 20],
                "B": [5, 7, 9, 11, 13, 15, 1000],
            }
        )

        # expected
        expected_df = pd.DataFrame(
            {"A": [10, 12, 14, 16, 18], "B": [5, 7, 9, 13, 15]}
        )

        # test
        df_outliers = DataHandler.handle_outliers(
            df, method="IQR", action="remove"
        )

        # validate
        assert_frame_equal(expected_df, df_outliers, check_dtype=True)

    def test_handle_outliers_series(self):
        series = pd.Series([10, 12, 14, 1000, 16, 18, 20])

        # expected
        expected = pd.Series([10, 12, 14, 16, 18, 20])

        # test
        series_outliers = DataHandler.handle_outliers(
            series, method="Z-score", factor=2, action="remove"
        )

        # validate
        self.assertEqual(list(series_outliers), list(expected))

    def test_split_data_default_ratio(self):
        train, test = DataHandler.split_data(self.sample_dataframe)
        # Default split ratio is 0.8
        expected_train_length = int(len(self.sample_dataframe) * 0.8)
        expected_test_length = (
            len(self.sample_dataframe) - expected_train_length
        )
        self.assertEqual(len(train), expected_train_length)
        self.assertEqual(len(test), expected_test_length)

    def test_lag_series_default(self):
        # Test with the default lag of 1
        lagged_series = DataHandler.lag_series(self.sample_series).reset_index(
            drop=True
        )
        expected_series = self.sample_series[:-1].reset_index(drop=True)
        self.assertTrue((lagged_series.values == expected_series.values).all())

    def test_lag_series_custom_lag(self):
        # Test with a custom lag of 5
        lag = 5
        lagged_series = DataHandler.lag_series(self.sample_series, lag=lag)
        self.assertEqual(len(lagged_series), len(self.sample_series) - lag)
        self.assertTrue(
            (lagged_series.values == self.sample_series[:-lag].values).all()
        )

    def test_split_data_custom_ratio(self):
        custom_ratio = 0.7
        train, test = DataHandler.split_data(
            self.sample_dataframe, train_ratio=custom_ratio
        )
        expected_train_length = int(len(self.sample_dataframe) * custom_ratio)
        expected_test_length = (
            len(self.sample_dataframe) - expected_train_length
        )
        self.assertEqual(len(train), expected_train_length)
        self.assertEqual(len(test), expected_test_length)

    def test_split_data_data_integrity(self):
        train, test = DataHandler.split_data(self.sample_dataframe)
        # Check if concatenated train and test sets equal the original data
        pd.testing.assert_frame_equal(
            pd.concat([train, test]).reset_index(drop=True),
            self.sample_dataframe,
        )

    # Type Check
    def test_check_null_type_error(self):
        with self.assertRaises(Exception):
            DataHandler.check_null([1, 2, 3, 4])

    def test_handle_null_series_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.align_timestamps(
                [1, 2, 2, 3], missing_values_strategy="fill_forward"
            )

    def test_check_duplicates_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.check_duplicates([123456, 3567])

    def test_handle_duplicates_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.handle_duplicates([123456, 3567])

    def test_check_outliers_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.check_outliers(
                [123456, 3567], method="Z-score", threshold=2
            )

    def test_normalize_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.normalize("123456")

    def test_standardize_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.standardize("123456")

    def test_sample_data_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.sample_data("series", frac=0.1, random_state=1)

    def test_handle_outliers_type_error(self):
        with self.assertRaises(TypeError):
            DataHandler.handle_outliers(
                [10, 12, 14, 16, 18, 20], method="IQR", factor=2, action="drop"
            )

    # Value check
    def test_handle_null_series_value_error(self):
        with self.assertRaises(ValueError):
            DataHandler.align_timestamps(
                pd.Series([1, 2, 2, 3]), missing_values_strategy="error"
            )

    def test_check_outliers_value_error(self):
        with self.assertRaises(ValueError):
            DataHandler.check_outliers(
                pd.Series([123456, 3567]), method="sdfgh", threshold=2
            )

    def test_normalize_value_error(self):
        with self.assertRaises(ValueError):
            DataHandler.normalize(pd.Series([10, 12, 14, 16, 18, 20]), "error")

    def test_standardize_value_error(self):
        with self.assertRaises(ValueError):
            DataHandler.standardize(
                pd.Series([10, 12, 14, 16, 18, 20]), "error"
            )

    def test_handle_outliers_value_error(self):
        with self.assertRaises(ValueError):
            DataHandler.handle_outliers(
                pd.Series([10, 12, 14, 16, 18, 20]),
                method="1",
                factor=2,
                action="remove",
            )

        with self.assertRaises(ValueError):
            DataHandler.handle_outliers(
                pd.Series([10, 12, 14, 16, 18, 20]),
                method="IQR",
                factor=2,
                action="drop",
            )

    # Edge Cases
    def test_check_null_type_empty(self):
        # test
        response = DataHandler.check_null(pd.Series([]))

        # validate
        self.assertFalse(response)

    def test_handle_null_fill_forward_null_first_value(self):
        response_missing_data = [
            {
                "id": 49252,
                "ts_event": "1651500000000000000",
                "symbol": "HE.n.0",
                "open": "104.0250",
                "close": "103.9250",
                "high": "104.2500",
                "low": "102.9500",
                "volume": 3553,
            },
            #   {"id":49253,"ts_event":"1651500000000000000","symbol":"ZC.n.0","open":"802.0000","close":"797.5000","high":"804.0000","low":"797.0000","volume":12195},
            {
                "id": 49256,
                "ts_event": "1651503600000000000",
                "symbol": "ZC.n.0",
                "open": "797.5000",
                "close": "798.2500",
                "high": "800.5000",
                "low": "795.7500",
                "volume": 7173,
            },
            {
                "id": 49257,
                "ts_event": "1651503600000000000",
                "symbol": "HE.n.0",
                "open": "103.8500",
                "close": "105.8500",
                "high": "106.6750",
                "low": "103.7750",
                "volume": 3489,
            },
            {
                "id": 49258,
                "ts_event": "1651507200000000000",
                "symbol": "HE.n.0",
                "open": "105.7750",
                "close": "104.7000",
                "high": "105.9500",
                "low": "104.2750",
                "volume": 2146,
            },
            {
                "id": 49259,
                "ts_event": "1651507200000000000",
                "symbol": "ZC.n.0",
                "open": "798.5000",
                "close": "794.2500",
                "high": "800.2500",
                "low": "794.0000",
                "volume": 9443,
            },
            {
                "id": 49262,
                "ts_event": "1651510800000000000",
                "symbol": "ZC.n.0",
                "open": "794.5000",
                "close": "801.5000",
                "high": "803.0000",
                "low": "794.2500",
                "volume": 8135,
            },
            {
                "id": 49263,
                "ts_event": "1651510800000000000",
                "symbol": "HE.n.0",
                "open": "104.7500",
                "close": "105.0500",
                "high": "105.2750",
                "low": "103.9500",
                "volume": 3057,
            },
        ]

        expected_df = pd.DataFrame(
            [
                {
                    "ts_event": "1651503600000000000",
                    "symbol": "HE.n.0",
                    "open": "103.8500",
                    "close": "105.8500",
                    "high": "106.6750",
                    "low": "103.7750",
                    "volume": 3489,
                },
                {
                    "ts_event": "1651503600000000000",
                    "symbol": "ZC.n.0",
                    "open": "797.5000",
                    "close": "798.2500",
                    "high": "800.5000",
                    "low": "795.7500",
                    "volume": 7173,
                },
                {
                    "ts_event": "1651507200000000000",
                    "symbol": "HE.n.0",
                    "open": "105.7750",
                    "close": "104.7000",
                    "high": "105.9500",
                    "low": "104.2750",
                    "volume": 2146,
                },
                {
                    "ts_event": "1651507200000000000",
                    "symbol": "ZC.n.0",
                    "open": "798.5000",
                    "close": "794.2500",
                    "high": "800.2500",
                    "low": "794.0000",
                    "volume": 9443,
                },
                {
                    "ts_event": "1651510800000000000",
                    "symbol": "HE.n.0",
                    "open": "104.7500",
                    "close": "105.0500",
                    "high": "105.2750",
                    "low": "103.9500",
                    "volume": 3057,
                },
                {
                    "ts_event": "1651510800000000000",
                    "symbol": "ZC.n.0",
                    "open": "794.5000",
                    "close": "801.5000",
                    "high": "803.0000",
                    "low": "794.2500",
                    "volume": 8135,
                },
            ]
        )

        expected_df["volume"] = expected_df["volume"].astype("float64")
        expected_df = expected_df.sort_values(
            by=["ts_event", "symbol"]
        ).reset_index(drop=True)

        # Test
        df = pd.DataFrame(response_missing_data)
        df.drop(columns=["id"], inplace=True)
        result = DataHandler.align_timestamps(
            data=df, missing_values_strategy="fill_forward"
        )

        # Validation
        assert_frame_equal(result, expected_df, check_dtype=True)


if __name__ == "__main__":
    unittest.main()
