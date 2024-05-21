import unittest
import pandas as pd
from datetime import datetime
from typing import List, Dict
from unittest.mock import Mock
from pandas.testing import assert_frame_equal

from quantAnalytics.dataprocessor import DataProcessor

def valid_process_data(db_response: List[Dict]):
    df = pd.DataFrame(db_response)
    df.drop(columns=['id'], inplace=True)

    # Convert OHLCV columns to floats
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    df[ohlcv_columns] = df[ohlcv_columns].astype(float)

    df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

    return df


class TestDataProcessor(unittest.TestCase):
    # Basic Validation
    def test_check_null_df_with_null(self):
        """Test check_missing with a dataframe containing missing values."""
        self.data_with_missing = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'values': [1, None, 3]
        })
        # test 
        response=DataProcessor.check_null(self.data_with_missing)

        # validate
        self.assertTrue(response)

    def test_check_null_series_with_null(self):
        # test series 
        self.data_with_missing = pd.Series([1, None, 3])
        # test 
        response=DataProcessor.check_null(self.data_with_missing)

        # validate
        self.assertTrue(response)

    def test_check_null_df_without_null(self):
        """Test check_missing with a dataframe without missing values."""
        self.data_without_missing = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'values': [1, 2, 3]
        })

        # test
        response = DataProcessor.check_null(self.data_without_missing)
        
        # validate
        self.assertFalse(response)

    def test_check_null_series_without_null(self):
        """Test check_missing with a dataframe without missing values."""
        self.data_without_missing = pd.Series([1, 2, 3])

        # test
        response = DataProcessor.check_null(self.data_without_missing)
        
        # validate
        self.assertFalse(response)
    
    def test_handle_null_df_fill_forward(self):
        response_missing_data = [{"id":49252,"timestamp":"1651500000000000000","symbol":"HE.n.0","open":"104.0250","close":"103.9250","high":"104.2500","low":"102.9500","volume":3553},
                                  {"id":49253,"timestamp":"1651500000000000000","symbol":"ZC.n.0","open":"802.0000","close":"797.5000","high":"804.0000","low":"797.0000","volume":12195},
                                  {"id":49256,"timestamp":"1651503600000000000","symbol":"ZC.n.0","open":"797.5000","close":"798.2500","high":"800.5000","low":"795.7500","volume":7173},
                                #   {"id":49257,"timestamp":"1651503600000000000","symbol":"HE.n.0","open":"103.8500","close":"105.8500","high":"106.6750","low":"103.7750","volume":3489},
                                  {"id":49258,"timestamp":"1651507200000000000","symbol":"HE.n.0","open":"105.7750","close":"104.7000","high":"105.9500","low":"104.2750","volume":2146},
                                  {"id":49259,"timestamp":"1651507200000000000","symbol":"ZC.n.0","open":"798.5000","close":"794.2500","high":"800.2500","low":"794.0000","volume":9443},
                                  {"id":49262,"timestamp":"1651510800000000000","symbol":"ZC.n.0","open":"794.5000","close":"801.5000","high":"803.0000","low":"794.2500","volume":8135},
                                  {"id":49263,"timestamp":"1651510800000000000","symbol":"HE.n.0","open":"104.7500","close":"105.0500","high":"105.2750","low":"103.9500","volume":3057},
        ]

        fill_forward_df = pd.DataFrame([{"timestamp":"1651500000000000000","symbol":"HE.n.0","open":"104.0250","close":"103.9250","high":"104.2500","low":"102.9500","volume":3553},
                                        {"timestamp":"1651500000000000000","symbol":"ZC.n.0","open":"802.0000","close":"797.5000","high":"804.0000","low":"797.0000","volume":12195},
                                        {"timestamp":"1651503600000000000","symbol":"ZC.n.0","open":"797.5000","close":"798.2500","high":"800.5000","low":"795.7500","volume":7173},
                                        {"timestamp":"1651503600000000000","symbol":"HE.n.0","open":"104.0250","close":"103.9250","high":"104.2500","low":"102.9500","volume":3553},
                                        {"timestamp":"1651507200000000000","symbol":"HE.n.0","open":"105.7750","close":"104.7000","high":"105.9500","low":"104.2750","volume":2146},
                                        {"timestamp":"1651507200000000000","symbol":"ZC.n.0","open":"798.5000","close":"794.2500","high":"800.2500","low":"794.0000","volume":9443},
                                        {"timestamp":"1651510800000000000","symbol":"ZC.n.0","open":"794.5000","close":"801.5000","high":"803.0000","low":"794.2500","volume":8135},
                                        {"timestamp":"1651510800000000000","symbol":"HE.n.0","open":"104.7500","close":"105.0500","high":"105.2750","low":"103.9500","volume":3057},
        ])
        
        # Expected df
        fill_forward_df['volume'] = fill_forward_df['volume'].astype('float64')
        fill_forward_df = fill_forward_df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)

        # Test 
        df = pd.DataFrame(response_missing_data)
        df.drop(columns=['id'], inplace=True)
        result = DataProcessor.handle_null(data=df, missing_values_strategy='fill_forward')

        # Validation
        assert_frame_equal(result, fill_forward_df, check_dtype=True)

    def test_handle_null_df_drop(self):
        response_missing_data = [{"id":49252,"timestamp":"1651500000000000000","symbol":"HE.n.0","open":"104.0250","close":"103.9250","high":"104.2500","low":"102.9500","volume":3553},
                                  {"id":49253,"timestamp":"1651500000000000000","symbol":"ZC.n.0","open":"802.0000","close":"797.5000","high":"804.0000","low":"797.0000","volume":12195},
                                  {"id":49256,"timestamp":"1651503600000000000","symbol":"ZC.n.0","open":"797.5000","close":"798.2500","high":"800.5000","low":"795.7500","volume":7173},
                                #   {"id":49257,"timestamp":"1651503600000000000","symbol":"HE.n.0","open":"103.8500","close":"105.8500","high":"106.6750","low":"103.7750","volume":3489},
                                  {"id":49258,"timestamp":"1651507200000000000","symbol":"HE.n.0","open":"105.7750","close":"104.7000","high":"105.9500","low":"104.2750","volume":2146},
                                  {"id":49259,"timestamp":"1651507200000000000","symbol":"ZC.n.0","open":"798.5000","close":"794.2500","high":"800.2500","low":"794.0000","volume":9443},
                                  {"id":49262,"timestamp":"1651510800000000000","symbol":"ZC.n.0","open":"794.5000","close":"801.5000","high":"803.0000","low":"794.2500","volume":8135},
                                  {"id":49263,"timestamp":"1651510800000000000","symbol":"HE.n.0","open":"104.7500","close":"105.0500","high":"105.2750","low":"103.9500","volume":3057},
        ]

        drop_df = pd.DataFrame([{"timestamp":"1651500000000000000","symbol":"HE.n.0","open":"104.0250","close":"103.9250","high":"104.2500","low":"102.9500","volume":3553},
                                {"timestamp":"1651500000000000000","symbol":"ZC.n.0","open":"802.0000","close":"797.5000","high":"804.0000","low":"797.0000","volume":12195},
                                {"timestamp":"1651507200000000000","symbol":"HE.n.0","open":"105.7750","close":"104.7000","high":"105.9500","low":"104.2750","volume":2146},
                                {"timestamp":"1651507200000000000","symbol":"ZC.n.0","open":"798.5000","close":"794.2500","high":"800.2500","low":"794.0000","volume":9443},
                                {"timestamp":"1651510800000000000","symbol":"ZC.n.0","open":"794.5000","close":"801.5000","high":"803.0000","low":"794.2500","volume":8135},
                                {"timestamp":"1651510800000000000","symbol":"HE.n.0","open":"104.7500","close":"105.0500","high":"105.2750","low":"103.9500","volume":3057},
        ])
        
        # Expected df
        drop_df['volume'] = drop_df['volume'].astype('float64')
        drop_df = drop_df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)

        # Test 
        df = pd.DataFrame(response_missing_data)
        df.drop(columns=['id'], inplace=True)
        result = DataProcessor.handle_null(data=df, missing_values_strategy='drop')

        # Validation
        assert_frame_equal(result, drop_df, check_dtype=True)
    
    def test_handle_null_series_fill_forward(self):
        data = pd.Series([1, 2, None,3])
    
        # Expected
        expected = pd.Series([1, 2, 2, 3])

        # Test 
        result = DataProcessor.handle_null(data, missing_values_strategy='fill_forward')

        # Validation
        self.assertEqual(list(result), list(expected))
    
    def test_handle_null_series_drop(self):
        data = pd.Series([1, 2, None,3])
    
        # Expected
        expected = pd.Series([1, 2, 3])

        # Test 
        result = DataProcessor.handle_null(data, missing_values_strategy='drop')

        # Validation
        self.assertEqual(list(result), list(expected))
    
    def test_check_duplicates_df_with_duplicates(self):
        self.data_with_duplicates = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-01', '2021-01-02'],
            'values': [1, 1, 3]
        })

        # test
        response = DataProcessor.check_duplicates(self.data_with_duplicates)

        # validate
        self.assertTrue(response)

    def test_check_duplicates_df_with_duplicates_subset(self):
        self.data_with_duplicates = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-01', '2021-01-02'],
            'values': [1, 2, 3]
        })

        # test
        response = DataProcessor.check_duplicates(self.data_with_duplicates, subset=['dates'])

        # validate
        self.assertTrue(response)

    def test_check_duplicates_df_without_duplicates(self):
        self.data_without_duplicates = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'values': [1, 2, 3]
        })
        
        # test
        response = DataProcessor.check_duplicates(self.data_without_duplicates)

        # validate
        self.assertFalse(response)

    def test_check_duplicates_series_with_duplicates(self):
        self.data_with_duplicates = pd.Series([1, 2, 2,3])

        # test
        response = DataProcessor.check_duplicates(self.data_with_duplicates)

        # validate
        self.assertTrue(response)

    def test_check_duplicates_series_without_duplicates(self):
        self.data_without_duplicates = pd.Series([1, 2, 3])

        # test
        response = DataProcessor.check_duplicates(self.data_without_duplicates)

        # validate
        self.assertFalse(response)

    def test_handle_duplicates_df(self):
        data_with_duplicates = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-01', '2021-01-02'],
            'values': [1, 1, 3]
        })

        # expected
        expected = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-02'],
            'values': [1, 3]
        })

        # test
        response = DataProcessor.handle_duplicates(data_with_duplicates)

        # validate
        assert_frame_equal(response, expected, check_dtype=True)

    def test_handle_duplicates_df_subset(self):
        data_with_duplicates = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-01', '2021-01-02'],
            'values': [1, 2, 3]
        })

        # expected
        expected = pd.DataFrame({
            'dates': ['2021-01-01', '2021-01-02'],
            'values': [1, 3]
        })

        # test
        response = DataProcessor.handle_duplicates(data_with_duplicates, subset=['dates'])

        # validate
        assert_frame_equal(response, expected, check_dtype=True)

    def test_handle_duplicates_df_without(self):
        data_without_duplicates = pd.DataFrame({
            'dates': ['2021-01-01', '2021-02-01', '2021-01-02'],
            'values': [1, 1, 3]
        })

        # test
        response = DataProcessor.handle_duplicates(data_without_duplicates)

        # validate
        assert_frame_equal(response, data_without_duplicates, check_dtype=True)

    def test_handle_duplicates_series(self):
        data = pd.Series([1, 2, 2,3])
    
        # Expected
        expected = pd.Series([1, 2, 3])

        # test
        response = DataProcessor.handle_duplicates(data)

        # Validation
        self.assertEqual(list(response), list(expected))

    def test_check_outliers_df(self):
        df = pd.DataFrame({
            'A': [10, 12, 14, 1000, 16, 18, 20],
            'B': [5, 7, 9, 11, 13, 15, 1000]
        })

        # expected
        expected = pd.DataFrame({
            'A': [False, False, False, True, False, False,False],
            'B': [False, False, False, False, False, False, True]
        })

        # test
        df_outliers_iqr = DataProcessor.check_outliers(df, method='IQR')
        
        # validate    
        assert_frame_equal(expected , df_outliers_iqr, check_dtype=True)
        
    def test_check_outliers_series(self):
        series = pd.Series([10, 12, 14, 1000, 16, 18, 20])

        # expected
        expected = pd.Series([False, False, False, True, False, False,False])

        # test
        series_outliers_zscore = DataProcessor.check_outliers(series, method='Z-score', threshold=2)

        #validate
        self.assertEqual(list(expected), list(series_outliers_zscore))

    def test_normalize_column_wise_df(self):
        df = pd.DataFrame({
            'A': [10, 12, 14, 16, 18, 20],
            'B': [5, 7, 9, 11, 13, 15]
        })

        # expected
        expected = pd.DataFrame({
            'A': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'B': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        })

        # test
        df_normalized = DataProcessor.normalize(df)

        # validate
        assert_frame_equal(expected, df_normalized)

    def test_normalize_global_df(self):
        df = pd.DataFrame({
            'A': [10, 12, 14, 16, 18, 20],
            'B': [5, 7, 9, 11, 13, 15]
        })

        # expected
        expected = pd.DataFrame({
            'A': [0.3333333333333333, 0.4666666666666667, 0.6, 0.7333333333333333, 0.8666666666666667, 1.0],
            'B':[0.0, 0.13333333, 0.2666666666, 0.4000000000, 0.53333333, 0.66666667]
        })

        # test
        df_normalized = DataProcessor.normalize(df, 'global')

        # validate
        assert_frame_equal(expected, df_normalized)

    def test_normalize_series(self):
        series = pd.Series([10, 12, 14, 16, 18, 20])
        
        # expected
        expected = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # test
        series_normalized = DataProcessor.normalize(series)
        
        # validate
        self.assertEqual(list(series_normalized), list(expected))

    def test_standardize_column_wise_df(self):
        df = pd.DataFrame({
            'A': [10, 12, 14, 16, 18, 20],
            'B': [5, 7, 9, 11, 13, 15]
        })

        # expected
        expected = pd.DataFrame({
            'A': [-1.336306,-0.801784, -0.267261,  0.267261, 0.801784, 1.336306],
            'B': [-1.336306,-0.801784, -0.267261,  0.267261, 0.801784, 1.336306]
        })

        # test
        df_standardized = DataProcessor.standardize(df)

        # validate
        assert_frame_equal(expected, df_standardized)

    def test_standardize_global_df(self):
        df = pd.DataFrame({
            'A': [10, 12, 14, 16, 18, 20],
            'B': [5, 7, 9, 11, 13, 15]
        })

        # expected
        expected = pd.DataFrame({
            'A': [-0.590624, -0.118125, 0.354375,  0.826874 , 1.299374, 1.771873 ],
            'B':[-1.771873, -1.299374, -0.826874,  -0.354375, 0.118125, 0.590624]
        })

        # test
        df_standardized = DataProcessor.standardize(df, 'global')

        print(df_standardized)

        # validate
        assert_frame_equal(expected, df_standardized)

    def test_standardize_series(self):
        series = pd.Series([10, 12, 14, 16, 18, 20])
        
        # expected
        expected =  [-1.3363062095621219, -0.8017837257372732, -0.2672612419124244, 0.2672612419124244, 0.8017837257372732, 1.3363062095621219]

        # test
        series_standardized = DataProcessor.standardize(series)
        
        # validate
        self.assertEqual(list(series_standardized), list(expected))

    def test_sample_data_df(self):
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200)
        })
        # test
        df_sampled = DataProcessor.sample_data(df, frac=0.1, random_state=1)

        # validate  
        self.assertEqual(len(df_sampled), len(df)*0.1)

    def test_sample_data_series(self):
        series = pd.Series(range(100))

        # test
        series_sampled = DataProcessor.sample_data(series, frac=0.1, random_state=1)

        # validate  
        self.assertEqual(len(series_sampled), len(series)*0.1)
   
    def test_handle_outliers_df(self):
        df = pd.DataFrame({
            'A': [10, 12, 14, 1000, 16, 18, 20],
            'B': [5, 7, 9, 11, 13, 15, 1000]
        })

        # expected
        expected_df = pd.DataFrame({
            'A': [10, 12, 14, 16, 18],
            'B': [5, 7, 9, 13, 15]
        })

        # test
        df_outliers = DataProcessor.handle_outliers(df, method='IQR', action='remove')
        print(df_outliers)
        # validate
        assert_frame_equal(expected_df, df_outliers, check_dtype=True)

    def test_handle_outliers_series(self):
        series = pd.Series([10, 12, 14, 1000, 16, 18, 20])

        # expected
        expected = pd.Series([10, 12, 14, 16, 18, 20])

        # test
        series_outliers = DataProcessor.handle_outliers(series, method='Z-score', factor=2, action='remove')
        
        # validate
        self.assertEqual(list(series_outliers), list(expected))

    # Type Check
    def test_check_null_type_error(self):
        with self.assertRaises(Exception):
            DataProcessor.check_null([1,2,3,4])

    def test_handle_null_series_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.handle_null([1, 2, 2, 3], missing_values_strategy='fill_forward')

    def test_check_duplicates_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.check_duplicates([123456,3567])

    def test_handle_duplicates_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.handle_duplicates([123456,3567])

    def test_check_outliers_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.check_outliers([123456,3567], method='Z-score', threshold=2)

    def test_normalize_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.normalize("123456")

    def test_standardize_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.standardize("123456")

    def test_sample_data_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.sample_data("series", frac=0.1, random_state=1)

    def test_handle_outliers_type_error(self):
        with self.assertRaises(TypeError):
            DataProcessor.handle_outliers([10, 12, 14, 16, 18, 20], method='IQR', factor=2, action='drop')
            

    # Value check
    def test_handle_null_series_value_error(self):
        with self.assertRaises(ValueError):
            result = DataProcessor.handle_null(pd.Series([1, 2, 2, 3]), missing_values_strategy='error')

    def test_check_outliers_type_error(self):
        with self.assertRaises(ValueError):
            DataProcessor.check_outliers(pd.Series([123456,3567]), method='sdfgh', threshold=2)
    
    def test_normalize_value_error(self):
        with self.assertRaises(ValueError):
            DataProcessor.normalize(pd.Series([10, 12, 14, 16, 18, 20]), "error")

    def test_standardize_value_error(self):
        with self.assertRaises(ValueError):
            DataProcessor.standardize(pd.Series([10, 12, 14, 16, 18, 20]), "error")

    def test_handle_outliers_value_error(self):
        with self.assertRaises(ValueError):
            DataProcessor.handle_outliers(pd.Series([10, 12, 14, 16, 18, 20]), method='1', factor=2, action='remove')

        with self.assertRaises(ValueError):
            DataProcessor.handle_outliers(pd.Series([10, 12, 14, 16, 18, 20]), method='IQR', factor=2, action='drop')
            
    # Edge Cases
    def test_check_null_type_empty(self):
        # test
        response=DataProcessor.check_null(pd.Series([]))

        # validate
        self.assertFalse(response)

    def test_handle_null_fill_forward_null_first_value(self):
        response_missing_data = [{"id":49252,"timestamp":"1651500000000000000","symbol":"HE.n.0","open":"104.0250","close":"103.9250","high":"104.2500","low":"102.9500","volume":3553},
                                #   {"id":49253,"timestamp":"1651500000000000000","symbol":"ZC.n.0","open":"802.0000","close":"797.5000","high":"804.0000","low":"797.0000","volume":12195},
                                  {"id":49256,"timestamp":"1651503600000000000","symbol":"ZC.n.0","open":"797.5000","close":"798.2500","high":"800.5000","low":"795.7500","volume":7173},
                                  {"id":49257,"timestamp":"1651503600000000000","symbol":"HE.n.0","open":"103.8500","close":"105.8500","high":"106.6750","low":"103.7750","volume":3489},
                                  {"id":49258,"timestamp":"1651507200000000000","symbol":"HE.n.0","open":"105.7750","close":"104.7000","high":"105.9500","low":"104.2750","volume":2146},
                                  {"id":49259,"timestamp":"1651507200000000000","symbol":"ZC.n.0","open":"798.5000","close":"794.2500","high":"800.2500","low":"794.0000","volume":9443},
                                  {"id":49262,"timestamp":"1651510800000000000","symbol":"ZC.n.0","open":"794.5000","close":"801.5000","high":"803.0000","low":"794.2500","volume":8135},
                                  {"id":49263,"timestamp":"1651510800000000000","symbol":"HE.n.0","open":"104.7500","close":"105.0500","high":"105.2750","low":"103.9500","volume":3057},
        ]

        expected_df = pd.DataFrame([
                                    {"timestamp":"1651503600000000000","symbol":"HE.n.0","open":"103.8500","close":"105.8500","high":"106.6750","low":"103.7750","volume":3489},
                                    {"timestamp":"1651503600000000000","symbol":"ZC.n.0","open":"797.5000","close":"798.2500","high":"800.5000","low":"795.7500","volume":7173},
                                    {"timestamp":"1651507200000000000","symbol":"HE.n.0","open":"105.7750","close":"104.7000","high":"105.9500","low":"104.2750","volume":2146},
                                    {"timestamp":"1651507200000000000","symbol":"ZC.n.0","open":"798.5000","close":"794.2500","high":"800.2500","low":"794.0000","volume":9443},
                                    {"timestamp":"1651510800000000000","symbol":"HE.n.0","open":"104.7500","close":"105.0500","high":"105.2750","low":"103.9500","volume":3057},
                                    {"timestamp":"1651510800000000000","symbol":"ZC.n.0","open":"794.5000","close":"801.5000","high":"803.0000","low":"794.2500","volume":8135},
        ])

        expected_df['volume'] = expected_df['volume'].astype('float64')
        expected_df = expected_df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)
        
        # Test    
        df = pd.DataFrame(response_missing_data)
        df.drop(columns=['id'], inplace=True)
        result = DataProcessor.handle_null(data=df, missing_values_strategy='fill_forward')

        # Validation
        assert_frame_equal(result, expected_df, check_dtype=True)


if __name__ =="__main__":
    unittest.main()
