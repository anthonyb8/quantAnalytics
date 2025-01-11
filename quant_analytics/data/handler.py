import numpy as np
import pandas as pd
from typing import List, Union, Tuple


class DataHandler:
    @staticmethod
    def lag_series(series: pd.Series, lag: int = 1) -> pd.Series:
        """
        Lags a pandas series by a given lag.

        Parameters:
        - series (pd.Series): The timeseries to be lagged.
        - lag (int): The number lags to be applied to the timeseries.

        Returns:
        - pd.Series : A pandas series representing the lagged series.
        """
        # Create a lagged version of the series
        series_lagged = series.shift(lag)

        # Remove NaN values and reset the index
        series_lagged = series_lagged[lag:].reset_index(drop=True)

        return series_lagged

    @staticmethod
    def split_data(
        data: pd.DataFrame, train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a pandas dataframe into a train and test dataframe.

        Parameters:
        - data (pd.DataFrame): The dataframe that is to be split.
        - train_ratio (float): The portion of original data to be in train dataframe.()

        Returns:
        - train_data (pd.DataFrame): Represents the portion of the data form the begining to the point equal to the train_ratio.
        - test_data (pd.DateFrame): Represents the portion of the data from the train_ratio point to the end of the data.
        """
        split_index = int(len(data) * train_ratio)
        train_data = data.iloc[:split_index].copy()
        test_data = data.iloc[split_index:].copy()
        return train_data, test_data

    @staticmethod
    def align_timestamps(
        data: Union[pd.DataFrame, pd.Series],
        missing_values_strategy: str = "fill_forward",
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles missing values in a DataFrame according to the specified strategy after pivoting the data for time series analysis.

        Parameters:
        - data (pd.DataFrame): The DataFrame with potential missing values.
        - missing_values_strategy (str): The strategy for handling missing values. Options are 'fill_forward' or 'drop'.

        Returns:
        - pd.DataFrame: The DataFrame with missing values handled according to the specified strategy.
        """

        if not isinstance(
            missing_values_strategy, str
        ) or missing_values_strategy not in ["fill_forward", "drop"]:
            raise ValueError(
                "'missing_values_strategy' must be either 'fill_forward' or 'drop' of type str."
            )

        if isinstance(data, pd.DataFrame):
            data = data.pivot(index="ts_event", columns="symbol")

            if missing_values_strategy == "drop":
                data.dropna(inplace=True)
            elif missing_values_strategy == "fill_forward":
                if data.iloc[0].isnull().any():
                    first_complete_index = data.dropna().index[0]
                    data = data.loc[first_complete_index:]
                data.ffill(inplace=True)

            return data.stack(level="symbol", future_stack=True).reset_index()

        elif isinstance(data, pd.Series):
            if missing_values_strategy == "drop":
                return data.dropna()
            elif missing_values_strategy == "fill_forward":
                return data.ffill()

        else:
            raise TypeError("Input data must be a DataFrame or Series.")

    @staticmethod
    def check_null(data: Union[pd.DataFrame, pd.Series]) -> bool:
        """
        Checks a DataFrame for any missing values.

        Parameters:
            - data (pd.DataFrame): The DataFrame to check for missing values.

        Returns:
            - bool: True if any missing values are found, otherwise False.
        """

        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input data must be a DataFrame or Series.")

        if isinstance(data, pd.DataFrame):
            return data.isna().any().any()
        elif isinstance(data, pd.Series):
            return data.isna().any()

    # else:
    #     raise TypeError("Input data must be a DataFrame or Series.")

    @staticmethod
    def check_duplicates(
        data: Union[pd.DataFrame, pd.Series], subset: List[str] = None
    ) -> bool:
        """
        Checks for and prints out any duplicate records in the data.

        Parameters:
        - data (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to check for duplicates.
        - subset (List[str], optional): List of columns to check for duplicates in a DataFrame. Defaults to None.

        Side Effect:
        - Prints out duplicate rows if found.

        Returns:
        - bool: True if any duplicate values are found, otherwise False.
        """
        if isinstance(data, pd.DataFrame):
            duplicates = data.duplicated(subset=subset, keep=False)
            if duplicates.any():
                print("Duplicates found in DataFrame:")
                print(data[duplicates])
                return True

            return False

        elif isinstance(data, pd.Series):
            duplicates = data.duplicated(keep=False)
            if duplicates.any():
                print("Duplicates found in Series:")
                print(data[duplicates])
                return True
            return False
        else:
            raise TypeError("Input data must be a DataFrame or Series.")

    @staticmethod
    def handle_duplicates(
        data: Union[pd.DataFrame, pd.Series],
        keep: str = "first",
        subset: List[str] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle duplicate rows in the data frame or series.

        Parameters:
        - data (Union[pd.DataFrame, pd.Series]): Input data frame or series.
        - keep (str): Which duplicates to keep ('first', 'last', or False to drop all duplicates).
        - subset (List[str], optional): List of columns to check for duplicates in a DataFrame. Defaults to None.

        Returns:
        - Union[pd.DataFrame, pd.Series]: Data frame or series with duplicates handled.

        Raises:
        - TypeError: If the input data is not a DataFrame or Series.
        """
        if isinstance(data, pd.DataFrame):
            result = data.drop_duplicates(
                subset=subset, keep=keep
            ).reset_index(drop=True)
        elif isinstance(data, pd.Series):
            result = data.drop_duplicates(keep=keep).reset_index(
                drop=True, level=0
            )
        else:
            raise TypeError("Input data must be a DataFrame or Series.")

        return result

    @staticmethod
    def check_outliers(
        data: Union[pd.DataFrame, pd.Series],
        method: str = "IQR",
        threshold: float = 1.5,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Checks for outliers in a DataFrame or Series.

        Parameters:
        - data (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to check for outliers.
        - method (str): Method to detect outliers ('IQR' or 'Z-score'). Defaults to 'IQR'.
        - threshold (float): Threshold for detecting outliers. For 'IQR', it's the multiplication factor for the IQR. For 'Z-score', it's the number of standard deviations.

        Returns:
        - Union[pd.DataFrame, pd.Series]: A boolean DataFrame or Series indicating the position of outliers.

        Raises:
        - TypeError: If the input data is not a DataFrame or Series.
        - ValueError: If an invalid method is provided.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input data must be a DataFrame or Series.")

        if method not in ["IQR", "Z-score"]:
            raise ValueError("Method must be 'IQR' or 'Z-score'.")

        if method == "IQR":
            if isinstance(data, pd.DataFrame):
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (data < (Q1 - threshold * IQR)) | (
                    data > (Q3 + threshold * IQR)
                )
            else:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (data < (Q1 - threshold * IQR)) | (
                    data > (Q3 + threshold * IQR)
                )

        elif method == "Z-score":
            if isinstance(data, pd.DataFrame):
                z_scores = (data - data.mean()) / data.std()
                outliers = z_scores.abs() > threshold
            else:
                z_scores = (data - data.mean()) / data.std()
                outliers = z_scores.abs() > threshold

        return outliers

    @staticmethod
    def handle_outliers(
        data: Union[pd.DataFrame, pd.Series],
        method: str = "IQR",
        factor: float = 1.5,
        action: str = "remove",
        replacement_value: float = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles outliers in the data using specified method and action.

        Parameters:
        - data (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to handle outliers.
        - method (str): Method to detect outliers ('IQR' or 'Z-score').
        - factor (float): The multiplication factor for the IQR method.
        - action (str): Action to take on outliers ('remove' or 'replace').
        - replacement_value (float, optional): The value to replace outliers with if action is 'replace'.

        Returns:
        - Union[pd.DataFrame, pd.Series]: Data with outliers handled.

        Raises:
        - TypeError: If the input data is not a DataFrame or Series.
        - ValueError: If an invalid method or action is provided.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input data must be a DataFrame or Series.")

        if method not in ["IQR", "Z-score"]:
            raise ValueError("Method must be 'IQR' or 'Z-score'.")

        if action not in ["remove", "replace"]:
            raise ValueError("Action must be 'remove' or 'replace'.")

        if method == "IQR":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (data < (Q1 - factor * IQR)) | (
                data > (Q3 + factor * IQR)
            )
        elif method == "Z-score":
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > factor

        if action == "remove":
            if isinstance(data, pd.DataFrame):
                data = data[~outliers.any(axis=1)].reset_index(drop=True)
            else:
                data = data[~outliers].reset_index(drop=True)
        elif action == "replace":
            if replacement_value is None:
                raise ValueError(
                    "replacement_value must be specified if action is 'replace'."
                )
            data[outliers] = replacement_value

        return data

    # -- Manipulate --
    @staticmethod
    def normalize(
        data: Union[pd.DataFrame, pd.Series], method: str = "column-wise"
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Normalizes the data to a range of [0, 1].

        Parameters:
        - data (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to normalize.
        - method (str): The method of normalization ('column-wise' or 'global').

        Returns:
        - Union[pd.DataFrame, pd.Series]: Normalized DataFrame or Series.

        Raises:
        - TypeError: If the input data is not a DataFrame or Series.
        - ValueError: If the method is not 'column-wise' or 'global'.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input data must be a DataFrame or Series.")

        if method not in ["column-wise", "global"]:
            raise ValueError("Method must be 'column-wise' or 'global'.")

        if method == "column-wise":
            return (data - data.min()) / (data.max() - data.min())
        elif method == "global":
            global_min = (
                data.min().min()
                if isinstance(data, pd.DataFrame)
                else data.min()
            )
            global_max = (
                data.max().max()
                if isinstance(data, pd.DataFrame)
                else data.max()
            )
            return (data - global_min) / (global_max - global_min)

    @staticmethod
    def standardize(
        data: Union[pd.DataFrame, pd.Series], method: str = "column-wise"
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Standardizes the data to have a mean of 0 and a standard deviation of 1.

        Parameters:
        - data (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to standardize.
        - method (str): The method of standardization ('column-wise' or 'global').

        Returns:
        - Union[pd.DataFrame, pd.Series]: Standardized DataFrame or Series.

        Raises:
        - TypeError: If the input data is not a DataFrame or Series.
        - ValueError: If the method is not 'column-wise' or 'global'.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input data must be a DataFrame or Series.")

        if method not in ["column-wise", "global"]:
            raise ValueError("Method must be 'column-wise' or 'global'.")

        if method == "column-wise":
            return (data - data.mean()) / data.std()
        elif method == "global":
            global_mean = (
                data.values.mean()
                if isinstance(data, pd.DataFrame)
                else data.mean()
            )
            global_std = (
                data.values.std()
                if isinstance(data, pd.DataFrame)
                else data.std()
            )
            return (data - global_mean) / global_std

    @staticmethod
    def sample_data(
        data: Union[pd.DataFrame, pd.Series],
        frac: float = 0.1,
        random_state: int = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Samples a fraction of the data.

        Parameters:
        - data (Union[pd.DataFrame, pd.Series]): The DataFrame or Series to sample.
        - frac (float): Fraction of data to sample.
        - random_state (int): Seed for the random number generator.

        Returns:
        - Union[pd.DataFrame, pd.Series]: Sampled DataFrame or Series.

        Raises:
        - TypeError: If the input data is not a DataFrame or Series.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Input data must be a DataFrame or Series.")

        return data.sample(frac=frac, random_state=random_state)
