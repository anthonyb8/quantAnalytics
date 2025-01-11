import numpy as np


class DataGenerator:
    @staticmethod
    def generate_mean_reverting_series(
        n: int = 2000,
        mu: float = 0,
        theta: float = 0.1,
        sigma: float = 0.2,
        start_value: float = 1,
    ) -> np.ndarray:
        """
        Generate a mean-reverting time series using the Ornstein-Uhlenbeck process.

        Parameters:
        - n (int): The number of observations in the time series.
        - mu (float): The long-term mean value towards which the time series reverts.
        - theta (float): The rate of reversion to the mean.
        - sigma (float): The volatility of the process.
        - start_value (float): The starting value of the time series.

        Returns:
        - np.array: A numpy array representing the generated time series.
        """
        time_series = [start_value]
        for _ in range(1, n):
            dt = 1  # Time step
            previous_value = time_series[-1]
            random_term = np.random.normal(loc=0.0, scale=np.sqrt(dt) * sigma)
            next_value = (
                previous_value
                + theta * (mu - previous_value) * dt
                + random_term
            )
            time_series.append(next_value)

        return np.array(time_series)

    @staticmethod
    def generate_trending_series(
        n: int = 2000,
        start_value: float = 0,
        trend: float = 5,
        step_std: float = 1,
    ) -> np.ndarray:
        """
        Generate a trending time series.

        Parameters:
        - n (int): The number of observations in the time series.
        - start_value (float): The starting value of the time series.
        - trend (float): The constant amount added to each step to create a trend.
        - step_std (float): The standard deviation of the step size.

        Returns:
        - np.array: A numpy array representing the generated time series.
        """
        time_series = [start_value]
        for _ in range(1, n):
            step = np.random.normal(scale=step_std) + trend
            next_value = time_series[-1] + step
            time_series.append(next_value)

        return np.array(time_series)

    @staticmethod
    def generate_random_walk_series(
        n: int = 2000, start_value: float = 0, step_std: float = 1
    ) -> np.ndarray:
        """
        Generate a random walk time series.

        Parameters:
        - n (int): The number of observations in the time series.
        - start_value (float): The starting value of the time series.
        - step_std (float): The standard deviation of the step size.

        Returns:
        - np.array: A numpy array representing the generated time series.
        """
        time_series = [start_value]
        for _ in range(1, n):
            step = np.random.normal(scale=step_std)
            next_value = time_series[-1] + step
            time_series.append(next_value)

        return np.array(time_series)
