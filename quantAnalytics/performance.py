import numpy as np
import pandas as pd

class PerformanceStatistics:
    @staticmethod
    def validate_trade_log(trade_log:pd.DataFrame) -> bool:
        """
        Validate the structure and data types of a trade log DataFrame.

        This method ensures that the provided DataFrame has the expected columns
        with the correct data types. If the 'start_date' or 'end_date' columns
        are not in datetime format, it attempts to convert them. Other columns
        are checked for their specified data types and converted if necessary.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame to validate. It should
                                  contain the following columns:
                                  - 'trade_id' (int64)
                                  - 'start_date' (object, convertible to datetime)
                                  - 'end_date' (object, convertible to datetime)
                                  - 'entry_value' (float64)
                                  - 'exit_value' (float64)
                                  - 'fees' (float64)
                                  - 'pnl' (float64)
                                  - 'gain/loss' (float64)

        Returns:
        - bool: True if the DataFrame is valid.
        """
        expected_columns = {
            'trade_id': 'int64',
            'start_date': 'object',  # Could enforce datetime dtype after initial check
            'end_date': 'object',    # Could enforce datetime dtype after initial check
            'entry_value': 'float64',
            'exit_value': 'float64',
            'fees': 'float64',
            'pnl': 'float64',
            'gain/loss': 'float64'
        }
        
        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check for the presence of all expected columns
        missing_columns = [col for col in expected_columns if col not in trade_log.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in trade_log DataFrame: {missing_columns}")
        
        # Check data types of columns
        for column, expected_dtype in expected_columns.items():
            if trade_log[column].dtype != expected_dtype:
                # Attempt to convert column types if not matching
                try:
                    if expected_dtype == 'object' and column in ['start_date', 'end_date']:
                        trade_log[column] = pd.to_datetime(trade_log[column])
                    else:
                        trade_log[column] = trade_log[column].astype(expected_dtype)
                except Exception as e:
                    raise ValueError(f"Error converting {column} to {expected_dtype}: {e}")
        
        return True
    
    @staticmethod
    def net_profit(trade_log:pd.DataFrame) -> float:
        """
        Calculate the net profit from a trade log DataFrame.

        This method calculates the net profit by summing up the 'pnl' (profit and loss)
        column of the provided DataFrame. It performs checks to ensure the DataFrame
        and required columns are present and valid before performing the calculation.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl'
                                  column which represents the profit and loss for each trade.

        Returns:
        - float: The total net profit, rounded to four decimal places. Returns 0 if the DataFrame is empty.
        """
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0
        
        # Perform calculation if checks pass
        return round(trade_log['pnl'].sum(), 4)
    
    @staticmethod
    def total_trades(trade_log:pd.DataFrame) -> int:
        """
        Calculate the total number of trades in the trade log DataFrame.

        This method returns the total number of trades by counting the rows in
        the provided DataFrame.

        Parameters:
        trade_log (pd.DataFrame): The trade log DataFrame.

        Returns:
        int: The total number of trades.
        """
        return len(trade_log)
    
    @staticmethod
    def total_winning_trades(trade_log:pd.DataFrame) -> int:
        """
        Calculate the total number of winning trades in the trade log DataFrame.

        This method returns the total number of winning trades by counting the rows
        where the 'pnl' column is greater than zero.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl'
                                  column which represents the profit and loss for each trade.

        Returns:
        - int: The total number of winning trades. Returns 0 if the DataFrame is empty.
        """
        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0
        
        # Perform calculation if checks pass
        return len(trade_log[trade_log['pnl'] > 0])
    
    @staticmethod
    def total_losing_trades(trade_log:pd.DataFrame) -> int:
        """
        Calculate the total number of losing trades in the trade log DataFrame.

        This method returns the total number of losing trades by counting the rows
        where the 'pnl' column is less than zero.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl'
                                  column which represents the profit and loss for each trade.

        Returns:
        - int: The total number of losing trades. Returns 0 if the DataFrame is empty.
        """
        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0
        
        # Perform calculation if checks pass
        return len(trade_log[trade_log['pnl'] < 0])
    
    @staticmethod
    def avg_win_return_rate(trade_log:pd.DataFrame) -> float:
        """
        Calculate the average return rate of winning trades in the trade log DataFrame.

        This method returns the average 'gain/loss' of winning trades (where 'pnl' > 0).
        The returned value is in decimal format, representing the average gain/loss.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl' and
                                  'gain/loss' columns.

        Returns:
        - float: The average return rate of winning trades, rounded to four decimal places.
               Returns 0 if there are no winning trades or if the DataFrame is empty.
        """
        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0
        else:
            winning_trades = round(trade_log[trade_log['pnl'] > 0], 4)
            return np.around(winning_trades['gain/loss'].mean(),decimals=4) if not winning_trades.empty else 0

    @staticmethod
    def avg_loss_return_rate(trade_log:pd.DataFrame) -> float:
        """
        Calculate the average return rate of losing trades in the trade log DataFrame.

        This method returns the average 'gain/loss' of losing trades (where 'pnl' < 0).
        The returned value is in decimal format, representing the average gain/loss.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl' and
                                  'gain/loss' columns.

        Returns:
        - float: The average return rate of losing trades, rounded to four decimal places.
               Returns 0 if there are no losing trades or if the DataFrame is empty.
        """
        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0
        else:
            losing_trades = round(trade_log[trade_log['pnl'] < 0],4)
            return np.around(losing_trades['gain/loss'].mean(),decimals=4) if not losing_trades.empty else 0
    
    @staticmethod
    def profitability_ratio(trade_log:pd.DataFrame) -> float:
        """
        Calculate the profitability ratio of the trades in the trade log DataFrame.

        This method returns the ratio of the number of winning trades to the total number
        of trades. The value is returned in decimal format.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl' column.

        Returns:
        - float: The profitability ratio, rounded to four decimal places. Returns 0.0 if there are
               no trades or if the DataFrame is empty.
        """
        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0.0
        else:
            total_winning_trades = PerformanceStatistics.total_winning_trades(trade_log)
            total_trades = len(trade_log)
            return round(total_winning_trades / total_trades, 4) if total_trades > 0 else 0.0
    
    @staticmethod
    def avg_trade_profit(trade_log:pd.DataFrame) -> float:
        """
        Calculate the average profit per trade in the trade log DataFrame.

        This method returns the average profit per trade by dividing the net profit
        by the total number of trades. The value is returned in dollars.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl' column.

        Returns:
        - float: The average profit per trade, rounded to four decimal places. Returns 0 if there
               are no trades or if the DataFrame is empty.
        """
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0

        net_profit = PerformanceStatistics.net_profit(trade_log)
        total_trades = len(trade_log)
        return round(net_profit / total_trades,4) if total_trades > 0 else 0
    
    @staticmethod
    def profit_factor(trade_log:pd.DataFrame) -> float:
        """
        Calculate the Profit Factor.

        This method calculates the profit factor by dividing the gross profits by the gross losses.
        Gross profits are the sum of profits from winning trades, and gross losses are the absolute
        sum of losses from losing trades.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl' column.

        Returns:
        - float: The profit factor, rounded to four decimal places. Returns 0 if there are no trades
               or if the DataFrame is empty.
        """
        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0

        gross_profits = trade_log[trade_log['pnl'] > 0]['pnl'].sum()
        gross_losses = abs(trade_log[trade_log['pnl'] < 0]['pnl'].sum())
        
        if gross_losses > 0:
            return round(gross_profits / gross_losses,4)
        return 0.0

    @staticmethod
    def profit_and_loss_ratio(trade_log:pd.DataFrame) -> float:
        """
        Calculate the ratio of average winning trade to average losing trade.

        This method returns the ratio of the average profit from winning trades to the average
        loss from losing trades. It measures the size of the average winning trade relative to
        the size of the average losing trade.

        Parameters:
        - trade_log (pd.DataFrame): The trade log DataFrame. It should contain the 'pnl' column.

        Returns:
        - float: The ratio of average winning trade to average losing trade, rounded to four decimal
               places. Returns 0 if there are no trades or if the DataFrame is empty.
        """

        # Check if the input is a pandas DataFrame
        if not isinstance(trade_log, pd.DataFrame):
            raise TypeError("trade_log must be a pandas DataFrame")
        
        # Check if the 'pnl' column exists in the DataFrame
        if 'pnl' not in trade_log.columns:
            raise ValueError("'pnl' column is missing in trade_log DataFrame")
        
        # Check for empty DataFrame
        if trade_log.empty:
            return 0

        # Calculate average win
        avg_win = trade_log[trade_log['pnl'] > 0]['pnl'].mean()
        avg_win = 0 if pd.isna(avg_win) else avg_win
        
        # Calculate average loss
        avg_loss = trade_log[trade_log['pnl'] < 0]['pnl'].mean()
        avg_loss = 0 if pd.isna(avg_loss) else avg_loss

        if avg_loss != 0:
            return round(abs(avg_win / avg_loss),4)
        
        return 0.0
