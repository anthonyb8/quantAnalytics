import pandas as pd
from quantAnalytics.statistics.results import Result


class RegressionResult(Result):
    def __init__(self, data: dict):
        super().__init__("Regression Analysis", "", data)
        self.footer = "** R-squared should be above the threshold and p-values should be below the threshold for model validity."

    def _to_dataframe(self) -> pd.DataFrame:
        # Flatten results
        flattened_data = {}
        for section, metrics in self.data.items():
            for metric, values in metrics.items():
                flattened_data[f"{metric}"] = values

        # Convert the flattened dictionary to a DataFrame
        df = pd.DataFrame(flattened_data).T
        df = df.reset_index()
        df.columns = ["Field", "Value", "Significant"]
        return df
