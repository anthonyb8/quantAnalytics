import pandas as pd
from quant_analytics.result import Result


class ADFResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("ADF Test", ts_name, data)
        self.footer = "** IF p-value < 0.05 and/or statistic < statistic @ confidence interval, then REJECT the Null that the time series posses a unit root (non-stationary)."

    def _to_dataframe(self) -> pd.DataFrame:
        # Convert ADF results to DataFrame
        adf_data = []
        # for ticker, values in self.data.items():
        # print(f"Ticker: {ticker}, Values: {values}, Type: {type(values)}")
        # print(ticker, values)
        row = {
            "Name": self.timeseries_name,
            "Lags": self.data["Lags"],
            "ADF Statistic": self.data["ADF Statistic"],
            "p-value": round(self.data["p-value"], 6),
        }
        row.update(
            {
                f"Critical Value ({key})": val
                for key, val in self.data["Critical Values"].items()
            }
        )
        row.update({"Stationarity": self.data["Stationarity"]})
        adf_data.append(row)

        return pd.DataFrame(adf_data)


class KPSSResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("KPSS Test", ts_name, data)
        self.footer = "\n** IF KPSS statistic > statistic @ confidence interval, then reject the NUll that time-series is stationary.\n"

    def _to_dataframe(self) -> pd.DataFrame:
        data = []

        row = {
            "Name": self.timeseries_name,
            "KPSS Statistic": self.data["KPSS Statistic"],
            "p-value": round(self.data["p-value"], 6),
        }
        row.update(
            {
                f"Critical Value ({key})": val
                for key, val in self.data["Critical Values"].items()
            }
        )
        row.update({"Stationarity": self.data["Stationarity"]})
        data.append(row)

        return pd.DataFrame(data)


class PPResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("Phillips Perron Test", ts_name, data)
        self.footer = "\n** IF p-value < 0.05, then REJECT the Null Hypothesis of a unit root (non-stationary time series)."

    def _to_dataframe(self) -> pd.DataFrame:
        # Convert PP results to DataFrame
        data = []
        # for ticker, values in self.data.items():
        row = {
            "Name": self.timeseries_name,
            "PP Statistic": self.data["PP Statistic"],
            "p-value": round(self.data["p-value"], 6),
        }
        row.update(
            {
                f"Critical Value ({key})": val
                for key, val in self.data["Critical Values"].items()
            }
        )
        row.update({"Stationarity": self.data["Stationarity"]})
        data.append(row)

        return pd.DataFrame(data)


class JohansenResult(Result):
    def __init__(self, data: dict, num_cointegrations: int, k_ar_diff: int):
        super().__init__("Johansen Test", "", data)
        self.header = f"Johansen Cointegration Test : (Ideal lag = {k_ar_diff})\n Number of cointerated realtionships : {num_cointegrations}"
        self.footer = "** IF Trace Statistic > Critical Value AND Max Eigenvalue > Critical Value then Reject Null of at most r cointegrating relationships.(r=0 in first test)"
        self.num_cointegrations = num_cointegrations

    def _to_dataframe(self) -> pd.DataFrame:
        # Creating DataFrame from the results
        johansen_df = pd.DataFrame(
            {
                "Hypothesis": [
                    f"H{i}" for i in range(len(self.data["Eigenvalues"]))
                ],
                "Eigenvalue": self.data["Eigenvalues"],
                "Trace Statistic": self.data["Trace Statistics"],
                "Critical Value (Trace)": self.data[
                    "Critical Values for Trace Statistic"
                ],
                "Max Eigenvalue Statistic": self.data[
                    "Max Eigenvalue Statistics"
                ],
                "Critical Value (Max Eigenvalue)": self.data[
                    "Critical Values for Max Eigenvalue Statistic"
                ],
            }
        )

        # Add decision columns based on comparisons
        johansen_df["Decision (Trace)"] = johansen_df.apply(
            lambda row: (
                "Reject"
                if row["Trace Statistic"] > row["Critical Value (Trace)"]
                else "Fail to Reject"
            ),
            axis=1,
        )
        johansen_df["Decision (Max Eigenvalue)"] = johansen_df.apply(
            lambda row: (
                "Reject"
                if row["Max Eigenvalue Statistic"]
                > row["Critical Value (Max Eigenvalue)"]
                else "Fail to Reject"
            ),
            axis=1,
        )

        return johansen_df


class DurbinWatsonResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("Durbin Watson Test", ts_name, data)
        self.footer = "** If the Durbin-Watson statistic is significantly different from 2 (either much less than 2 or much greater than 2), it suggests the presence of autocorrelation in the residuals.\n"

    def _to_dataframe(self) -> pd.DataFrame:
        # Creating DataFrame from the results
        data = []
        for ticker, values in self.data.items():
            row = {
                "Series": ticker,
                "Durbin-Watson Statistic": values["Durbin-Watson Statistic"],
                "Autocorrelation": values["Autocorrelation"],
            }
            data.append(row)

        return pd.DataFrame(data)


class LjungBoxResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("Ljung-Box Test", ts_name, data)
        self.footer = "** IF p-value < 0.05, then REJECT the Null Hypothesis of no autocorrelation (i.e., autocorrelation is present).\n"

    def _to_dataframe(self) -> pd.DataFrame:
        data = []
        # for ticker, values in self.data.items():
        row = {
            "Name": self.timeseries_name,
            "Test Statistic": self.data["test_statistic"][0],
            "p-value": round(self.data["p_value"][0], 6),
        }
        row.update(
            {
                "Autocorrelation": (
                    "Absent"
                    if not self.data["significance"][0]  # == False
                    else "Present"
                )
            }
        )
        data.append(row)

        return pd.DataFrame(data)


class ShapiroWilkResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("Shapiro Wilk Test", ts_name, data)
        self.footer = "** If p-value < 0.05, then REJECT the Null Hypothesis of normality (i.e., data is not normally distributed).\n"

    def _to_dataframe(self) -> pd.DataFrame:
        # Convert Shapiro-Wilk results to DataFrame
        data = []
        # for ticker, values in self.data.items():
        row = {
            "Name": self.timeseries_name,
            "Shapiro-Wilk Statistic": self.data["Statistic"],
            "p-value": self.data["p-value"],
            "Normality": self.data["Normality"],
        }
        data.append(row)

        return pd.DataFrame(data)


class BreuschPaganResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("Breusch Pagan Test", ts_name, data)
        self.footer = "** IF p-value < 0.05, then REJECT the Null Hypothesis of homoscedasticity (constant variance) in favor of heteroscedasticity (varying variance).\n"

    def _to_dataframe(self) -> pd.DataFrame:
        # Convert Breusch-Pagan results to DataFrame
        data = []
        # for ticker, values in self.data.items():
        row = {
            "Name": self.timeseries_name,
            "Breusch-Pagan Test Statistic": self.data[
                "Breusch-Pagan Test Statistic"
            ],
            "p-value": round(self.data["p-value"], 6),
            "Heteroscedasticity": self.data["Heteroscedasticity"],
        }
        data.append(row)

        return pd.DataFrame(data)


class WhiteResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("White Test", ts_name, data)
        self.footer = "** IF p-value < 0.05, then REJECT the Null Hypothesis of homoscedasticity (constant variance) in favor of heteroscedasticity (varying variance).\n"

    def _to_dataframe(self) -> pd.DataFrame:
        # Convert White test results to DataFrame
        data = []
        # for ticker, values in self.data.items():
        row = {
            "Name": self.timeseries_name,
            "White Test Statistic": self.data["White Test Statistic"],
            "p-value": round(self.data["p-value"], 6),
            "Heteroscedasticity": self.data["Heteroscedasticity"],
        }
        data.append(row)

        return pd.DataFrame(data)


class GrangerCausalityResult(Result):
    def __init__(self, ts_name: str, data: dict):
        super().__init__("Granger Causality Test", ts_name, data)
        self.footer = "\n** IF p-value < significance level then REJECT the NUll and conclude that the lagged values of one time series can provide useful information for predicting the other time series."

    def _to_dataframe(self) -> pd.DataFrame:
        # Creating DataFrame from the results
        df = pd.DataFrame(
            [
                {
                    "Variable Pair": f"{pair[0]} -> {pair[1]}",
                    "Min P-Value": details["Min P-Value"],
                    "Granger Causality": (
                        "Yes"
                        if details["Granger Causality"] == "Causality"
                        else "No"
                    ),
                    "Significance Level": details["Significance Level"],
                }
                for pair, details in self.data.items()
            ]
        )
        return df
