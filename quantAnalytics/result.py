from abc import ABC, abstractmethod


class Result(ABC):
    def __init__(self, test_name: str, ts_name: str, data: dict):
        self.test_name = test_name
        self.timeseries_name = ts_name
        self.data = data
        self.header = f"{test_name} Results"
        self.footer = ""

    @abstractmethod
    def _to_dataframe(self) -> str:
        """Abstract method for converts data to dataframe."""
        pass

    def to_html(self, index=False) -> str:
        df = self._to_dataframe()

        # Convert DataFrame to HTML table and add explanation
        html_table = df.to_html(index=index, border=1)

        html_title = f"<h4>{self.header}</h4>\n"
        html_footer = (
            f"<p class='{self.test_name}_footnote'>{self.footer}</p>\n"
        )
        html_output = f"<div class='{self.test_name}_test'>\n{html_title}{html_table}\n{html_footer}</div>"

        return html_output

    def __repr__(self) -> str:
        df = self._to_dataframe()

        output = f"\n{self.header}"
        output += f"\n{'=' * len(self.header)}"
        output += f"\n{df.to_string(index=False)}"
        output += f"\n{self.footer}"
        return output
