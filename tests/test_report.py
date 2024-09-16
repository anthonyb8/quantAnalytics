import os
import unittest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unittest.mock import Mock, MagicMock
from quantAnalytics.report import ReportGenerator, Header
import shutil


# Helper functions
def generate_random_data(rows=100):
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="D")
    data = np.random.randn(rows)
    return dates, data


def simple_plot():
    plt.plot([1, 2, 3], [4, 5, 6])


def line_plot(x, y, title="", x_label="", y_label=""):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()


def create_plot_image(image_path: str) -> None:
    """
    Generates a simple plot and saves it to the specified path.

    Parameters:
    - image_path (str): The full path where the image will be saved (e.g., 'output_plot.png').
    """
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], label="Sample Plot")
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Sample Plot")
    plt.legend()

    # Save the plot as an image
    plt.savefig(image_path)
    plt.close()


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.output_directory = "test_report"
        self.html_name = "test_report.html"
        self.pdf_name = self.html_name.replace(".html", ".pdf")
        self.report = ReportGenerator(self.html_name, self.output_directory)

    # @classmethod
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists(self.output_directory):
            shutil.rmtree(self.output_directory)
        # html = os.path.join(cls.output_directory, cls.html_name)
        # if os.path.exists(cls.output_directory):
        #     os.rmdir(cls.output_directory)
        #
        # pdf = os.path.join(self.output_directory, self.pdf_name)
        # if os.path.exists(pdf):
        #     os.remove(pdf)

    def test_add_header(self):
        # Test
        self.report.add_header("Header Test", Header.H1, "test")
        self.report.add_header("Header Test", Header.H2)

        # Validate
        self.assertIn('<h1 class="test">Header Test</h1>', self.report._html_content)
        self.assertIn("<h2>Header Test</h2>", self.report._html_content)

    def test_add_paragraph(self):
        # Test
        self.report.add_paragraph("Hello this is a paragraph", "para")

        # Validate
        self.assertIn('<p class="para">Hello this is a paragraph</p>', self.report._html_content)

    def test_add_unorderdlist_dict(self):
        # Test
        summary_dict = {"key1": "value1", "key2": "value2"}
        self.report.add_unorderedlist_dict(summary_dict, True, "dict1")

        summary_dict = {"key1": "value1", "key2": "value2"}
        self.report.add_unorderedlist_dict(summary_dict, bold_key=False, css_class="dict2")

        # Validate
        self.assertIn(
            f"""<ul class="dict1">\n<li><strong>key1:</strong> value1</li>\n<li><strong>key2:</strong> value2</li>\n</ul>""",
            self.report._html_content,
        )
        self.assertIn(
            f"""<ul class="dict2">\n<li>key1: value1</li>\n<li>key2: value2</li>\n</ul>""",
            self.report._html_content,
        )

    def test_add_unorderedlist(self):
        # Test
        ordered_list = ["hello", "world"]
        self.report.add_unorderedlist(ordered_list, css_class="list1")

        # Validate
        self.assertIn(
            f"""<ul class="list1">\n<li>hello</li>\n<li>world</li>\n</ul>""",
            self.report._html_content,
        )

    def test_add_orderedlist(self):
        # Test
        ordered_list = ["hello", "world"]
        self.report.add_orderedlist(ordered_list, css_class="list2")

        # Validate
        self.assertIn(
            f"""<ol class="list2">\n<li>hello</li>\n<li>world</li>\n</ol>""",
            self.report._html_content,
        )

    def test_add_table(self):
        # Test
        df = pd.DataFrame({"Column1": [1, 2], "Column2": [3, 4]})
        self.report.add_table(df, "Title", Header.H4, "title_css", "table_css")

        # Validate
        self.assertIn(
            f"""<h4 class=\'h4\'>Title</h4>\n<table class="table_css"><table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th>Column1</th>\n      <th>Column2</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>1</td>\n      <td>3</td>\n    </tr>\n    <tr>\n      <td>2</td>\n      <td>4</td>\n    </tr>\n  </tbody>\n</table>\n</table>""",
            self.report._html_content,
        )

    def test_add_image(self):
        create_plot_image("test_report/output_plot.png")

        # Test
        self.report.add_image("output_plot.png", css_class="image1")

        # Validate
        self.assertIn(f"""<img src="output_plot.png" class="image1" alt="Plot Image"><br>""", self.report._html_content)

    def test_complete_report(self):
        # Setup
        mock_html_content = "<html><body>Report content</body></html>"
        self.report._html_content = mock_html_content
        self.report._generate_html = Mock()
        self.report._generate_pdf = Mock()

        # test
        self.report.complete_report()

        # Validate HTML file write
        self.report._generate_html.assert_called_once()
        self.report._generate_pdf.assert_called_once()

    # error handling
    def test_complete_report_errors(self):
        # Setup
        self.report._generate_html = MagicMock(side_effect=Exception("HTML generation error"))
        self.report._generate_pdf = Mock()

        # Test
        with self.assertRaises(Exception) as context:
            self.report.complete_report()

        # Validate
        self.assertTrue("Error generating report" in str(context.exception))
        self.report._generate_html.assert_called_once()
        self.report._generate_pdf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
