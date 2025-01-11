import os
import unittest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unittest.mock import Mock, MagicMock
from quant_analytics.report.report import ReportBuilder, DivBuilder, Header
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
        self.report = ReportBuilder(self.html_name, self.output_directory)

    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists(self.output_directory):
            shutil.rmtree(self.output_directory)

    def test_add_header(self):
        # Test
        self.report.add_header("Header Test", Header.H1, "test")
        self.report.add_header("Header Test", Header.H2)

        # Validate
        report_content = self.report._content
        self.assertIn('<h1 class="test">Header Test</h1>', report_content)
        self.assertIn("<h2>Header Test</h2>", report_content)

    def test_add_paragraph(self):
        # Test
        self.report.add_paragraph("Hello this is a paragraph", "para")

        # Validate
        report_content = self.report._content
        self.assertIn(
            '<p class="para">Hello this is a paragraph</p>', report_content
        )

    def test_add_unorderdlist_dict(self):
        # Test
        summary_dict = {"key1": "value1", "key2": "value2"}
        self.report.add_unorderedlist_dict(summary_dict, True, "dict1")

        summary_dict = {"key1": "value1", "key2": "value2"}
        self.report.add_unorderedlist_dict(
            summary_dict, bold_key=False, css_class="dict2"
        )

        # Validate
        report_content = self.report._content
        self.assertIn(
            f"""<ul class="dict1">\n<li><strong>key1:</strong> value1</li>\n<li><strong>key2:</strong> value2</li>\n</ul>""",
            report_content,
        )
        self.assertIn(
            f"""<ul class="dict2">\n<li>key1: value1</li>\n<li>key2: value2</li>\n</ul>""",
            report_content,
        )

    def test_add_unorderedlist(self):
        # Test
        ordered_list = ["hello", "world"]
        self.report.add_unorderedlist(ordered_list, css_class="list1")

        # Validate
        report_content = self.report._content
        self.assertIn(
            f"""<ul class="list1">\n<li>hello</li>\n<li>world</li>\n</ul>""",
            report_content,
        )

    def test_add_orderedlist(self):
        # Test
        ordered_list = ["hello", "world"]
        self.report.add_orderedlist(ordered_list, css_class="list2")

        # Validate
        report_content = self.report._content
        self.assertIn(
            f"""<ol class="list2">\n<li>hello</li>\n<li>world</li>\n</ol>""",
            report_content,
        )

    def test_add_table(self):
        # Test
        df = pd.DataFrame({"Column1": [1, 2], "Column2": [3, 4]})
        self.report.add_table(df, "Title", Header.H4, "title_css", "table_css")

        # Validate
        report_content = self.report._content
        self.assertIn(
            f"""<h4 class=\'h4\'>Title</h4>\n<table class="table_css"><table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th>Column1</th>\n      <th>Column2</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>1</td>\n      <td>3</td>\n    </tr>\n    <tr>\n      <td>2</td>\n      <td>4</td>\n    </tr>\n  </tbody>\n</table>\n</table>""",
            report_content,
        )

    def test_add_image(self):
        create_plot_image("test_report/output_plot.png")

        # Test
        self.report.add_image("output_plot.png", css_class="image1")

        # Validate
        report_content = self.report._content
        self.assertIn(
            f"""<img src="output_plot.png" class="image1" alt="Plot Image"><br>""",
            report_content,
        )

    def test_complete_report(self):
        # Setup
        mock_html_content = "<html><body>Report content</body></html>"
        self.report._content = mock_html_content
        self.report._generate_html = Mock()
        self.report._generate_pdf = Mock()

        # test
        self.report.build()

        # Validate HTML file write
        self.report._generate_html.assert_called_once()
        self.report._generate_pdf.assert_called_once()

    def test_add_html_block(self):
        html = report_content = "<h2>Header Test</h2>"
        html += '<p class="para">Hello this is a paragraph</p>'

        # Test
        self.report.add_html_block(html)

        # Validate
        report_content = self.report._content
        self.assertIn(
            '\n<h2>Header Test</h2><p class="para">Hello this is a paragraph</p>',
            report_content,
        )

    def test_full_build(self):
        # Test
        ordered_list = ["hello", "world"]
        summary_dict = {"key1": "value1", "key2": "value2"}

        # fmt: off
        self.report.add_header("Header1", Header.H1) \
            .add_paragraph("Hello this is a paragraph", "para") \
            .add_unorderedlist(ordered_list, css_class="list1") \
            .add_unorderedlist_dict(summary_dict, True, "dict1") \
            .add_div(DivBuilder().add_header("DivHeader", Header.H2, "div_header").build()) \
            .build()
        # fmt: on

        # Validate
        if os.path.exists(
            self.output_directory
        ):  # Make sure you reference the correct file path
            # Open and read the generated file
            with open(self.report.file_path, "r") as f:
                content = f.read()

            # Optionally, validate content (simple check for specific elements)
            assert (
                "Header1" in content
            ), "Header is missing in the generated file."
            assert (
                "Hello this is a paragraph" in content
            ), "Paragraph is missing in the generated file."
            assert (
                '<ul class="list1">' in content
            ), "Unordered list is missing in the generated file."
            assert (
                "<strong>key1:</strong>" in content
            ), "Dictionary values are missing in the generated file."
            assert (
                '<h2 class="div_header">DivHeader</h2>' in content
            ), "Div missing from file."

    # error handling
    def test_complete_report_errors(self):
        # Setup
        self.report._generate_html = MagicMock(
            side_effect=Exception("HTML generation error")
        )
        self.report._generate_pdf = Mock()

        # Test
        with self.assertRaises(Exception) as context:
            self.report.build()

        # Validate
        self.assertTrue("Error generating report" in str(context.exception))
        self.report._generate_html.assert_called_once()
        self.report._generate_pdf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
