import os
import unittest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from unittest.mock import patch, mock_open, Mock, MagicMock

from quantAnalytics.report import ReportGenerator

def generate_random_data(rows=100):
    dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
    data = np.random.randn(rows)
    return dates, data

def simple_plot():
    plt.plot([1, 2, 3], [4, 5, 6])

def line_plot(x, y, title="", x_label="", y_label=""):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.report_path = "test_report.html"
        self.pdf_path = self.report_path.replace(".html", ".pdf")
        self.report = ReportGenerator(self.report_path)

    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists(self.report_path):
            os.remove(self.report_path)
        if os.path.exists(self.pdf_path):
            os.remove(self.pdf_path)

    # basic validation
    def test_add_title(self):
        # test
        self.report.add_title("Test Title")
        
        # validate
        self.assertIn("<h2>Test Title</h2>", self.report.html_content)
    
    def test_add_text(self):
        html_content = "<p>Test HTML Content</p>"
        # test 
        self.report.add_text(html_content)
        
        # validate
        self.assertIn(html_content, self.report.html_content)

    def test_add_list(self):
        summary_dict = {"key1": "value1", "key2": "value2"}
        # test 
        self.report.add_list(summary_dict)
        # validate
        self.assertIn("<li><strong>key1:</strong> value1</li>", self.report.html_content)
        self.assertIn("<li><strong>key2:</strong> value2</li>", self.report.html_content)

    def test_add_table(self):
        headers = ["Column1", "Column2"]
        rows = [["Row1Cell1", "Row1Cell2"], ["Row2Cell1", "Row2Cell2"]]
        # Test 
        self.report.add_table(headers, rows)

        # Validate
        self.assertIn("<th>Column1</th>", self.report.html_content)
        self.assertIn("<td>Row1Cell1</td>", self.report.html_content)

    def test_add_image(self):
        # Test
        self.report.add_image(simple_plot)
        # Validate
        self.assertIn('<img src="data:image/png;base64,', self.report.html_content)

    def test_add_dataframe(self):
        df = pd.DataFrame({'Column1': [1, 2], 'Column2': [3, 4]})
        # Test
        self.report.add_dataframe(df, "DataFrame Title")
        # Validate
        self.assertIn("<h2>DataFrame Title</h2>", self.report.html_content)
        self.assertIn("Column1", self.report.html_content)
        self.assertIn("1", self.report.html_content)

    def test_complete_report(self):
        # Setup
        mock_html_content = "<html><body>Report content</body></html>"
        self.report.html_content = mock_html_content 
        self.report._generate_html = Mock()
        self.report._generate_pdf = Mock()

        # test
        self.report.complete_report()
        
        # Validate HTML file write
        self.report._generate_html.assert_called_once()
        self.report._generate_pdf.assert_called_once()
    
    # type constraints
    def test_add_title_type_check(self):
        with self.assertRaises(TypeError):
            self.report.add_title(9999)

    def test_add_text_type_check(self):
        with self.assertRaises(TypeError):
            self.report.add_text(9999)

    def test_add_text_type_check(self):
        with self.assertRaises(TypeError):
            self.report.add_text(9999)

    def test_add_list_type_check(self):
        with self.assertRaises(TypeError):
            self.report.add_list("asdfghgfdsdfgfdsdf")

    def test_add_table_type_check(self):
        headers = ["Column1", "Column2"]
        rows = [["Row1Cell1", "Row1Cell2"], ["Row2Cell1", "Row2Cell2"]]

        with self.assertRaises(TypeError):
            self.report.add_table("headers", rows)

        with self.assertRaises(TypeError):
            self.report.add_table(headers, "rows")
    
        with self.assertRaises(TypeError):
            self.report.add_table(headers, rows, "0")

    def test_add_image_type_check(self):
        with self.assertRaises(TypeError):
            self.report.add_image("simple_plot")
    
        with self.assertRaises(TypeError):
            self.report.add_image(simple_plot, indent="1")

    def test_add_dataframe_type_check(self):
        df = pd.DataFrame({'Column1': [1, 2], 'Column2': [3, 4]})

        with self.assertRaises(TypeError):
            self.report.add_dataframe(1234, "DataFrame Title")

        with self.assertRaises(TypeError):
            self.report.add_dataframe(df, 1234)

    # value constraints
    def test_add_table_value_check(self):
        headers = ["Column1", "Column2"]
        rows = [["Row1Cell1", "Row1Cell2"], ["Row2Cell1"]]

        with self.assertRaises(ValueError):
            self.report.add_table(headers, rows)

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

    # integration
    def test_integration(self):
        # Set-up Report
        report_path = "isolated_test_report.html"
        # custom_css = "tests/test_styles.css"
        report = ReportGenerator(report_path)

        # Generate Random Data
        dates, data = generate_random_data()

        # Add Section and Plot
        report.add_title("Random Data Visualization")
        report.add_image(line_plot, 0, x=dates, y=data, title="Random Data Line Plot", x_label="Date", y_label="Value")

        # Add a Random Data Table
        random_df = pd.DataFrame({'Date': dates, 'Value': data})
        report.add_dataframe(random_df, "Random Data Table")

        # Add Another Section and Plot
        report.add_title("Another Random Data Visualization")
        more_dates, more_data = generate_random_data()
        report.add_image(line_plot, 0, x=more_dates, y=more_data, title="More Random Data Line Plot", x_label="Date", y_label="Value")

        # Complete and Generate Report
        report.complete_report()



if __name__ =="__main__":
    unittest.main()