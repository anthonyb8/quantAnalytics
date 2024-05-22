
import base64
import pandas as pd
from io import BytesIO
from weasyprint import HTML
from textwrap import dedent
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
from typing import List, Dict, Any, Callable

class ReportGenerator:   
    """
    A class to generate dynamic HTML and PDF reports for visualizing and presenting data analysis results, especially suited for financial backtesting reports.

    The class supports adding HTML snippets, tables, images (from matplotlib plots), and pandas DataFrames into a structured HTML document. 
    It allows for custom CSS styling and handles basic layout and formatting through embedded CSS or an external stylesheet.

    Attributes:
    - file_path (str): The path where the final HTML file will be saved.
    - css_link (str): A string containing the link tag for an optional external CSS file.
    - base_style (str): A string containing default CSS styles if no external CSS is provided.
    - html_content (str): A string that builds up the HTML content to be written to file.

    Methods:
    - base_style(): Returns default CSS styles for the report.
    - add_html(html): Adds raw HTML to the report.
    - add_image(plot_func, indent, *args, **kwargs): Adds an image from a matplotlib plot to the report.
    - _get_plot_base64(plot_func, *args, **kwargs): Converts a matplotlib plot to a Base64 string.
    - add_table(headers, rows, indent): Adds a table to the report with specified headers and rows.
    - complete_report(): Finalizes the HTML content and writes it to the file specified by file_path.
    - add_section_title(title): Adds a section title to the report.
    - add_list(summary_dict): Adds a bullet-point list summarizing key points or data.
    - add_dataframe(df, title): Adds a pandas DataFrame as an HTML table to the report.
    """
    def __init__(self, file_path:str, custom_css_path:str=None):
        """
        Initializes the HTMLReportGenerator with a file path and optional custom CSS.

        Parameters:
        - file_path (str): The file path where the HTML report will be saved.
        - custom_css_path (str, optional): The path to a custom CSS file. If provided, default styles are omitted.
        """
        self.file_path = file_path
        
        if custom_css_path:
            self.css_link = f'<link rel="stylesheet" href="{custom_css_path}">'
        else:
            # Locate the internal styles.css file using pkg_resources
            css_path = resource_filename(__name__, 'styles.css')
            self.css_link = f'<link rel="stylesheet" href="{css_path}">'
            
        self.html_content = dedent(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    {self.css_link}
</head>
<body>
    """)
   
    def add_main_title(self, title:str) -> None:
        """
        Adds a section title to the report.

        Parameters:
        - title (str): The title text to be added as an HTML <h2> element.
        """
        if not isinstance(title, str):
            raise TypeError(f"Expected title to be a string, but got {type(title).__name__}")

        self.html_content += f"<h1>{title}</h1>\n"
   
    def add_section_title(self, title:str) -> None:
        """
        Adds a section title to the report.

        Parameters:
        - title (str): The title text to be added as an HTML <h2> element.
        """
        if not isinstance(title, str):
            raise TypeError(f"Expected title to be a string, but got {type(title).__name__}")

        self.html_content += f"<h3>{title}</h3>\n"
    
    def add_text(self, html:str) -> None:
        """
        Appends raw HTML to the report.

        Parameters:
        - html (str): A string of HTML to add to the report.
        """
        if not isinstance(html, str):
            raise TypeError(f"Expected text to be a string, but got {type(html).__name__}")
        
        self.html_content += html + "\n"

    def add_list(self, items:Dict) -> None:
        """
        Adds a bullet-point list to the report summarizing key points or data.

        Parameters:
        - summary_dict (Dict): A dictionary where each key-value pair represents a bullet point, with the key being the topic and the value being the detail.
        """
        if not isinstance(items, dict):
            raise TypeError(f"Expected items to be a string, but got {type(items).__name__}")
        
        self.html_content += "<ul>\n"
        for key, value in items.items():
            self.html_content += f"<li><strong>{key}:</strong> {value}</li>\n"
        self.html_content += "</ul>\n"

    def add_table(self, headers:List[Any], rows:List[List[Any]], indent:int=0) -> None:
        """
        Adds an HTML table to the report content with specified headers and rows.

        Parameters:
        - headers (List[Any]): A list of column headers for the table.
        - rows (List[List[Any]]): A list of rows, where each row is a list of cell values.
        - indent (int): The indentation level for the table HTML in the report for better readability.

        Side Effect:
        - Modifies the html_content attribute by appending the table HTML.
        """
        if not isinstance(headers, list):
            raise TypeError(f"Expected headers to be a list, but got {type(headers).__name__}")

        if not all(isinstance(row, list) for row in rows):
            raise TypeError("Expected rows to be a list of lists")

        if not isinstance(indent, int):
            raise TypeError(f"Expected indent to be an integer, but got {type(indent).__name__}")

        if any(len(row) != len(headers) for row in rows):
            raise ValueError("All rows must have the same length as headers")
        
        base_indent = "    " * indent
        next_indent = "    " * (indent + 1)  

        self.html_content += f"{base_indent}<table  border='1' class='dataframe'>\n"
        self.html_content += f"{next_indent}<thead>\n"
        self.html_content += f"{next_indent + base_indent}<tr>\n"

        for header in headers:
            self.html_content += f"{next_indent + (base_indent*2)}<th>{header}</th>\n"
        self.html_content += f"{next_indent + base_indent}</tr>\n{next_indent}</thead>\n{next_indent}<tbody>\n"

        for row in rows:
            self.html_content += f"{next_indent + base_indent}<tr>\n"
            for cell in row:
                self.html_content += f"{next_indent + (base_indent*2)}<td>{cell}</td>\n"
            self.html_content += f"{next_indent + base_indent}</tr>\n"
        self.html_content += f"{next_indent}</tbody>\n{base_indent}</table>\n"

    def add_image(self, plot_func:Callable, indent:int=0, *args:Any, **kwargs:Any) -> None:
        """
        Adds an image to the report by executing a plot function and capturing its output as a Base64 encoded image.

        Parameters:
        - plot_func (Callable): A function that generates a matplotlib plot.
        - indent (int): The indentation level for the HTML image tag in the report.
        - *args, **kwargs: Additional arguments to pass to the plot function.
        """
        if not callable(plot_func):
            raise TypeError(f"Expected plot_func to be callable, but got {type(plot_func).__name__}")

        if not isinstance(indent, int):
            raise TypeError(f"Expected indent to be an integer, but got {type(indent).__name__}")

        # Define the base indentation as a string of spaces
        base_indent = "    " * indent

        try:
            image_data = self._get_plot_base64(plot_func, *args, **kwargs)
            self.html_content += f'{base_indent}<img src="data:image/png;base64,{image_data}"><br>\n'
        except Exception as e:
            raise ValueError(f"Failed to generate plot: {e}")

    def _get_plot_base64(self, plot_func:Callable, *args:Any, **kwargs:Any) -> None:
        """
        Executes a plotting function to generate a plot and converts the plot into a Base64-encoded PNG image.

        Paremeters:
        - plot_func (Callable): A function that generates a matplotlib plot.
        - *args: Positional arguments to be passed to the plot function.
        - **kwargs: Keyword arguments to be passed to the plot function.

        Returns:
        - str: The Base64-encoded string of the plot image.
        """
        buf = BytesIO()
        try:
            plot_func(*args, **kwargs)
            plt.savefig(buf, format='png')
            plt.close()
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            raise ValueError(f"Failed to create plot: {e}")

    def add_dataframe(self, df:pd.DataFrame, title:str=None, index:bool=True) -> None:
        """
        Adds a pandas DataFrame as an HTML table to the report. Optionally includes a title for the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be converted to an HTML table and added to the report.
        - title (str, optional): The title for the DataFrame section. If provided, it precedes the table as an <h2> element.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected df to be a pandas DataFrame, but got {type(df).__name__}")

        if title and not isinstance(title, str):
            raise TypeError(f"Expected title to be a string, but got {type(title).__name__}")

        if title:
            self.html_content  += f"<h4>{title}</h4>\n"

        html_table = df.to_html(index=index, border=1)
        self.html_content += html_table + "\n"

    def complete_report(self) -> None:
        """
        Generates both the HTML and PDF versions of the report.

        Raises:
        - Exception: If an error occurs during the report generation process.
        """
        try:
            self._generate_html()
            self._generate_pdf()
        except Exception as e:
            raise Exception(f"Error generating report : {e}")

    def _generate_html(self) -> None:
        """
        Finalizes the HTML report content and writes it to the specified file path.

        Side Effect:
        - Writes the complete HTML content to a file, overwriting any existing file with the same name.
        """
        self.html_content += "</body>\n</html>"

        try:
            with open(self.file_path, "w") as file:
                file.write(self.html_content)
        except IOError as e:
            raise IOError(f"Failed to write HTML file: {e}")
        
    def _generate_pdf(self) -> None:
        """
        Converts the HTML report to a PDF file.

        Raises:
        - IOError: If an I/O error occurs during file writing.
        - RuntimeError: If an error occurs during PDF generation.
        """
        pdf_path = self.file_path.replace(".html", ".pdf")
        try:
            HTML(self.file_path).write_pdf(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to generate PDF: {e}")
    
    