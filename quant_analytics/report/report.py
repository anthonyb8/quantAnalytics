import pandas as pd
from weasyprint import HTML
from typing import List, Dict, Optional
import importlib.resources as resources
from enum import Enum
import os


class Header(Enum):
    H1 = "h1"
    H2 = "h2"
    H3 = "h3"
    H4 = "h4"
    H5 = "h5"
    H6 = "h6"


class ComponentBuilder:
    def __init__(self):
        self._content = ""

    def add_header(
        self, text: str, header: Header, css_class: Optional[str] = None
    ) -> "ComponentBuilder":
        """Add a html header block."""
        header_str = header.value

        if css_class:
            self._content += f"""\n<{header_str} class="{css_class}">{text}</{header_str}>"""
        else:
            self._content += f"""\n<{header_str}>{text}</{header_str}>"""

        return self

    def add_paragraph(
        self, text: str, css_class: Optional[str] = None
    ) -> "ComponentBuilder":
        """
        Adds a paragraph of text to the report with an optional CSS class.

        Parameters:
        - text (str): The text content for the paragraph.
        - css_class (Optional[str]): An optional CSS class to apply to the paragraph.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Expected text to be a string, but got {type(text).__name__}"
            )

        if css_class:
            # Add the class to the paragraph if provided
            self._content += f'<p class="{css_class}">{text}</p>\n'
        else:
            # Add a plain paragraph if no class is provided
            self._content += f"<p>{text}</p>\n"
        return self

    def add_unorderedlist_dict(
        self,
        items: Dict,
        bold_key: bool = True,
        css_class: Optional[str] = None,
    ) -> "ComponentBuilder":
        """
        Adds a bullet-point list to the report summarizing key-value pairs, with an optional CSS class.

        Parameters:
        - items (Dict): A dictionary where each key-value pair represents a bullet point.
        - bold_key (bool): If True, makes the key bold. Default is True.
        - css_class (Optional[str]): An optional CSS class to apply to the unordered list.
        """
        if not isinstance(items, dict):
            raise TypeError(
                f"Expected items to be a dictionary, but got {type(items).__name__}"
            )

        # Start the unordered list with a CSS class if provided
        if css_class:
            self._content += f'\n<ul class="{css_class}">'
        else:
            self._content += "\n<ul>"

        # Iterate over the dictionary items and create list items
        for key, value in items.items():
            if bold_key:
                self._content += f"\n<li><strong>{key}:</strong> {value}</li>"
            else:
                self._content += f"\n<li>{key}: {value}</li>"

        self._content += "\n</ul>"

        return self

    def add_unorderedlist(
        self, items: list, css_class: Optional[str] = None
    ) -> "ComponentBuilder":
        """
        Adds a simple bullet-point list to the report, with an optional CSS class.

        Parameters:
        - items (list): A list of strings to be added as bullet points.
        - css_class (Optional[str]): An optional CSS class to apply to the unordered list.
        """
        if not isinstance(items, list):
            raise TypeError(
                f"Expected items to be a list, but got {type(items).__name__}"
            )

        # Start the unordered list with a CSS class if provided
        if css_class:
            self._content += f'\n<ul class="{css_class}">'
        else:
            self._content += "\n<ul>"

        # Iterate over the list items and create list items
        for item in items:
            self._content += f"\n<li>{item}</li>"

        self._content += "\n</ul>"
        return self

    def add_orderedlist(
        self, items: List, css_class: Optional[str] = None
    ) -> "ComponentBuilder":
        if not isinstance(items, list):
            raise TypeError(
                f"Expected items to be a list, but got {type(items).__name__}"
            )

        if css_class:
            # Add the class to the unordered list if provided
            self._content += f'\n<ol class="{css_class}">'
        else:
            # Add a plain unordered list if no class is provided
            self._content += "\n<ol>"

        # Iterate over the dictionary items and create list items
        for value in items:
            self._content += f"\n<li>{value}</li>"

        self._content += "\n</ol>"
        return self

    def add_table(
        self,
        df: pd.DataFrame,
        header: str = "",
        header_size: Header = Header.H4,
        header_css_class: Optional[str] = None,
        table_css_class: Optional[str] = None,
        index: bool = False,
    ) -> "ComponentBuilder":
        """
        Adds a pandas DataFrame as an HTML table to the report. Optionally includes a title for the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be converted to an HTML table and added to the report.
        - title (str, optional): The title for the DataFrame section. If provided, it precedes the table as an <h2> element.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected df to be a pandas DataFrame, but got {type(df).__name__}"
            )

        if header and not isinstance(header, str):
            raise TypeError(
                f"Expected header to be a string, but got {type(header).__name__}"
            )

        if header:
            header_str = header_size.value
            if header_css_class:
                self._content += f"\n<{header_str} class='{header_str}'>{header}</{header_str}>"
            else:
                self._content += f"\n<{header_str} class='{header_str}'>{header}</{header_str}>"

        # Add the table with a CSS class if provided
        if table_css_class:
            self._content += f'\n<table class="{table_css_class}">'
        else:
            self._content += "\n<table>"

        self._content += df.to_html(index=index, border=1)
        self._content += "\n</table>"
        return self

    def add_image(
        self, image_filename: str, css_class: Optional[str] = None
    ) -> "ComponentBuilder":
        """
        Adds an image to the report with an optional CSS class.
        """
        if not isinstance(image_filename, str):
            raise TypeError(
                f"Expected image_filename to be a string, but got {type(image_filename).__name__}"
            )

        # Add the <img> tag with optional CSS class
        if css_class:
            self._content += f'\n<img src="{image_filename}" class="{css_class}" alt="Plot Image"><br>'
        else:
            self._content += (
                f'\n<img src="{image_filename}" alt="Plot Image"><br>'
            )
        return self

    def add_html_block(self, html: str) -> "ComponentBuilder":
        """
        Adds a prebuilt HTML block to the report or div. Optionally wraps it in a div.

        Parameters:
        - html (str): The raw HTML block to be added.
        """
        if not isinstance(html, str):
            raise TypeError(
                f"Expected prebuilt_html to be a string, but got {type(html).__name__}"
            )

        self._content += f"\n{html}"
        return self

    def get_content(self) -> str:
        """Return the generated content."""
        return self._content


class DivBuilder(ComponentBuilder):
    def __init__(self, css_class: str = ""):
        super().__init__()
        self.div_class = css_class

    def set_class(self, div_class: str) -> "DivBuilder":
        """Set the CSS class for the div."""
        self.div_class = div_class
        return self

    def build(self) -> str:
        """Construct the div and return the HTML."""

        div_class_str = f' class="{self.div_class}"' if self.div_class else ""

        wrapped_content = f"""\n<div{div_class_str}> \n{self.get_content().strip()} \n</div>"""
        return wrapped_content


class ReportBuilder(ComponentBuilder):
    def __init__(
        self,
        file_name: str,
        output_directory: str = "report",
        css_path: str = "",
    ):
        super().__init__()
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.output_directory = output_directory
        self.file_path = os.path.join(output_directory, file_name)

        if not css_path:
            with resources.as_file(
                resources.files("quant_analytics") / "styles.css"
            ) as css_file:
                self.css_path = str(css_file)
        else:
            self.css_path = css_path

        self._content = f"""<!DOCTYPE html>\n<html>\n<head>\n    <title>Report</title>\n    <link rel="stylesheet" href="{self.css_path}">\n</head>\n<body>"""

    def add_div(self, div_content: str) -> "ReportBuilder":
        """Add a div block to the report."""
        self._content += div_content
        return self

    def build(self) -> None:
        """Finalize and return the report content."""
        self._content += "\n</body>\n</html>"
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
        try:
            with open(self.file_path, "w") as file:
                file.write(self._content)
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
