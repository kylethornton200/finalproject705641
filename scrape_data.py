import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os


def scrape_crs_reports():
    """
        Download and parse the complete CRS reports CSV file from EveryCRSReport.com.

        This function fetches the master CSV file containing metadata for all Congressional
        Research Service (CRS) reports available on EveryCRSReport.com.

        Returns:
            pandas.DataFrame: A dataframe containing report metadata with columns:
                - number: Report ID
                - url: JSON metadata URL
                - sha1: Report hash
                - latestPubDate: Publication date
                - title: Report title
                - latestPDF: PDF file path
                - latestHTML: HTML file path
    """
    # Direct CSV download
    df = pd.read_csv('https://www.everycrsreport.com/reports.csv')
    print("Successfully downloaded CSV file")
    return df

def fetch_report_content(url, report_id):
    """
    Fetch and extract text content from a single CRS report webpage.

    Args:
        url (str): Complete URL to the report's HTML page
        report_id (str): Unique identifier for the report (e.g., 'IN12522')

    Returns:
        str: Extracted text content from the report with preserved formatting
             and paragraph breaks
    """
    # Add polite delay
    time.sleep(1)

    print(f"Fetching report: {report_id}")
    response = requests.get(url)
    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get content directly from paragraphs and headers
    content = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    full_text = '\n\n'.join([elem.get_text(strip=True) for elem in content])
    #print("\nFirst 300 characters of the report:")
    #print(full_text[:300] + "...")
    return full_text


def get_report_content(row):
    """
        Process a single report row and retrieve its content.
        Args:
            row (pandas.Series): A row from the CRS reports dataframe.
            e.g. df.iloc[0]
        Returns:
            - full_text: The complete text content of the report
            - url: The full URL used to fetch the report
    """
    # Construct full URL
    base_url = "https://www.everycrsreport.com/"
    html_url = base_url + row['latestHTML']

    # Fetch the content
    full_text = fetch_report_content(html_url, row['number'])
    return full_text, html_url

# Example usage:
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('crs_reports.csv', header=0, names=[
        'number', 'url', 'sha1', 'latestPubDate',
        'title', 'latestPDF', 'latestHTML'
    ])

    # Process first report as example
    text, url = get_report_content(df.iloc[0])
    if text:
        print(f"Successfully processed report from: {url}")