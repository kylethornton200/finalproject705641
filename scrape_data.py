import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

from sentence_transformers import SentenceTransformer, util
from langchain.schema import Document
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

milestones = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000]
max_reports = 30000

class EmbeddedClass:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = []
        self.docs = []


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

def fetch_report_content(url):
    """
    Fetch and extract text content from a single CRS report webpage.

    Args:
        url (str): Complete URL to the report's HTML page

    Returns:
        str: Extracted text content from the report with preserved formatting
             and paragraph breaks
    """
    try:
        # Send a request to the webpage
        response = requests.get(url, timeout=10)

        # Check if request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all text from the page
            text = soup.get_text()

            # Skip pages with no meaningful content 
            if text and len(text.strip()) > 0:
                return text
            else:
                print(f"Skipping {url}: No content found")
                return None
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        # catch connection errors, timeouts, and other request-related exceptions
        print(f"Request error: {e}")
        return None


def get_report_content(row, embedder):
    """
        Process a single report row and retrieve its content.
        Args:
            row (pandas.Series): A row from the CRS reports dataframe.
            e.g. df.iloc[0]
        Returns:
            - full_text: The complete text content of the report
            - url: The full URL used to fetch the report
    """
    # Check if HTML link exists
    if pd.isna(row['latestHTML']):
        print(f"Skipping row: No HTML link available for {row['number']}")
        return None, None
        
    # Construct full URL
    base_url = "https://www.everycrsreport.com/"
    html_url = base_url + str(row['latestHTML'])

    # Fetch the content
    full_text = fetch_report_content(html_url)
    #print("\nFirst 100 characters of the report:")
    #print(full_text[:100] + "...")
    docs = embed_page(html_url, full_text, embedder)
    embedder.docs.extend(docs)
    return full_text, html_url

def get_all_reports(df, embedder):
    for i, row in enumerate(df.itertuples(index=False)):

        if max_reports is not None and i >= max_reports:
            print(f"[INFO] Early stop reached at {i} rows.")
            break

        full_text, html_url = get_report_content(row._asdict(), embedder)

        if not full_text:
            print(f"[WARN] Row {i}: failed to fetch report (url={html_url})")

        if i in milestones:
            print(f"Reached milestone: collected {i} reports so far")
            


def embed_page(url, raw_text, embedder: EmbeddedClass):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    vecs   = embedder.model.encode(chunks, show_progress_bar=False)
    embedder.embeddings.extend(vecs)

    docs = [Document(page_content=chunk, metadata={"source_url": url}) for chunk in chunks]
    return docs

def create_vector_store(docs: list[Document], embedder: EmbeddedClass, store_name="CRS_Reports"):
    embedding_matrix = np.asarray(embedder.embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    docstore            = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
    index_to_docstore   = {i: str(i) for i in range(len(docs))}

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vstore = FAISS(
        embedding_function   = embeddings,
        index                = index,
        docstore             = docstore,
        index_to_docstore_id = index_to_docstore
    )
    vstore.save_local(store_name)
    return vstore

# Example usage:
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('crs_reports.csv', header=0, names=[
        'number', 'url', 'sha1', 'latestPubDate',
        'title', 'latestPDF', 'latestHTML'
    ])

    embedder = EmbeddedClass()
    print("="*12+"Starting scrape of crs reports" + "="*12)
    get_all_reports(df, embedder)

    print("="*12+"Creating vector store for embedding indexes" + "="*12)
    v_store = create_vector_store(embedder.docs, embedder)

    if v_store:
        print(f"Successfully created vector db CRS_Reports")
