# Political Misinformation Classifier
Misinformation has been a prevelant issue throughout social media as users can blatantly make up information about current events and spread this misinformation to others.
Our goal is to provide users a way to combat opinionated/biased/and false information using a Chatbot. This involves a RAG pipeline with a knowledge base consisting of nearly 20,000
congressional reports that include domestic and foreign information that are carefully chunked and embedded in order to get proper retrieval from our vector database. From here, users can input their statements or questions and we will retrieve relevant documents to their inquiry and determine from this context whether the given query was misleading or supported.

## Dataset
Congressional Research Service (CRS) Reports Data Collection [EveryCRSReport.com](https://www.everycrsreport.com/)
- 22,195 CRS reports available.
  
### Tools for scraping CRS reports text data.
```markdown
├── crs_reports.csv      # Dataset containing CRS report metadata
├── requirements.txt     # Required Python packages for scraping
└── scrape_data.py       # Scraping script for report content
```

### Data Collection Usage

```python
# Import scraping functions
from scrape_data import get_report_content

# Read the CSV file
df = pd.read_csv('crs_reports.csv', header=0, names=[
     'number', 'url', 'sha1', 'latestPubDate',
     'title', 'latestPDF', 'latestHTML'])

# Process reports
text, url = get_report_content(df.iloc[0])
```

## Requirements
pip install -r requirements.txt

## Installation

Data scraping and vector db creation: python scrape_data.py (Automatically will run and create the vector db locally)
RAG and LLM message generation (main api feature): 
- python workflow.py False (Indicates to use a dataset to create a confusion matrix and determine overall accuracy from simple True or False outputs from the LLM)
- python workflow.py True (Gives details to any responses and allows user inqueries into the LLM for misinformation classification)

## Team Members
- Kyle Thornton
- Mengyu Tu
- Arvind Kumar