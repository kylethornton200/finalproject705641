# Final Project
Final Project

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

# Process reports
text, url = get_report_content(df.iloc[0])
```

## Requirements

Dependencies for data collection:
- pandas==2.2.1
- requests==2.31.0 
- beautifulsoup4==4.12.3
- lxml==4.9.3
- urllib3==2.2.0

## Installation

