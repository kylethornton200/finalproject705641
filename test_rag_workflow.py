import pytest
import pandas as pd
from unittest.mock import MagicMock
import numpy as np
from workflow import preprocess_dfs, start_rag_workflow  # replace with actual file name

@pytest.fixture
def sample_dataframes():
    fake = pd.DataFrame({
        'title': ['Fake news 1', 'Fake news 2'],
        'text': ['This is fake content.', 'More fake content.']
    })
    true = pd.DataFrame({
        'title': ['True news 1', 'True news 2'],
        'text': ['This is real content.', 'More real content.']
    })
    return fake, true

def test_preprocess_dfs(sample_dataframes):
    fake, true = sample_dataframes
    result = preprocess_dfs(fake, true)

    # Check length
    assert len(result) == 4

    # Check both labels are present
    assert result["Label"].isin([True, False]).all()

    # Check columns
    assert "title" in result.columns or 0 in result.columns  # account for unnamed columns
    assert "text" in result.columns or 1 in result.columns

def test_start_rag_workflow_full(monkeypatch, sample_dataframes):
    # Mock vector store
    class MockVectorStore:
        def search(self, text, method, k):
            return [MagicMock(page_content="Example content", metadata={"source_url": "http://example.com"}) for _ in range(k)]

    mock_vector = MockVectorStore()
    fake, true = sample_dataframes
    df = preprocess_dfs(fake, true)

    # Mock the chat function from ollama
    monkeypatch.setattr("workflow.chat", lambda model, messages, options: {"message": {"content": "The claim is **supported** by evidence."}})
    
    results = start_rag_workflow(vector_store=mock_vector, df=df, full_output=True)
    
    assert len(results) > 0
    for item in results:
        assert isinstance(item[0], str)
        assert isinstance(item[1], (bool, np.bool_))