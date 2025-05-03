import shutil
from pathlib import Path

import numpy as np
import pytest
import requests
from langchain.schema import Document

from scrape_data import (EmbeddedClass, create_vector_store, embed_page,
                         fetch_report_content, get_report_content)


class MockResponse:
    """Minimal mock of requests.Response for fetch_report_content tests."""

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


@pytest.fixture()
def dummy_embedder(monkeypatch):
    """EmbeddedClass with encode() mocked to deterministic 4-dim vectors."""
    emb = EmbeddedClass()

    def fake_encode(texts, show_progress_bar=False):
        # return len(texts) × 4 matrix of incremental floats
        return np.arange(len(texts) * 4, dtype="float32").reshape(len(texts), 4)

    monkeypatch.setattr(emb.model, "encode", fake_encode)
    return emb


def test_fetch_report_content_success(monkeypatch):
    html = "<html><body><p>Hello CRS</p></body></html>"
    monkeypatch.setattr("requests.get", lambda url, timeout: MockResponse(200, html))
    text = fetch_report_content("http://example.com/report")
    assert "Hello CRS" in text


def test_fetch_report_content_no_content(monkeypatch):
    # response text with only whitespace
    monkeypatch.setattr(
        "requests.get", lambda url, timeout: MockResponse(200, "   \n ")
    )
    result = fetch_report_content("http://example.com/empty")
    assert result is None


def test_fetch_report_content_http_error(monkeypatch):
    monkeypatch.setattr("requests.get", lambda url, timeout: MockResponse(500, "error"))
    result = fetch_report_content("http://example.com/fail")
    assert result is None


def test_fetch_report_content_exception(monkeypatch):
    def raiser(url, timeout):
        raise requests.exceptions.Timeout("timeout")

    monkeypatch.setattr("requests.get", raiser)
    result = fetch_report_content("http://example.com/timeout")
    assert result is None


def test_get_report_content_skips_na_html(dummy_embedder):
    row = {"latestHTML": np.nan, "number": "R1"}
    text, url = get_report_content(row, dummy_embedder)
    assert text is None and url is None
    assert dummy_embedder.docs == []


def test_get_report_content_success(monkeypatch, dummy_embedder):
    # stub fetch_report_content and embed_page
    monkeypatch.setattr("scrape_data.fetch_report_content", lambda url: "PAGE TEXT")
    docs_out = [Document(page_content="chunk1", metadata={"source_url": "u"})]
    monkeypatch.setattr("scrape_data.embed_page", lambda url, txt, emb: docs_out)

    row = {"latestHTML": "path.html", "number": "R2"}
    text, url = get_report_content(row, dummy_embedder)
    assert text == "PAGE TEXT"
    assert url.endswith("path.html")
    # since embed_page returned docs_out, embedder.docs should equal that list
    assert dummy_embedder.docs == docs_out


def test_embed_page_returns_docs_and_vectors(dummy_embedder):
    url = "http://example.com/page"
    text = "A " * 5000  # will split into multiple chunks

    before = len(dummy_embedder.embeddings)
    docs = embed_page(url, text, dummy_embedder)
    after = len(dummy_embedder.embeddings)

    # embeddings grew by exactly the number of docs returned
    assert after == before + len(docs)

    # each returned doc is a Document with correct metadata
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.metadata["source_url"] == url for d in docs)


def test_create_vector_store(tmp_path: Path, dummy_embedder):
    # 1) Build 5 dummy docs
    docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(5)]

    # 2) Give the embedder 5 matching 4-dim vectors
    dummy_embedder.embeddings = np.arange(5 * 4, dtype="float32").reshape(5, 4).tolist()

    # 3) Create the store
    store_dir = tmp_path / "faiss_store"
    vstore = create_vector_store(docs, dummy_embedder, store_name=str(store_dir))

    # Index should report exactly 5 vectors
    assert hasattr(vstore.index, "ntotal")
    assert vstore.index.ntotal == len(docs)

    # Directory should have been created
    assert store_dir.exists()

    # 4) Reload with LangChain and verify count
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS as LCFAISS

    emb_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    lc_store = LCFAISS.load_local(
        str(store_dir), emb_fn, allow_dangerous_deserialization=True
    )
    assert lc_store.index.ntotal == len(docs)

    # Cleanup
    shutil.rmtree(store_dir)
