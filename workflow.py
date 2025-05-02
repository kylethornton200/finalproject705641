import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from langchain.schema import Document
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from sklearn.utils import shuffle

def preprocess_dfs(fake:pd.DataFrame, true:pd.DataFrame) -> pd.DataFrame:
    fake["Label"] = [False]*len(fake)
    true["Label"] = [True]*len(true)
    df = pd.concat([fake, true]).reset_index(drop =True)
    df = shuffle(df)
    df.reset_index(inplace = True, drop = True)

def start_rag_workflow(vector_store: FAISS, df: pd.DataFrame, full_output: bool = True):
    if full_output:
        indeces = np.random.randint(0,len(df)-1, [20])
        for ind in indeces:
            title = df.iloc[ind, 0]
            contents = df.iloc[ind,1]
            labels = df.loc[ind, "Label"]

            system_prompt = f"""
            You are a fact checker.
            Use this context as ground truth.
            Do not use any other data besides the context given.
            Here is the context:
            {[d.page_content for d in D]}
            """
            user_prompt = f"""
            Determine whether this article contains misinformation: {contents}.
            Use specific quotes from the context to back up your claim.
            Here are the urls for each on of the context sources: {[d.metadata["source_url"] for d in D]}
            """
    else:
        return None
    return None