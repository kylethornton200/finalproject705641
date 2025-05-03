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
from rag_call import get_output
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse

def preprocess_dfs(fake:pd.DataFrame, true:pd.DataFrame) -> pd.DataFrame:
    fake["Label"] = [False]*len(fake)
    true["Label"] = [True]*len(true)
    df = pd.concat([fake, true]).reset_index(drop =True)
    df = shuffle(df)
    df.reset_index(inplace = True, drop = True)

def start_rag_workflow(vector_store: FAISS, df: pd.DataFrame, full_output: bool = True):
    if full_output:
        indeces = np.random.randint(0,len(df)-1, [20])
        final = []
        for ind in indeces:
            title = df.iloc[ind, 0]
            contents = df.iloc[ind,1]
            labels = df.loc[ind, "Label"]
            D = vector_store.search(title, "similarity", k = 2)
            E = vector_store.search(contents, "similarity", k = 2)
            system_prompt = f"""
            You are a fact checker.
            Use this context as ground truth.
            Do not use any other data besides the context given.
            Here is the context:
            {[e.page_content for e in E]}
            {[d.page_content for d in D]}
            """
            user_prompt = f"""
            Determine whether this query is article is true: Title: {title} \n Article:{contents}.
            If you don't have the information, give the best guess based on the context.
            """
            output = get_output({"system": system_prompt, "user": user_prompt})
            final.append((output, [d.metadata["source_url"] for d in D],[d.metadata["source_url"] for d in E], labels))
            time.sleep(1)
        return final
    else:
        test = []
        for item, row in df[:15:].iterrows():
            title = df.iloc[item, 0]
            contents = df.iloc[item,1]
            labels = df.loc[item, "Label"]
            title_docs = vector_store.search(title, "similarity", k = 3)
            conent_docs = vector_store.search(contents, "similarity", k = 3)

            system_prompt = f"""
            You are a fact checker.
            Use this context as ground truth.
            Do not use any other data besides the context given.
            Here is the context:
            {[doc.page_content for doc in title_docs]}
            {[doc.page_content for doc in conent_docs]}
            """
            user_prompt = f"Determine whether this query is article is fake or real: Title: {title} \n Article:{contents}. Output True if you think the article is real or output False if you think it is fake. Only ouput True or False do not explain"
            response: ChatResponse = chat(model='llama3.2', messages=[
            {
                "role": "system",
                "content": system_prompt,
                'role': 'user',
                'content': user_prompt,
            },
            ],
            options = {
                'temperature': .3
            })
            print(response['message']['content'], labels)
            test.append(response['message']['content'] == str(labels))
    return None