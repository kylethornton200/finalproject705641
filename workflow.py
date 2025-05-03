import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from sklearn.utils import shuffle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ollama import chat
from ollama import ChatResponse
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt
import argparse

def preprocess_dfs(fake:pd.DataFrame, true:pd.DataFrame) -> pd.DataFrame:
    fake["Label"] = [False]*len(fake)
    true["Label"] = [True]*len(true)
    df = pd.concat([fake, true]).reset_index(drop =True)
    df = shuffle(df)
    df.reset_index(inplace = True, drop = True)
    return df

def start_rag_workflow(vector_store: FAISS, df: pd.DataFrame, full_output: bool = True):
    """
    Starts the rag workflow based upon if we want to test against our data set.

    Args:
        url (str): Complete URL to the report's HTML page

    Returns:
        str: Extracted text content from the report with preserved formatting
             and paragraph breaks
    """
    if full_output:
        indeces = np.random.randint(0,len(df)-1, [20])
        final = []
        for ind in indeces:
            title = df.iloc[ind, 0]
            contents = df.iloc[ind,1]
            labels = df.loc[ind, "Label"]
            title_docs = vector_store.search(title, "similarity", k = 5)
            content_docs = vector_store.search(contents, "mmr", k = 5)

            system_prompt = f"""You are a concise fact-checker specialising in political misinformation.
                    
                    You will receive:
                    • A **user claim**.
                    • A **context block** (peer-reviewed or otherwise trustworthy sources).
                    
                    Your job
                    --------
                    1. Scan the context for statements that *directly* support or contradict the user’s claim.  
                    2. Reply in 2-3 short sentences:
                    
                    – Start with a plain-English verdict (e.g., “Based on the sources above, this claim appears **incorrect**.”  
                        or “The claim is **supported** by…”)  
                    
                    – Briefly quote or paraphrase the decisive evidence in quotation marks, followed by a parenthetical
                        hint such as “(Source: Article A)” if metadata is present.  
                    
                    – If nothing in the context settles the matter, say  
                        “I don’t have enough information to verify this.” and stop.
                    
                    Rules
                    -----
                    * **Use ONLY the provided context**—no outside facts or speculation.  
                    * Do **not** analyse writing style; focus on factual alignment.
                    Context: {[doc.page_content for doc in title_docs + content_docs]}
                    URLS: {[doc.metadata["source_url"] for doc in title_docs + content_docs]}
                    """
            response: ChatResponse = chat(model='llama3.2', messages=[
            {
                "role": "system",
                "content": system_prompt
            },{
                'role': 'user',
            'content': title + contents,
            }
            ],
            options = {
                'temperature': .3
            })
            final.append((response['message']['content'], labels))
        return final
    else:
        test = []
        for item, row in df[:100 :].iterrows():
            title = df.iloc[item, 0]
            contents = df.iloc[item,1]
            labels = df.loc[item, "Label"]
            title_docs = vector_store.search(title, "similarity", k =5)
            conent_docs = vector_store.search(contents, "mmr", k = 5)

            system_prompt = f"""
            You are a fact checker.
            Use this context as ground truth.
            Do not use any other data besides the context given.
            Do **not** analyse writing style; focus on factual alignment.
            Here is the context:
            {[doc.page_content for doc in title_docs]}
            {[doc.page_content for doc in conent_docs]}
            """
            user_prompt = f"Determine whether this article is fake or real: Title: {title} \n Article:{contents}. Output True if you think the article is real or output False if you think it is fake. Only ouput True or False do not explain"
            response: ChatResponse = chat(model='llama3.2', messages=[
            {
                "role": "system",
                "content": system_prompt,
            },{
                'role': 'user',
                'content': user_prompt,},
            ],
            options = {
                'temperature': .3
            })
            test.append((response['message']['content'], str(labels)))
        return test

def main():
    parser = argparse.ArgumentParser(description="Arguments for our main function.")
    parser.add_argument("fulloutput", help="Whether to show all of our outputs.")
    args = parser.parse_args()
    Embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = FAISS.load_local("CRS_Reports2", Embeddings, allow_dangerous_deserialization=True)
    df = preprocess_dfs(pd.read_csv("Fake.csv"), pd.read_csv("True.csv"))
    if args.fulloutput == str(False):
        outputs = start_rag_workflow(vector_store=vector_store, df= df, full_output= False)
        outputs = np.array(outputs)
        acc = accuracy_score(outputs[:,0], outputs[:,1])
        print(acc)
        ax= plt.subplot()
        c = confusion_matrix(outputs[:,0], outputs[:,1])
        normed_c = c / np.sum(c, axis=1, keepdims=True)
        df_cm = pd.DataFrame(normed_c, range(2), range(2))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted")
        ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['False', 'True'])
        plt.show()
    else:
        outputs = start_rag_workflow(vector_store=vector_store, df= df, full_output= True)
        for out in outputs:
            print(f"{out[0]}\n\nLabel:{out[1]}\n\n")
if __name__ == "__main__":
    main()