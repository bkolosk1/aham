import pandas as pd
from datasets import load_dataset

def load_ml_arxiv_data(split="train"):
    """
    Loads the ML-ArXiv Papers dataset and returns abstracts and titles.
    For testing, only the first 100 records are used.
    """
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")[split]
    abstracts = dataset["abstract"][:100]
    titles = dataset["title"][:100]
    return abstracts, titles

def load_ida_dataset():
    """
    Loads a dataset from a local TSV file 'id2tiab.tsv'.
    """
    dataset = pd.read_csv('id2tiab.tsv', delimiter='\t').dropna().reset_index(drop=True)
    abstracts = dataset['abstract'].tolist()
    titles = dataset["title"].tolist()
    return abstracts, titles
