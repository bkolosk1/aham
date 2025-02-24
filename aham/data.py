import pandas as pd
from datasets import load_dataset

def load_ml_arxiv_data(split="train"):
    """
    Loads the ML-ArXiv Papers dataset and returns abstracts and titles.
    
    Parameters:
      - split (str): Dataset split to load (default "train").
    
    Returns:
      - abstracts (list[str]): Abstracts from the dataset.
      - titles (list[str]): Titles from the dataset.
    """
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")[split]
    abstracts = dataset["abstract"][:100]
    titles = dataset["title"][:100]
    return abstracts, titles

def load_ida_dataset():
    dataset = pd.read_csv('id2tiab.tsv', delimiter = '\t').dropna().reset_index(drop=True)
    abstracts = dataset['abstract']
    titles = dataset["title"]
    return abstracts, titles
