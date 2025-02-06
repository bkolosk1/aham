# topic_eval_grid/data.py
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
    abstracts = dataset["abstract"]
    titles = dataset["title"]
    return abstracts, titles
