# topic_eval_grid/evaluation.py
import numpy as np
import itertools
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

def load_sem_model(sem_model_name):
    """
    Loads and returns a SentenceTransformer model for semantic similarity.
    
    Parameters:
      - sem_model_name (str): The model name.
    
    Returns:
      - sem_model: Loaded SentenceTransformer model.
    """
    return SentenceTransformer(sem_model_name)

def semantic_similarity(text_a, text_b, sem_model):
    """
    Computes cosine similarity between two texts using the provided semantic model.
    
    Parameters:
      - text_a (str): First text.
      - text_b (str): Second text.
      - sem_model: A SentenceTransformer model.
    
    Returns:
      - float: Cosine similarity between text embeddings.
    """
    emb_a = sem_model.encode(text_a, convert_to_numpy=True)
    emb_b = sem_model.encode(text_b, convert_to_numpy=True)
    return np.dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b))

def compute_topic_similarity(topic_a, topic_b, sem_model):
    """
    Computes the similarity between two topic labels using the provided semantic model.
    
    Parameters:
      - topic_a (str): Label for topic A.
      - topic_b (str): Label for topic B.
      - sem_model: A SentenceTransformer model.
    
    Returns:
      - float: Similarity score.
    """
    return semantic_similarity(topic_a, topic_b, sem_model)

def compute_aham_objective(topic_names, outlier_topic_ids, sem_model):
    """
    Computes the AHAM objective.
    
    AHAM = 2 * (|outliers|/|topics|) * (average pairwise topic similarity)
    
    Parameters:
      - topic_names (dict): Mapping of topic id to topic label.
      - outlier_topic_ids (iterable): Set or list of topic ids considered as outliers (typically {-1}).
      - sem_model: A SentenceTransformer model for semantic similarity.
      
    Returns:
      - float: The AHAM objective score.
    """
    topic_ids = list(topic_names.keys())
    k = len(topic_ids)
    if k < 2:
        raise ValueError("At least two topics are required to compute pairwise similarity.")
    num_outliers = sum(1 for tid in topic_ids if tid in outlier_topic_ids)
    sim_scores = []
    for tid1, tid2 in itertools.combinations(topic_ids, 2):
        sim = compute_topic_similarity(topic_names[tid1], topic_names[tid2], sem_model)
        sim_scores.append(sim)
    avg_similarity = np.mean(sim_scores) if sim_scores else 0.0
    aham_objective = 2 * (num_outliers / k) * avg_similarity
    return aham_objective
