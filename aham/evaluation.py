import numpy as np
import itertools
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_sem_model(sem_model_name):
    """
    Loads and returns a SentenceTransformer model for semantic similarity.
    """
    logger.info(f"Loading semantic model: {sem_model_name}")
    return SentenceTransformer(sem_model_name)

def semantic_similarity(text_a, text_b, sem_model):
    """
    Computes cosine similarity between two texts using the provided semantic model.
    """
    emb_a = sem_model.encode(text_a, convert_to_numpy=True)
    emb_b = sem_model.encode(text_b, convert_to_numpy=True)
    sim = np.dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b))
    return sim

def fuzzy_similarity(text_a, text_b):
    """
    Computes normalized fuzzy matching similarity between two texts.
    """
    return fuzz.ratio(text_a, text_b) / 100.0

def compute_topic_similarity(topic_a, topic_b, sem_model=None, method="semantic"):
    """
    Computes the similarity between two topic labels using the specified method.
    """
    if method == "semantic":
        if sem_model is None:
            raise ValueError("sem_model must be provided for semantic similarity")
        return semantic_similarity(topic_a, topic_b, sem_model)
    elif method == "fuzzy":
        return fuzzy_similarity(topic_a, topic_b)
    else:
        raise ValueError(f"Unsupported topic similarity method: {method}")

def compute_aham_objective(topic_names, outlier_topic_ids, sem_model=None, method="semantic"):
    """
    Computes the AHAM objective:
      AHAM = 2 * (|outliers| / |topics|) * (average pairwise topic similarity)
    """
    topic_ids = list(topic_names.keys())
    k = len(topic_ids)
    if k < 2:
        raise ValueError("At least two topics are required to compute pairwise similarity.")
    num_outliers = sum(1 for tid in topic_ids if tid in outlier_topic_ids)
    sim_scores = []
    for tid1, tid2 in itertools.combinations(topic_ids, 2):
        sim = compute_topic_similarity(topic_names[tid1], topic_names[tid2], sem_model, method)
        sim_scores.append(sim)
    avg_similarity = np.mean(sim_scores) if sim_scores else 0.0
    aham = avg_similarity
    if num_outliers != 0:
        aham = 2 * (num_outliers / k) * avg_similarity    
    logger.info(f"Computed AHAM objective: {aham}")
    return aham
