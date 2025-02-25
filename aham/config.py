# aham/config.py

import itertools
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
You are a helpful, respectful, and honest research assistant for labeling topics.
<</SYS>>
"""
DEFAULT_EXAMPLE_PROMPT = """
I have a topic that contains the following documents:
- Bisociative Knowledge Discovery by Literature Outlier Detection.
- Evaluating Outliers for Cross-Context Link Discovery.
- Exploring the Power of Outliers for Cross-Domain Literature Mining.
The topic is described by the following keywords: bisociative, knowledge discovery, outlier detection, ,data mining, cross-context, link discovery, cross-domain,
machine learning’.
Based on the information about the topic above, please create a simple, short,
and concise computer science label for this topic. Make sure you only return the
label and nothing more.
[/INST]: Outlier-based knowledge discovery
"""
DEFAULT_MAIN_PROMPT = """
I have a topic that contains the following documents [DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]
Based on the information about the topic above, please create a simple, short
and concise computer science label for this topic. Make sure you only return the
label and nothing more.
[/INST]
"""

DEFAULT_LLAMA_GEN_PARAMS = {"temperature": 0.1, "max_new_tokens": 50, "repetition_penalty": 1.1}
DEFAULT_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
]
DEFAULT_UMAP_PARAMS = {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42}

HDBSCAN_PARAMS_GRID = [
    {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "eom"},
    {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"},
]

DEFAULT_SEM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOPIC_SIMILARITY_METHOD = "semantic"

def get_grid():
    """
    Returns a list of configuration dictionaries for grid search.
    Each configuration includes:
      - system_prompt, example_prompt, main_prompt,
      - llama_gen_params,
      - embedding_model_name,
      - umap_params,
      - hdbscan_params,
      - sem_model_name,
      - topic_similarity_method.
    
    Four example configurations are provided.
    """
    grid = []

    for emb_model, hdb_params in itertools.product(DEFAULT_EMBEDDING_MODELS, HDBSCAN_PARAMS_GRID):
        config = {
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "example_prompt": DEFAULT_EXAMPLE_PROMPT,
            "main_prompt": DEFAULT_MAIN_PROMPT,
            "llama_gen_params": DEFAULT_LLAMA_GEN_PARAMS,
            "embedding_model_name": emb_model,
            "umap_params": DEFAULT_UMAP_PARAMS,
            "hdbscan_params": hdb_params,
            "sem_model_name": emb_model,  # Use same model for evaluation as embedding
            "topic_similarity_method": DEFAULT_TOPIC_SIMILARITY_METHOD,
        }
        grid.append(config)
    if len(grid) < 2:
        raise ValueError("At least two grid configurations are required.")
    logger.info(f"Generated {len(grid)} grid configurations.")
    return grid
