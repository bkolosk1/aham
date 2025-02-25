import itertools
import logging
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TOKEN = "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" 
DEFAULT_LLAMA_GEN_PARAMS = {"temperature": 0.1, "max_new_tokens": 300, "repetition_penalty": 1.1}

DEFAULT_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
]

DEFAULT_UMAP_PARAMS = {"n_neighbors": 15, "n_components": 32, "min_dist": 0.0, "metric": "cosine", "random_state": 42}
UMAP_PARAMS_GRID = [
    DEFAULT_UMAP_PARAMS,
    {"n_neighbors": 10, "n_components": 32, "min_dist": 0.1, "metric": "cosine", "random_state": 42},
    {"n_neighbors": 20, "n_components": 32, "min_dist": 0.0, "metric": "cosine", "random_state": 42},
]

HDBSCAN_PARAMS_GRID = [
    {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "eom"},
    {"min_cluster_size": 5,  "metric": "euclidean", "cluster_selection_method": "eom"},
    {"min_cluster_size": 15, "metric": "euclidean", "cluster_selection_method": "eom"},
    {"min_cluster_size": 8,  "metric": "euclidean", "cluster_selection_method": "eom"},
    {"min_cluster_size": 12, "metric": "euclidean", "cluster_selection_method": "eom"},
]

DEFAULT_CHAT_TEMPLATE = [
    {"role": "system", "content": "You are a helpful, respectful, and honest research assistant for labeling topics."},
    {"role": "user", "content": (
        "I have a topic that contains the following documents:\n"
        "- Bisociative Knowledge Discovery by Literature Outlier Detection.\n"
        "- Evaluating Outliers for Cross-Context Link Discovery.\n"
        "- Exploring the Power of Outliers for Cross-Domain Literature Mining.\n"
        "The topic is described by the following keywords: bisociative, knowledge discovery, "
        "outlier detection, data mining, cross-context, link discovery, cross-domain, machine learning.\n"
        "Based on the information above, please create a simple, short and concise computer science label for this topic. "
        "Make sure you only return the label."
    )},
    {"role": "assistant", "content": "Outlier-based knowledge discovery"},
    {"role": "user", "content": (
        "\n"
        "I have a topic that contains the following documents: [DOCUMENTS]\n"
        "The topic is described by the following keywords: '[KEYWORDS]'.\n"
        "Based on the information above, please create a simple, short and concise computer science label for this topic. "
        "Make sure you only return the label."
    )}
]

DEFAULT_TOPIC_SIMILARITY_METHOD = "semantic"

def get_grid():
    """
    Returns a list of configuration dictionaries for grid search.
    Each configuration includes:
      - chat_template,
      - model_id, token, llama_gen_params,
      - embedding_model_name,
      - umap_params,
      - hdbscan_params,
      - sem_model_name (for evaluation),
      - topic_similarity_method.
    
    The grid is produced as the Cartesian product of the embedding models,
    UMAP parameter sets, and HDBSCAN parameter sets.
    """
    grid = []
    for emb_model, umap_params, hdb_params in itertools.product(
        DEFAULT_EMBEDDING_MODELS, UMAP_PARAMS_GRID, HDBSCAN_PARAMS_GRID
    ):
        config = {
            "chat_template": DEFAULT_CHAT_TEMPLATE,
            "model_id": DEFAULT_MODEL_ID,
            "token": DEFAULT_TOKEN,
            "llama_gen_params": DEFAULT_LLAMA_GEN_PARAMS,
            "embedding_model_name": emb_model,
            "umap_params": umap_params,
            "hdbscan_params": hdb_params,
            "sem_model_name": emb_model,
            "topic_similarity_method": DEFAULT_TOPIC_SIMILARITY_METHOD
        }
        grid.append(config)
    if len(grid) < 2:
        raise ValueError("At least two grid configurations are required.")
    logger.info(f"Generated {len(grid)} grid configurations.")
    return grid

def load_config_from_yaml(file_path):
    """
    Loads a configuration (or a list of configurations) from a YAML file.
    """
    with open(file_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {file_path}")
    return cfg