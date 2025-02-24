DEFAULT_PROMPT = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information above, please create a short label for this topic. Only return the label.
[/INST]
"""

DEFAULT_LLAMA_GEN_PARAMS = {"temperature": 0.1, "max_new_tokens": 50, "repetition_penalty": 1.1}
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_UMAP_PARAMS = {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine", "random_state": 42}
DEFAULT_SEM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HDBSCAN_PARAMS_GRID = [
    {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "eom"},
    {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"},
]

DEFAULT_TOPIC_SIMILARITY_METHOD = "semantic"

def get_grid():
    """
    Returns a list of configuration dictionaries for grid search.
    Only the HDBSCAN parameters (and the topic similarity method) vary,
    while all other parameters remain fixed.
    """
    grid = []
    for hdbscan_params in HDBSCAN_PARAMS_GRID:
        config = {
            "prompt": DEFAULT_PROMPT,
            "llama_gen_params": DEFAULT_LLAMA_GEN_PARAMS,
            "embedding_model_name": DEFAULT_EMBEDDING_MODEL,
            "umap_params": DEFAULT_UMAP_PARAMS,
            "hdbscan_params": hdbscan_params,
            "sem_model_name": DEFAULT_SEM_MODEL,
            "topic_similarity_method": DEFAULT_TOPIC_SIMILARITY_METHOD,
        }
        grid.append(config)
    if len(grid) < 2:
        raise ValueError("At least two grid configurations are required.")
    return grid
