# topic_eval_grid/grid_search.py
import itertools
import pandas as pd
from sentence_transformers import SentenceTransformer
from .modeling import run_topic_modeling
from .evaluation import compute_aham_objective, load_sem_model

def grid_search(abstracts, grid):
    """
    Runs a grid search over the provided configurations.
    
    Parameters:
      - abstracts (list[str]): List of documents.
      - grid (list[dict]): List of configuration dictionaries.
    
    Returns:
      - results (list[dict]): Each entry contains the configuration, computed AHAM score, and topic info.
    """
    results = []
    total = len(grid)
    for idx, config in enumerate(grid):
        print(f"Running configuration {idx+1}/{total}")
        try:
            topic_model, topic_info = run_topic_modeling(abstracts, config)
            topic_names = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
            # Typically, BERTopic marks outliers as -1.
            outlier_topic_ids = {-1}
            sem_model = load_sem_model(config["sem_model_name"])
            aham_score = compute_aham_objective(topic_names, outlier_topic_ids, sem_model)
            results.append({
                "config": config,
                "aham_score": aham_score,
                "topic_info": topic_info
            })
            print(f"AHAM Score: {aham_score}")
        except Exception as e:
            print(f"Error with configuration {idx}: {e}")
    return results

def select_best_configuration(results, higher_better=False):
    """
    Selects the best configuration from grid search results based on the AHAM score.
    
    Parameters:
      - results (list[dict]): List of result dictionaries.
      - higher_better (bool): If True, higher AHAM scores are better; otherwise, lower scores are better.
    
    Returns:
      - best_config (dict): The best configuration result.
    """
    valid_results = [r for r in results if r["aham_score"] is not None]
    if not valid_results:
        raise ValueError("No valid configurations found.")
    if higher_better:
        best_config = max(valid_results, key=lambda r: r["aham_score"])
    else:
        best_config = min(valid_results, key=lambda r: r["aham_score"])
    return best_config
