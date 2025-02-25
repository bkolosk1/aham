# aham/grid_search.py

from .modeling import run_topic_modeling
from .evaluation import compute_aham_objective, load_sem_model
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def grid_search(abstracts, grid):
    """
    Runs grid search over provided configurations.
    Returns a list of dicts containing config, AHAM score, topic info, and fitted topic_model.
    """
    results = []
    total = len(grid)
    for idx, config in enumerate(grid):
        logger.info(f"Running configuration {idx+1}/{total}")
        try:
            topic_model, topic_info = run_topic_modeling(abstracts, config)
            topic_names = {row["Topic"]: row["Llama2"] for _, row in topic_info.iterrows()}
            outlier_topic_ids = {-1}
            sem_model = load_sem_model(config["sem_model_name"])
            aham_score = compute_aham_objective(
                topic_names,
                outlier_topic_ids,
                sem_model,
                method=config.get("topic_similarity_method", "semantic")
            )
            results.append({
                "config": config,
                "aham_score": aham_score,
                "topic_info": topic_info,
                "topic_model": topic_model
            })
            logger.info(f"Config {idx+1}: AHAM Score = {aham_score}")
        except Exception as e:
            logger.error(f"Error with configuration {idx+1}: {e}")
    return results

def select_best_configuration(results, higher_better=True):
    """
    Selects and returns the best configuration based on the AHAM score.
    """
    valid_results = [r for r in results if r["aham_score"] is not None]
    if not valid_results:
        raise ValueError("No valid configurations found.")
    best_config = max(valid_results, key=lambda r: r["aham_score"]) if higher_better else min(valid_results, key=lambda r: r["aham_score"])
    return best_config
