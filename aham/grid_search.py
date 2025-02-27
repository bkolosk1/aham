from aham.modeling import run_topic_modeling
from aham.evaluation import compute_aham_objective, load_sem_model
from aham.llm import cleanup_llm_models
import logging
import gc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def grid_search(abstracts, grid):
    """
    Runs grid search over the provided configurations.
    Returns a list of dictionaries containing:
      - "config": The configuration used.
      - "aham_score": The computed AHAM score.
      - "topic_info": The topic information DataFrame.
      - "topic_model": The fitted BERTopic model.
    """
    results = []
    total = len(grid)
    for idx, config in enumerate(grid):
        logger.info(f"Running configuration {idx+1}/{total}")
        try:
            topic_model, topic_info = run_topic_modeling(abstracts, config)
            topic_names = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
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
                "topic_names": topic_names
            })
            logger.info(f"Configuration {idx+1}: AHAM Score = {aham_score}")
        except Exception as e:
            logger.error(f"Error with configuration {idx+1}: {e}")
        finally:
            cleanup_llm_models()
            del topic_model
            del topic_info
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
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
