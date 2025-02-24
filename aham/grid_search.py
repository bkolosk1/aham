from .modeling import run_topic_modeling
from .evaluation import compute_aham_objective, load_sem_model

def grid_search(abstracts, grid):
    """
    Runs a grid search over the provided configurations.
    
    Returns a list of dictionaries, each containing:
      - "config": The configuration used.
      - "aham_score": The computed AHAM score.
      - "topic_info": Topic info DataFrame.
      - "topic_model": The fitted BERTopic model.
    """
    results = []
    total = len(grid)
    for idx, config in enumerate(grid):
        print(f"Running configuration {idx+1}/{total}")
        #try:
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
        print(f"AHAM Score: {aham_score}")
        print(f"Current topics are: {topic_names}")

    return results

def select_best_configuration(results, higher_better=True):
    """
    Selects the best configuration based on the AHAM score.
    
    If higher_better is True, the configuration with the highest AHAM score is chosen.
    """
    valid_results = [r for r in results if r["aham_score"] is not None]
    if not valid_results:
        raise ValueError("No valid configurations found.")
    if higher_better:
        best_config = max(valid_results, key=lambda r: r["aham_score"])
    else:
        best_config = min(valid_results, key=lambda r: r["aham_score"])
    return best_config
