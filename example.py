import logging
from aham.data import load_ida_dataset
from aham.config import get_grid, load_config_from_yaml
from aham.grid_search import grid_search, select_best_configuration
from aham.aham_topic_modeling import AHAMTopicModeling
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
def main(n_grids = 3):
    abstracts, _ = load_ida_dataset()
    
    grid = random.sample(get_grid(), n_grids)
    logging.info(f"Total grid configurations: {len(grid)}")
    
    results = grid_search(abstracts, grid)
    best_result = select_best_configuration(results, higher_better=False)

    best_topic_names = best_result["topic_info"]["Llama2"].tolist()
    logging.info("Best Topic Names from the best configuration:")
    for name in best_topic_names:
        logging.info(name[0])
    
    logging.info(f"Best AHAM Score from grid search: {best_result['aham_score']}")
    
    estimator = AHAMTopicModeling(config=best_result["config"], topic_similarity_method="fuzzy")
    estimator.fit(abstracts[:-3])
    score = estimator.score()
    logging.info(f"Estimator AHAM Score: {score}")
    
    new_docs = abstracts[-3:]
    predicted_topics = estimator.predict(new_docs)
    logging.info(f"Predicted topics for new documents: {predicted_topics}")

if __name__ == "__main__":
    main()
