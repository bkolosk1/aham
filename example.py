# run_example.py

import logging
from aham.data import load_ml_arxiv_data
from aham.config import get_grid
from aham.aham_topic_modeling import AHAMTopicModeling

# Set up basic logging configuration.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main():
    abstracts, _ = load_ml_arxiv_data()
    grid = get_grid()
    logging.info(f"Total grid configurations: {len(grid)}")
    
    # Initialize the estimator with a grid of configurations.
    model = AHAMTopicModeling(grid=grid)
    model.fit(abstracts)
    logging.info("Best configuration:")
    for key, value in model.best_config_.items():
        logging.info(f"{key}: {value}")
    logging.info(f"Best AHAM Score: {model.best_aham_score_}")
    
    # Predict topics for new documents.
    new_docs = [
        "Recent advances in machine learning have led to breakthroughs in natural language processing.",
        "The study of climate change shows significant effects on global agriculture."
    ]
    topics = model.predict(new_docs)
    logging.info(f"Predicted topics for new documents: {topics}")

if __name__ == "__main__":
    main()
