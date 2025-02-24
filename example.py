from aham import AHAMTopicModeling, get_grid, load_ida_dataset

def main():
    abstracts, _ = load_ida_dataset()
    grid = get_grid()
    print(f"Total grid configurations: {len(grid)}")
    
    # Initialize the estimator with a grid of configurations.
    model = AHAMTopicModeling(grid=grid)
    model.fit(abstracts)
    print("\nBest configuration:")
    print(model.best_config_)
    print("AHAM Score:", model.best_aham_score_)
    
    # Predict topics for new documents.
    new_docs = [
        "Recent advances in machine learning have led to breakthroughs in natural language processing.",
        "The study of climate change shows significant effects on global agriculture."
    ]
    topics = model.predict(new_docs)
    print("\nPredicted topics for new documents:", topics)

if __name__ == "__main__":
    main()
