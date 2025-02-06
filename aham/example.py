# run_grid_search.py
from aham.data import load_ml_arxiv_data
from aham.config import get_grid
from aham.grid_search import grid_search, select_best_configuration

def main():
    # Load dataset
    abstracts, _ = load_ml_arxiv_data()
    
    # Get grid configurations
    grid = get_grid()
    print(f"Total configurations to test: {len(grid)}")
    
    # Run grid search
    results = grid_search(abstracts, grid)
    
    # Select the best configuration (assuming lower AHAM is preferred)
    best_result = select_best_configuration(results, higher_better=False)
    print("\n=== Best configuration ===")
    print("AHAM Score:", best_result["aham_score"])
    print("Configuration:")
    for key, value in best_result["config"].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()