from sklearn.base import BaseEstimator, TransformerMixin
from aham.grid_search import grid_search, select_best_configuration
from aham.modeling import run_topic_modeling
from aham.evaluation import load_sem_model, compute_aham_objective

class AHAMTopicModeling(BaseEstimator, TransformerMixin):
    """
    A scikit‑learn estimator for topic modeling using the AHAM objective.
    
    If a grid (list of configurations) is provided, grid search is performed
    to select the configuration with the highest AHAM score. Otherwise, the provided configuration is used.
    """
    def __init__(self, config=None, grid=None, outlier_topic_ids={-1}):
        """
        Parameters:
          - config: A configuration dictionary (if grid search is not used).
          - grid: A list of configuration dictionaries for grid search.
          - outlier_topic_ids: Set of topic IDs considered as outliers (default {-1}).
        """
        self.config = config
        self.grid = grid
        self.outlier_topic_ids = outlier_topic_ids
        self.best_model_ = None
        self.best_config_ = None
        self.best_aham_score_ = None

    def fit(self, X, y=None):
        """
        Fit the topic model on the given documents X.
        
        Parameters:
          - X: List of documents (strings).
        
        After fitting, the best model (with the highest AHAM score) is stored.
        """
        if self.grid is not None:
            results = grid_search(X, self.grid)
            best_result = select_best_configuration(results, higher_better=True)
            self.best_config_ = best_result["config"]
            self.best_aham_score_ = best_result["aham_score"]
            self.best_model_ = best_result["topic_model"]
        elif self.config is not None:
            self.best_model_, topic_info = run_topic_modeling(X, self.config)
            sem_model = load_sem_model(self.config["sem_model_name"])
            topic_names = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
            self.best_aham_score_ = compute_aham_objective(topic_names, self.outlier_topic_ids, sem_model,
                                                            method=self.config.get("topic_similarity_method", "semantic"))
            self.best_config_ = self.config
        else:
            raise ValueError("Either 'config' or 'grid' must be provided.")
        print(f"Fitted model with AHAM score: {self.best_aham_score_}")
        return self

    def predict(self, X):
        """
        Predict topic assignments for new documents using the best fitted model.
        
        Parameters:
          - X: List of new documents.
          
        Returns:
          - topics: List of predicted topic IDs.
        """
        if self.best_model_ is None:
            raise ValueError("The model is not fitted yet.")
        topics, _ = self.best_model_.transform(X)
        return topics
