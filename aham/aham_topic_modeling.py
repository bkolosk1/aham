from sklearn.base import BaseEstimator, TransformerMixin
from .grid_search import grid_search, select_best_configuration
from .modeling import run_topic_modeling
from .evaluation import load_sem_model, compute_aham_objective
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AHAMTopicModeling(BaseEstimator, TransformerMixin):
    """
    A scikit‑learn estimator for topic modeling using the AHAM objective.
    
    If a grid (list of configurations) is provided, grid search is performed
    to select the configuration with the highest AHAM score.
    """
    def __init__(self, config=None, grid=None, outlier_topic_ids={-1}):
        self.config = config
        self.grid = grid
        self.outlier_topic_ids = outlier_topic_ids
        self.best_model_ = None
        self.best_config_ = None
        self.best_aham_score_ = None

    def fit(self, X, y=None):
        """
        Fit the topic model on a list of documents.
        """
        if self.grid is not None:
            logger.info("Running grid search...")
            results = grid_search(X, self.grid)
            best_result = select_best_configuration(results, higher_better=True)
            self.best_config_ = best_result["config"]
            self.best_aham_score_ = best_result["aham_score"]
            self.best_model_ = best_result["topic_model"]
            logger.info(f"Grid search complete. Best AHAM Score: {self.best_aham_score_}")
        elif self.config is not None:
            self.best_model_, topic_info = run_topic_modeling(X, self.config)
            sem_model = load_sem_model(self.config["sem_model_name"])
            topic_names = {row["Topic"]: row["Llama2"] for _, row in topic_info.iterrows()}
            self.best_aham_score_ = compute_aham_objective(topic_names, self.outlier_topic_ids, sem_model,
                                                            method=self.config.get("topic_similarity_method", "semantic"))
            self.best_config_ = self.config
            logger.info(f"Fitted model with AHAM Score: {self.best_aham_score_}")
        else:
            raise ValueError("Either 'config' or 'grid' must be provided.")
        return self

    def predict(self, X):
        """
        Predict topic assignments for new documents using the best fitted model.
        """
        if self.best_model_ is None:
            raise ValueError("The model is not fitted yet.")
        topics, _ = self.best_model_.transform(X)
        return topics
