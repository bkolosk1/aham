from sklearn.base import BaseEstimator, TransformerMixin
from .modeling import run_topic_modeling
from .evaluation import load_sem_model, compute_aham_objective
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AHAMTopicModeling(BaseEstimator, TransformerMixin):
    """
    A scikit‑learn estimator for topic modeling built on top of BERTopic
    """
    def __init__(self, config, topic_similarity_method="semantic"):
        """
        Parameters:
          - config: A configuration dictionary 
          - topic_similarity_method: "semantic" or "fuzzy" (default "semantic").
        """
        self.config = config
        self.topic_similarity_method = topic_similarity_method
        self.topic_model_ = None
        self.topic_names_ = None
        self.best_aham_score_ = None

    def fit(self, X, y=None):
        """
        Fit the topic model on a list of documents.
        """
        logger.info("Fitting AHAMTopicModeling...")
        topic_model, topic_info = run_topic_modeling(X, self.config)
        self.topic_model_ = topic_model
        self.topic_names_ = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
        logger.info("Model fitted; topic names stored.")
        return self

    def predict(self, X):
        """
        Predict topic assignments for new documents using the fitted topic model.
        """
        if self.topic_model_ is None:
            raise ValueError("The model is not fitted yet.")
        topics, _ = self.topic_model_.transform(X)
        return topics

    def score(self, X=None, y=None):
        """
        Computes the AHAM score using the stored topic names.
        """
        if self.topic_names_ is None:
            raise ValueError("The model is not fitted yet.")
        outlier_topic_ids = {-1}
        if self.topic_similarity_method == "semantic":
            sem_model = load_sem_model(self.config["sem_model_name"])
            aham = compute_aham_objective(self.topic_names_, outlier_topic_ids, sem_model, method="semantic")
        elif self.topic_similarity_method == "fuzzy":
            aham = compute_aham_objective(self.topic_names_, outlier_topic_ids, method="fuzzy")
        else:
            raise ValueError("Unsupported topic similarity method")
        self.best_aham_score_ = aham
        logger.info(f"Computed AHAM score: {aham}")
        return aham
