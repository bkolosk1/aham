from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer as STModel
from aham.llm import get_llm_generator, cleanup_llm_models
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_topic_modeling(abstracts, config):
    """
    Builds and fits a BERTopic model using configuration parameters.
    Returns the fitted topic_model and topic_info DataFrame.
    """
    logger.info(f"Running topic modeling with embedding model: {config['embedding_model_name']}")
    embedding_model = STModel(config["embedding_model_name"])
    umap_model = UMAP(**config["umap_params"])
    hdbscan_model = HDBSCAN(**config["hdbscan_params"], prediction_data=True)
    
    generator, full_prompt = get_llm_generator(config)
    
    logger.info("Constructed full prompt for topic labeling.")
    
    llama_rep = TextGeneration(generator, prompt=full_prompt)
    
    keybert = KeyBERTInspired()
    mmr = MaximalMarginalRelevance(diversity=0.3)
    representation_model = {
        "KeyBERT": keybert,
        "Llama2": llama_rep,
        "MMR": mmr,
    }
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=False
    )
    
    abstract_embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    topics, _ = topic_model.fit_transform(abstracts, embeddings=abstract_embeddings)
    
    llama_topics = topic_model.get_topics(full=True).get("Llama2", {})
    labels = {}
    for topic_id, label_info in llama_topics.items():
        if label_info and len(label_info) > 0:
            label = label_info[0][0].split("\n")[0].strip()
            labels[topic_id] = label
        else:
            labels[topic_id] = ""
    topic_model.set_topic_labels(labels)
    logger.info(f"Extracted topics: {labels}")
    
    topic_info = topic_model.get_topic_info()

    logger.info("Topic modeling complete.")
    
    cleanup_llm_models()    
    return topic_model, topic_info
