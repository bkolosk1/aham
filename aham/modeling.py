# aham/modeling.py

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer as STModel
from .llama import get_llama_generator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_topic_modeling(abstracts, config):
    """
    Builds and fits a BERTopic model using the configuration parameters.
    Returns the fitted topic_model and topic_info DataFrame.
    """
    logger.info(f"Running topic modeling with embedding model: {config['embedding_model_name']}")

    embedding_model = STModel(config["embedding_model_name"])
    
    umap_model = UMAP(**config["umap_params"])
    hdbscan_model = HDBSCAN(**config["hdbscan_params"], prediction_data=True)
    
    generator = get_llama_generator(config["llama_gen_params"])
    
    system_prompt = config.get("system_prompt", "")
    example_prompt = config.get("example_prompt", "")
    main_prompt = config.get("main_prompt", "")
    full_prompt = system_prompt + "\n" + example_prompt + "\n" + main_prompt
    logger.info("Constructed full prompt for topic labeling.")
    
    llama2_rep = TextGeneration(generator, prompt=full_prompt)
    
    keybert = KeyBERTInspired()
    mmr = MaximalMarginalRelevance(diversity=0.3)
    
    representation_model = {
        "KeyBERT": keybert,
        "Llama2": llama2_rep,
        "MMR": mmr,
    }
    
    # Create and fit the BERTopic model.
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
    
    # Extract and set Llama2-generated labels.
    llama2_topics = topic_model.get_topics(full=True).get("Llama2", {})
    llama2_labels = {}
    for topic_id, label_info in llama2_topics.items():
        if label_info and len(label_info) > 0:
            label = label_info[0][0].split("\n")[0].strip()
            llama2_labels[topic_id] = label
        else:
            llama2_labels[topic_id] = ""
    topic_model.set_topic_labels(llama2_labels)
    
    topic_info = topic_model.get_topic_info()
    logger.info("Topic modeling complete.")
    return topic_model, topic_info
