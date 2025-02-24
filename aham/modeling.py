from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer as STModel
from .llama import get_llama_generator

def run_topic_modeling(abstracts, config):
    """
    Builds and fits a BERTopic model based on the supplied configuration.
    
    Returns:
      - topic_model: The fitted BERTopic model.
      - topic_info: DataFrame with topic information.
    """
    # Instantiate the embedding model.
    embedding_model = STModel(config["embedding_model_name"])
    
    # Configure UMAP and HDBSCAN.
    umap_model = UMAP(**config["umap_params"])
    hdbscan_model = HDBSCAN(**config["hdbscan_params"], prediction_data=True)
    
    # Build the Llama2 generator.
    generator = get_llama_generator(config["llama_gen_params"])
    
    # Build a full prompt for topic naming.
    system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""
    example_prompt = """
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] Environmental impacts of eating meat
"""
    main_prompt = config["prompt"]
    full_prompt = system_prompt + example_prompt + main_prompt
    
    # Build the TextGeneration representation model.
    llama2_rep = TextGeneration(generator, prompt=full_prompt)
    
    # Instantiate additional representation models.
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
    
    # Compute embeddings for abstracts.
    abstract_embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    topics, _ = topic_model.fit_transform(abstracts, embeddings=abstract_embeddings)
        
    llama2_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Llama2"].values()]
    topic_model.set_topic_labels(llama2_labels)

    
    
    topic_info = topic_model.get_topic_info()
    return topic_model, topic_info
