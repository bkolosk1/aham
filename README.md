# AHAM [![arXiv](https://img.shields.io/badge/arXiv-2312.15784-b31b1b.svg)](https://arxiv.org/abs/2312.15784) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fkMqefRjq4P78XvaYNnRaZi_30SdXeCA?usp=sharing)

AHAM adapts a topic modeling framework to a specific domain by minimizing the AHAM metric, as introduced in [AHAM: Adapt, Help, Ask, Model - Harvesting LLMs for literature mining](https://arxiv.org/pdf/2312.15784). By doing so, it reduces the proportion of outlier topics and lowers the lexical or semantic similarity between the generated topic labels, resulting in more distinct and domain-relevant topics.

### AHAM METRIC

The AHAM metric is defined as:

$$
\text{AHAM} = 2 \times \left(\frac{|\text{outliers}|}{|\text{topics}|}\right) \times \text{(average pairwise topic similarity)}
$$

This metric combines the ratio of outlier topics to total topics with the average pairwise similarity of topic labels. Minimizing this metric drives the adaptation process, ensuring that topics are both distinct and well-aligned with the target domain.

## Installation

### Using pip

Install the package in editable mode:
```
pip install -e .
```
Or directly:
```
pip install git+https://github.com/bkolosk1/aham
```

### Using Poetry

1. Clone the repository:
    ```
    git clone https://github.com/bkolosk1/aham
    cd aham
    ```
2. Install dependencies and enter the Poetry shell:
    ```
    poetry install
    ```

## Hugging Face Setup

Before running the example, please ensure you have a valid Hugging Face token and have subscribed to the required model.

1. **Set your Hugging Face token:**

   Export your token as an environment variable by running:
   ```
   export HF_TOKEN="your_token_here"
   ```
   
   To generate a token, log in to your Hugging Face account and navigate to [Hugging Face Tokens](https://huggingface.co/settings/tokens). Create a new token with the necessary scopes (typically, read access to models is sufficient).

2. **Subscribe to the model:**

   If the model you intend to use requires subscription or acceptance of specific terms, go to the model’s page on Hugging Face. Click on the **Subscribe** (or **Accept Model License**) button to gain access. This step is necessary if the model is behind a subscription or access gate.

## Usage

An example script (`exampl.py`) is provided. Below is an example of how to run AHAM:

```python
import logging
from aham.data import load_ida_dataset
from aham.config import get_grid, load_config_from_yaml
from aham.grid_search import grid_search, select_best_configuration
from aham.aham_topic_modeling import AHAMTopicModeling

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main():
    abstracts, _ = load_ida_dataset()
    
    grid = get_grid()[:3]

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
```

Run the example with:
```
python example.py
```

This can result these topic names:
```
Extracted topics: {-1: 'Literature-based Discovery Systems (LDS) for Medical Problem Solving', 0: 'Literature-Based Discovery Systems for Cross-Domain Knowledge Discovery', 1: 'Gene-Disease Literature Mining for Biomarker Discovery and Drug Repositioning', 2: 'Neural Network-based Literature-based Discovery (LBD) for Knowledge Graph Prediction and Association Discovery', 3: 'Semantic Graph Database-based Literature-Based Discovery (LBD)', 4: 'Biomedical Semantic-based Novel Connection Discovery from Literature', 5: 'Literature-Based Discovery Systems in Biomedical Domain', 6: 'Biomedical Literature Connection Discovery System (LitLinker)', 7: 'Literature-based Future Connection Prediction via Temporal KCNs and LSTM', 8: '"Complementary Noninteractive Literature Search Algorithm"', 9: 'Predication-based Semantic Indexing (PSI) for Analogical Reasoning and Therapeutic Discovery', 10: 'Literature-based Discovery Approaches for Biological Relationship Extraction', 11: 'Literature-based Drug Discovery & Repurposing'}
```

## Citation

If you use this code; please cite our work:
```
@InProceedings{10.1007/978-3-031-58547-0_21,
author="Koloski, Boshko
and Lavra{\v{c}}, Nada
and Cestnik, Bojan
and Pollak, Senja
and {\v{S}}krlj, Bla{\v{z}}
and Kastrin, Andrej",
editor="Miliou, Ioanna
and Piatkowski, Nico
and Papapetrou, Panagiotis",
title="AHAM: Adapt, Help, Ask, Model Harvesting LLMs for Literature Mining",
booktitle="Advances in Intelligent Data Analysis XXII",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="254--265",
isbn="978-3-031-58547-0"
}
