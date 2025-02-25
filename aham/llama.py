# aham/llama.py

import transformers
from torch import cuda, bfloat16
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_ID = 'google/gemma-2-2b'
TOKEN = "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"

def load_llama_model(token=TOKEN):
    """
    Loads and returns the Llama-2 model and tokenizer.
    """
    logger.info("Loading Llama-2 model...")
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
        token=token
    )
    model.eval()
    logger.info("{MODEL_ID} model loaded.")
    return model, tokenizer

def get_llama_generator(gen_params):
    """
    Returns a text-generation pipeline using Llama-2 with specified parameters.
    """
    model, tokenizer = load_llama_model()
    generator = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        temperature=gen_params.get("temperature", 0.1),
        max_new_tokens=gen_params.get("max_new_tokens", 500),
        repetition_penalty=gen_params.get("repetition_penalty", 1.1)
    )
    return generator
