import transformers
from torch import cuda, bfloat16
import logging
import gc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GLOBAL_LLAMA_MODEL_CACHE = {}

def load_gen_model(model_id, token=""):
    """
    Loads and returns the text-generation model and tokenizer for the given model_id.
    Uses a global cache to ensure the model is loaded only once.
    """
    if model_id in GLOBAL_LLAMA_MODEL_CACHE:
        logger.info(f"Using cached model for: {model_id}")
        return GLOBAL_LLAMA_MODEL_CACHE[model_id]
    logger.info(f"Loading model: {model_id}")
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
        token=token
    )
    model.eval()
    GLOBAL_LLAMA_MODEL_CACHE[model_id] = (model, tokenizer)
    logger.info(f"Model {model_id} loaded and cached.")
    return model, tokenizer

def get_llama_generator(config):
    """
    Returns a text-generation pipeline using the model specified in config.
    """
    model_id = config.get("model_id")
    token = config.get("token", "")
    model, tokenizer = load_gen_model(model_id, token)
    gen_params = config.get("llama_gen_params", {})
    generator = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        temperature=gen_params.get("temperature", 0.1),
        max_new_tokens=gen_params.get("max_new_tokens", 500),
        repetition_penalty=gen_params.get("repetition_penalty", 1.1)
    )
    prompt = tokenizer.apply_chat_template(config.get("chat_template"), tokenize=False)
    return generator, prompt

def cleanup_llama_models():
    """
    Deletes all loaded Llama models from the cache and clears GPU memory.
    """
    global GLOBAL_LLAMA_MODEL_CACHE
    logger.info("Cleaning up loaded Llama models...")
    for key in list(GLOBAL_LLAMA_MODEL_CACHE.keys()):
        del GLOBAL_LLAMA_MODEL_CACHE[key]
    GLOBAL_LLAMA_MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Cleanup complete.")
