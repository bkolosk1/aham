# topic_eval_grid/llama.py
import transformers
from torch import cuda, bfloat16

# Define model id and token (replace with your actual token)
MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'
TOKEN = "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"

def load_llama_model(token=TOKEN):
    """
    Loads and returns the Llama-2 model and tokenizer.
    
    Parameters:
      - token (str): Your Hugging Face token.
    
    Returns:
      - model: The Llama-2 model.
      - tokenizer: The corresponding tokenizer.
    """
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
    return model, tokenizer

def get_llama_generator(gen_params):
    """
    Returns a text-generation pipeline using Llama-2 with the specified generation parameters.
    
    Parameters:
      - gen_params (dict): Dictionary of generation parameters (temperature, max_new_tokens, etc.)
    
    Returns:
      - generator: A Hugging Face text-generation pipeline.
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
