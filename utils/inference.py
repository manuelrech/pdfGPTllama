import path
import sys
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from constants import LLM_MODEL

from transformers import AutoTokenizer,TextStreamer, pipeline
from llama_index.llms import HuggingFaceLLM
#from langchain.llms import HuggingFacePipeline
import torch

def load_llm(): 
    print('\nLLM model:', LLM_MODEL) 
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL) 
    llm = HuggingFaceLLM (
            context_window = 8192, # 2048,
            max_new_tokens = 720, # 512, # 256,
            generate_kwargs = {"temperature": 0.1, "repetition_penalty": 1.1, "do_sample": True, 'pad_token_id': tokenizer.eos_token_id},
            query_wrapper_prompt = "{query_str}",
            model_name = LLM_MODEL,
            device_map = "auto",
            tokenizer_kwargs = {"max_length": 4000},
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs = {"torch_dtype": torch.bfloat16},
            tokenizer = tokenizer,
    )
 
    return llm


def load_general_llm():
    print('\nLLM model:', LLM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        repetition_penalty=1.1,
        temperature=0.1,
        do_sample = True,
        trust_remote_code=True,
        device_map="auto",
        max_new_tokens=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    return pipe
