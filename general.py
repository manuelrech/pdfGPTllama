from transformers import AutoTokenizer,TextStreamer, pipeline
import torch
from constants import LLM_MODEL
    
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


pipe = load_general_llm()

while True:
    query = input(' > Enter a query: ')
    if query == 'stop':
        break
    pipe(f'<s>[INST]{query}[/INST]')

