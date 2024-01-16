### INGESTION & INFERENCE ###
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
#COLLECTION_NAME = 'llm800'
#COLLECTION_NAME = 'polymers7000'
#COLLECTION_NAME = 'stocks3'
COLLECTION_NAME = 'polydec9'

### INGESTION ###
## define pkl file with lc_docs
#PKL_FILE = '/home/ec2-user/grobid_parser/data/grobid_parsed_docs/polymers7000.pkl'
PKL_FILE = '/home/ec2-user/grobid_parser/data/grobid_parsed_docs/polymers6500_7dec.pkl'

### INFERENCE ###
## custom prompt template for llamaindex retrieval
PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "<s>[INST]Given the context information and not prior knowledge, answer 'The given chunks does not provide information to answer such question' if you cannot find the answer in the contex information[/INST]" 
    "Query: [INST]{query_str}[/INST]\n"
    "Answer: "
)

## similarity cutoff value
SIMILARITY_CUTOFF = 0.7
