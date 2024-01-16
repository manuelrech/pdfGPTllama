from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

def load_embeddings():
    embed_model = HuggingFaceBgeEmbeddings (
        model_name = 'BAAI/bge-large-en-v1.5',
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'normalize_embeddings': True},
        #query_instruction='Generate a representation for this sentence that can be used to retrieve related articles:'
        )
    return embed_model
