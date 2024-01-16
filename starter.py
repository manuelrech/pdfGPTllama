import os
import sys
import pickle
import torch
import logging
import chromadb

from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import AsyncIteratorCallbackHandler

from llama_index import Document, VectorStoreIndex, SummaryIndex, ServiceContext, StorageContext, load_index_from_storage, get_response_synthesizer, global_service_context
from llama_index.retrievers import VectorIndexRetriever, VectorIndexAutoRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores import ChromaVectorStore

from transformers import AutoTokenizer,TextStreamer, TextIteratorStreamer, pipeline

from constants import COLLECTION_NAME

data = None
documents = []

# logging.basicConfig (stream = sys.stdout, level = logging.INFO)
# logging.getLogger ().addHandler (logging.StreamHandler (stream = sys.stdout))

def load_llm():
    model = "mistralai/Mistral-7B-Instruct-v0.1" # LLM_MODEL
    tokenizer = AutoTokenizer.from_pretrained (model)
    streamer = TextStreamer (tokenizer, skip_prompt = True, skip_special_tokens = True)
    # streamer = TextIteratorStreamer (tokenizer, skip_prompt = True, skip_special_tokens = True)

    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        torch_dtype = torch.bfloat16,
        repetition_penalty = 1.1,
        temperature = 0.0, # 0.1,
        trust_remote_code = True,
        device_map = "auto",
        max_new_tokens = 1024,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.eos_token_id,
        streamer = streamer,
    )

    llm = HuggingFacePipeline (pipeline = pipe)

    return llm

def corpus_parser(file_dict):
    corpus = file_dict["corpus"]

    corpus_text = ""
    for section in corpus:

        corpus_text += section["section_text"]

    return corpus_text

with open ("/home/ec2-user/grobid_parser/data/grobid_parsed_docs/llm800.pkl", "rb") as f:
    data = pickle.load (f)
    # doi = 10000
    for pdf in data:
        # doi += 1
        for section in pdf["corpus"]:
            # doctext = corpus_parser (pdf)
            corpusDoc = Document (
                text = section["section_text"], # doctext,
                metadata = {
                    'filename': pdf['file'],
                    'title': pdf['title'],
                    'pub_date': pdf['pub_date'],
                    'doi': pdf['doi'],
                }
            )
            documents.append (corpusDoc)

llm = load_llm ()

embed_model = HuggingFaceBgeEmbeddings (
    model_name = 'BAAI/bge-large-en-v1.5',
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True},
    query_instruction='Generate a representation for this sentence that can be used to retrieve related articles:'
)

# initialize client
db = chromadb.PersistentClient(path="/home/ec2-user/pdfGPTllama/DBs")
chroma_collection = db.get_or_create_collection (COLLECTION_NAME)

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection = chroma_collection)

service_context = ServiceContext.from_defaults(
        llm = llm, 
        embed_model = embed_model,
        )

index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context)

query_engine = index.as_query_engine (similarity_top_k = 5)


print ("Query engine ready")
while True:
    try:
        query = input('\n> Enter a query: ')
        if (len (query) == 0):
            continue
        print ("Answering your question")
        response = query_engine.query (query)
        print (response)
        # sources = response.get_formatted_sources ()
        print ("\n>>>>>>> SOURCES <<<<<<<\n")
        sources = response.source_nodes
        for doc in sources:
            print ("\nFilename: %s\nTitle: %s\nDOI: %s\nSimilarity score: %f\nChunk: %s\n\n" % (doc.metadata["filename"], doc.metadata["title"], doc.metadata["doi"], float (doc.score), doc.text))
    except EOFError:
        print ("Goodbye!")
        break
