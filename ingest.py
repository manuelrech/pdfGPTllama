from transformers import AutoTokenizer
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from utils.ingestion import abstract_parser, corpus_parser
from llama_index import VectorStoreIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from constants import LLM_MODEL, PKL_FILE, COLLECTION_NAME
from llama_index.text_splitter import TokenTextSplitter
import pickle


def load_and_split(PKL_FILE = PKL_FILE):
    with open(PKL_FILE, 'rb') as file:
        data = pickle.load(file)
     
    docs = []
    for d in data:
        abstractDoc = abstract_parser(d)
        corpusDocs = corpus_parser(d)
        documents = [abstractDoc] + corpusDocs
        docs.extend(documents)
    
    return docs

def load_textSplitter(LLM_MODEL = LLM_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=500,
        chunk_overlap=100,
        backup_separators=["\n"],
        tokenizer=tokenizer.encode
    )

    return text_splitter

# loading documents from PKL and token text splitter
docs = load_and_split()
text_splitter = load_textSplitter()

embed_model = HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-large-en-v1.5',
        model_kwargs={'device':'cuda'},
        encode_kwargs={'normalize_embeddings':True},
        query_instruction='Generate a representation for this sentence that can be used to retrieve related articles:'
        )

# loading vectorstore structure
db = chromadb.PersistentClient(path="./DBs")
chroma_collection = db.get_or_create_collection(
        name = COLLECTION_NAME, 
        metadata = {"hnsw:space": "cosine"}, 
        )
vector_store = ChromaVectorStore(
        chroma_collection = chroma_collection,
        persist_dir = f'./DBs/{COLLECTION_NAME}',
        )

# adding details to vectorstore
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
        embed_model = embed_model,
        text_splitter = text_splitter,
        llm = None,
        )

# creating the index from the vectorstore
index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, show_progress=True, service_context=service_context,
)

