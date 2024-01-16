from pathlib import Path
from llama_hub.file.unstructured import UnstructuredReader
import os
from transformers import AutoTokenizer
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from llama_index import VectorStoreIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from constants import LLM_MODEL, PKL_FILE, COLLECTION_NAME
from llama_index.text_splitter import TokenTextSplitter

folder_path = '/home/ec2-user/PDF/stocks3'

loader = UnstructuredReader()
docs = []
for pdf in os.listdir(folder_path):
    document = loader.load_data(file=Path(folder_path + '/' + pdf))[0]
    docs.append(document)

print(docs[0])

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

text_splitter = load_textSplitter()

embed_model = HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-large-en-v1.5',
        model_kwargs={'device':'cuda'},
        encode_kwargs={'normalize_embeddings':True},
        query_instruction='Generate a representation for this sentence that can be used to retrieve related articles:'
        )

db = chromadb.PersistentClient(path="./DBs")
chroma_collection = db.get_or_create_collection(
        name = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
        )
vector_store = ChromaVectorStore(
        chroma_collection = chroma_collection,
        persist_dir = f'./DBs/{COLLECTION_NAME}',
        )

storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
        embed_model = embed_model,
        text_splitter = text_splitter,
        llm = None,
        )

index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, show_progress=True, service_context=service_context,
)
