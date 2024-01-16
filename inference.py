import path
import sys
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from constants import COLLECTION_NAME, PROMPT_TEMPLATE, SIMILARITY_CUTOFF

from utils.inference import load_llm
from utils.general import load_embeddings

import torch
import chromadb
from transformers import AutoTokenizer 

from llama_index.prompts import PromptTemplate
from llama_index.response.schema import Response
from llama_index.vector_stores import ChromaVectorStore
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.postprocessor import SimilarityPostprocessor

########## INFERENCE ##########

### Loading models
llm = load_llm ()
embed_model = load_embeddings()

### Loading DB
db = chromadb.PersistentClient(path="/home/ec2-user/pdfGPTllama/DBs")
chroma_collection = db.get_or_create_collection (COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection = chroma_collection)

service_context = ServiceContext.from_defaults(
        llm = llm, 
        embed_model = embed_model,
        )

index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context = service_context,
        )

### Initialize query engine
similarity_filter = SimilarityPostprocessor(similarity_cutoff = SIMILARITY_CUTOFF)
query_engine = index.as_query_engine(
        similarity_top_k = 5,
        streaming = True,
        text_qa_template = PromptTemplate( PROMPT_TEMPLATE ),
        node_postprocessors = [similarity_filter], 
        )

### Actual QA logic
if __name__ == '__main__':
    while True:
        try:
            query = input('\n> Enter a query: ')
            if (len (query) == 0):
                continue
            print ("\nRetrieving references ... ")

            response = query_engine.query (query)

            if type(response) == Response:
                print("\nI haven't found anything in the database with a similarity higher than {SIMILARITY_CUTOFF} ... ")
                print("This is how i would respond to this question without external information ...")
                response = llm.stream_complete(query)

                for token in response:
                    print(token, end="")

            else:
                response.print_response_stream()
            
            
                for doc in response.source_nodes:
                    print("\nFilename:", doc.metadata["filename"])
                    print("\nTitle:", doc.metadata["title"])
                    print("\nDOI:", doc.metadata["doi"])
                    print("\nSimilarity score:", doc.score)
                    print("\nChunk:", doc.text)

        except EOFError:
            print ("Goodbye!")
            break
