from llama_index.response.schema import Response
from inference import *

if __name__ == '__main__':
    while True:
        try:
            query = input('\n> Enter a query: ')
            if (len (query) == 0):
                continue
            print ("\nRetrieving references ... ")

            response = query_engine.query (query)
            
            if type(response) == Response:
                print(f"\nI haven't found anything in the database with a similarity higher than {SIMILARITY_CUTOFF} ... ")
                print("This is how i would respond to this question without external information ...")
                response = llm.stream_complete (query)
                # print (response)  
                for token in response:
                    print (token, end = "")

            else:
                response.print_response_stream()
                print('\n\n')
                
                for doc in response.source_nodes:
                    print('\nFILENAME:', doc.metadata['filename'].split('/')[-1] )
                    print('SIMILARITY:', doc.score)
                    print('CHUNK:', doc.text )
                    print('---')

        except EOFError:
            print ("Goodbye!")
            break
