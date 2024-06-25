# the model that will be used within our application
# create a class that holds the  necessary functions to get embeddings, 
#     generate responses, have conversational chat, and more

from typing import List, Dict

class RAGModel:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, index_name: str):
        """
        initialize the model with necessary names and api keys
        """
        pass

    def get_embeddings(self, text: str, model: str) -> List[float]:
        """
        this will embed user query so it will be used to find similar documents in db

        Parameters: 
        text (str): input text to be embedded
        model (str): the embedding model to use (this will be local for now...)

        Returns: 
        List[float]: the embedding vector 
        """
        pass