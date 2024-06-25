# the model that will be used within our application
# create a class that holds the  necessary functions to get embeddings, 
#     generate responses, have conversational chat, and more

from typing import List, Dict
from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class RAGModel:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, index_name: str):
        """
        initialize the model with necessary names and api keys
        """

    def connect_pinecone(self, pinecone_api_key: str, index_name: str):
        pc = Pinecone(api_key=pinecone_api_key)

        cloud = "aws"
        region = "us-east-1"

        spec = ServerlessSpec(cloud=cloud, region=region)

        # check if index already exists
        if index_name not in pc.list_indexes().names():
            # if does not exist, create index
            pc.create_index(
                index_name,
                dimension=768,
                metric="cosine",
                spec=spec,
            )

        # connect to index
        return pc.Index(index_name)

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

    def get_prompt(self, query: str, index, limit: int = 3750):
        """
        This will embed the input text then perform a similarity search
        in Pinecone to find similar documents for context

        Parameters:
        text (str): user query to find context for

        Returns: 
        str: context from multiple documents formatted into one prompt
        """
        
        ### Embed query
        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                              device="cpu") # choose device to load model to
        
        # make sure model set to CPU
        embedding_model.to("cpu")

        # get relevant documents
        res = embedding_model.encode(query)
        xq = res.tolist()
        res = index.query(vector=xq, top_k=2, include_metadata=True) # returns top k context

        contexts = [
            x["metadata"]["text"] for x in res["matches"]
        ]

        # build prompt with retrieved context
        prompt_start = (
            "Answer the question based on the context below. \n\n"+
            "Context: \n"
        )
        prompt_end = (
            f"\n\nQuestion: {query}\nAnswer:"
        )

        # append context until hitting limit
        for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= limit:
                prompt = (
                    prompt_start + 
                    "\n\n---\n\n".join(contexts[:i]) +
                    prompt_end
                )
                break
            elif i == len(contexts)-1:
                prompt = (
                    prompt_start + 
                    "\n\n---\n\n".join(contexts) + 
                    prompt_end
                )

        return prompt

    def generate_response(self, context: str, engine: str = "gpt-3.5-turbo", max_tokens: int = 500) -> str: 
        """
        Generate a response to the user query relying on the context provided

        Parameters: 
        context (str): The context retrieved from Pinecone
        prompt (str): The user input prompt
        engine (str): model that will be used to generate
        max_tokens (int): max number of tokens for generated response

        Returns: 
        str: the generated response
        """
        client = OpenAI()

        # define query as a list of message objects
        query = [
            {"role": "system", "content": "You are a helpful flight instructor that will use the context below from the user to answer their question as accurate to your ability."},
            {"role": "user", "content": context}
        ]

        completion = client.chat.completions.create(
            model=engine,
            messages=query
        )

        return completion.choices[0].message.content


    def chat(self, input_text: str) -> str:
        """
        Handle chat interaction by getting embeddings, querying Pinecone, and generating response.

        Parameters: 
        input_text (str): the user's input text

        Returns: 
        str: the generated response from the model
        """

        index = self.connect_pinecone(pinecone_api_key="", index_name="rag-retrieval-v1")
        
        prompt = self.get_prompt(query=input_text, index=index)

        response = self.generate_response(context=prompt)

        return response
