import os
from  dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# load dotenv
load_dotenv()

# Setup streamlit page
st.set_page_config(page_title="CFI App", page_icon="🛩️")

# Streamlit app
st.title("CFI Application")

# init Pinecone
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY)")) # configure client

    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    spec = ServerlessSpec(cloud=cloud, region=region)

    index_name = "rag-retriever-v2"

    # check if index already exists
    if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=768,
        metric="cosine",
        spec=spec,
    )
    #connect to index
    index = pc.Index(index_name)
    return index

# init embedding model
@st.cache_resource
def init_embedding_model():
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                          device="cpu")
    embedding_model.to("cpu")

    return embedding_model

# init llm
@st.cache_resource
def init_llm():
    return OpenAI()

# Input for context
context = st.text_area("Enter context here: ", height=200)

# Input for question
question = st.text_input("Enter your question: ")

if st.button("Get Answer"): 
    if context and question: 
        with st.spinner("Thinking..."):
            result = "Context: " + context + "\n" + "Question: " + question

        st.success("Here is your result: ")
        st.write(result)
    else:
        st.warning("Please provide context and a question")

# Add info about app
st.sidebar.header("About")
st.sidebar.info(
    "This application is under development"
)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Cachet Aviation")