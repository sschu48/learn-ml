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

def retrieve(query, embedding_model, index):
    res = embedding_model.encode(query)
    xq = res.tolist()
    res = index.query(vector=xq, top_k=2, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    limit = 3750

    # build prompt with the retrieved context included
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

# init llm
@st.cache_resource
def init_llm():
    return OpenAI()

# generate answer
def generate_answer(prompt):
    client = init_llm()

    query = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=query
    )

    return completion.choices[0].message.content


# Connect to Pinecone
index = init_pinecone()
pinecone_stats = str(index.describe_index_stats()) # view index stats

# Initialize embedding model
embedding_model = init_embedding_model()

# Input for context
context = st.text_area("Enter context here: ", height=200)

# Input for question
question = st.text_input("Enter your question: ")

if st.button("Get Answer"): 
    if question: 
        with st.spinner("Thinking..."):
            prompt = retrieve(question, embedding_model, index)
            result = generate_answer(prompt)

        st.success("Here is your result: ")
        st.write(result)
    else:
        st.warning("Please provide context and a question")

# Add info about app
st.sidebar.header("About")
st.sidebar.info(
    "This application is under development"
)
st.sidebar.info(pinecone_stats)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Cachet Aviation")