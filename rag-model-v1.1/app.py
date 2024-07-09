import streamlit as st

# Setup streamlit page
st.set_page_config(page_title="CFI App", page_icon="🛩️")

# Initialize question answering pipeline
# @st.cache_resource
    # here is where a pipeline from hugging face could be implemented
    # learn more about pipelines and their usage on hugging face's site

# Streamlit app
st.title("CFI Application")

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