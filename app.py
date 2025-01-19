import streamlit as st
import os
import time
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Custom CSS for the sidebar
sidebar_css = """
<style>
/* Change the background color of the sidebar */
[data-testid="stSidebar"] {
    background-color: #add8e6; /* Light blue */
    color: black; /* Text color */
}

/* Customize the font and padding of the sidebar elements */
[data-testid="stSidebar"] .css-1d391kg {
    font-size: 18px;
    padding: 10px;
}

[data-testid="stAppViewContainer"] {
    background-color: #f0f8ff;
}
[data-testid="stHeader"] {
    background-color: #f0f8ff;
}
[data-testid="stHeadingWithActionElements"] {
    text-align: center
}
[data-testid="stMarkdownContainer"] {
    font-size: 20px
}
[data-testid="stMarkdownContainer"] h1 {
    text-align: center;
}
[data-testid="stButton"] {
    text-align: center;
}
[data-testid="stButton"] button {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    font-weight: 600;
    background-color: green;
    color: white;
    transition: transform 0.3s ease;
    margin-top: 20px;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: white;
    color: white;
    transform: scale(1.1);
}
textarea::placeholder {
    font-size: 20px;
}
[data-testid="stVerticalBlock"] {
    gap: 0;
}
[data-testid="stSpinner"] {
    margin-top: 60px;
}
[data-testid="stSpinner"] p {
    font-size: 28px;
}
.stMainBlockContainer [data-testid="stElementContainer"] [data-testid="stTextInputRootElement"] {
    height: 80px;
}
.stMainBlockContainer [data-testid="stElementContainer"] [data-testid="stTextInputRootElement"] input {
    padding: 25px;
    height: 50px;
    font-size: 24px;
    box-sizing: border-box;
}
</style>
"""

# Inject the custom CSS
st.markdown(sidebar_css, unsafe_allow_html=True)

st.title("Search Tool 🔎")
st.sidebar.title("Enter URLs for Context & press GO")

urls = []
for i in range(3):
    url = st.sidebar.text_input("", placeholder=f"Context URL {i+1}")
    urls.append(url)

clicked = st.sidebar.button("GO")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
processed = 1
file_path = "faiss_store_openai.pkl"

if clicked:
    # processed = 0
    loader = UnstructuredURLLoader(urls=urls)
    with st.spinner("Loading data from the URLs 🏃‍➡️"):
        data = loader.load()
        time.sleep(2)
    with st.spinner("Recursively splitting the text 🏃‍➡️🏃‍➡️"):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=500
        )
        docs = text_splitter.split_documents(data)
        time.sleep(2)
    with st.spinner("Preparing embeddings and vector database 🏃‍➡️🏃‍➡️🏃‍➡️"):
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        time.sleep(2)

    with open(file_path, "wb") as f:
        vectorstore_openai.save_local("faiss_index")
        # processed = 1

if (processed == 1):
    query = main_placeholder.text_input(
        "", 
        placeholder="Message Search Tool"
    )

    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = FAISS.load_local(
                    "faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True
                )
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=vectorstore.as_retriever()
                )
                result = chain(
                    {"question": query}, 
                    return_only_outputs=True
                )

                st.header("Reply")
                st.write(result["answer"])

                # Displaying sources
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")

                    for source in sources_list:
                        st.write(source)