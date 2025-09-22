import os
import streamlit as st
from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sidebar
st.sidebar.title("📄 RAG Document QA")
st.sidebar.write("Upload documents into the `docs/` folder before running.")

# 1. Load documents
@st.cache_resource
def load_and_index():
    loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
        retriever=retriever
    )
    return qa

qa = load_and_index()

# 2. Chat interface
st.title("🔎 RAG Q&A with Your Docs")

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
    st.success(answer)
