import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Load documents (PDFs inside docs/ folder)
loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# 3. Create embeddings & vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever()

# 4. Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    retriever=retriever
)

# 5. Ask questions interactively
while True:
    query = input("Ask a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print("Answer:", result)
