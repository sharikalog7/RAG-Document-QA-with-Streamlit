# RAG Document QA with Streamlit

A **Retrieval-Augmented Generation (RAG) system** for querying and interacting with your document collection.  
This project uses **LangChain** for document processing, **ChromaDB** for semantic search, and provides a **Streamlit web interface** for interactive Q&A.

---

## 🚀 Features

- 📂 Load and query documents (PDF, text, etc.) from a folder.  
- 🔎 Semantic search with vector embeddings for accurate retrieval.  
- 💬 Chat-style interface for asking questions about your documents.  
- ⚡ Supports both **OpenAI API** and **local LLMs** (GPT4All or HuggingFace models) — works offline with local models.  
- 🖥 Easy to run locally or deploy on **Streamlit Cloud**.

---

## 🧰 Tech Stack

- Python 3.10+  
- [LangChain](https://www.langchain.com/) (`langchain`, `langchain-community`, `langchain-openai`)  
- [ChromaDB](https://www.trychroma.com/) (vector store)  
- [Streamlit](https://streamlit.io/) (UI)  
- [OpenAI API](https://platform.openai.com/) or **local LLMs** (GPT4All / HuggingFace models)  

---

## Notes
Ensure your network allows connections to OpenAI API if using cloud LLM.

Cache embeddings to speed up repeated runs.

Streamlit Cloud is a convenient deployment option if local network blocks API calls.

## License
MIT License


