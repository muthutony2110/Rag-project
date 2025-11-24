# 🧠 RAG API Service (Retrieval-Augmented Generation)

A lightweight RAG-based API that accepts documents, indexes them using embeddings + FAISS, and answers user questions using retrieval + DeepSeek LLM.

This project satisfies all task requirements:

- Document upload + text extraction + chunking  
- Embeddings using **BGE-small v1.5**  
- Local vector store (**FAISS**)  
- Query endpoint with retrieval + DeepSeek LLM  
- **LangChain LCEL pipeline** (retriever → LLM → output)  
- Citations included  
- Safety checks  
- Clean and production-like code  

---

# 🚀 Features

## ✔ Document Upload
- Accepts **PDF, TXT, Markdown**
- Extracts text
- Splits into chunks
- Embeds using **BAAI/bge-small-en-v1.5**
- Stores vectors in **FAISS**
- Automatically **deletes old vector DB** before indexing a new file  
  (only the latest document is used)

---

## ✔ Query Endpoint
- Retrieves relevant chunks from FAISS
- Sends them to **DeepSeek Chat** (OpenRouter)
- Produces final answer using only the document context
- Includes:
  - `answer`
  - `citations`
  - `confidence`

---

# 📦 Project Structure

