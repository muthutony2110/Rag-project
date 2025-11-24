import os
import shutil
import tempfile
import requests
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# -------- SAFE RETRIEVER WRAPPER --------
def safe_invoke_retriever(retriever, query, k=4):
    """Safe wrapper that supports: get_relevant_documents(), retrieve(), invoke()."""
    
    # 1) Standard LangChain
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)

    # 2) Newer LangChain
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query)

    # 3) LCEL invoke()
    if hasattr(retriever, "invoke"):
        out = retriever.invoke(query)

        # may return list
        if isinstance(out, list):
            return out

        # may return dict
        if isinstance(out, dict):
            if "documents" in out:
                return out["documents"]
            if "results" in out:
                return out["results"]

        try:
            return list(out)
        except:
            raise RuntimeError("Retriever.invoke() returned unsupported structure")

    raise RuntimeError("Unsupported retriever API in this LangChain version")


class RAGService:
    def __init__(self):
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.vector_store_path = os.getenv("VECTORSTORE_PATH", "vector_store")

        self.llm_url = os.getenv("DEEPSEEK_API_URL")
        self.llm_key = os.getenv("DEEPSEEK_API_KEY")
        self.llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat")

        if not self.llm_key:
            raise EnvironmentError("Missing DEEPSEEK_API_KEY in .env")

        self.embedder = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        if os.path.exists(self.vector_store_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.vector_store_path,
                    self.embedder,
                    allow_dangerous_deserialization=True
                )
            except:
                self.vectorstore = None
        else:
            self.vectorstore = None

    # -------- Extract PDF --------
    def extract_pdf(self, pdf_bytes: bytes):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()

        doc = fitz.open(tmp.name)
        text = "\n".join([p.get_text() for p in doc])
        doc.close()
        os.remove(tmp.name)
        return text

    # -------- Process Document --------
    def process_document(self, filename, content, content_type):
        # delete old DB
        if os.path.exists(self.vector_store_path):
            shutil.rmtree(self.vector_store_path)

        os.makedirs(self.vector_store_path, exist_ok=True)

        # extract
        if content_type == "application/pdf":
            text = self.extract_pdf(content)
        else:
            text = content.decode("utf-8")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_text(text)

        docs = [
            Document(page_content=c, metadata={"filename": filename, "chunk": i})
            for i, c in enumerate(chunks)
        ]

        self.vectorstore = FAISS.from_documents(docs, self.embedder)
        self.vectorstore.save_local(self.vector_store_path)

        return len(chunks)

    # -------- Query --------
    def answer_query(self, query: str, k=4):
        if not self.vectorstore:
            raise ValueError("No documents uploaded.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        docs = safe_invoke_retriever(retriever, query, k)

        if not docs:
            raise ValueError("No relevant content found.")

        # format context
        context = "\n\n".join(
            f"Source: {d.metadata.get('filename')} | chunk {d.metadata.get('chunk')}\n{d.page_content}"
            for d in docs
        )

        prompt = (
            "Use ONLY the context below to answer. If not found, say 'Not in document.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
        )

        answer = self.ask_llm(prompt)

        confidence = "high" if len(docs) >= 3 else "medium"

        return {
            "answer": answer,
            "confidence": confidence,
            "citations": [{"filename": d.metadata["filename"], "chunk": d.metadata["chunk"]} for d in docs],
        }

    # -------- LLM Call --------
    def ask_llm(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.llm_key}",
            "Content-Type": "application/json",
            "X-Title": "RAG-FastAPI",
        }

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }

        response = requests.post(self.llm_url, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"LLM error: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]
