from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag_service import RAGService
from app.models import QueryRequest

app = FastAPI(title="RAG FastAPI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service (will raise if env/config is missing)
try:
    rag = RAGService()
except Exception as e:
    # raise at import time so server won't start silently with broken config
    raise RuntimeError(f"Failed to initialize RAGService: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF / TXT / MD file. This will wipe existing vector store and index only this file.
    """
    allowed = ["application/pdf", "text/plain", "text/markdown"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    try:
        content = await file.read()
        chunks = rag.process_document(file.filename, content, file.content_type)
        return {"message": "File indexed successfully", "chunks_indexed": chunks}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # LLM/network issues during indexing (unlikely) — surface friendly msg
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        # Hide raw trace for safety, return friendly message
        raise HTTPException(status_code=500, detail="Error processing file")


@app.post("/query")
async def query_api(query: QueryRequest):
    """
    Query the indexed document. Returns answer, confidence, and citations.
    """
    try:
        result = rag.answer_query(query.query)
        return result
    except ValueError as e:
        # user-caused (no docs / no relevant chunks)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # external provider (LLM) problems
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        # generic
        raise HTTPException(status_code=500, detail="Internal error while answering the query")
