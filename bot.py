# bot.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from charset_normalizer import from_path
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

app = FastAPI(title="RAG Chat", version="0.1.0")

# CORS: let the middleware handle preflight automatically
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # or restrict to your frontend origin(s)
    allow_credentials=False, # keep False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "BotWeb")
if not os.path.isdir(DOCS_DIR):
    DOCS_DIR = SCRIPT_DIR

class RobustTextLoader(BaseLoader):
    """Load text using detected encoding and replace undecodable bytes."""
    def __init__(self, path: str):
        self.path = path

    def load(self):
        result = from_path(self.path).best()
        encoding = result.encoding if result else "utf-8"
        with open(self.path, encoding=encoding, errors="replace") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.path})]

# Build RAG artifacts
loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.txt",
    loader_cls=RobustTextLoader,
    show_progress=True,
)
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)

emb = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True},
)
vs = FAISS.from_documents(chunks, emb)

llm = OllamaLLM(model="llama3.1:latest")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vs.as_retriever(search_kwargs={"k": 5}),
)

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"ok": True, "docs_dir": DOCS_DIR, "chunks": len(chunks)}

# No need for a manual OPTIONS handler—CORS middleware handles it.
@app.post("/chat")
def chat(query: Query):
    answer = qa.run(query.question)
    return {"answer": answer}

# IMPORTANT: run the in-memory app object, not "app:app"
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
