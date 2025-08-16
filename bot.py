"""FastAPI application that exposes a retrieval-augmented chatbot.

This module wires together a local Ollama model with a FAISS vector store so
that questions sent to the ``/chat`` endpoint are answered using both the model
and the documents found in ``knowledge_base.txt``.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from charset_normalizer import from_path
import os
# ----- FastAPI setup -----
app = FastAPI()

# Allow local frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----- RAG bot setup -----


def auto_loader(path: str) -> TextLoader:
    """Detect file encoding and return a TextLoader using that encoding.

    Some of the knowledge-base files include characters that are not decodable
    with the default Windows CP1252 codec. ``charset_normalizer`` inspects each
    file and suggests an encoding so that all text can be read safely.
    """

    result = from_path(path).best()
    encoding = result.encoding if result else "utf-8"
    return TextLoader(path, encoding=encoding, errors="replace")


llm = OllamaLLM(model="llama3.1:latest")

script_dir = os.path.dirname(os.path.realpath(__file__))
# Load all ``.txt`` files in the repository directory, such as ``knowledge_base.txt``.
docs_path = script_dir
loader = DirectoryLoader(
    docs_path,
    glob="**/*.txt",
    loader_cls=auto_loader,
    show_progress=True,
)
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)

emb = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True},
)
vs = FAISS.from_documents(chunks, emb)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vs.as_retriever(search_kwargs={"k": 5})
)

# ----- API endpoint -----
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = qa.run(query.question)
    return {"answer": answer}
