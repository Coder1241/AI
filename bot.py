# file: app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
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
llm = Ollama(model="llama3.1:latest")

script_dir = os.path.dirname(os.path.realpath(__file__))
docs_path = os.path.join(script_dir, "BotWeb")
loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)
print(script_dir)
emb = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})
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
