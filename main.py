from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI()

class Question(BaseModel):
    question: str

qa_chain = None
initialized = False

def initialize_model():
    global qa_chain, initialized
    if qa_chain is None:
        try:
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.from_documents(split_docs, embeddings)
            retriever = vectordb.as_retriever()
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            initialized = True
            print("✅ Modelo e vetores carregados com sucesso.")
        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")

@app.post("/ask")
def ask(q: Question):
    global initialized

    # Mensagem de boas-vindas apenas quando receber "__init__"
    if q.question.strip() == "__init__":
        if not initialized:
            initialize_model()  
        return {
            "answer": "Olá! Sou seu assistente financeiro. Como posso te ajudar hoje?",
            "sources": []
        }

    if qa_chain is None:
        initialize_model()

    result = qa_chain({"query": q.question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:300] + "..."
            } for doc in result["source_documents"]
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
