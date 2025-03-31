import uvicorn
import os
from inference import predict_rag
from fastapi import FastAPI
from models import Request, Response
from dotenv import load_dotenv
from datetime import datetime
from build_rag import RAG  # Import the RAG class

load_dotenv()

app = FastAPI()
rag = RAG()  # Initialize RAG to interact with ChromaDB


@app.post("/", response_model=Response)
async def predict_api(prompt: Request):
    """API to get predictions from the RAG model"""
    response = predict_rag(prompt.prompt)  
    return Response(response=response, timestamp=str(datetime.utcnow()))

@app.get("/health")
async def health_check():
    """API to check if the service is running"""
    return {"status": "ok", "message": "RAG API is running!"}

@app.get("/documents")
async def get_document_count():
    """API to check the number of documents stored in ChromaDB"""
    db = rag.load_vector_db()  
    num_docs = db._collection.count()
    return {"total_documents": num_docs}

@app.delete("/documents")
async def delete_all_documents():
    """API to delete all stored documents from ChromaDB"""
    db = rag.load_vector_db()
    db.delete_collection()  # This will clear the database
    return {"message": "All documents deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host=os.getenv("UVICORN_HOST"),  
        port=int(os.getenv("UVICORN_PORT"))
    )
