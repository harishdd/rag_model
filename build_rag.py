from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

class RAG:
    def __init__(self) -> None:
        self.pdf_folder_path = os.getenv('SOURCE_DATA')  # Path to PDF files
        self.emb_model_path = os.getenv('EMBED_MODEL')  # Path to embedding model
        self.vector_store_path = os.getenv('VECTOR_STORE')  # Path to Chroma DB

        # Load embedding model
        self.emb_model = self.get_embedding_model(self.emb_model_path)

    def load_docs(self, path: str):
        print(f"Loading documents from: {path}")
        loader = PyPDFDirectoryLoader(path)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        return docs
    
    def get_embedding_model(self, emb_model):
        print(f"Loading embedding model: {emb_model}")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings_model = HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embeddings_model
    
    def split_docs(self, docs):
        print(f"Splitting {len(docs)} documents")
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        print(f"Split into {len(documents)} chunks")
        return documents
    
    def populate_vector_db(self):
        """Load documents, split them, and store embeddings in ChromaDB."""
        docs = self.load_docs(self.pdf_folder_path)
        if not docs:
            print("No documents found. Check your PDF folder path.")
            return
        
        documents = self.split_docs(docs)
        if not documents:
            print("Document splitting failed.")
            return
        
        print("Populating ChromaDB...")
        db = Chroma.from_documents(documents,
                                   embedding=self.emb_model,
                                   persist_directory=self.vector_store_path)
        db.persist()
        print("✅ ChromaDB populated successfully!")
    
    def load_vector_db(self):
        """Load embeddings from disk"""
        print("Loading ChromaDB from disk...")
        db = Chroma(persist_directory=self.vector_store_path, embedding_function=self.emb_model)
        return db
    
    def get_retriever(self):
        return self.load_vector_db().as_retriever()  

# ✅ Initialize and populate the vector database
rag = RAG()
rag.populate_vector_db()

# 🔍 Check if documents are stored correctly
db = rag.load_vector_db()
num_docs = db._collection.count()
print(f"Number of documents in ChromaDB: {num_docs}")

if num_docs == 0:
    print("⚠️ Warning: No documents found in ChromaDB. Check your document loading and processing.")
