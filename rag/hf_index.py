import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

load_dotenv()  # Load environment variables from .env file

pdf_path = Path(__file__).parent / "tactics_course.pdf"

# Load this file using the PyPDFLoader
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Splits the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(docs)

# NEW FREE BGE EMBEDDINGS SYSTEM (Runs locally, automatically handles downloading)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},  # Change to 'cuda' if you have an Nvidia GPU
    encode_kwargs={
        "normalize_embeddings": True
    },  # Essential for BGE cosine distance calculation
)

# VECTOR STORE
vector_store = QdrantVectorStore.from_documents(
    # url=os.environ["QDRANT_URL"], # for Qdrant URL cloud
    # api_key=os.environ["QDRANT_API_KEY"], # For cloud Qdrant API key
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",  # Qdrant local server URL
    collection_name="chess_tactics_hf",  # Name of the collection in Qdrant
)

print("✅ Vector store created successfully with BGE embeddings and Qdrant.")
