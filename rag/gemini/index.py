# import os
# from dotenv import load_dotenv
# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
# load_dotenv()  # Load environment variables from .env file

# pdf_path = Path(__file__).parent / "final_thesis.pdf"

# #load this file using the PyPDFLoader
# loader = PyPDFLoader(file_path=pdf_path)
# docs = loader.load()

# #Splits the docs in to smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
# chunks = text_splitter.split_documents(docs)

# #Vector Embeddings
# # Point LangChain's OpenAI client to the Google Gemini endpoint
# embeddings = OpenAIEmbeddings(
#     model="gemini-embedding-001",  # Use the valid Gemini embedding model name
# )

# #Vector Store
# vector_store = QdrantVectorStore.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     url="http://localhost:6333",  # Qdrant local server URL 
#     collection_name="thesis_collection"  # Name of the collection in Qdrant)
# )

import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# IMPORT THE GOOGLE GENAI EMBEDDINGS CLASS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()  # Load environment variables from .env file

pdf_path = Path(__file__).parent / "final_thesis.pdf"

# Load this file using the PyPDFLoader
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Splits the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(docs)

# NATIVE GOOGLE EMBEDDINGS SYSTEM
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.getenv("GOOGLE_API_KEY"),  # Google AI Studio API key
    embedding_dimensions=768  # Set the embedding dimensions to match the Gemini model
)

# VECTOR STORE
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",  # Qdrant local server URL 
    collection_name="thesis_collection"  # Name of the collection in Qdrant
)
