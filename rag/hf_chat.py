import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_qdrant import QdrantVectorStore
load_dotenv()  # Load environment variables from .env file
from openai import OpenAI

# Client for Google Gemini API initialization from the OpenAI class
client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# NEW FREE BGE EMBEDDINGS SYSTEM (Runs locally, automatically handles downloading)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have an Nvidia GPU
    encode_kwargs={'normalize_embeddings': True} # Essential for BGE cosine distance calculation
)

# Connect to the new BGE-compatible collection
# Note: You must run your ingestion/PDF-parsing script first using this new collection name
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",  # Qdrant local server URL
    collection_name="thesis_collection"  # Re-ingested collection name for 384 dimensions
)

while True:
    # Take the user input and query the vector store
    user_input = input("Ask Something: ")
    if user_input.strip().lower() in ['exit', 'quit']:
        break

    # Relevant chunks from the vector db
    search_results = vector_db.similarity_search(user_input) # k is the number of relevant chunks to retrieve

    context = "\n\n\n".join([
        f"Page Content:{result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nFile Location: {result.metadata.get('source', 'N/A')}"
        for result in search_results
    ])

    SYSTEM_PROMPT = f"""You are a helpful assistant who answers questions based on the available context
    retrieved from a PDF file along with page contents and page numbers.

    You only answer the user based on the following context and navigate the
    user to the open the right page number to know more.

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )

    print(f"🤖: {response.choices[0].message.content}\n")
