# from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
load_dotenv()  # Load environment variables from .env file
from openai import OpenAI

# Client for Google Gemini API intialization from the OpenAI class
client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# NATIVE GOOGLE EMBEDDINGS SYSTEM
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),  # Google AI Studio API key
    embedding_dimensions=768  # Set the embedding dimensions to match the Gemini model
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",  # Qdrant local server URL
    collection_name="thesis_collection"  # Name of the collection in Qdrant
)
while True:
    #Take the user input and query the vector store
    user_input = input("Ask Something: ")

    # Relevant chunks from the vector db
    search_results = vector_db.similarity_search(user_input, k=5) # here k is the number of relevant chunks to retrieve

    context = "\n\n\n".join([
        f"Page Content:{result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
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

    print(f"🤖: {response.choices[0].message.content}")