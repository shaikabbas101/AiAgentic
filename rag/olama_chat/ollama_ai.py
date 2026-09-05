import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from ollama import chat
from langchain_ollama import OllamaEmbeddings

load_dotenv()

embedding_model = OllamaEmbeddings(
    model="all-minilm:22m",
)

qclient = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    check_compatibility=False,
)  # Qdrant local server URL

vector_db = QdrantVectorStore(
    embedding=embedding_model,
    client=qclient,  # Use the Qdrant client for connection
    collection_name="thesis_hf_cloud",  # Re-ingested collection name for 384 dimensions
)


while True:
    # Take the user input and query the vector store
    user_input = input("Ask Something: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break
    # Relevant chunks from the vector db
    search_results = vector_db.similarity_search(
        user_input, k=3
    )  # k is the number of relevant chunks to retrieve

    context = "\n\n\n".join(
        [
            f"Page Content:{result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nFile Location: {result.metadata.get('source', 'N/A')}"
            for result in search_results
        ]
    )

    SYSTEM_PROMPT = f"""You are a helpful assistant who answers questions based on the available context
    retrieved from a PDF file along with page contents and page numbers.

    You only answer the user based on the following context and navigate the
    user to the open the right page number to know more.

    Context:
    {context}
    """
    print(f"🧔‍♂️: {user_input}\n")
    print("Agent thinking...\n")
    response = chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        # stream=True,
    )
    print(f"🤖: {response.message.content}")
    # for chunk in response:
    #     print(chunk["message"]["content"], end="", flush=True)
