import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from transformers import GenerationConfig, pipeline

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is missing. Add it to your .env or environment variables."
    )

# Local embedding model for Qdrant similarity search.
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Use the local Qdrant instance for semantic retrieval.
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="thesis_hf",
)

# Local generation pipeline. This avoids using an invalid OpenAI-compatible base URL.
text_generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B",
    tokenizer="HuggingFaceTB/SmolLM2-1.7B",
    device=-1,
    clean_up_tokenization_spaces=False,
)

generation_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    pad_token_id=text_generator.tokenizer.eos_token_id,
)

while True:
    user_input = input("Ask Something: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break

    search_results = vector_db.similarity_search(user_input, k=3)
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

    prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_input}\n\nAnswer:"
    response = text_generator(prompt, generation_config=generation_config)
    print(response[0]["generated_text"][len(prompt) :].strip())
