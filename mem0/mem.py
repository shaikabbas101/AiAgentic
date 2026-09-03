import os

os.environ["MEM0_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

import warnings
from dotenv import load_dotenv
load_dotenv()
from mem0 import Memory
from openai import OpenAI
import json

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
api_key= GOOGLE_API_KEY,
base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

warnings.simplefilter("ignore", DeprecationWarning)

config = {
    "version": "v1.1",
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": GOOGLE_API_KEY,
            "model": "gemini-3.1-flash-lite",
            },
        },
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": GOOGLE_API_KEY,
            "model": "gemini-embedding-001",
            "embedding_dims": 768,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.getenv("NEO4J_URI"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD"),
            "database": os.getenv("NEO4J_DATABASE"),
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host":"localhost",
            "port": 6333,
            "embedding_model_dims": 768,
            "collection_name": "mem_users",
        },
    },

}

memory_client = Memory.from_config(config)

while True:
    user_query = input(">: ")

    if user_query.strip().lower() in {"exit", "quit"}:
        break

    search_response = memory_client.search(query=user_query,user_id= "grag")
    memories = [f"ID: {mem.get('id')}\nMemory: {mem.get('memory')}" 
           for mem in search_response.get("results", [])]
   
    SYSTEM_MESSAGE =f"""You are a helpful Ai Expert assistant named rabbit.You need answer the user very polite and friendly way.
    Here is the context about the user from his memories:
    {json.dumps(memories)}
    """

    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_query}
        ]
    )

    ai_response = response.choices[0].message.content
    print("AI: ", ai_response)

    memory_result = memory_client.add(
        messages=[
            {"role": "user", "content": user_query}, 
            {"role": "assistant","content": ai_response}
        ],
        user_id="grag",
    ) 
    print("Memory saved:", json.dumps(memory_result))