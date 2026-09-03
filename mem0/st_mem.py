import os
import time
import json
import warnings
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

os.environ["MEM0_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()

st.set_page_config(
    page_title="Rabbit AI",
    page_icon="🐇",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_secret(name: str):
    """Read Streamlit Cloud secrets, with .env fallback for local development."""
    return st.secrets.get(name) or os.getenv(name)


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
required_secrets = {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "NEO4J_URI": get_secret("NEO4J_URI"),
    "NEO4J_USERNAME": get_secret("NEO4J_USERNAME"),
    "NEO4J_PASSWORD": get_secret("NEO4J_PASSWORD"),
    "NEO4J_DATABASE": get_secret("NEO4J_DATABASE"),
    "QDRANT_URL": get_secret("QDRANT_URL"),
    "QDRANT_API_KEY": get_secret("QDRANT_API_KEY"),
}
missing_secrets = [name for name, value in required_secrets.items() if not value]
if missing_secrets:
    st.error("Missing secrets: " + ", ".join(missing_secrets))
    st.stop()

from mem0 import Memory
from openai import OpenAI
from qdrant_client import QdrantClient

warnings.simplefilter("ignore", DeprecationWarning)

# Vector store configuration
vector_store_config = {
    "embedding_model_dims": 768,
    "collection_name": "mem_users",
}

if QDRANT_URL:
    qdrant_client = QdrantClient(
        url=QDRANT_URL.rstrip("/"),
        api_key=QDRANT_API_KEY,
        timeout=60,
        prefer_grpc=False,
        check_compatibility=False,
    )
    vector_store_config.update(
        {
            "client": qdrant_client,
        }
    )
else:
    qdrant_client = QdrantClient(
        host="localhost",
        port=6333,
        timeout=60,
        prefer_grpc=False,
        check_compatibility=False,
    )
    vector_store_config.update(
        {
            "client": qdrant_client,
        }
    )

# LLM and Embedder configuration
config = {
    "version": "v1.1",
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": GOOGLE_API_KEY,
            "model": "gemini-3.5-flash-lite",
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
            "url": get_secret("NEO4J_URI"),
            "username": get_secret("NEO4J_USERNAME"),
            "password": get_secret("NEO4J_PASSWORD"),
            "database": get_secret("NEO4J_DATABASE"),
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": vector_store_config,
    },
}


@st.cache_resource(show_spinner=False)
def get_clients():
    return Memory.from_config(config), OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


try:
    memory_client, client = get_clients()
except Exception as error:
    st.error(
        "Could not initialize Mem0. Check Qdrant Cloud URL/API key and Neo4j "
        "secrets. Details: " + str(error)
    )
    st.stop()

styles_path = Path(__file__).with_name("styles.css")
st.markdown(
    f"<style>{styles_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


# with st.sidebar:
#     st.markdown(
#         '<div class="brand"><div class="brand-mark">🐇</div>'
#         '<div><div class="brand-name">Rabbit</div>'
#         "</div>"
#         "",
#         unsafe_allow_html=True,
#     )
st.title("🐰 Rabbit Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="🐇" if message["role"] == "assistant" else "🧔‍♂️",
    ):
        st.markdown(message["content"])


user_query = st.chat_input("Write a message to Rabbit...")


def stream_text(text: str, delay: float = 0.01):
    """Generate the text with a typing effect."""
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(displayed)
        time.sleep(delay)


if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="user"):
        st.markdown(user_query)

    with st.spinner("Thinking🐇..."):
        search_response = memory_client.search(query=user_query, user_id="abbas")
        recalled = search_response.get("results", [])
        memories = [memory.get("memory", "") for memory in recalled]
        system_message = f"""You are Rabbit, a warm and helpful AI assistant. Answer politely and clearly.
        -Note: If user corrects any data or information, you should acknowledge it and update your knowledge memory graph accordingly.
                     Use these relevant memories of the user and give the best answer you can the below are the memories of the user to help you answer the question.
                    - Memories:
                    {json.dumps(memories)}"""

        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            messages=[
                {"role": "system", "content": system_message},
                *st.session_state.messages,
            ],
        )

    ai_response = response.choices[0].message.content or ""
    with st.chat_message("assistant", avatar="assistant"):
        stream_text(ai_response)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    memory_client.add(
        messages=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response},
        ],
        user_id="abbas",
    )

    # st.rerun()
