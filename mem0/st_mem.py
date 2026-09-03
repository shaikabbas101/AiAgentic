import os
import time

os.environ["MEM0_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

import json
import warnings
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from mem0 import Memory
from openai import OpenAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
            "collection_name": "mem_users",
        },
    },
}


@st.cache_resource(show_spinner=False)
def get_clients():
    return Memory.from_config(config), OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


memory_client, client = get_clients()

st.set_page_config(
    page_title="Rabbit AI",
    page_icon="🐇",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
                     Use these relevant memories of the user and give the best answer you can the below are the memories of the user to help you answer the question.
                    - Memories:
                    {json.dumps(memories)}"""

        response = client.chat.completions.create(
            model="gemini-3.1-flash-lite",
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
