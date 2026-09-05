import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="HF RAG Chat", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #0b1220;
        color: #e5e7eb;
    }
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
    }
    .stChatMessage {
        padding: 0.8rem 0.9rem;
        border-radius: 18px;
        margin: 0.4rem 0;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: rgba(59,130,246,0.16);
        border-color: rgba(96,165,250,0.25);
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background: rgba(255,255,255,0.02);
    }
    [data-testid="stChatMessageContent"] {
        color: #e5e7eb;
    }
    .stTextInput > div > div > input {
        background: #111827;
        color: #f9fafb;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
    }
    .stButton > button {
        border-radius: 12px;
        background: #2563eb;
        color: white;
        border: none;
    }
    h1, h2, h3 {
        color: #f9fafb;
    }
    .caption {
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Client for Google Gemini API initialization from the OpenAI class
client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# NEW FREE BGE EMBEDDINGS SYSTEM (Runs locally, automatically handles downloading)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Connect to the new BGE-compatible collection
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="chess_tactics_hf",
)


@st.cache_resource(show_spinner=False)
def get_clients():
    return OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


client = get_clients()


def generate_answer(user_input: str) -> str:
    search_results = vector_db.similarity_search(user_input)

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

    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content


def stream_text(text: str, delay: float = 0.01):
    """Generate the text with a typing effect."""
    placeholder = st.empty()
    displayed = ""

    for char in text:
        displayed += char
        placeholder.markdown(displayed)
        time.sleep(delay)


if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    "<div style='padding: 0.5rem 0 1rem 0;'><h1 style='margin:0; font-size:2.2rem;'>PDF Q&A</h1><div class='caption'>Semantic search from Qdrant + Gemini</div></div>",
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="🐇" if message["role"] == "assistant" else "🧔‍♂️",
    ):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the PDF...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking🐇..."):
        answer = generate_answer(user_input)
        with st.chat_message("assistant", avatar="🐇"):
            stream_text(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
