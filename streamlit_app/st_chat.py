import json
import time

import streamlit as st

# from utils import AGENT_SYSTEM_PROMPT
# from modals import MyOutputFormat
# from tools import available_tools
# from ai_server import client


def stream_text(text: str, delay: float = 0.01):
    """Generate the text with a typing effect."""
    placeholder = st.empty()
    displayed = ""

    for char in text:
        displayed += char
        placeholder.markdown(displayed)
        time.sleep(delay)


st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🐰 Rabbit Agent")

st.markdown(
    """
    <style>
    .stChatMessage {
        font-size: 20px !important;
    }
    .stChatMessage p {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 24px !important;
        min-height: 50px !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        font-size: 24px !important;
    }
    div[data-testid="stBlockContainer"] {
        max-width: 800px !important;
        padding: 20px 30px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    st.session_state.llm_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]


def show_msgs(role: str, content: str, avatar: str = None, stream: bool = False):
    """Append a message to chat history and display it."""
    if role == "assistant":
        avatar = "🐰"

    st.session_state.messages.append(
        {"role": role, "content": content, "avatar": avatar}
    )

    with st.chat_message(role, avatar=avatar):
        if stream:
            stream_text(content)
        else:
            st.markdown(content)


for message in st.session_state.messages:
    if message["role"] in ("user", "assistant"):
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.llm_messages.append({"role": "user", "content": prompt})
    show_msgs("user", prompt)

    final_answer = ""

    with st.spinner("Agent is thinking..."):
        for _ in range(12):
            try:
                response = client.chat.completions.parse(
                    model="gemini-3.5-flash-lite",
                    response_format=MyOutputFormat,
                    messages=st.session_state.llm_messages,
                )
            except Exception as e:
                final_answer = f"Error calling model: {e}"
                break

            raw = response.choices[0].message.content
            parsed = response.choices[0].message.parsed
            st.session_state.llm_messages.append({"role": "assistant", "content": raw})

            if parsed.step == "START":
                # show_msgs("assistant", f"🔥 {parsed.content}")
                st.session_state.llm_messages.append(
                    {"role": "user", "content": "Continue."}
                )

            elif parsed.step == "PLAN":
                # show_msgs("assistant", f"🧠 {parsed.content}")
                st.session_state.llm_messages.append(
                    {"role": "user", "content": "Continue."}
                )

            elif parsed.step == "TOOL":
                tool_name = parsed.tool
                tool_input = parsed.input or {}
                # show_msgs("assistant", f"🛠️ `{tool_name}` with `{tool_input}`")

                try:
                    result = available_tools[tool_name](**tool_input)
                except Exception as e:
                    result = f"Error: {e}"

                show_msgs("assistant", f"📊 `{result}`")
                st.session_state.llm_messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "step": "OBSERVE",
                                "tool": tool_name,
                                "output": result,
                            }
                        ),
                    }
                )

            elif parsed.step == "OUTPUT":
                final_answer = parsed.content
                show_msgs("assistant", final_answer, stream=True)
                break

    if not final_answer:
        final_answer = "Sorry, I couldn't get a final answer."
        show_msgs("assistant", final_answer, stream=True)
