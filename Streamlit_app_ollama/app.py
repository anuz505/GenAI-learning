from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_groq import ChatGroq

load_dotenv()

# Set LangSmith configs
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "ollama simple app"

# Streamlit UI
st.title("Chat with Gemma:2b")
st.write("⚠️ Remember to set a chat name first!")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with a charismatic and funny tone."),
    ("user", "{input}")
])

# Chat history management
if "history" not in st.session_state:
    st.session_state["history"] = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["history"]:
        st.session_state["history"][session_id] = ChatMessageHistory()
    return st.session_state["history"][session_id]

# Sidebar inputs
model_name = st.sidebar.selectbox("Select Open Source model", ["llama-3.3-70b-versatile"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)
api_key = st.sidebar.text_input("API Key", type="password")
chat_sessions = st.sidebar.text_input("Chat name (Session ID)")

# User input
st.write("Go ahead and ask me anything!")
user_input = st.text_input("You:")

# Token trimming setup (if needed)
trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=lambda messages: sum(len(m.content.split()) for m in messages if hasattr(m, "content")),
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# Response handler
def get_response(question, model_name, temperature, max_tokens, api_key, chat_sessions):
    llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Attach memory/history
    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": chat_sessions}}
    return chain_with_history.invoke({"input": question}, config=config)

# Execution
if chat_sessions and user_input:
    response = get_response(user_input, model_name, temperature, max_tokens, api_key, chat_sessions)
    st.write(f"**Gemma:** {response}")
    # Append to Streamlit session history
    history = get_history(chat_sessions)
    history.add_user_message(user_input)
    history.add_ai_message(response)
elif not chat_sessions:
    st.warning("⚠️ Please enter a chat name to continue.")
