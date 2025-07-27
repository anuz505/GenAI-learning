from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "ollama simple app"

st.title("Chat with me Gemma:2b")

# prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Hey, you are a helpful assistant. Please respond to the question asked. ask with some charisma funny"),
    ("user", "{question}")
])

input_text = st.text_input("What do wanna ask?")
#ollama model
llm = Ollama(model = "gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
if input_text:
    st.write(chain.invoke({"question":input_text}))
