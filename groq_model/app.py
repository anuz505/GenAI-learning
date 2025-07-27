from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI

load_dotenv()
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGSMITH_PROJECT"] = "ollama simple app"
groq_api = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api)

system_template = "Translate the following into this particular language."
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])
parser = StrOutputParser()

chain = chat_prompt | llm | parser

#app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain runnable interfaces."
)
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost", port=8080)