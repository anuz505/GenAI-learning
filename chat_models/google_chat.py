from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model= "gemini-2.5-pro")

result = model.invoke("Hey gemini give me a joke a sarcastic one for learning langchain")

print(result)