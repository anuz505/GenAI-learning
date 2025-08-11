import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="Math shite")
st.title("Let's Do Math lads")

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")


# let's create a chain for math alright

math_chain = LLMMathChain.from_llm(llm= llm)
calc = Tool(
    name="calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)
# why not a wikipedia
wikipedia_wrapper=WikipediaAPIWrapper()
wiki = Tool(
    name="wiki",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"

)
prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below. Also answer in the a funny witty tone as possible some british slangs as well.
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# final chain
reasoning_chain = LLMChain(llm = llm, prompt = prompt_template)

reasoning_tool = Tool(
    name="reasoning_shite",
    func=reasoning_chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

math_agent = initialize_agent(
    tools=[calc,wiki,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state or st.sidebar.button("Clear chat"):
    st.session_state["messages"] = [{"role":"assistant","content":"Hey lad how can i help you"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_input("enter your question mate")

if st.button("Ask"):
    if question:
        with st.spinner("wait a sec aight mate....."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            res = math_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":res})
            st.write("You see here lad: ")
            st.success(res)
    else:
        st.error("Enter question")