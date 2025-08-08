import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCAL_DB = "USE_LOCAL"
MYSQL = "USE_MYSQL"

radio_options = ["use local db","use mysql"]

selected_option = st.sidebar.radio(label="Choose the DB u want to connect with",options = radio_options)

if radio_options.index(selected_option) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("provide the mysql host")
    mysql_user = st.sidebar.text_input("provide the mysql user")
    mysql_password = st.sidebar.text_input("provide the mysql passowrd",type="password")
    mysql_db = st.sidebar.text_input("provide the database")
else:
    db_uri = LOCAL_DB

api_key_groq = st.sidebar.text_input("enter your groq api")

if not db_uri:
    st.info("Enter your database info and uri")

if not api_key_groq:
    st.info("enter the api key for groq")

llm=ChatGroq(groq_api_key=api_key_groq,model_name="Llama3-8b-8192",streaming=True)

@st.cache_resource(ttl="2h")
def db_config(db_uri,mysql_host = None, mysql_user = None, mysql_password = None, mysql_db = None):
    if db_uri == LOCAL_DB:
        db_path = (Path(__file__).parent/"student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("enter the sql information. there is a connection errorr")
            st.stop()
    return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))   

if db_uri == LOCAL_DB:
    db =  db_config(db_uri)
elif db_uri == MYSQL:
    db = db_config(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
# toolkit
toolkit = SQLDatabaseToolkit(db = db, llm=llm)

agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)


if "messages" not in st.session_state or st.sidebar.button("clear message history"):
    st.session_state["messages"] = [{"role":"assistant","content":"How can i help you"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input(placeholder="Your message to AI")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)