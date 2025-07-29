import streamlit as st 
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# environments setup
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "Rag App"

# streamlit setup
st.title("Welcome to a RAG Q&A application")
st.write("Upload Pdf's and chat with their content")

#embeddings setup
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key = st.text_input("Enter your GROQ API key",type="password")

model = st.selectbox("select the opensource model",["llama-3.3-70b-versatile","Gemma2-9b-It"])

if groq_api_key:
    llm = ChatGroq(model=model, api_key=groq_api_key)

    # session_id
    session_id = st.text_input("Chat name", value="default")

    if "store" not in st.session_state:
        st.session_state.store = {}
    
    #upload files
    uploaded_files = st.file_uploader(label="choose a pdf file", type="pdf",  accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "/temp.pdf"
            with open(temp_pdf,"wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = f.name
            
            loader = PyPDFLoader(temp_pdf)
            docs_loader  = loader.load()
            documents.extend(docs_loader)

    # splitting the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(splits,embeddings)
        retriever = vector_db.as_retriever()
    
    # prompt and promt templates 
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_ret = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        system_prompt = (
                "You are an funny assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "Answer all the questions with charisma maybe add genz slangs and tones"
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        
        QNA_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        qna_chain = create_stuff_documents_chain(llm,QNA_prompt)

        # RAG chain
        rag_chain = create_retrieval_chain(history_aware_ret,qna_chain)

        def get_session_history(session_id:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Final Conversational RAG Chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        ) 

        user_input = st.text_input("Your query brotha")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {
                    "input": user_input
                }, config={"configurable":{"session_id":session_id}}
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("chat_History:",session_history.messages)
else:
    st.warning("Enter your groq api key")
