import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import boto3
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA


# data ingestion part
def data_ingestion():
    loader = DirectoryLoader("../test_pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter =  RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    docs = splitter.split_documents(documents=documents)
    return docs

# vector embeddings part
bedrock_client = boto3.client(service_name = "bedrock-runtime",region_name = "ap-south-1")
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock_client)
# vector store
def getVectorDB(docs):
    vectorDB = FAISS.from_documents(docs,embedding=embeddings)
    vectorDB.save_local("index")
    return vectorDB

def getLLM():
    llm = BedrockLLM(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={
            "maxTokenCount": 512,
            "temperature": 0.7
        }
    )
    return llm

# prompt
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant: 
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

def get_res_from_llm(llm,vector_store,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents = True,
        chain_type_kwargs={"prompt":prompt}
        )
    answers = qa({"query":query})
    return answers["result"]


def app():
    st.set_page_config(page_title="QNA RAG AWS")
    st.title("QnA RAG using AWS Bedrock and Titan model")


    st.write("Initialize Vector DB")
    if st.button("Vector DB"):
        with st.spinner("creating....."):
            docs = data_ingestion()
            getVectorDB(docs=docs)
            st.session_state["vector_db"] = True
            st.success("Done")

    if st.session_state.get("vector_db",False):
        user_question = st.text_input("Ask what u want from the PDFs")

        if st.button("Enter"):
            with st.spinner("In progress...."):
                faiss_index = FAISS.load_local("index",embeddings=embeddings,allow_dangerous_deserialization=True)
                llm_res = getLLM()
                st.write(get_res_from_llm(llm=llm_res, vector_store=faiss_index,query=user_question))
                st.success("done")

if __name__ == "__main__":
    app()    