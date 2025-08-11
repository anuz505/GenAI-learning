import validators, streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="Simple Youtube and website summarizer")
llm = ChatGroq(model="llama-3.3-70b-versatile",streaming= True)

url = st.text_input(placeholder="Enter the url maybe a youtube link perhaps",label="URL FFS")

if st.button("Summarize the Content form youtube or website"):
    if not url.strip():
        st.error("Enter the URL mate")
    elif not validators.url(url):
        st.error("Enter the correct url mate come on")
    else:
        try:
            with st.spinner("wait a sec lad"):
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
                else:
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers=headers)
                docs=loader.load()
                
                # Split large documents into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=200
                )
                split_docs = text_splitter.split_documents(docs)
                
                prompt_template = """
                                    Write a summary of the following content : {text} make it  as fun and witty as u can.
                                    """
                prompt = PromptTemplate(input_variables=["text"],template=prompt_template)
                chain = load_summarize_chain(llm = llm, chain_type="map_reduce" ,verbose=True)
                summary = chain.run(split_docs)   
                st.success(summary)
        except Exception as e:
            st.exception(e)
