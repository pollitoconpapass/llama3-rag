import os
import time
import tempfile
import streamlit as st
from backend.llama import rag_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader


def main_process(data_src):
    loader = PyPDFLoader(data_src) if data_src.endswith(".pdf") else WebBaseLoader(data_src)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    return retriever

def webpage_process():
    start = time.time()

    # Get the webpage URL from the user
    webpage_url = st.text_input("Enter Webpage URL", type="default")

    if webpage_url:
        # Load the data
        retriever = main_process(webpage_url)
        st.success(f"Loaded {webpage_url} successfully!")

        with st.form("chat-form"):
            # Ask a question about the webpage
            prompt = st.text_input("Ask any question about the webpage")
            submit_button = st.form_submit_button("Send")

            # Chat with the webpage
            if submit_button: 
                result = rag_chain(prompt, retriever)
                st.write(result)

    st.caption(f"Time taken: {time.time() - start} seconds")

def pdf_process():
    start = time.time()

    # Get the PDF file from the user
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file:
        # Save the uploaded file to a temporary location
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file_path = os.path.join(tmp_dir, pdf_file.name)
            with open(tmp_file_path, 'wb') as f:
                f.write(pdf_file.getvalue())

            # Load the data
            retriever = main_process(tmp_file_path)
            st.success(f"Loaded {pdf_file.name} successfully!")

            with st.form("chat-form"):
                # Ask a question about the PDF
                prompt = st.text_input("Ask any question about the PDF")
                submit_button = st.form_submit_button("Send")

                # Chat with the PDF
                if submit_button:
                    result = rag_chain(prompt, retriever)
                    st.write(result)

    st.caption(f"Time taken: {time.time() - start} seconds")


# ==== STREAMLIT COMPONENTS ====
st.title("LLAMA-3 RAG Chatbot 🦙 🤖")
st.caption("This app allows you to chat with a webpage using local Llama-3 and RAG")

choice = st.selectbox("Pick an option: ", ["Webpage", "PDF"])

if choice == "Webpage":
    webpage_process()
elif choice == "PDF":
    pdf_process()
