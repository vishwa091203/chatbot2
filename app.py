from dotenv import load_dotenv
import os
import streamlit as st

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Workout & Diet Chatbot", page_icon="💪")
st.title("💪 Workout & Diet Chatbot")
st.caption("Ask questions from your PDFs")


# -----------------------------
# STEP 1 — LOAD PDFs
# -----------------------------
@st.cache_resource
def load_pdfs():
    pdf_files = ["workoutinfo.pdf", "dietinfo.pdf"]

    docs = []
    for file in pdf_files:
        if not os.path.exists(file):
            st.error(f"Missing file: {file}")
            st.stop()

        loader = PyPDFLoader(file)
        docs.extend(loader.load())

    return docs


# -----------------------------
# STEP 2 — CHUNKING
# -----------------------------
@st.cache_resource
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


# -----------------------------
# STEP 3 — EMBEDDINGS
# -----------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -----------------------------
# STEP 4 — VECTOR DB (CHROMA)
# -----------------------------
@st.cache_resource
def create_vectorstore(chunks, embeddings):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectordb


# -----------------------------
# STEP 5 — RAG PIPELINE
# -----------------------------
def build_rag(vectordb):

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{input}
""")

    doc_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return rag_chain


# -----------------------------
# LOAD PIPELINE
# -----------------------------
docs = load_pdfs()
chunks = chunk_docs(docs)
embeddings = get_embeddings()
vectordb = create_vectorstore(chunks, embeddings)
rag_chain = build_rag(vectordb)


# -----------------------------
# CHAT UI
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]

            st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))
