from dotenv import load_dotenv
import os
import streamlit as st

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found")
    st.stop()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Workout & Diet Chatbot")
st.title("Workout & Diet Chatbot")
st.caption("Ask questions from your PDFs")


# -----------------------------
# STEP 1 — LOAD PDFs
# -----------------------------
@st.cache_resource
def load_pdfs():
    pdf_files = ["workoutinformation.pdf", "dietinformation.pdf"]

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
# STEP 4 — VECTOR DB (FAISS)
# -----------------------------
def create_vectorstore(chunks, embeddings):
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
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

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")

    return llm, retriever, prompt


# -----------------------------
# LOAD PIPELINE
# -----------------------------
docs = load_pdfs()
chunks = chunk_docs(docs)
embeddings = get_embeddings()
vectordb = create_vectorstore(chunks, embeddings)


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

            llm, retriever, prompt = build_rag(vectordb)

            docs = retriever.invoke(user_input)

            context = "\n\n".join([doc.page_content for doc in docs])

            # Limit context size (VERY IMPORTANT)
            context = context[:3000]
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful fitness and diet assistant. Answer only from the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"
                }
            ]
            
            response = llm.invoke(messages)
            
            answer = response.content

            st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))
