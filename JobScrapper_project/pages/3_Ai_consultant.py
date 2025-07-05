import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Page config
st.set_page_config(page_title="AI Job Consultant", layout="wide")
st.title("HireGenei:AI Job Consultant")
st.markdown("Ask questions related to your resume, skills, and suitable job roles.")

# Validate session state
required_keys = ["extracted_roles", "extracted_skills", "loader"]
if not all(k in st.session_state for k in required_keys):
    st.error("❌ Resume not processed yet. Please upload it on the home page first.")
    st.stop()

# Load resume documents and embed them
loader = st.session_state["loader"]
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """You are an intelligent job consultant helping the user.
The user has the following resume-based skills and job roles:

Always answer in short bullet points with clarity and avoid long paragraphs.

Question: {input}
Context: {context}
"""
)

# Set up the LLM
llm = ChatGroq(
    api_key=os.getenv("Groq_api_key"),
    model_name="Llama3-8b-8192"
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_query = st.chat_input("Ask me anything about jobs, roles, or career advice...")

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Handle new input
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append(("user", user_query))

    context_vars = {
        "input": user_query,
    }

    # Run retrieval chain
    response = retrieval_chain.invoke(context_vars)
    answer = response.get("answer", "⚠️ No relevant answer found.")

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))
