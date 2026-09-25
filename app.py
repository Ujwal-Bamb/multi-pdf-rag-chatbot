import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# ----------------------------
# Set up Google Gemini API key
# ----------------------------
# It's recommended to use secrets or environment variables in production
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------------
# Function: Extract text from PDF files
# ------------------------------------
def get_pdf_text(pdf_paths):
    text = ""
    for path in pdf_paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle cases where text may be None
    return text

# -------------------------------------------------
# Function: Split text into manageable chunks
# -------------------------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# -------------------------------------------------------------------
# Function: Generate and save vector embeddings using FAISS and Gemini
# -------------------------------------------------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    
    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )
    
    vector_store.save_local("faiss_index")

# ------------------------------------------------------------------
# Function: Load the conversational chain with custom prompt template
# ------------------------------------------------------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say "answer is not available in the context".

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    # Initialize the Gemini Pro chat model
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    # Create a custom prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load a simple Q&A chain with the model and prompt
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ----------------------------------------------------------------
# Function: Search PDFs and answer user question using embeddings
# ----------------------------------------------------------------
def chat_with_pdf(question):
    # Load saved embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform similarity search on question
    docs = db.similarity_search(question)
    # Get the conversational chain
    chain = get_conversational_chain()
    # Run the chain with documents and question
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return result["output_text"]

# ------------------------
# Streamlit Web Interface
# ------------------------

# App Title
st.title("💬 Chat with Your multiple PDFs using RAG and Gemini")

# Sidebar: File Upload Section
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Process PDFs and create vector store on button click
    if st.button("Process PDFs") and uploaded_files:
        file_paths = []
        for file in uploaded_files:
            # Save uploaded files temporarily
            with open(file.name, "wb") as f:
                f.write(file.read())
                file_paths.append(file.name)

        # Extract text, split, and generate vector store
        raw_text = get_pdf_text(file_paths)
        chunks = get_text_chunks(raw_text)
        get_vector_store(chunks)

        st.success("✅ PDFs processed and indexed successfully!")

# Chat history management
if "history" not in st.session_state:
    st.session_state.history = []

# Main chat section
st.header("Ask a Question")
question = st.text_input("Your Question")  # User input

# Handle submit button
if st.button("Submit") and question:
    # Get answer using chat_with_pdf
    answer = chat_with_pdf(question)
    # Store Q&A in session history
    st.session_state.history.append((question, answer))

# Display chat history (latest first)
for q, a in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")