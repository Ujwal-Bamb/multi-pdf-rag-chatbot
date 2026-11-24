import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

try:
    from langchain_google_genai.chat_models import GoogleAPIError, ServerError
except ImportError:
    try:
        from google.api_core.exceptions import GoogleAPIError, ServerError
    except ImportError:
        GoogleAPIError = Exception
        ServerError = Exception

API_ERRORS = (GoogleAPIError, ServerError)
try:
    from google.api_core.exceptions import GoogleAPIError as CoreGoogleAPIError
    if CoreGoogleAPIError not in API_ERRORS:
        API_ERRORS = API_ERRORS + (CoreGoogleAPIError,)
except ImportError:
    pass

load_dotenv()

# Support local .env or Streamlit Community Cloud Secrets
if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
FALLBACK_CHAT_MODEL = os.getenv("GEMINI_FALLBACK_CHAT_MODEL", "gemini-3.5-flash-lite")


def get_pdf_text(pdf_paths: list[str]) -> str:
    text = ""
    for path in pdf_paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks: list[str]) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)


def get_conversational_chain(model=None):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    if model is None:
        model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def chat_with_pdf(question: str) -> str:
    if not Path(FAISS_INDEX_DIR).exists():
        return "Please upload and process at least one PDF first."

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    try:
        result = chain.invoke({"input_documents": docs, "question": question})
    except API_ERRORS:
        try:
            fallback_model = ChatGoogleGenerativeAI(model=FALLBACK_CHAT_MODEL, temperature=0.3)
            fallback_chain = get_conversational_chain(model=fallback_model)
            result = fallback_chain.invoke({"input_documents": docs, "question": question})
        except Exception:
            return "Gemini is temporarily experiencing a server error (500/503). Please try again in a moment."
    except Exception as exc:
        if any(err_code in str(exc) for err_code in ("500", "503", "ServerError", "overloaded", "ResourceExhausted")):
            try:
                fallback_model = ChatGoogleGenerativeAI(model=FALLBACK_CHAT_MODEL, temperature=0.3)
                fallback_chain = get_conversational_chain(model=fallback_model)
                result = fallback_chain.invoke({"input_documents": docs, "question": question})
            except Exception:
                return "Gemini is temporarily experiencing a server error (500/503). Please try again in a moment."
        else:
            raise

    if isinstance(result, dict):
        return result.get("output_text", str(result))
    return str(result)


st.set_page_config(page_title="Multi-PDF RAG Chatbot", layout="wide")
st.title("💬 Chat with Your Multiple PDFs using RAG and Gemini")

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Set `GOOGLE_API_KEY` in `.env` locally or in Streamlit Cloud **Settings → Secrets**.")

with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Process PDFs") and uploaded_files:
        temp_paths: list[str] = []
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for file in uploaded_files:
                    safe_name = Path(file.name).name
                    temp_file = Path(tmp_dir) / safe_name
                    temp_file.write_bytes(file.getbuffer())
                    temp_paths.append(str(temp_file))

                raw_text = get_pdf_text(temp_paths)
                chunks = get_text_chunks(raw_text)
                if not chunks:
                    st.error("No extractable text found in the uploaded PDFs.")
                else:
                    get_vector_store(chunks)
                    st.success("✅ PDFs processed and indexed successfully!")
        except Exception as exc:
            st.error(f"Error processing PDFs: {exc}")

if "history" not in st.session_state:
    st.session_state.history = []

st.header("Ask a Question")
question = st.text_input("Your Question")

if st.button("Submit") and question:
    with st.spinner("Searching documents and generating response..."):
        answer = chat_with_pdf(question)
        st.session_state.history.append((question, answer))

for q, a in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")