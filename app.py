import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def _resolve_api_key() -> str | None:
    key = None
    try:
        if hasattr(st, "secrets"):
            key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


api_key = _resolve_api_key()
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


@st.cache_resource
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def get_pdf_text(pdf_files_or_paths) -> str:
    text_parts = []
    for item in pdf_files_or_paths:
        pdf_reader = PdfReader(item)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def get_text_chunks(text: str):
    length = len(text)
    if length < 15_000:
        chunk_size, chunk_overlap = 1500, 200
    elif length < 80_000:
        chunk_size, chunk_overlap = 2000, 200
    else:
        chunk_size, chunk_overlap = 4000, 400

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text), chunk_size, chunk_overlap


def get_vector_store(text_chunks: list[str]):
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.vector_store = vector_store


def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_retries=2,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def _load_db():
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        return st.session_state.vector_store
    if os.path.exists("faiss_index"):
        embeddings = get_embeddings_model()
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.session_state.vector_store = db
        return db
    return None


def chat_with_pdf(question: str) -> str:
    if not api_key:
        return "Missing `GOOGLE_API_KEY` or `GEMINI_API_KEY` in environment or Streamlit secrets."

    db = _load_db()
    if db is None:
        return "Please upload and process PDFs first."

    docs = db.similarity_search(question, k=4)
    chain = get_conversational_chain()

    try:
        result = chain.invoke({"input_documents": docs, "question": question})
        return result["output_text"]
    except Exception as exc:
        return f"Error generating answer: {exc}"


st.title("💬 Chat with Your multiple PDFs using RAG and Gemini")

with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if not api_key:
            st.error("Please set `GOOGLE_API_KEY` or `GEMINI_API_KEY` before processing PDFs.")
        elif not uploaded_files:
            st.warning("Please choose at least one PDF file.")
        else:
            with st.spinner("Extracting text, chunking, and indexing PDFs..."):
                temp_paths = []
                try:
                    for file in uploaded_files:
                        file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            temp_paths.append(tmp.name)

                    raw_text = get_pdf_text(temp_paths)
                    if not raw_text.strip():
                        st.error("No extractable text found in the uploaded PDF(s).")
                    else:
                        chunks, c_size, c_overlap = get_text_chunks(raw_text)
                        get_vector_store(chunks)
                        st.success(
                            f"✅ Indexed {len(chunks)} chunks (chunk_size={c_size}, overlap={c_overlap})!"
                        )
                finally:
                    for p in temp_paths:
                        if os.path.exists(p):
                            os.remove(p)

if "history" not in st.session_state:
    st.session_state.history = []

st.header("Ask a Question")
question = st.text_input("Your Question")

if st.button("Submit") and question:
    if _load_db() is None:
        st.warning("Please upload and process PDFs first.")
    else:
        with st.spinner("Searching indexed chunks and generating answer..."):
            answer = chat_with_pdf(question)
        st.session_state.history.append((question, answer))

for q, a in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
