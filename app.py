import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = (
    (st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None)
    or (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None)
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


def get_pdf_text(pdf_paths):
    text = ""
    for path in pdf_paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


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
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def chat_with_pdf(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    try:
        result = chain.invoke({"input_documents": docs, "question": question})
        return result["output_text"]
    except Exception:
        return "Gemini is temporarily busy (500/503 server error). Please try again in a moment."


st.title("💬 Chat with Your multiple PDFs using RAG and Gemini")

with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process PDFs") and uploaded_files:
        file_paths = []
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())
                file_paths.append(file.name)

        raw_text = get_pdf_text(file_paths)
        chunks = get_text_chunks(raw_text)
        get_vector_store(chunks)
        st.success("✅ PDFs processed and indexed successfully!")

if "history" not in st.session_state:
    st.session_state.history = []

st.header("Ask a Question")
question = st.text_input("Your Question")

if st.button("Submit") and question:
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process PDFs first.")
    else:
        answer = chat_with_pdf(question)
        st.session_state.history.append((question, answer))

for q, a in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
