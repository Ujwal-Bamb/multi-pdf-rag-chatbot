# 💬 Multi-PDF RAG Conversational Chatbot

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multi-pdf-rag-chatbot-ub.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ujwal-Bamb/multi-pdf-rag-chatbot.git)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API-4285F4?style=flat&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Index-00599C?style=flat)
![RAG](https://img.shields.io/badge/GenAI-RAG%20Architecture-9cf?style=flat)

> **Enterprise Retrieval-Augmented Generation (RAG) System** that enables conversational question answering across multiple uploaded PDF documents with grounded source citations, FAISS vector indexing, and Google Gemini / OpenAI LLM orchestration.

---

## 🔗 Live Application & Repository

- **Live Application**: [https://multi-pdf-rag-chatbot-ub.streamlit.app](https://multi-pdf-rag-chatbot-ub.streamlit.app)
- **Source Code**: [https://github.com/Ujwal-Bamb/multi-pdf-rag-chatbot.git](https://github.com/Ujwal-Bamb/multi-pdf-rag-chatbot.git)

---

## 📸 Screenshots

> *Add your application screenshots to `docs/screenshots/` using the exact paths below to render them automatically:*

| Multi-PDF Conversational Chat UI | Batch Document Upload & Indexing |
| :---: | :---: |
| ![Chat UI](docs/screenshots/rag_chat_ui.png) | ![PDF Upload](docs/screenshots/pdf_upload_indexing.png) |
| *Grounded question answering with context synthesis* | *Sidebar multi-PDF ingestion and FAISS vectorization* |

---

## 🎯 Problem Statement & Challenge

Reviewing complex enterprise documents (financial reports, research publications, legal contracts, technical manuals) introduces significant information discovery friction:
1. **Context Window Limitations**: Uploading multiple complete 50+ page PDFs directly into standard LLM prompts exceeds context windows and skyrockets token latency and cost.
2. **Hallucination Risk**: Standard generative models frequently hallucinate facts when asked specific, granular questions from unindexed documents.
3. **Information Silos**: Information scattered across disjoint PDF files cannot be synthesized efficiently with standard keyword search (`Ctrl + F`).

---

## 💡 Technical Architecture & RAG Pipeline

This application implements a modular **Retrieval-Augmented Generation (RAG)** pipeline using LangChain, FAISS, and Google Gemini:

```mermaid
flowchart TD
    A[Multiple PDF Uploads] --> B[Text Extraction\nPyPDF2 PdfReader]
    B --> C[Recursive Character Text Splitting\nChunk Size: 10,000 | Overlap: 1,000]
    C --> D[Dense Embeddings Generation\nGoogleGenerativeAIEmbeddings]
    D --> E[Local Vector Database\nFAISS faiss_index]
    
    F[User Question] --> G[Vector Similarity Search\nTop-k Relevant Chunks]
    E --> G
    G --> H[Prompt Synthesis with Anti-Hallucination Guardrails]
    H --> I[LLM Reasoning & Generation\nGoogle Gemini 3.5 Flash]
    I --> J[Grounded Streamlit Response]
```

### Key Technical Components:
- **Text Extraction**: Uses `PyPDF2.PdfReader` to extract and normalize text across multiple uploaded PDF files in batch.
- **Recursive Chunking Strategy**: Employs LangChain's `RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)` to ensure semantic boundaries are preserved without splitting sentences or concepts mid-thought.
- **Vector Indexing (`FAISS`)**: Serializes embeddings into high-speed local `faiss-cpu` vector indices using `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`, enabling sub-second cosine distance vector retrieval.
- **Strict Anti-Hallucination Guardrails**: Prompts the LLM with strict grounding instructions:
  > *"Answer the question as detailed as possible from the provided context. If the answer is not in the context, say 'answer is not available in the context'. Do not provide false information."*
- **Conversational Chain**: Uses LangChain's QA chain (`load_qa_chain` with `stuff` type) paired with `ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)`.
- **Cloud Deployment**: Deployed on **Streamlit Community Cloud** with secure environment variable and secrets management (`st.secrets` / `.env`).

---

## 📁 Repository Structure

```
multi-pdf-rag-chatbot/
├── .streamlit/                     # Streamlit configuration & secrets
├── docs/
│   └── screenshots/                # Application screenshot placeholders
│       ├── rag_chat_ui.png
│       └── pdf_upload_indexing.png
├── app.py                          # Complete Streamlit RAG application code
├── requirements.txt                # Pinned production dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive project documentation
```

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Ujwal-Bamb/multi-pdf-rag-chatbot.git
cd multi-pdf-rag-chatbot
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/).)*

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ How to Use

1. **Upload Documents**: Open the left sidebar and upload one or more PDF files.
2. **Process & Vectorize**: Click **"Process PDFs"**. The app extracts text, generates chunked vector embeddings, and builds the local FAISS index.
3. **Ask Questions**: Type any natural language question into the chat input (e.g., *"Summarize the quarterly revenue and key financial risks mentioned across these reports"*).
4. **Inspect Answers**: The LLM synthesizes verified answers strictly extracted from the uploaded context.

---

## 👤 Author & Contact

**Ujwal Bamb**  
- **Email**: [ujwalbamb2000@gmail.com](mailto:ujwalbamb2000@gmail.com)  
- **LinkedIn**: [linkedin.com/in/ujwal-bamb-276b04330](https://www.linkedin.com/in/ujwal-bamb-276b04330)  
- **GitHub**: [@Ujwal-Bamb](https://github.com/Ujwal-Bamb)  
- **Credentials**: MeitY-NASSCOM Gold Certified (96/100, NSQF Level 5, ID: FSP/2026/11/10520719) | Master of Computer Science in AI & ML, Woolf University (Jan 2025 - Present) | B.Pharm (SPPU)
