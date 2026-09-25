# Multi-PDF RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) application
that allows users to upload multiple PDF documents and ask questions
about their content.

## Tech Stack

- Python
- Streamlit
- LangChain
- Gemini
- Gemini Embeddings
- FAISS
- PyPDF2

## Architecture

PDF Documents
↓
Text Extraction
↓
Text Chunking
↓
Gemini Embeddings
↓
FAISS Vector Database
↓
Similarity Search
↓
Gemini LLM
↓
Answer

## Features

- Upload multiple PDF documents
- Extract text from PDFs
- Split documents into chunks
- Generate embeddings
- Store embeddings using FAISS
- Perform semantic similarity search
- Generate answers using Gemini
- Interactive Streamlit interface

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Multipdf_project