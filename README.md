
# 🧪 Medical Report Interpreter (RAG + Groq + FAISS)

A Streamlit-powered AI app that intelligently interprets lab test reports using Retrieval-Augmented Generation (RAG). The app extracts values from uploaded medical PDFs, compares them with standard reference ranges, and uses Groq's LLaMA 3.1 model to explain results in simple, understandable language.

---

## 🚀 Features

- 🔍 Upload medical **PDF lab reports**
- 📑 Extract lab test values using regex
- 📘 Compare against **standard medical reference ranges**
- 🧠 Uses **FAISS** + **Sentence-Transformers** to retrieve relevant information
- 💬 Uses **Groq API (LLaMA-3.1-8B-Instant)** to explain lab results in layman's terms
- 🎨 Simple and clean **Streamlit UI**
- ✅ Designed for **educational purposes** — not a diagnostic tool

---

## 📸 Demo Preview

https://huggingface.co/spaces/ilsa15/Report-Interpreter

---

## 🛠️ Tech Stack

| Layer       | Tool/Library                         |
|-------------|--------------------------------------|
| Frontend    | Streamlit                            |
| LLM         | [Groq API](https://console.groq.com) (LLaMA 3.1) |
| Embeddings  | `sentence-transformers` (MiniLM-L6-v2) |
| Vector DB   | `FAISS` (Flat L2 Index)              |
| File Parsing| `PyPDF2`                             |








