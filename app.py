import streamlit as st
import os
import re
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import PyPDF2

# ------------------ Setup ------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Sample Medical Ranges ------------------
reference_data = {
    "Hemoglobin": "13.5-17.5 g/dL for men, 12.0-15.5 g/dL for women",
    "Cholesterol": "Below 200 mg/dL is desirable",
    "WBC": "4,500 to 11,000 cells/mcL",
    "Platelets": "150,000 to 450,000 platelets/mcL",
    "Blood Sugar": "Fasting <100 mg/dL, 2-hr post-meal <140 mg/dL"
}
reference_chunks = [f"{k}: {v}" for k, v in reference_data.items()]
reference_embeddings = embedder.encode(reference_chunks)
faiss_index = faiss.IndexFlatL2(reference_embeddings.shape[1])
faiss_index.add(np.array(reference_embeddings).astype("float32"))

# ------------------ Helper Functions ------------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_lab_values(text):
    results = {}
    for test in reference_data.keys():
        pattern = re.compile(fr"{test}\s*[:\-]?\s*(\d+\.?\d*)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            results[test] = match.group(1)
    return results

def retrieve_medical_info(query):
    query_embed = embedder.encode([query])
    _, indices = faiss_index.search(np.array(query_embed).astype("float32"), 1)
    return reference_chunks[indices[0][0]]

def query_llm(context, question):
    prompt = f"""You are a medical assistant. Based on this context:
    
{context}

Answer this question in very simple words:
{question}"""
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant"
    )
    return response.choices[0].message.content

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="🧪 Medical Report Interpreter", layout="centered")
st.title("🧪 Medical Report Interpreter with RAG + Groq")

uploaded = st.file_uploader("Upload a Lab Report (PDF)", type=["pdf"])
if uploaded:
    with st.spinner("Reading and processing your medical report..."):
        text = extract_text(uploaded)
        labs = extract_lab_values(text)

    st.success("Report processed. Here's what I found:")
    st.write(labs)

    question = st.text_input("Ask a question about your lab results")

    if question:
        combined_context = ""
        for lab, value in labs.items():
            info = retrieve_medical_info(lab)
            combined_context += f"\n{lab} value is {value}. Reference: {info}"

        with st.spinner("Thinking..."):
            answer = query_llm(combined_context, question)

        st.subheader("📋 Explanation:")
        st.write(answer)

st.info("This app is for educational purposes only. Always consult a doctor for medical decisions.")
