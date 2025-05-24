# app.py
import streamlit as st
from rag import RAGPipeline
from model import GeminiModel

# CONFIG
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG with Google Gemini")

# Init
api_key = st.text_input("Enter your Google API Key", type="password")
if api_key:
    model = GeminiModel(api_key=api_key)
    rag = RAGPipeline()

    # Upload and process PDFs
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Process PDFs"):
        file_paths = []
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())
            file_paths.append(file.name)
        rag.load_pdfs(file_paths)
        rag.create_embeddings()
        st.success("Documents processed and indexed!")

    # Ask question
    query = st.text_input("Ask something about your documents:")
    if query:
        docs = rag.retrieve(query)
        context = "\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        answer = model.chat_completion(prompt)

        st.subheader("📜 Answer")
        st.write(answer)

        st.subheader("📂 Context Used")
        for i, doc in enumerate(docs, 1):
            st.text_area(f"Doc #{i}", doc[:1000], height=150)
