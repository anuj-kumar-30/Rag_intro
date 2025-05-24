# rag.py
import os
import faiss
import pickle
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, db_path="vector_db.index", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.index = faiss.IndexFlatL2(384)
        self.documents = []

        if os.path.exists(db_path):
            self.load_db()

    def load_pdfs(self, pdf_paths: List[str]):
        for path in pdf_paths:
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                self.documents.append(text)


    def create_embeddings(self):
    # Clean the documents list
        clean_docs = [doc for doc in self.documents if isinstance(doc, str) and doc.strip()]
    
        if not clean_docs:
            raise ValueError("No valid documents found to embed.")
    
        embeddings = self.model.encode(clean_docs)
        self.index.add(embeddings)
        self.documents = clean_docs  # update stored documents
        self.save_db()


    def save_db(self):
        faiss.write_index(self.index, self.db_path)
        with open("documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load_db(self):
        self.index = faiss.read_index(self.db_path)
        with open("documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
