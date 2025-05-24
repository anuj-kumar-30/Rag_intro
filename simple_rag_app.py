import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Page setting
st.set_page_config(layout="wide")

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

# Initialize session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

def process_pdfs(uploaded_files):
    """Process uploaded PDF files and extract text chunks"""
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(pages)
            
            # Add file name to metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = uploaded_file.name
            
            all_chunks.extend(chunks)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return all_chunks

def find_relevant_chunks(query, chunks, top_k=3):
    """Simple keyword-based chunk retrieval"""
    query_words = set(query.lower().split())
    
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.lower()
        # Simple scoring based on keyword matches
        score = sum(1 for word in query_words if word in content)
        if score > 0:
            chunk_scores.append((score, i, chunk))
    
    # Sort by score and return top chunks
    chunk_scores.sort(reverse=True)
    return [chunk for _, _, chunk in chunk_scores[:top_k]]

def generate_answer(query, relevant_chunks):
    """Generate answer using LLM and relevant chunks"""
    if not relevant_chunks:
        return "I couldn't find relevant information in your documents to answer this question."
    
    # Combine relevant chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    prompt = ChatPromptTemplate.from_template("""
Based on the following context from the uploaded documents, please answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question based on the provided documents."

Context:
{context}

Question: {question}

Answer:
""")
    
    try:
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": query})
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    st.header("📗 Chat with PDF (Simple RAG)")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        accept_multiple_files=True, 
        type="pdf"
    )
    
    # Process files when uploaded
    if uploaded_files:
        # Check if files have changed
        current_file_names = [f.name for f in uploaded_files]
        if current_file_names != st.session_state.uploaded_files:
            st.session_state.uploaded_files = current_file_names
            with st.spinner("Processing PDF files..."):
                st.session_state.chunks = process_pdfs(uploaded_files)
            st.success(f"Processed {len(uploaded_files)} files into {len(st.session_state.chunks)} chunks")
    
    # Display current status
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Document Status")
        if st.session_state.chunks:
            st.write(f"📄 **Files processed:** {len(st.session_state.uploaded_files)}")
            st.write(f"📝 **Text chunks:** {len(st.session_state.chunks)}")
            
            # Show chunk preview
            if st.checkbox("Show chunk preview"):
                for i, chunk in enumerate(st.session_state.chunks[:3]):  # Show first 3 chunks
                    with st.expander(f"Chunk {i+1} - {chunk.metadata.get('source_file', 'Unknown')}"):
                        st.write(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
        else:
            st.write("No documents uploaded yet")
    
    with col2:
        st.subheader("Ask Questions")
        
        if st.session_state.chunks:
            # Question input
            query = st.text_input(
                "Ask a question about your documents:",
                placeholder="What is the main topic discussed in the documents?"
            )
            
            if st.button("Get Answer", type="primary") and query:
                with st.spinner("Finding relevant information and generating answer..."):
                    # Find relevant chunks
                    relevant_chunks = find_relevant_chunks(query, st.session_state.chunks)
                    
                    # Generate answer
                    answer = generate_answer(query, relevant_chunks)
                    
                    # Display answer
                    st.markdown("### Answer:")
                    st.markdown(answer)
                    
                    # Show sources
                    if relevant_chunks:
                        st.markdown("### Sources:")
                        for i, chunk in enumerate(relevant_chunks):
                            source_file = chunk.metadata.get('source_file', 'Unknown file')
                            page = chunk.metadata.get('page', 'Unknown page')
                            with st.expander(f"Source {i+1}: {source_file} (Page {page})"):
                                st.write(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
        else:
            st.info("Please upload PDF files first to start asking questions.")

if __name__ == '__main__':
    main()