import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Load environment variables
load_dotenv()

# Page setting
st.set_page_config(layout="wide")

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Create chroma directory if it doesn't exist
os.makedirs("./chroma_db", exist_ok=True)

# Initialize embedding function
embedding_function = SentenceTransformerEmbeddingFunction()

# Load Vector database
client = chromadb.PersistentClient(path="./chroma_db")
db = Chroma(
    client=client,
    collection_name="chat-with-pdf",
    embedding_function=embedding_function
)

# Init langchain
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
prompt = ChatPromptTemplate.from_template("""
Based on the provided context only, find the best answer for my question. Format the answer in markdown format
<context>
{context}
</context>
Question:{input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

if "question" not in st.session_state:
    st.session_state.question = None

if "old_filenames" not in st.session_state:
    st.session_state.old_filenames = []

# Create temp directory if it doesn't exist
os.makedirs("./temp", exist_ok=True)

@st.cache_resource
def get_collection():
    print("DEBUG: call get_collection()")
    try:
        # Delete all documents
        client.delete_collection("chat-with-pdf")
    except:
        pass
    finally:
        collection = client.get_or_create_collection(
            name="chat-with-pdf",
            embedding_function=embedding_function  # Use the same embedding function
        )
    return collection

# Load, transform and embed new files into Vector Database
def add_files(uploaded_files):
    collection = get_collection()

    # old_filenames: contains a list of names of files being used
    # uploaded_filenames: contains a list of names of uploaded files
    old_filenames = st.session_state.old_filenames
    uploaded_filename = [file.name for file in uploaded_files]
    new_files = [file for file in uploaded_files if file.name not in old_filenames]

    for file in new_files:
        try:
            # Step 1: load uploaded file
            temp_file = f"./temp/{file.name}"
            with open(temp_file, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_file)
            pages = loader.load()

            # Step 2: split content in to chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents(pages)

            # Step 3: embed chunks into Vector Store
            for index, chunk in enumerate(chunks):
                # Ensure the text is properly formatted
                text = chunk.page_content.strip()
                if not text:  # Skip empty chunks
                    continue
                    
                try:
                    # Convert text to string and ensure it's properly encoded
                    text = str(text).encode('utf-8', errors='ignore').decode('utf-8')
                    
                    # Remove any non-printable characters
                    text = ''.join(char for char in text if char.isprintable() or char.isspace())
                    
                    # Skip if text is too short after cleaning
                    if len(text.strip()) < 10:  # Skip very short chunks
                        continue
                        
                    collection.upsert(
                        ids=[f"{file.name}_{index}"],
                        metadatas=[{"source": file.name, "page": chunk.metadata.get("page", 0)}],
                        documents=[text]
                    )
                except Exception as e:
                    st.error(f"Error processing chunk {index} from {file.name}: {str(e)}")
                    continue

        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass

# Remove all relevant chunks of the removed files
def remove_files(uploaded_files):
    collection = get_collection()

    # old_filenames: contains a list of names of files being used
    # uploaded_filenames: contains a list of names of uploaded files
    old_filenames = st.session_state.old_filenames
    uploaded_filename = [file.name for file in uploaded_files]

    # Step 1: Get the list of file that was removed from upload files
    deleted_filenames = [name for name in old_filenames if name not in uploaded_filename]

    # Step 2: Remove all relevant chunks of the removed files
    if len(deleted_filenames) > 0:
        try:
            all_chunks = collection.get()
            ids = all_chunks["ids"]
            metadatas = all_chunks["metadatas"]

            if len(metadatas) > 0:
                deleted_ids = []
                for name in deleted_filenames:
                    for index, metadata in enumerate(metadatas):
                        if metadata and metadata.get('source') == name:
                            deleted_ids.append(ids[index])
                if deleted_ids:
                    collection.delete(ids=deleted_ids)
        except Exception as e:
            st.error(f"Error removing files: {str(e)}")

# Return chunks after having any change in the file list
def refresh_chunks(uploaded_files):
    if uploaded_files is None:
        uploaded_files = []
        
    # old_filenames: contains a list of names of files being used
    # uploaded_filenames: contains a list of names of uploaded files
    old_filenames = st.session_state.old_filenames
    uploaded_filename = [file.name for file in uploaded_files]

    if len(old_filenames) < len(uploaded_filename):
        add_files(uploaded_files)
    elif len(old_filenames) > len(uploaded_filename):
        remove_files(uploaded_files)

    # Step 3: Save the state
    st.session_state.old_filenames = uploaded_filename

def main_page():
    st.header("📗 Chat with PDF (RAG version)")

    uploaded_files = st.file_uploader("Choose a PDF", accept_multiple_files=True, type="pdf",
                                    label_visibility="collapsed")
    refresh_chunks(uploaded_files)

    col1, col2 = st.columns([4, 6])
    collection = get_collection()
    chunk_count = collection.count()
    
    with col1:
        st.write(f"TOTAL CHUNKS: {chunk_count}")
        if st.session_state.question is not None:
            try:
                relevant_chunks = retriever.invoke(input=st.session_state.question)
                st.write("RELEVANT CHUNKS:")
                st.write(relevant_chunks)
            except Exception as e:
                st.error(f"Error retrieving chunks: {str(e)}")
        else:
            try:
                all_chunks = collection.get()
                if all_chunks and all_chunks.get("documents"):
                    st.write("All chunks in database:")
                    for i, doc in enumerate(all_chunks["documents"]):
                        st.write(f"Chunk {i+1}:")
                        st.write(doc[:200] + "..." if len(doc) > 200 else doc)
            except Exception as e:
                st.error(f"Error displaying chunks: {str(e)}")
    
    with col2:
        if chunk_count > 0:
            query = st.text_input(label="Question", placeholder="Please ask me anything related to your files",
                                value="")
            ask = st.button("Send message", type="primary")
            if len(query) > 0 and ask:
                with st.spinner("Processing your question..."):
                    st.session_state.question = query
                    try:
                        response = retriever_chain.invoke({"input": query})
                        st.markdown(response['answer'])
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

if __name__ == '__main__':
    main_page()