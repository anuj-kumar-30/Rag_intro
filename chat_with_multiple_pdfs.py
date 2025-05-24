# Imports
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Load environment variables
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Creating a collection named `chat-with-pdf` into chroma db
native_db = chromadb.PersistentClient('./chroma_db')
db = Chroma(client=native_db,
            collection_name='chat-with-pdfs',
            embedding_function=SentenceTransformerEmbeddingFunction())

# In order to work with the collection we will use the function get_collection()
@st.cache_resource
def get_collection():
    print("DEBUG: Call get_collection()")
    collection = None
    try:
        # delete all documents
        native_db.delete_collection('chat-with-pdfs')
    except:
        pass
    finally:
        collection = native_db.get_or_create_collection('chat-with-pdfs', embedding_function=SentenceTransformerEmbeddingFunction())
    return collection

# get_collection()
def add_files(upload_files):
    collections = get_collection() # deletes the prev data collections and creates the cache of old uploded files

    old_filenames = st.session_state.old_filesnames
    upload_filename = [file.name for file in upload_files]

add_files('test.pdf')