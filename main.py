import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv


# load api keys
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    print('API has been found')
else:
    print('There has been some error in API fetching')

# reading pdf with PyPDF2.PdfReader
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

text_data = get_pdf_text([r'./test.pdf'])
# print(text_data)

from langchain.text_splitter import RecursiveCharacterTextSplitter
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

chunks_data = get_text_chunks(text_data)
print(len(chunks_data))

import chromadb

# in disk storage
client = chromadb.PersistentClient('./chorma_db')
# Chroma lets you manage collections of embeddings, using the collection primitive.

# create collection
collection = chromadb.Collection(name='my_pdf_collections')
# get collection
collection = client.get_collection(name='my_pdf_collections')

# collection