from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    content = ""
    for page in pages:
        content = content + "\n\n" + page.page_content.encode('utf-8', errors='ignore').decode('utf-8')
    return content

print(load_pdf('test.pdf'))
print(len(load_pdf('test.pdf')))

prompt = ChatPromptTemplate.from_messages(  
    [("system", "You are a very helpful assistant"),  
     ("user",  
      "Based on my Pdf content:{content}. Please answer my question: {question}. Please use the language that I used in the question")]  
)  
llm = ChatOpenAI(api_key=OPENAI_API_KEY)  
output_parser = StrOutputParser()

# The beauty of langchain is this simple line of code  
chain = prompt | llm | output_parser