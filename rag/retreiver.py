from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama 
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load a PDF file using PyPDFLoader
loader = PyPDFLoader('attention.pdf')
pdf_docs = loader.load()

# Split the text into smaller chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(pdf_docs)

FAISS_vector_db = FAISS.from_documents(text_chunks[:30], OllamaEmbeddings())

query = "What is an attention mechanism in transformers?"

llm_model = Ollama(model='llama2')

# Design Chat prompt template
# 'context' here refers to the documents from the vector store
# 'input' refers to the user query(question)
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the following
       question based on the given context. Think step by step
       and provide a detailed answer.
       <context>
       {context}
       </context>
       Question: {input}""")
       
# Create a chain that combines the vector store and the LLM
document_chain = create_stuff_documents_chain(llm_model, prompt)

# Creater a retriever from the vector store
retriever = FAISS_vector_db.as_retriever()

# Create a chain that combines the retriever and the LLM
retreival_chain = create_retrieval_chain(
    retriever,
    document_chain,
)

response = retreival_chain.invoke({
    "input": "How is the attention matrix calculated in transformers?"    
})
print(response['answer'])