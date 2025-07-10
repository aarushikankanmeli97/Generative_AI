from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS, LanceDB
import bs4

# Load a text file using TextLoader
loader = TextLoader('speech.txt')
text_document = loader.load()
#print(text_document)

# Load a web page using WebBaseLoader
loader = WebBaseLoader(("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=('post-title', 'post-content', 'post-header') 
                       )),)

web_document = loader.load()
#print(web_document)

# Load a PDF file using PyPDFLoader
loader = PyPDFLoader('attention.pdf')
pdf_document = loader.load()
#print(pdf_document)

# Split the text into smaller chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(pdf_document)
#print(text_chunks[:3])

# Create embeddings for the text chunks using OpenAIEmbeddings and store the embeddings in a vector store using Chroma
vector_db = Chroma.from_documents(text_chunks[:20], OpenAIEmbeddings())

query = "What is the attention mechanism in transformers?"
result = vector_db.similarity_search(query)
print(result)

# Create embeddings for the text chunks using OpenAIEmbeddings and store the embeddings in a vector store using FAISS
vector_db_FAISS = FAISS.from_documents(text_chunks[:20], OpenAIEmbeddings())
FAISS_result = vector_db_FAISS.similarity_search(query)
#print(FAISS_result)

# Create embeddings for the text chunks using OpenAIEmbeddings and store the embeddings in a vector store using LanceDB
vector_db_LanceDB = LanceDB.from_documents(text_chunks[:20], OpenAIEmbeddings())
LanceDB_result = vector_db_LanceDB.similarity_search(query)
#print(LanceDB_result)

