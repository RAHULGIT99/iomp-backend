import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# loading pdf
PDF_PATH = "./DotCom-CompanyProfile.pdf"
loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()
print(f"Total pages loaded: {len(raw_docs)}")

# splitting to chunks
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunked_docs = text_splitter.split_documents(raw_docs)
print(f"Total chunks: {len(chunked_docs)}")

# initializing embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Gemini embedding model
    google_api_key=GEMINI_API_KEY
)

# initializing pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)


# embedding and uploading to Pinecone
vector_store = PineconeVectorStore.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)

print(f"Successfully uploaded {len(chunked_docs)} chunks to Pinecone index '{PINECONE_INDEX_NAME}'")
