import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

PDF_PATH = "./Lic_doc.pdf"
loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()

# # Dimensions
# # 768 for all-mpnet-base-v2

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunked_docs = text_splitter.split_documents(raw_docs)

# new url is https://rahulbro123-embedding-model.hf.space/get_embeddings
def get_embeddings(texts):
    response = requests.post("https://rahulbro123-embedding-model.hf.space/get_embeddings", json={"texts": texts})
    response.raise_for_status()
    return response.json()["embeddings"]


vectors = []
for i, doc in enumerate(chunked_docs):
    emb = get_embeddings([doc.page_content])[0]
    vectors.append({
        "id": f"doc_{i}",
        "values": emb,
        "metadata": {
            "text": doc.page_content,           # ✅ store chunk text here
            "source": doc.metadata.get("source", "")  # optional
        }
    })

index.upsert(vectors=vectors)
print(f"Successfully uploaded {len(vectors)} chunks to Pinecone index '{PINECONE_INDEX_NAME}'")


# Delete all vectors inside the index
# index.delete(delete_all=True)

# print(" All vectors deleted successfully.")