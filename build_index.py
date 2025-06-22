from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv("file.env")

# Load PDF data from 'data/' folder
documents = SimpleDirectoryReader('data').load_data()

# Build the index
index = VectorStoreIndex.from_documents(documents)

# Persist the index to disk
index.storage_context.persist(persist_dir="query-engine-storage")

print("✅ Index built and saved to 'query-engine-storage/'")
