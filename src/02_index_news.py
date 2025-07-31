import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set up Hugging Face embedding model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    token=hf_token  # only needed for gated/private models
)

# Set the embedding model in global settings
Settings.embed_model = embed_model

# Load documents from 'articles/' directory
documents = SimpleDirectoryReader('articles').load_data()

# Build vector index using Hugging Face embeddings
index = VectorStoreIndex.from_documents(documents)

# Persist index to disk
index.storage_context.persist()