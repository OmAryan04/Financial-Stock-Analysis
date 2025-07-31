import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings


# Load environment variables
load_dotenv()

# Set up huggingface API key
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Re-set the same embedding model used during index creation
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Set the LLM to None to avoid using OpenAI
# This is necessary if you don't want to use OpenAI for querying

Settings.llm = None # llm


# Load storage from the default "storage/" directory
storage_context = StorageContext.from_defaults(persist_dir="./storage")
#Load the index from disk
index = load_index_from_storage(storage_context)

# new version of llama index uses query_engine.query()
query_engine = index.as_query_engine()

# response = query_engine.query("What are some near-term risks to Nvidia?")
# print(response)


response = query_engine.query("Tell me about Google's new supercomputer.")
print(response)