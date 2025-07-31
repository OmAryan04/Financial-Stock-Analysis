import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import load_index_from_storage
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.settings import Settings
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Load environment variables
load_dotenv()

# Get Hugging Face API key
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set up Hugging Face Inference API client
# This is an example of how to use the Hugging Face Inference API client
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

completion = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM3-3B",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    stream=True,
)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Set up Hugging Face LLM (hosted API, no local model needed)
llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceTB/SmolLM3-3B",  
    token=hf_token,
    temperature=0.7,
    max_new_tokens=512
)

Settings.llm = llm  # Set to None if you don't want to use llm

# Load vector index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# Streamlit UI
st.title('Financial Stock Analysis using LlamaIndex')
st.header("Reports:")

report_type = st.selectbox(
    'What type of report do you want?',
    ('Single Stock Outlook', 'Competitor Analysis'))

if report_type == 'Single Stock Outlook':
    symbol = st.text_input("Stock Symbol")
    if symbol:
        with st.spinner(f'Generating report for {symbol}...'):
            response = query_engine.query(
                f"Write a report on the outlook for {symbol} stock from the years 2023-2027. Be sure to include potential risks and headwinds."
            )
            st.write(str(response))

if report_type == 'Competitor Analysis':
    symbol1 = st.text_input("Stock Symbol 1")
    symbol2 = st.text_input("Stock Symbol 2")
    if symbol1 and symbol2:
        with st.spinner(f'Generating report for {symbol1} vs. {symbol2}...'):
            response = query_engine.query(
                f"Write a report on the competition between {symbol1} stock and {symbol2} stock."
            )
            st.write(str(response))
