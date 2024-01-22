# embeddings/__init__.py

# API Key as module level variable
api_key = None
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

openai_key = None

# data on the models that we host

models_info = {
    "BAAI/bge-small-en-v1.5": {
        "full_name": "BAAI/bge-small-en-v1.5",
        "type": "text",
        "description": "General purpose embedding model",
        "dimensions": 384,
        "token_limit": 512,
        "pricing": "0.0001 per embedding",
    },
    "BAAI/bge-base-en-v1.5": {
        "full_name": "BAAI/bge-base-en-v1.5",
        "type": "text",
        "description": "General purpose embedding model",
        "dimensions": 768,
        "token_limit": 512,
        "pricing": "0.0001 per embedding",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "full_name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "text",
        "description": "General purpose embedding model",
        "dimensions": 384,
        "token_limit": 512,
        "pricing": "0.00001 per embedding",
    },
    "jinaai/jina-embeddings-v2-base-en": {
        "full_name": "jinaai/jina-embeddings-v2-base-en",
        "type": "text",
        "description": "General purpose embedding model",
        "dimensions": 512,
        "token_limit": 8192,
        "pricing": "0.00001 per embedding",
    }
}

def init(key):
    global api_key
    api_key = key

def get_api_key():
    return api_key

def set_model(model):
    global embedding_model
    embedding_model = model
    
def get_model():
    return embedding_model

def set_openai_key(key):
    openai_key = key

def get_openai_key():
    return openai_key

# Import necessary components
from .database import Database
from .collection import Collection
from .generator import Generator