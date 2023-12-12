# embeddings/__init__.py

# API Key as module level variable
api_key = None
embedding_model = "MiniLM"

# data on the models that we host

models_info = {
    "bge-small": {
        "full_name": "BAAI/bge-small-en-v1.5",
        "type": "text",
        "description": "General purpose embedding model",
        "dimensions": 384,
        "token_limit": 512,
        "pricing": "0.0001 per embedding",
    },
    "bge-base": {
        "full_name": "BAAI/bge-base-en-v1.5",
        "type": "text",
        "description": "General purpose embedding model",
        "dimensions": 768,
        "token_limit": 512,
        "pricing": "0.0001 per embedding",
    },
    "MiniLM": {
        "full_name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "text",
        "description": "General purpose embedding model",
        "dimensions": 384,
        "token_limit": 512,
        "pricing": "0.00001 per embedding",
    },
    "jina-small": {
        "full_name": "jinaai/jina-embeddings-v2-small-en",
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

# Import necessary components
from .database import Database
from .collection import Collection
from .generator import Generator