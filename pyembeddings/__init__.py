# embeddings/__init__.py

# API Key as module level variable
api_key = None
embedding_model = "MiniLM"

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