# embeddings/__init__.py

# API Key as module level variable
api_key = None

def init(key):
    global api_key
    api_key = key

def get_api_key():
    return api_key

# Import necessary components
from .database import Database
from .collection import Collection
from .generator import Generator