import requests
from .collection import Collection
from . import get_api_key, models_info, set_model

API_URL = "https://storage.silverarrow.ai"


class Database:
    # Creates a new collection in ChromaDB and returns a Collection object.
    # :param name: Name of the collection to be created
    # :param model_name: Name of the model to be used for embeddings (optional).
    # :return: Collection object.
    def create_collection(self, name, model=None):
        api_key = get_api_key()
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")
        if model is None:
            raise Exception(
                "Model to be used for the collection not provided.")
        if model not in models_info:
            raise ValueError("Model not found. Please choose a valid model.")
        set_model(model)
        modelInfo = models_info[model]
        # Make API request to create a new collection
        response = requests.post(
            f"{API_URL}/create_collection",
            json={"collection_name": name, "embedding_model": modelInfo["full_name"]},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            return Collection(name)
        else:
            raise Exception(f"Failed to create collection: {response.text}")

    # Connects to an existing collection in ChromaDB and returns a Collection object.
    # :param name: Name of the collection to be loaded.
    # :return: Collection object.
    def get_collection(self, name):
        api_key = get_api_key()
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        # Make API request to get the collection
        response = requests.post(
            f"{API_URL}/get_collection",
            json={"collection_name": name},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        model_full_name = requests.post(
            f"{API_URL}/get_collection_model",
            json={"collection_name": self.collection_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        model_full_name = model_full_name.json()

        for model in models_info:
            if models_info[model]["full_name"] == model_full_name:
                set_model(model)
                break

        if response.status_code == 200:
            return Collection(name)
        elif response.status_code == 404:
            # TODO: Add a 404 if the collection is not found
            raise Exception(f"Collection '{name}' not found.")
        else:
            raise Exception(f"Failed to load collection: {response.text}")

    # Deletes a collection from ChromaDB.
    # :param name: Name of the collection to be deleted.
    def delete_collection(self, name):
        api_key = get_api_key()
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        # Making the API request to delete the collection
        response = requests.post(
            f"{API_URL}/delete_collection",
            json={"collection_name": name},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Failed to delete collection: {response.text}")
