import requests
from http.client import HTTPResponse
from io import BytesIO
import re
import os

from . import get_api_key, get_model, models_info

# TODO: Replace with hosted API URL
API_URL = "https://storage.silverarrow.ai"


class Collection:
    def __init__(self, collection_name):
        self.collection_name = collection_name

    # Renames a collection in the db.
    # :param new_collection_name: The name to rename the collection to.
    def rename(self, new_collection_name):
        api_key = get_api_key()
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/rename_collection",
            json={"collection_name": self.collection_name, "new_collection_name": new_collection_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Returns the number of items in the collection.
    def count(self):
        api_key = get_api_key()
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/count_collection",
            json={"collection_name": self.collection_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Returns the embedding model used by the collection
    def model(self):
        api_key = get_api_key()
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/get_collection_model",
            json={"collection_name": self.collection_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Error fetching collection model: {response.text}")

        return response.json()

    # Adds embeddings to the collection.
    # :param documents: List of texts corresponding to the embeddings to be added.
    # :param metadatas: List of dictionaries of metadata corresponding to the embeddings to be added.
    # :param ids: List of ids corresponding to the embeddings to be added.
    # :param embeddings: List of embeddings to be added.
    # TODO: Add check for data types of parameters
    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        api_key = get_api_key()
        if api_key is None:
            raise Exception(
                "API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/add_embeddings",
            json={
                "collection_name": self.collection_name,
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids,
                "embeddings": embeddings,
                "model_name": get_model()
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    def add_file(self, file_path: str):
        api_key = get_api_key()
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")

        collection_name = self.collection_name
        model_full_name = models_info[get_model()]["full_name"]
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        response = requests.post(
            f"{API_URL}/add_file",
            files={
                "file": (os.path.basename(file_path), file_content),
            },
            data={
                "collection_name": collection_name,
                "model_name": model_full_name
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Error in file addition: {response.text}")

        return response.json()

    # Queries the collection for similar embeddings.
    # :param query_embeddings: List of embeddings to be queried.
    # :param n_results: Number of results to return.
    # !!! TODO: CURRENTLY ONLY SUPPORTS ONE DICTIONARY FOR FILTER
    # :param (optional) where: Dictionary of metadata filters to be applied.
    # :param (optional) include_embeddings: Whether to include embeddings in the response.
    def query(self, embedding, n_results=10, where=None, include_embeddings=False):
        api_key = get_api_key()
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        include = None
        if include_embeddings:
            include = ["metadatas", "documents", "embeddings"]

        response = requests.post(
            f"{API_URL}/query_collection",
            json={
                "collection_name": self.collection_name,
                "query_embeddings": [embedding],
                "n_results": n_results,
                "where": where,
                "include": include
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Gets the embeddings for the given ids, search string or where clause.
    # :param ids: List of ids to get embeddings for.
    # :param (optional) where: Dictionary of metadata filter to be applied.
    # :param (optional) search_string: Search string to be applied.
    # :param (optional) include_embeddings: Whether to include embeddings in the response.
    def get(self, ids=None, where=None, search_string=None, include_embeddings=False):
        api_key = get_api_key()
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        include = None
        if include_embeddings:
            include = ["metadatas", "documents", "embeddings"]

        response = requests.post(
            f"{API_URL}/get_collection_items",
            json={
                "collection_name": self.collection_name,
                "ids": ids,
                "where": where,
                "search_string": search_string,
                "include": include
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()

    # Deletes the given ids, or where clause from the collection.
    # :param ids: List of ids to delete.
    # :param (optional) where: Dictionary of metadata filter to be applied.
    def delete(self, ids=None, where=None):
        api_key = get_api_key()
        response = requests.post(
            f"{API_URL}/delete_from_collection",
            json={"collection_name": self.collection_name, "ids": ids, "where": where},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()    

    # Retrieval augmented generation returns a string in response to query both in language and in context of the query.
    # :param query: Query string.
    # :param n_results: Number of results to return.
    # :return: prompt_response: String generated in response to query.
    # :return: results: List of raw retrieval results.
    def retrieval_augmented_generation(self, query, n_results=10):
        api_key = get_api_key()
        model_full_name = models_info[get_model()]["full_name"]
        response = requests.post(
            f"{API_URL}/retrieval_augmented_generation",
            json={"collection_name": self.collection_name, "embedding_model": model_full_name, "query": query, "n_results": n_results},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        # Return just prompt_response 
        return response.json()['prompt_response']
    

    # Helper function to download a single file from a specified endpoint.
    def download_file(self, endpoint, filename):
        api_key = get_api_key()
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        response = requests.post(
            f"{API_URL}/{endpoint}",
            json={"collection_name": self.collection_name},
            headers={"Authorization": f"Bearer {api_key}"},
            stream=True
        )

        if response.status_code == 200:
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                filenames = re.findall('filename="([^"]+)"', content_disposition)
                if filenames:
                    filename = os.path.join(os.path.dirname(filename), filenames[0])

            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)

            return filename
        else:
            raise Exception(f"Failed to download file: {response.status_code} - {response.text}")


    # Export function to download both vectors and metadata TSV files.
    def export(self, directory=''):
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        saved_files = []
        for endpoint, default_filename in [("get_vectors_tsv", "vectors.tsv"), ("get_metadata_tsv", "metadata.tsv")]:
            full_path = os.path.join(directory, default_filename)
            filename = self.download_file(endpoint, full_path)
            saved_files.append(filename)
        print(f"Saved files: {saved_files}")
        return saved_files
