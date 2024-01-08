import requests
import os
from transformers import AutoTokenizer
from typing import Union, List
from . import get_api_key, set_model, get_model, models_info

# Constants
GENERATION_SERVER_URL = "https://generation-cpu-bln3a2qo6a-uc.a.run.app"

class Generator:
    # Initializes the Generator
    # TODO: Should make an api call to set to user's chosen model
    def __init__(self):
        self.tokenizers = {}
        
        # Preload tokenizers for all models
        for model in models_info:
            full_name = models_info[model]["full_name"]
            self.tokenizers[model] = AutoTokenizer.from_pretrained(full_name)

    # List all available embedding models and their details
    def list_models(self):
        return list(models_info.values())

    # Set the model to be used for embedding generation
    def set_model(self, model_name: str):
        if model_name not in models_info:
            raise ValueError("Model not found. Please choose a valid model.")
        set_model(model_name)

    # Get the details of the set model
    def get_model_info(self):
        return models_info.get(get_model(), {})

    # Generates embeddings for texts or files
    def embed(self, input_data: Union[str, List[str]]):
        api_key = get_api_key()
        if api_key is None:
            raise Exception("API key not set. Use embeddings.api_key = API_KEY to set the API key.")

        # Check if the input is a single string and a path to a PDF file
        if isinstance(input_data, str) and input_data.endswith('.pdf'):
            # Handle PDF file input
            return self._embed_pdf(input_data, api_key)
        elif isinstance(input_data, str) or isinstance(input_data, list):
            # Handle text input (both single string and list of strings)
            return self._embed_text(input_data, api_key)
        else:
            raise ValueError("Invalid input data type. Expected a pdf file, a string or a list of strings.")


    # Helper function to generate embeddings for text
    def _embed_text(self, texts: Union[str, List[str]], api_key: str):
        # Convert text to a list if it's a single string
        texts = [texts] if isinstance(texts, str) else texts

        # Check if the text(s) are within the token limit
        for t in texts:
            is_within_limit, message = self.within_token_limit(t)
            if not is_within_limit:
                raise ValueError(message)

        model_full_name = models_info[get_model()]["full_name"]
        response = requests.post(
            f"{GENERATION_SERVER_URL}/embed",
            json={"text": texts, "model_name": model_full_name},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Error in embedding generation: {response.text}")

        return response.json()["embeddings"]

    def _embed_pdf(self, pdf_path: str, api_key: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        with open(pdf_path, 'rb') as file:
            pdf_content = file.read()

        model_full_name = models_info[get_model()]["full_name"]
        response = requests.post(
            f"{GENERATION_SERVER_URL}/embed_pdf",
            files={
                "file": (os.path.basename(pdf_path), pdf_content),
            },
            data={
                "model_name": model_full_name
            },
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Error in PDF embedding generation: {response.text}")

        return response.json()


    # Counts the number of tokens in the given text with respect to the set embedding model
    def count_tokens(self, text: str):
        tokenizer = self.tokenizers[get_model()]
        inputs = tokenizer(text)
        return len(inputs["input_ids"])

    # Checks if the given text is within the token limit of the set embedding model
    def within_token_limit(self, text: str):
        token_count = self.count_tokens(text)
        max_tokens = models_info[get_model()]["token_limit"]
        if token_count > max_tokens:
            return False, f"Text exceeds the token limit of {max_tokens}.\nCurrent token count is {token_count}."
        return True, "Text is within the token limit."
