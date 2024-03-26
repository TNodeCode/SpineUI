import requests


class ModelEndpoint:
    @staticmethod
    def get_available_models(url: str):
        response = requests.get(f"{url}/available_models")
        return response.json()