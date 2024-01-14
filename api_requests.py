import requests
import os
from dotenv import load_dotenv

load_dotenv(".env")

def make_gapi_request():
    api_key = os.getenv('API_KEY')
    symbol = "XAU"
    curr = "EUR"
    date = ""

    url = f"https://www.goldapi.io/api/{symbol}/{curr}{date}"
    
    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        result = response.text
        return result
    except requests.exceptions.RequestException as e:
        print("Error:", str(e))
        
# result = make_gapi_request()
# print(result)