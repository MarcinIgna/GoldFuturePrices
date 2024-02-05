import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np

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

        result = response.json()  # Convert response to JSON

        # Ensure 'Date' is in the correct format and type
        result['Date'] = int(result['timestamp'])  # Remove [0] since 'timestamp' is now an integer

        # Create DataFrame with the desired columns
        api_df = pd.DataFrame({
            'Date': [result['Date']],
            'Open': [result['open_price']],
            'High': [result['high_price']],
            'Low': [result['low_price']],
            'Close': [result['price']]
        })
        print("return form API: ",api_df)
        return api_df
    except requests.exceptions.RequestException as e:
        print("Error:", str(e))
        return None  # Handle error gracefully