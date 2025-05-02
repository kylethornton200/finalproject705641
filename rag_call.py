import os
from typing import Union, Dict, Tuple
try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("Please install the dotenv package using pip")
import requests
load_dotenv()
API_KEY = os.getenv("OPEN_AI_KEY")

if not API_KEY:
    raise ValueError("API Key not found. Ensure OPEN_AI_KEY is set in environment variables or .env file")

def get_output(prompts: Dict[str, str], url: str = "https://api.openai.com/v1/chat/completions", temperature: float = 0.1, model: str = "gpt-4o") -> str:
    """
    Calls the OpenAI API for chat completion.
    Args:
        prompts: Dict[str,str] the prompt message.
        full_output (bool, optional): Whether to give an answer besides true or false/ give sources.
        temperature (float, optional): The creativity setting for responses. Defaults to 0.1.
        model(string, optional): What model of chatgpt do we want to use. Defaults to gpt-4o
    Returns:
        str: The response content.
    Raises:
        ValueError: If API key is missing or prompts are incorrectly formatted.
        Various Request exceptions.
    """
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    if not isinstance(prompts, dict) or "system" not in prompts or "user" not in prompts:
        raise ValueError("For system prompts, provide a dictionary with 'system' and 'user' keys.")
    
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]}
    ]

    if model == "o1" or model == "o1-mini":
        data = {
        "model": model,
        "messages": messages,
        }

    else:
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises an error for HTTP error responses
        json_response = response.json()

        if "error" in json_response:
            raise requests.exceptions.HTTPError(json_response["error"]["message"])
        
        if "choices" not in json_response or not json_response["choices"]:
            raise ValueError("Unexpected API response: missing 'choices' key.")
        
        return json_response['choices'][0]['message']['content']
    
    except requests.exceptions.HTTPError as http_err:
        raise requests.exceptions.HTTPError(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.ConnectionError("Failed to connect to OpenAI API. Check your internet connection.")
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout("Request timed out. Try again later.")
    except requests.exceptions.RequestException as err:
        raise requests.exceptions.RequestException(f"An unexpected error occurred: {err}")
