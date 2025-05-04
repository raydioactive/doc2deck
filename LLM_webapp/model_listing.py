"""
Functions for fetching available models directly from LLM provider APIs.
This approach eliminates the need for hardcoded model information.
"""

from typing import List, Dict, Any, Optional
import os

def list_gemini_models(api_key: str) -> List[Dict[str, Any]]:
    """
    List available Gemini models directly from the Google Gen AI API.
    
    Args:
        api_key (str): The API key for Google Gen AI.
        
    Returns:
        List[Dict[str, Any]]: A list of model information dictionaries.
    """
    try:
        # Import here to avoid dependency issues if Google SDK isn't installed
        from google import genai
        from google.genai import types
        
        # Initialize the client with the provided API key and explicit API version
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version='v1')
        )
        
        print("Successfully initialized Gemini client")
        models_data = []
        
        # Use the models.list() method to fetch all available models
        print("Fetching models...")
        model_iterator = client.models.list()
        models_list = list(model_iterator)
        print(f"Found {len(models_list)} total models")
        
        # Diagnostic printing of model properties
        if models_list and len(models_list) > 0:
            sample_model = models_list[0]
            print(f"Sample model attributes: {dir(sample_model)}")
        
        # Filter for Gemini models using name pattern
        for model in models_list:
            model_name = str(model.name).lower()
            
            # Filter by 'gemini' in the name
            if 'gemini' in model_name:
                try:
                    # Extract model information with better error handling
                    model_info = {
                        'name': model.name,
                        'display_name': getattr(model, 'display_name', model.name),
                        'input_token_limit': getattr(model, 'input_token_limit', 0),
                        'output_token_limit': getattr(model, 'output_token_limit', 0)
                    }
                    models_data.append(model_info)
                    print(f"Added Gemini model: {model.name}")
                except Exception as model_err:
                    print(f"Error extracting data for model {model.name}: {model_err}")
        
        print(f"Filtered to {len(models_data)} Gemini models")
        return models_data
    
    except Exception as e:
        import traceback
        print(f"Error listing Gemini models: {str(e)}")
        traceback.print_exc()
        return []

def list_claude_models(api_key: str) -> List[Dict[str, Any]]:
    """
    List available Claude models from Anthropic API.
    
    Args:
        api_key (str): The API key for Anthropic.
        
    Returns:
        List[Dict[str, Any]]: A list of model information dictionaries.
    """
    import requests
    
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            models = []
            
            for model in data.get("data", []):
                name = model.get("name", "")
                if "claude" in name.lower():
                    models.append({
                        "name": name,
                        "display_name": name,
                        "context_window": model.get("context_window", 0),
                        "max_tokens": model.get("max_tokens_to_sample", 0)
                    })
            return models
        else:
            print(f"Error listing Claude models: HTTP {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"Error listing Claude models: {e}")
        import traceback
        traceback.print_exc()
        return []

def list_openai_models(api_key: str) -> List[Dict[str, Any]]:
    """
    List available OpenAI models from OpenAI API.
    
    Args:
        api_key (str): The API key for OpenAI.
        
    Returns:
        List[Dict[str, Any]]: A list of model information dictionaries.
    """
    import requests
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            models = []
            
            # Filter for chat models only
            for model in data.get("data", []):
                id = model.get("id", "")
                # Include only relevant models for our use case
                if any(prefix in id for prefix in ["gpt-4", "gpt-3.5"]) and "vision" not in id:
                    # Parse out token limits if available
                    context_window = 0
                    if "gpt-4-turbo" in id or "gpt-4o" in id:
                        context_window = 128000
                    elif "gpt-4-32k" in id:
                        context_window = 32768
                    elif "gpt-4" in id:
                        context_window = 8192
                    elif "gpt-3.5-turbo-16k" in id:
                        context_window = 16385
                    elif "gpt-3.5" in id:
                        context_window = 4096
                        
                    models.append({
                        "name": id,
                        "display_name": id,
                        "context_window": context_window
                    })
            
            return models
        else:
            print(f"Error listing OpenAI models: HTTP {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"Error listing OpenAI models: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_available_models(provider: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Get available models for the specified provider.
    
    Args:
        provider (str): The provider name ('gemini', 'claude', or 'openai').
        api_key (str): The API key for the provider.
        
    Returns:
        List[Dict[str, Any]]: A list of model information dictionaries.
    """
    if provider == 'gemini':
        return list_gemini_models(api_key)
    elif provider == 'claude':
        return list_claude_models(api_key)
    elif provider == 'openai':
        return list_openai_models(api_key)
    else:
        return []