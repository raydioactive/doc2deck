# clients.py
import requests
import json
from typing import List, Dict, Optional
from google import genai  # Updated import for the new SDK
from google.genai import types  # Import types from new SDK
from google.api_core import exceptions as google_exceptions  # Import SDK exceptions

# Default models (can be overridden via constructor)
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro-latest"
DEFAULT_CLAUDE_MODEL = "claude-3-opus-20240229"
DEFAULT_OPENAI_MODEL = "gpt-4-turbo"

# Define your consistent, optimized prompt here
# ESCAPE LITERAL BRACES in the example JSON by using {{ and }}
FLASHCARD_PROMPT_TEMPLATE = """Create comprehensive Anki flashcards from the following document content.
Generate question-answer pairs that cover key concepts, definitions, facts, and relationships.
Focus on creating atomic cards (one main idea per card) for effective learning.
Format the entire output strictly as a single JSON array of objects, where each object has a "front" key (the question or term) and a "back" key (the answer or definition).
Example format: [{{"front": "Question 1?", "back": "Answer 1."}}, {{"front": "Term A", "back": "Definition A."}}]
Ensure the output is only the JSON array, without any introductory text, explanations, or markdown formatting like ```json ... ```.

Document content:
{content}"""


class LLMClient:
    """Base class for LLM API clients"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name # Specific model set by implementation

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        """Generates flashcards using the specific LLM API."""
        raise NotImplementedError

    def _parse_llm_response(self, response_text: str) -> List[Dict[str, str]]:
        """Attempts to parse the LLM response text into a list of flashcard dictionaries."""
        try:
            # Basic cleanup for markdown code blocks
            text_to_parse = response_text.strip()
            if text_to_parse.startswith("```json"):
                text_to_parse = text_to_parse.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif text_to_parse.startswith("```"):
                 text_to_parse = text_to_parse.split("```", 1)[1].rsplit("```", 1)[0].strip()

            # Parse the JSON
            flashcards = json.loads(text_to_parse)

            # Validate structure
            if isinstance(flashcards, list) and all(isinstance(card, dict) and 'front' in card and 'back' in card for card in flashcards):
                # Basic content check (optional): Filter out empty cards
                return [card for card in flashcards if card.get('front', '').strip() or card.get('back', '').strip()]
            else:
                raise ValueError("Parsed JSON is not a list of {'front': ..., 'back': ...} objects.")

        except json.JSONDecodeError as json_e:
            # Provide context for debugging
            raise ValueError(f"Failed to parse JSON response. Error: {json_e}. Response text received:\n{response_text[:500]}...") from json_e
        except Exception as e:
             # Catch unexpected parsing issues
            raise ValueError(f"An unexpected error occurred during response parsing: {e}") from e


# --- Updated Gemini Client using new API ---
class GeminiClient(LLMClient):
    """Client for Google's Gemini API using the google-generativeai SDK"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        super().__init__(api_key)
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        try:
            # Create a client instance instead of configuring globally
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            # Catch potential configuration errors early
            raise RuntimeError(f"Failed to initialize Google Gemini client: {e}") from e

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=content)

        try:
            # Generate content using the new API pattern
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_mime_type='application/json'  # Request JSON formatting
                )
            )

            # Check if response is empty
            if not response.text:
                raise ValueError("Gemini response was empty. Check safety settings or prompt.")

            response_text = response.text
            return self._parse_llm_response(response_text)

        except google_exceptions.GoogleAPIError as e:
            # Handle API errors
            error_message = f"Google AI SDK API error: {e}"
            if hasattr(e, 'message'): 
                error_message += f" Details: {e.message}"
            raise RuntimeError(error_message) from e
        except ValueError as e:
            # Re-raise parsing errors
            raise e
        except Exception as e:
            # Catch other unexpected errors
            raise RuntimeError(f"An unexpected error occurred with the Google AI SDK: {e}") from e


# --- Claude Client (Remains the same - uses requests) ---
class ClaudeClient(LLMClient):
    """Client for Anthropic's Claude API"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        super().__init__(api_key)
        self.model_name = model_name or DEFAULT_CLAUDE_MODEL
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=content)
        request_body = {
            "model": self.model_name, "max_tokens": max_output_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=request_body, timeout=180)
            response.raise_for_status()
            data = response.json()
            if data.get("content") and isinstance(data["content"], list) and data["content"][0].get("type") == "text":
                 response_text = data["content"][0]["text"]
                 return self._parse_llm_response(response_text)
            else: raise ValueError(f"Unexpected Claude response structure: {data.keys()}")
        except requests.exceptions.HTTPError as e:
             error_message = f"HTTP error calling Claude API ({e.response.status_code}): {e}"
             try: error_message += f"\nResponse: {json.dumps(e.response.json(), indent=2)}"
             except json.JSONDecodeError: error_message += f"\nResponse: {e.response.text}"
             raise RuntimeError(error_message) from e
        except requests.exceptions.RequestException as e: raise RuntimeError(f"Network error calling Claude API: {e}") from e
        except Exception as e: raise RuntimeError(f"An error occurred during Claude API interaction: {e}") from e


# --- OpenAI Client (Remains the same - uses requests) ---
class OpenAIClient(LLMClient):
    """Client for OpenAI's API"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        super().__init__(api_key)
        self.model_name = model_name or DEFAULT_OPENAI_MODEL
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        system_prompt = "You are an assistant that generates flashcards strictly in the specified JSON array format."
        user_prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=content)
        request_body = {
            "model": self.model_name, "temperature": temperature, "max_tokens": max_output_tokens,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }
        if "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
             request_body["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(self.base_url, headers=headers, json=request_body, timeout=180)
            response.raise_for_status()
            data = response.json()
            if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                response_text = data["choices"][0]["message"]["content"]
                return self._parse_llm_response(response_text)
            else: raise ValueError(f"Unexpected OpenAI response structure: {data}")
        except requests.exceptions.HTTPError as e:
            error_message = f"HTTP error calling OpenAI API ({e.response.status_code}): {e}"
            try: error_message += f"\nResponse: {json.dumps(e.response.json(), indent=2)}"
            except json.JSONDecodeError: error_message += f"\nResponse: {e.response.text}"
            raise RuntimeError(error_message) from e
        except requests.exceptions.RequestException as e: raise RuntimeError(f"Network error calling OpenAI API: {e}") from e
        except Exception as e: raise RuntimeError(f"An error occurred during OpenAI API interaction: {e}") from e