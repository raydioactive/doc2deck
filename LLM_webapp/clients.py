# clients.py
import requests
import json
import time
import re
from typing import List, Dict, Optional, Tuple, Any, Mapping
from google import genai  # Updated import for the new SDK
from google.genai import types  # Import types from new SDK

# Default models (can be overridden via constructor)
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"  # Updated from 2.5-pro-latest
DEFAULT_CLAUDE_MODEL = "claude-3-opus-20240229"
DEFAULT_OPENAI_MODEL = "gpt-4-turbo"

# Model context window cache
MODEL_CONTEXT_CACHE = {}

# Define your consistent, optimized prompt
FLASHCARD_PROMPT_TEMPLATE = """Create high-quality Anki flashcards from the following document content.
Focus on the most important concepts for long-term retention.

GUIDELINES:
1. Create atomic cards (one concept per card)
2. Front side should contain a clear, specific question or cue
3. Back side should contain a concise answer without including the question
4. Favor applied understanding over pure memorization
5. Include 15-30 cards depending on content density

FORMAT:
Output must be ONLY a JSON array of objects with "front" and "back" keys.
Do not include any explanatory text, markdown formatting, or code block syntax.

EXAMPLES OF GOOD CARDS:
[
  {{"front": "What is the principle of locality in computer systems?", "back": "The tendency of a processor to access the same set of memory locations repetitively over a short period of time, forming the basis for cache design."}},
  {{"front": "Symptoms of hyperkalemia", "back": "- Muscle weakness\\n- Paresthesia\\n- ECG changes (peaked T waves, widened QRS)\\n- Cardiac arrhythmias"}},
  {{"front": "Why is encapsulation important in OOP?", "back": "It protects the internal state of an object by hiding implementation details, reducing dependencies, and providing a controlled interface for interaction."}}
]

DOCUMENT CONTENT:
{content}"""


def get_model_limits(model_name: str, api_key: str) -> Tuple[int, int]:
    """
    Retrieve token limits for a specific model using the Gen AI SDK.
    Returns (input_token_limit, output_token_limit) tuple.
    """
    # Check cache first
    if model_name in MODEL_CONTEXT_CACHE:
        return MODEL_CONTEXT_CACHE[model_name]
    
    # Provider-specific limit retrieval
    if model_name.startswith("gemini"):
        return _get_gemini_limits(model_name, api_key)
    elif model_name.startswith("claude"):
        return _get_claude_limits(model_name)
    elif "gpt" in model_name:
        return _get_openai_limits(model_name)
    else:
        # Default conservative limits
        return (4096, 4096)

def _get_gemini_limits(model_name: str, api_key: str) -> Tuple[int, int]:
    """Get limits for Gemini models using the SDK."""
    try:
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        # List models to find the specified one
        for model in client.models.list():
            if model.name.endswith(model_name) or model_name in model.name:
                # Extract limits from model metadata
                input_token_limit = getattr(model, "input_token_limit", 30720)
                output_token_limit = getattr(model, "output_token_limit", 4096)
                
                # Cache the result
                MODEL_CONTEXT_CACHE[model_name] = (input_token_limit, output_token_limit)
                return (input_token_limit, output_token_limit)
        
        # Model not found in the list, use defaults based on known models
        if "gemini-1.5" in model_name:
            limits = (1000000, 8192)  # ~1M token context for Gemini 1.5
        elif "gemini-2.5" in model_name:
            limits = (1000000, 8192)  # ~1M token context for Gemini 2.5
        else:
            limits = (30720, 4096)    # Default for older models
            
        MODEL_CONTEXT_CACHE[model_name] = limits
        return limits
    except Exception as e:
        # Fall back to conservative defaults if API call fails
        print(f"Warning: Error retrieving Gemini model limits: {e}")
        limits = (30720, 4096)  # Conservative default
        MODEL_CONTEXT_CACHE[model_name] = limits
        return limits

def _get_claude_limits(model_name: str) -> Tuple[int, int]:
    """Get limits for Claude models based on known specifications."""
    if "claude-3-opus" in model_name:
        limits = (200000, 4096)
    elif "claude-3-sonnet" in model_name:
        limits = (200000, 4096)
    elif "claude-3-haiku" in model_name:
        limits = (150000, 4096)
    else:
        limits = (100000, 4096)  # Conservative default
    
    MODEL_CONTEXT_CACHE[model_name] = limits
    return limits

def _get_openai_limits(model_name: str) -> Tuple[int, int]:
    """Get limits for OpenAI models based on known specifications."""
    if "gpt-4-turbo" in model_name:
        limits = (128000, 4096)
    elif "gpt-4o" in model_name:
        limits = (128000, 4096)
    elif "gpt-3.5-turbo" in model_name:
        limits = (16385, 4096)
    else:
        limits = (8192, 4096)  # Conservative default
    
    MODEL_CONTEXT_CACHE[model_name] = limits
    return limits


def estimate_tokens_from_chars(text: str) -> int:
    """Estimate token count from character count using conservative ratio."""
    # Most tokenizers average 4-6 chars per token
    # Using 4 for a conservative estimate
    return len(text) // 4


def chunk_content(content: str, max_chunk_size: int) -> List[str]:
    """Split content into chunks based on token limits."""
    if estimate_tokens_from_chars(content) <= max_chunk_size:
        return [content]
    
    # Simple paragraph-based chunking
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para_tokens = estimate_tokens_from_chars(para)
        current_chunk_tokens = estimate_tokens_from_chars(current_chunk)
        
        if current_chunk_tokens + para_tokens + 2 <= max_chunk_size:
            current_chunk += (para + "\n\n")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            
            # Handle paragraphs that exceed chunk size individually
            if para_tokens > max_chunk_size:
                # Split by sentences if a paragraph is too large
                sentences = para.split('. ')
                sentence_chunk = ""
                
                for sentence in sentences:
                    if estimate_tokens_from_chars(sentence_chunk + sentence + '. ') <= max_chunk_size:
                        sentence_chunk += sentence + '. '
                    else:
                        if sentence_chunk:
                            chunks.append(sentence_chunk.strip())
                        sentence_chunk = sentence + '. '
                
                if sentence_chunk:
                    current_chunk = sentence_chunk
                else:
                    current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


class LLMClient:
    """Base class for LLM API clients"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name  # Specific model set by implementation
        self.input_token_limit = 0    # Will be set by subclass
        self.output_token_limit = 0   # Will be set by subclass

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        """Generates flashcards using the specific LLM API."""
        raise NotImplementedError
        
    def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists available models for this provider."""
        raise NotImplementedError

    def _parse_llm_response(self, response_text: str) -> List[Dict[str, str]]:
        """Improved parser for LLM responses into flashcard dictionaries."""
        import re
        try:
            # Clean up potential markdown, code blocks, etc.
            cleaned_text = response_text.strip()
            
            # Handle code blocks with or without language specifier
            json_block_patterns = [
                r"```json\s*([\s\S]*?)\s*```",  # ```json block
                r"```\s*([\s\S]*?)\s*```",      # ``` block without language
                r"\{[\s\S]*\}"                  # Just find JSON-like content
            ]
            
            for pattern in json_block_patterns:
                matches = re.findall(pattern, cleaned_text)
                if matches:
                    for potential_json in matches:
                        try:
                            parsed = json.loads(potential_json)
                            if self._validate_cards_structure(parsed):
                                return parsed
                        except json.JSONDecodeError:
                            continue
            
            # Final attempt: try parsing the whole text
            parsed = json.loads(cleaned_text)
            if self._validate_cards_structure(parsed):
                return parsed
                
            raise ValueError("Couldn't extract valid JSON from response")
            
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}") from e
            
    def _validate_cards_structure(self, data) -> bool:
        """Validate the parsed JSON has the expected flashcard structure."""
        if not isinstance(data, list):
            return False
            
        # Empty list is technically valid
        if not data:
            return True
            
        # Check that all items have front/back keys
        for item in data:
            if not isinstance(item, dict) or 'front' not in item or 'back' not in item:
                return False
                
        return True


# --- Updated Gemini Client using new API ---
class GeminiClient(LLMClient):
    """Client for Google's Gemini API using the google-generativeai SDK"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        super().__init__(api_key)
        self.client = genai.Client(api_key=self.api_key)
        
        # If model not specified, find best available model
        if not model_name:
            available_models = self.list_available_models()
            # Default fallback models to try in order
            fallback_models = [
                "gemini-1.5-pro-latest",
                "gemini-1.5-pro",
                "gemini-1.5-flash-latest",
                "gemini-1.0-pro",
                "gemini-pro"
            ]
            
            model_found = False
            available_model_names = [m['name'] for m in available_models]
            
            # Try each fallback model in order
            for fallback in fallback_models:
                for available in available_model_names:
                    if fallback in available:
                        model_name = available
                        model_found = True
                        break
                if model_found:
                    break
                    
            # If still no model, use first available
            if not model_found and available_models:
                model_name = available_models[0]['name']
        
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        
        try:
            # Get token limits for this model
            self.input_token_limit, self.output_token_limit = get_model_limits(self.model_name, self.api_key)
            
        except Exception as e:
            # Catch potential configuration errors early
            raise RuntimeError(f"Failed to initialize Google Gemini client: {e}") from e

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        # Use the smaller of requested tokens and model limit
        max_output_tokens = min(max_output_tokens, self.output_token_limit)
        
        # Calculate conservative token estimate
        estimated_tokens = estimate_tokens_from_chars(content)
        
        # If content exceeds context window, chunk it
        if estimated_tokens > self.input_token_limit * 0.75:  # 75% of limit to be safe
            print(f"Content exceeds 75% of token limit ({estimated_tokens} vs {self.input_token_limit}), chunking...")
            chunks = chunk_content(content, int(self.input_token_limit * 0.75))
            all_cards = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} ({estimate_tokens_from_chars(chunk)} estimated tokens)")
                chunk_prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=chunk)
                chunk_cards = self._generate_cards_for_chunk(chunk_prompt, temperature, max_output_tokens)
                all_cards.extend(chunk_cards)
            
            return all_cards
        else:
            # Process normally for content within limits
            prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=content)
            return self._generate_cards_for_chunk(prompt, temperature, max_output_tokens)

    def _generate_cards_for_chunk(self, prompt: str, temperature: float, max_output_tokens: int) -> List[Dict[str, str]]:
        # Add retry logic
        max_retries = 3
        backoff_base = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Generate content using the API
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

            except (ValueError, RuntimeError) as e:
                # Re-raise parsing errors on final attempt
                if attempt == max_retries - 1:
                    raise e
                
                # Otherwise, retry with backoff
                backoff_time = backoff_base ** attempt
                print(f"API error: {e}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                
            except Exception as e:
                # Catch other unexpected errors
                raise RuntimeError(f"An unexpected error occurred with the Google AI SDK: {e}") from e
                
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available Gemini models."""
        try:
            models = []
            for model in self.client.models.list():
                # Filter to only include generative models (not embedding models)
                if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                    name = getattr(model, 'name', '')
                    if name and ('gemini' in name.lower()):
                        # Extract version and capabilities
                        display_name = getattr(model, 'display_name', name)
                        version = getattr(model, 'version', 'unknown')
                        input_token_limit = getattr(model, 'input_token_limit', 0)
                        output_token_limit = getattr(model, 'output_token_limit', 0)
                        
                        models.append({
                            'name': name,
                            'display_name': display_name,
                            'version': version,
                            'input_token_limit': input_token_limit,
                            'output_token_limit': output_token_limit
                        })
            return models
        except Exception as e:
            print(f"Error listing Gemini models: {e}")
            return []


# --- Claude Client (with improved retry logic) ---
class ClaudeClient(LLMClient):
    """Client for Anthropic's Claude API"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        super().__init__(api_key)
        self.model_name = model_name or DEFAULT_CLAUDE_MODEL
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        # Get token limits for this model
        self.input_token_limit, self.output_token_limit = get_model_limits(self.model_name, api_key)

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        # Use the smaller of requested tokens and model limit
        max_output_tokens = min(max_output_tokens, self.output_token_limit)
        
        # Calculate conservative token estimate
        estimated_tokens = estimate_tokens_from_chars(content)
        
        # If content exceeds context window, chunk it
        if estimated_tokens > self.input_token_limit * 0.75:  # 75% of limit to be safe
            print(f"Content exceeds 75% of token limit ({estimated_tokens} vs {self.input_token_limit}), chunking...")
            chunks = chunk_content(content, int(self.input_token_limit * 0.75))
            all_cards = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} ({estimate_tokens_from_chars(chunk)} estimated tokens)")
                chunk_prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=chunk)
                chunk_cards = self._generate_cards_for_chunk(chunk_prompt, temperature, max_output_tokens)
                all_cards.extend(chunk_cards)
            
            return all_cards
        else:
            # Process normally for content within limits
            prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=content)
            return self._generate_cards_for_chunk(prompt, temperature, max_output_tokens)

    def _generate_cards_for_chunk(self, prompt: str, temperature: float, max_output_tokens: int) -> List[Dict[str, str]]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        request_body = {
            "model": self.model_name, 
            "max_tokens": max_output_tokens, 
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Add retry logic
        max_retries = 3
        backoff_base = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=request_body, timeout=180)
                response.raise_for_status()
                data = response.json()
                
                if data.get("content") and isinstance(data["content"], list) and data["content"][0].get("type") == "text":
                    response_text = data["content"][0]["text"]
                    return self._parse_llm_response(response_text)
                else: 
                    raise ValueError(f"Unexpected Claude response structure: {data.keys()}")
                    
            except requests.exceptions.HTTPError as e:
                error_message = f"HTTP error calling Claude API ({e.response.status_code}): {e}"
                try: 
                    error_message += f"\nResponse: {json.dumps(e.response.json(), indent=2)}"
                except json.JSONDecodeError: 
                    error_message += f"\nResponse: {e.response.text}"
                
                # On final attempt, raise the error
                if attempt == max_retries - 1:
                    raise RuntimeError(error_message) from e
                
                # Otherwise, retry with backoff
                backoff_time = backoff_base ** attempt
                print(f"API error: {error_message}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                
            except requests.exceptions.RequestException as e: 
                # Network errors may be transient, retry
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Network error calling Claude API: {e}") from e
                
                backoff_time = backoff_base ** attempt
                print(f"Network error: {e}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                
            except Exception as e: 
                # Other errors are likely not retriable
                raise RuntimeError(f"An error occurred during Claude API interaction: {e}") from e
                
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available Claude models."""
        try:
            headers = {
                "x-api-key": self.api_key,
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
                return []
        except Exception as e:
            print(f"Error listing Claude models: {e}")
            # Fall back to hardcoded models
            return [
                {"name": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "context_window": 200000, "max_tokens": 4096},
                {"name": "claude-3-sonnet-20240229", "display_name": "Claude 3 Sonnet", "context_window": 200000, "max_tokens": 4096},
                {"name": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku", "context_window": 150000, "max_tokens": 4096}
            ]


# --- OpenAI Client (with improved retry logic) ---
class OpenAIClient(LLMClient):
    """Client for OpenAI's API"""
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        super().__init__(api_key)
        self.model_name = model_name or DEFAULT_OPENAI_MODEL
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        # Get token limits for this model
        self.input_token_limit, self.output_token_limit = get_model_limits(self.model_name, api_key)

    def generate_flashcards(self, content: str, temperature: float = 0.7, max_output_tokens: int = 4096) -> List[Dict[str, str]]:
        # Use the smaller of requested tokens and model limit
        max_output_tokens = min(max_output_tokens, self.output_token_limit)
        
        # Calculate conservative token estimate
        estimated_tokens = estimate_tokens_from_chars(content)
        
        # If content exceeds context window, chunk it
        if estimated_tokens > self.input_token_limit * 0.75:  # 75% of limit to be safe
            print(f"Content exceeds 75% of token limit ({estimated_tokens} vs {self.input_token_limit}), chunking...")
            chunks = chunk_content(content, int(self.input_token_limit * 0.75))
            all_cards = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} ({estimate_tokens_from_chars(chunk)} estimated tokens)")
                chunk_prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=chunk)
                chunk_cards = self._generate_cards_for_chunk(chunk_prompt, temperature, max_output_tokens)
                all_cards.extend(chunk_cards)
            
            return all_cards
        else:
            # Process normally for content within limits
            prompt = FLASHCARD_PROMPT_TEMPLATE.format(content=content)
            return self._generate_cards_for_chunk(prompt, temperature, max_output_tokens)

    def _generate_cards_for_chunk(self, prompt: str, temperature: float, max_output_tokens: int) -> List[Dict[str, str]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        system_prompt = "You are an assistant that generates flashcards strictly in the specified JSON array format."
        
        request_body = {
            "model": self.model_name, 
            "temperature": temperature, 
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": prompt}
            ]
        }
        
        if "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
            request_body["response_format"] = {"type": "json_object"}
        
        # Add retry logic
        max_retries = 3
        backoff_base = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=request_body, timeout=180)
                response.raise_for_status()
                data = response.json()
                
                if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                    response_text = data["choices"][0]["message"]["content"]
                    return self._parse_llm_response(response_text)
                else: 
                    raise ValueError(f"Unexpected OpenAI response structure: {data}")
                    
            except requests.exceptions.HTTPError as e:
                error_message = f"HTTP error calling OpenAI API ({e.response.status_code}): {e}"
                try: 
                    error_message += f"\nResponse: {json.dumps(e.response.json(), indent=2)}"
                except json.JSONDecodeError: 
                    error_message += f"\nResponse: {e.response.text}"
                
                # On final attempt, raise the error
                if attempt == max_retries - 1:
                    raise RuntimeError(error_message) from e
                
                # Otherwise, retry with backoff
                backoff_time = backoff_base ** attempt
                print(f"API error: {error_message}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                
            except requests.exceptions.RequestException as e: 
                # Network errors may be transient, retry
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Network error calling OpenAI API: {e}") from e
                
                backoff_time = backoff_base ** attempt
                print(f"Network error: {e}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                
            except Exception as e: 
                # Other errors are likely not retriable
                raise RuntimeError(f"An error occurred during OpenAI API interaction: {e}") from e
                
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                models = []
                
                # Filter for chat models only
                chat_models = []
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
                            
                        chat_models.append({
                            "name": id,
                            "display_name": id,
                            "context_window": context_window
                        })
                
                return chat_models
            else:
                print(f"Error listing OpenAI models: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            # Fall back to hardcoded models
            return [
                {"name": "gpt-4-turbo", "display_name": "GPT-4 Turbo", "context_window": 128000},
                {"name": "gpt-4o", "display_name": "GPT-4o", "context_window": 128000},
                {"name": "gpt-3.5-turbo", "display_name": "GPT-3.5 Turbo", "context_window": 16385}
            ]