# Doc2Deck: Document to Anki Flashcard Generator

Convert documents (DOCX, PPTX, PDF) into Anki flashcard decks using LLMs like Google Gemini, Anthropic Claude, and OpenAI GPT.

## Features

* **Multiple Input Formats:** Supports `.docx`, `.pptx`, and `.pdf` files
* **LLM Providers:** Google Gemini, Anthropic Claude, and OpenAI GPT
* **Dynamic Model Selection:** Fetches available models directly from provider APIs
* **Multiple Output Formats:**
  * Anki Package (`.apkg`) - Ready to import into Anki
  * Tab-Separated Values (`.txt`) - Plain text format
  * Comma-Separated Values (`.csv`) - Standard spreadsheet format
* **Dual Interfaces:**
  * Command-Line Interface (`doc2deck`) for batch processing
  * Web Interface for interactive editing and preview
* **Advanced LLM Options:**
  * Temperature control (creativity vs. determinism)
  * Token limit customization (control output size)
  * Model-specific parameters

## Installation

### Prerequisites

* Python 3.8+ 
* pip (Python package installer)

### Method 1: Install from GitHub

```bash
pip install git+https://github.com/raydioactive/doc2deck.git
```

### Method 2: Manual Installation

```bash
git clone https://github.com/raydioactive/doc2deck.git
cd doc2deck
pip install -r requirements.txt
```

## Usage

### API Keys

Doc2Deck requires an API key from at least one provider:

* **Environment Variable (Recommended):**
  ```bash
  export LLM_API_KEY=your_api_key_here
  ```

* **Command Line Option:**
  ```bash
  doc2deck --api-key YOUR_API_KEY ...
  ```

### Command-Line Interface

#### Basic Usage

```bash
doc2deck generate notes.docx --provider gemini --output flashcards.apkg
```

#### Advanced Options

```bash
doc2deck generate lecture.pdf slides.pptx notes.docx \
    --provider gemini \
    --model gemini-2.5-pro-preview-03-25 \
    --deck-name "Biology Course" \
    --output bio_cards.apkg \
    --temperature 0.7 \
    --max-tokens 12000 \
    --filter-cards
```

#### List Available Models

```bash
doc2deck list-models --provider gemini
```

### Web Interface

1. Start the server:
   ```bash
   python -m LLM_webapp.app
   ```

2. Open your browser to `http://127.0.0.1:5000`

3. Use the interface to:
   * Upload files
   * Select LLM provider and model
   * Configure parameters
   * Generate, review, and edit flashcards
   * Export to desired format

## Model Selection

Doc2Deck dynamically fetches available models from each provider's API. The list of models varies based on:

* Your API key's permissions
* Provider account type
* Regional availability
* Model versions

### Gemini Models

```
doc2deck list-models --provider gemini
```

Typically includes models like:
* `gemini-2.5-pro-preview-03-25`
* `gemini-1.5-pro-latest`
* `gemini-1.5-flash-latest`

### Claude Models

```
doc2deck list-models --provider claude
```

Typically includes models like:
* `claude-3-opus-20240229`
* `claude-3-sonnet-20240229`
* `claude-3-haiku-20240307`

### OpenAI Models

```
doc2deck list-models --provider openai
```

Typically includes models like:
* `gpt-4-turbo`
* `gpt-4o`
* `gpt-3.5-turbo`

## Output Formats

### Anki Package (.apkg)

* Standard Anki deck format
* Compatible with Anki Desktop, AnkiDroid, and AnkiMobile
* Preserves Markdown formatting for hierarchical lists and basic formatting

### Tab-Separated Values (.txt)

* Simple text format
* Compatible with most flashcard apps
* One card per line: `Question<tab>Answer`

### CSV Format (.csv)

* Standard comma-separated values format
* Compatible with spreadsheets and many flashcard apps
* Includes header row: `Front,Back`

## Advanced Configuration

### Token Limit Optimization

The `--max-tokens` parameter controls the maximum output size:

```bash
doc2deck generate document.pdf --max-tokens 12000
```

Higher values (8000-12000+) produce more cards at the cost of increased API usage and potentially longer processing time.

### Temperature Control

The `--temperature` parameter (0.0-1.0) controls output randomness:

```bash
# More deterministic output
doc2deck generate document.pdf --temperature 0.3

# More creative output
doc2deck generate document.pdf --temperature 0.8
```

### Card Filtering

Use `--filter-cards` to remove low-quality flashcards:

```bash
doc2deck generate document.pdf --filter-cards
```

## Batch Processing

The included `batch_process.sh` script demonstrates processing multiple files:

```bash
#!/bin/bash
INPUT_DIR="/path/to/documents"
OUTPUT_DIR="/path/to/output"
export LLM_API_KEY="your_api_key"

for file in "$INPUT_DIR"/*.{docx,pptx,pdf}; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    name="${filename%.*}"
    doc2deck generate "$file" \
      --provider gemini \
      --model gemini-2.5-pro-preview-03-25 \
      --deck-name "$name" \
      --output "$OUTPUT_DIR/$name.apkg" \
      --max-tokens 10000
  fi
done
```

## Architecture

Doc2Deck consists of several integrated components:

* **CLI Interface** (`cli.py`): Command-line processing
* **Web Interface** (`app.py`): Flask-based UI
* **Client Modules** (`clients.py`): LLM API integrations
* **Utilities** (`utils.py`): Document processing and card formatting
* **Model Listing** (`model_listing.py`): Dynamic API-based model discovery

The application flow:
1. Parse documents to extract text
2. Connect to LLM provider with appropriate parameters
3. Generate flashcards using optimized prompts
4. Process and format the result
5. Output to selected file format

## Troubleshooting

### Common Issues

**API Key Issues**
```
Error: API Key not provided
```
→ Set the `LLM_API_KEY` environment variable or use `--api-key`

**Model Availability**
```
No models found for provider or API error occurred
```
→ Check your API key permissions or try another provider

**JSON Parsing Errors**
```
Failed to parse response: Expecting property name
```
→ Try with a different temperature or model

**Content Size Limitations**
```
Content exceeds token limit, chunking...
```
→ Normal for large documents; the tool automatically segments content

## Contributing

Contributions welcome! Current priorities:

* Image extraction support
* Image occlusion card generation
* Additional LLM providers
* UI improvements for web interface

## License

MIT License