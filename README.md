# Doc2Deck: Document to Anki Flashcard Generator

Convert your documents (DOCX, PPTX, PDF) into Anki flashcard decks (.apkg), TSV (.txt), or CSV (.csv) files using the power of Large Language Models (LLMs) like Google Gemini, Anthropic Claude, and OpenAI GPT.

This tool provides both a command-line interface (`doc2deck`) for quick conversions and batch processing, and a web interface for a more interactive experience with card review and editing.

## Features

* **Multiple Input Formats:** Supports `.docx`, `.pptx`, and `.pdf` files.
* **Multiple LLM Providers:** Integrates with Google Gemini, Anthropic Claude, and OpenAI APIs.
* **Flexible Model Selection:** Choose specific models supported by your API key (e.g., `gemini-1.5-pro-latest`, `claude-3-opus-20240229`, `gpt-4-turbo`).
* **Multiple Output Formats:** Export flashcards as:
    * Anki Package (`.apkg`) - Ready to import into Anki.
    * Tab-Separated Values (`.txt`) - Plain text, easily editable.
    * Comma-Separated Values (`.csv`) - Standard spreadsheet format.
* **Command-Line Interface:** Efficiently generate decks directly from your terminal using the `doc2deck` command.
* **Web Interface:** An easy-to-use Flask web app for uploading files, selecting options, generating cards, and reviewing/editing/exporting them.
* **Batch Processing:** Includes an example shell script (`batch_process.sh`) for converting multiple files in a directory.
* **Customizable Generation:** Control LLM creativity (`--temperature`) and output length (`--max-tokens`).

## Prerequisites

* Python 3.8 or higher
* `pip` (Python package installer)
* Git (for installation/cloning)
* API Key for at least one supported LLM provider (Google AI Studio, Anthropic, OpenAI Platform).

## Installation

You can install the command-line tool directly using pip and git:

```bash
pip install git+[https://github.com/raydioactive/doc2deck.git](https://github.com/raydioactive/doc2deck.git)
