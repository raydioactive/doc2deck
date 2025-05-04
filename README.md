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

## Setup Guide: Getting Started from Scratch

Follow these steps to get Doc2Deck running on your computer, even if you don't have Python or Git installed yet.

**1. Install Python (Version 3.8 or Higher)**

Python is the programming language Doc2Deck is written in. `pip`, the Python package installer, usually comes bundled with Python.

* **Windows:**
    * Download the latest stable Python 3 installer from the [official Python website](https://www.python.org/downloads/windows/).
    * Run the installer. **Crucially, check the box that says "Add Python X.X to PATH"** during installation. This makes Python accessible from your command prompt.
    * Follow the installer prompts.
* **macOS:**
    * Download the macOS installer from the [official Python website](https://www.python.org/downloads/macos/).
    * Run the installer package and follow the prompts. Python should be added to your PATH automatically.
    * Alternatively, if you use [Homebrew](https://brew.sh/): `brew install python`
* **Linux:**
    * Python 3 is often pre-installed. Check by opening your terminal and typing `python3 --version`.
    * If not installed or you need a newer version, use your distribution's package manager. Examples:
        * Debian/Ubuntu: `sudo apt update && sudo apt install python3 python3-pip python3-venv`
        * Fedora: `sudo dnf install python3 python3-pip`

* **Verify Installation:** Open your terminal (Command Prompt, PowerShell, or Terminal) and type:
    ```bash
    python --version
    # or on some systems
    python3 --version
    pip --version
    # or on some systems
    pip3 --version
    ```
    You should see the installed versions printed (e.g., `Python 3.10.4`). If you get an error like "command not found", ensure Python was added to your PATH (you might need to restart your terminal or computer).

**2. Install Git**

Git is needed to download the Doc2Deck code directly from its repository.

* Go to the [official Git website](https://git-scm.com/downloads).
* Download the installer for your operating system (Windows, macOS, Linux).
* Run the installer, accepting the default options is usually fine.
* **Verify Installation:** Open a *new* terminal window and type:
    ```bash
    git --version
    ```
    You should see the installed Git version (e.g., `git version 2.34.1`).

**3. Get an LLM API Key**

Doc2Deck needs to communicate with an AI service (LLM) to generate flashcards. You need an API key from at least one supported provider.

* **Google Gemini:** Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
* **Anthropic Claude:** Request API access via the [Anthropic Console](https://console.anthropic.com/).
* **OpenAI GPT:** Sign up and get an API key from the [OpenAI Platform](https://platform.openai.com/api-keys).

**Important:** Keep your API key secure and private. These services often have associated costs based on usage.
**quick TBH:** I've only tested this using gemini. I do not know that this works with the openAI/Claude APIs. That would require some more testing, that I think is irrelevant because as of now (5/4/2025) Gemini is best for this use case. Just do google with your free account and it should be fine - I've been testing this for a while and they still haven't charged me anything. 

**4. Install Doc2Deck**

Now that you have Python, pip, and Git, you can install Doc2Deck.

* Open your terminal.
* Run the following command:
    ```bash
    pip install git+https://github.com/raydioactive/doc2deck.git
    ```
    * **What this does:** `pip` uses `git` to download the code from the specified GitHub repository and then installs the `doc2deck` package and all its required Python library dependencies (like `python-docx`, `genanki`, `requests`, etc.).

**5. Set Your API Key**

You need to tell Doc2Deck how to authenticate with the LLM provider. You have two main options:

* **Option A (Recommended): Environment Variable**
    * Set an environment variable named `LLM_API_KEY` to your API key value. How to do this varies by OS:
        * **Windows (Temporary - for current session):**
            ```bash
            set LLM_API_KEY=YOUR_API_KEY_HERE
            ```
        * **Windows (Permanent):** Search for "Environment Variables" in the Start menu and add `LLM_API_KEY` as a new User variable.
        * **macOS/Linux (Temporary - for current session):**
            ```bash
            export LLM_API_KEY='YOUR_API_KEY_HERE'
            ```
        * **macOS/Linux (Permanent):** Add the `export` line above to your shell profile file (e.g., `~/.bashrc`, `~/.zshrc`, `~/.profile`) and restart your terminal or run `source ~/.your_profile_file`.
    * The tool will automatically detect and use this variable if the `--api-key` argument isn't provided.

* **Option B: Command-Line Argument**
    * You can pass the key directly when running the command using the `--api-key` flag (see "Running Doc2Deck" below). This is less secure if your commands are logged or shared.

## Running Doc2Deck

You can use Doc2Deck either via the command line or the web interface. Make sure you've completed the Setup Guide first.

**1. Command-Line Interface (CLI)**

The CLI is ideal for quick conversions or scripting.

* **Get Help:** See all available options:
    ```bash
    doc2deck --help
    ```
* **Example Usage:**
    ```bash
    # Generate an Anki deck from a DOCX using Gemini (API key from environment)
    doc2deck my_notes.docx --provider gemini --deck-name "My Study Notes" --output study_notes.apkg

    # Generate a CSV from a PDF using Claude (API key passed as argument)
    doc2deck report.pdf --provider claude --api-key YOUR_CLAUDE_API_KEY --output report_cards.csv --export-format csv

    # Generate from multiple files using OpenAI GPT-4 Turbo
    doc2deck chapter1.pptx chapter2.docx --provider openai --model gpt-4-turbo --output combined_chapters.apkg
    ```

**2. Web Interface**

The web interface provides a graphical way to upload files, set options, and review/edit cards before exporting.

* **Navigate to the Code:** You first need the actual code files (not just the installed package). If you haven't already, clone the repository:
    ```bash
    git clone https://github.com/raydioactive/doc2deck.git
    cd doc2deck # Navigate into the downloaded folder
    ```
    *(If you installed via pip earlier, you might need to find where pip installed the files, or just clone it fresh as shown above).*
* **Ensure Dependencies (if cloned fresh):** If you cloned the repository directly instead of using `pip install`, you need to install the dependencies manually:
    ```bash
    pip install -r requirements.txt
    ```
* **Run the Web App:** From inside the `doc2deck` directory (the one containing `LLM_webapp`), run:
    ```bash
    python -m LLM_webapp.app
    ```
    *(Make sure the `templates` directory with `index.html` and `results.html` is present inside `LLM_webapp`)*
* **Access in Browser:** Open your web browser and go to `http://127.0.0.1:5000` (or the address shown in the terminal).
* **Use the Interface:**
    * Upload your `.docx`, `.pptx`, or `.pdf` file(s).
    * Select the LLM provider and specific model.
    * Paste your API key.
    * Choose an output name and export format (.apkg, .csv, .txt).
    * Adjust temperature and max tokens if desired.
    * Click "Generate Flashcards".
    * On the results page, review, edit, add, or delete cards.
    * Click "Export" to download your deck/file.

## Batch Processing

A sample script `batch_process.sh` (for Linux/macOS/WSL) is included in the repository, demonstrating how to convert all `.docx` files in a directory. Adapt it for your needs.

```bash
#!/bin/bash
# Example batch processing script

# --- Configuration ---
INPUT_DIR="path/to/your/documents"
OUTPUT_DIR="path/to/your/output/decks"
PROVIDER="gemini" # Or claude, openai
MODEL="gemini-1.5-pro-latest" # Optional: specify model
# Ensure LLM_API_KEY environment variable is set, or add --api-key below

# --- Script ---
mkdir -p "$OUTPUT_DIR" # Create output dir if it doesn't exist

shopt -s nullglob # Prevent errors if no files match

# Process DOCX files
for infile in "$INPUT_DIR"/*.docx; do
  filename=$(basename "$infile" .docx)
  outfile="$OUTPUT_DIR/${filename}.apkg"
  echo "Processing $infile -> $outfile"
  doc2deck "$infile" --provider "$PROVIDER" ${MODEL:+--model "$MODEL"} --deck-name "$filename" --output "$outfile"
  # Add error checking if needed
done

# Add similar loops for .pptx and .pdf if desired
# for infile in "$INPUT_DIR"/*.pptx; do ... done
# for infile in "$INPUT_DIR"/*.pdf; do ... done

echo "Batch processing complete."
shopt -u nullglob # Reset nullglob option
(Remember to make the script executable: chmod +x batch_process.sh)

```

Contributing

Contributions, issues, and feature requests are welcome! I want to make it work with pictures and maybe figure out a way to get image occlusion cards as well