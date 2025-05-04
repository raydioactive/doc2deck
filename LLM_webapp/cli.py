# cli.py
# Provides a command-line interface (CLI) for generating decks.
# Useful for batch processing or if you don't need the web UI.

import argparse 
import sys
import requests
import json 
import os 

from LLM_webapp.clients import ClaudeClient, GeminiClient, OpenAIClient 
from utils import (
    extract_text_from_docx,
    extract_text_from_pptx,
    extract_text_from_pdf, 
    save_flashcards_as_apkg
    )

def run_cli():
    """Sets up and runs the command-line interface."""
    # --- Configure Argument Parser ---
    # Describes the script and defines the arguments it accepts
    parser = argparse.ArgumentParser(description='Generate Anki .apkg decks from DOCX, PPTX, or PDF files using LLM APIs')
    # Required argument: one or more input files
    parser.add_argument('files', nargs='+', help='Path(s) to one or more DOCX, PPTX, or PDF files') # Updated help text
    # Optional arguments with defaults
    parser.add_argument('--provider', choices=['claude', 'gemini', 'openai'], default='gemini', help='LLM provider (default: gemini)')
    parser.add_argument('--model', default=None, help='Specific LLM model name (e.g., gemini-1.5-pro-latest). Overrides client default.')
    parser.add_argument('--api-key', required=False, default=None, help='API key (reads LLM_API_KEY env var if not provided)')
    parser.add_argument('--deck-name', default=None, help='Name for the Anki deck (defaults to first valid filename)')
    parser.add_argument('--output', default='anki_deck.apkg', help='Output file path for the .apkg deck (default: anki_deck.apkg)')
    # LLM generation parameters
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature (0.0-1.0, default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=8190, help='Max output tokens (default: 8190)') # Note: clients.py uses 16384 default, consider aligning

    # Parse the arguments provided by the user when running the script
    args = parser.parse_args()

    # --- Get API Key ---
    api_key = args.api_key
    if not api_key:
        # If not given via argument, try getting it from an environment variable
        api_key = os.environ.get('LLM_API_KEY')
        if not api_key:
            # If still no key, print error and exit
            print("Error: API Key not provided via --api-key or LLM_API_KEY environment variable.", file=sys.stderr)
            return 1 # Exit code 1 indicates an error
        else:
            print("Using API Key from LLM_API_KEY environment variable.")

    # --- Input Validation ---
    if not args.output.lower().endswith('.apkg'):
        print(f"Warning: Output filename '{args.output}' doesn't end with .apkg. GenAnki will still create an .apkg file.", file=sys.stderr)

    # Validate temperature and token ranges
    if not (0.0 <= args.temperature <= 1.0):
        print("Error: Temperature must be between 0.0 and 1.0.", file=sys.stderr)
        return 1
    if args.max_tokens < 50:
        print("Error: Max tokens must be at least 50.", file=sys.stderr)
        return 1
    print(f"Using Temperature: {args.temperature}, Max Tokens: {args.max_tokens}")


    # --- Process Input Files ---
    all_text = [] # Store extracted text here
    valid_files_processed = [] # Track successful files
    first_valid_filename_base = None # For default deck name
    allowed_extensions = ('.docx', '.pptx', '.pdf') # Add PDF support

    # Loop through each file path provided by the user
    for file_path in args.files:
        try:
            filename_lower = file_path.lower()
            print(f"Processing file: '{file_path}'")

            # Basic check: does the file exist?
            if not os.path.exists(file_path):
                 print(f"Error: Input file not found at '{file_path}'", file=sys.stderr)
                 continue # Skip this file, try the next one

            # Check if the extension is one we support
            if not filename_lower.endswith(allowed_extensions):
                print(f"Warning: Skipping unsupported file type '{file_path}'. Only {', '.join(allowed_extensions)} are supported.", file=sys.stderr)
                continue # Skip this file

            # Grab the name of the first file we successfully process
            if first_valid_filename_base is None:
                first_valid_filename_base = os.path.splitext(os.path.basename(file_path))[0]

            # Extract text using the appropriate function from utils.py
            document_text = None
            file_type = None
            if filename_lower.endswith('.docx'):
                document_text = extract_text_from_docx(file_path)
                file_type = "DOCX"
            elif filename_lower.endswith('.pptx'):
                document_text = extract_text_from_pptx(file_path)
                file_type = "PPTX"
            elif filename_lower.endswith('.pdf'): # Handle PDF
                document_text = extract_text_from_pdf(file_path)
                file_type = "PDF"

            # If we got text, add it to our list
            if document_text is not None and document_text.strip():
                all_text.append(document_text)
                valid_files_processed.append(file_path)
                print(f"  Extracted {len(document_text)} characters from {file_type} file.")
            else:
                 print(f"Warning: Text extraction returned little or no content for '{file_path}'.", file=sys.stderr)

        # Handle errors during file reading/extraction
        except IOError as e:
             print(f"Error processing file '{file_path}': {e}", file=sys.stderr)
             continue # Skip file if extraction fails
        except Exception as e:
             print(f"Unexpected error processing file '{file_path}': {e}", file=sys.stderr)
             continue # Skip on other unexpected errors

    # If we couldn't process any files at all
    if not valid_files_processed:
         print(f"Error: No valid {', '.join(allowed_extensions)} files could be processed.", file=sys.stderr)
         return 1 # Exit with error

    # Combine text from all processed files
    combined_text = "\n\n--- End of Document / Start of Next ---\n\n".join(all_text)
    print(f"Total combined text length: {len(combined_text)} characters from {len(valid_files_processed)} file(s).")

    # --- Determine Deck Name ---
    deck_name = args.deck_name
    # If user didn't specify a name, use the name of the first processed file
    if not deck_name and first_valid_filename_base:
        deck_name = first_valid_filename_base
    # Fallback if something went weirdly wrong
    elif not deck_name:
        deck_name = "Default Anki Deck"
    print(f"Using Anki deck name: '{deck_name}'")

    # --- Initialize LLM Client ---
    print(f"Using provider: {args.provider}")
    # Use the specific model if provided, otherwise the client will use its default
    if args.model:
        print(f"Using specified model: {args.model}")
    # else: # Optional: Log the default model being used by the client
    #     default_model = ""
    #     if args.provider == 'claude': default_model = "claude-3-opus-20240229" # Example
    #     elif args.provider == 'gemini': default_model = "gemini-1.5-pro-latest" # Example
    #     elif args.provider == 'openai': default_model = "gpt-4-turbo" # Example
    #     if default_model: print(f"(Using default model for provider: {default_model})")

    try:
        # Create the appropriate client instance
        if args.provider == 'claude':
            client = ClaudeClient(api_key, model_name=args.model)
        elif args.provider == 'gemini':
            client = GeminiClient(api_key, model_name=args.model)
        elif args.provider == 'openai':
            client = OpenAIClient(api_key, model_name=args.model)
        else:
             print(f"Error: Invalid provider '{args.provider}'") # Should be caught by argparse
             return 1

        # --- Generate Flashcards ---
        print(f"Generating flashcards via API ({client.model_name})...") # Show actual model
        # Call the LLM API
        flashcards = client.generate_flashcards(
            combined_text,
            temperature=args.temperature,
            max_output_tokens=args.max_tokens
            )
        print(f"Generated {len(flashcards)} flashcards.")
        if not flashcards:
             print("Warning: No flashcards were generated by the LLM.", file=sys.stderr)
             # Decide whether to create an empty deck or not. Current code does.

        # --- Save Flashcards to .apkg ---
        print(f"Saving Anki deck to '{args.output}'...")
        # Use the utility function to create the Anki package file
        save_flashcards_as_apkg(flashcards, deck_name, args.output)
        if flashcards:
            print(f"Anki deck '{args.output}' generated successfully.")
        else:
            print(f"Anki deck '{args.output}' created (it's empty as no flashcards were generated).")

    # Catch errors during API calls or file saving
    except (requests.exceptions.RequestException, RuntimeError, ValueError, IOError) as e:
        print(f"\nError during API call or file saving: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}", file=sys.stderr)
        return 1

    return 0 # Indicate success

# This makes the script runnable directly using `python cli.py ...`
if __name__ == "__main__":
    exit_code = run_cli()
    sys.exit(exit_code) # Exit the script with the return code from run_cli