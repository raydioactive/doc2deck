# app.py
# This is the main file for the Flask web application.
# It handles user requests, file uploads, calls the LLM, and shows results.

import os
import tempfile
import sys
import logging
from typing import List, Dict, Optional
import requests # Used by clients, though not directly here usually
import json # Used potentially in error handling
import io # Used in utils for generating tsv/csv in memory

from flask import Flask, request, render_template, redirect, url_for, flash, send_file, Response

# Import our custom modules
from LLM_webapp.clients import ClaudeClient, GeminiClient, OpenAIClient # Handles talking to different LLMs
# Import all our helper functions for file processing and saving
from LLM_webapp.utils import (
    extract_text_from_docx,
    extract_text_from_pptx,
    extract_text_from_pdf, # Added PDF extractor
    save_flashcards_as_apkg,
    # save_flashcards_as_tsv, # Not directly used here, but generate_tsv_content is
    generate_tsv_content,
    # save_flashcards_as_csv, # Not directly used here, but generate_csv_content is
    generate_csv_content
)
# from cli import run_cli # We'll use Flask's built-in runner usually

# --- Flask App Setup ---
app = Flask(__name__)
# Secret key needed for Flask sessions and flash messages. Change this in production!
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key__CHANGE_ME")

# --- Global State (Simple) ---
# These variables store the data from the *last* generation request.
# This is simple but won't work well if multiple users access the app simultaneously.
current_flashcards: List[Dict[str, str]] = [] # Holds the generated question/answer pairs
current_deck_name: Optional[str] = None # Name chosen for the export file/deck
current_export_format: Optional[str] = None # 'apkg', 'tsv' or 'csv'
current_model_used: Optional[str] = None # Which LLM model was used (e.g., gemini-1.5-pro)
current_provider_used: Optional[str] = None # Which LLM provider (e.g., gemini)

# --- Logging Setup ---
def setup_logging():
    # Basic logging to see what's happening during requests
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask Routes ---

# This is the main page '/'
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the file upload form and triggers the flashcard generation."""
    # Make sure we can modify the global variables
    global current_flashcards, current_deck_name, current_export_format
    global current_model_used, current_provider_used

    # This block runs when the user submits the form (POST request)
    if request.method == 'POST':
        # Clear out results from any previous run
        current_flashcards = []
        current_deck_name = None
        current_export_format = None
        current_model_used = None
        current_provider_used = None
        app.logger.info("Received POST request on / - Starting generation process")

        # --- Get Files and Validate ---
        uploaded_files = request.files.getlist('file') # Get all files uploaded
        if not uploaded_files or all(f.filename == '' for f in uploaded_files):
            # Tell user if they forgot to select files
            flash('No files selected. Please select one or more .docx, .pptx, or .pdf files.', 'error')
            app.logger.warning("No files selected in upload attempt.")
            return redirect(request.url) # Reload the page
        app.logger.info(f"Received {len(uploaded_files)} file part(s).")

        # --- Get API Key and Other Form Data ---
        api_key = request.form.get('api_key', '').strip()
        if not api_key:
            flash('API Key is required.', 'error')
            app.logger.warning("API Key was missing in request.")
            return redirect(request.url)

        provider = request.form.get('provider', 'gemini') # Which LLM service
        model = request.form.get('model') # Specific model name (optional)
        output_name = request.form.get('output_name', '').strip() # Desired name for output
        export_format = request.form.get('export_format', 'apkg') # How to save the cards
        try:
            # Get LLM creativity/randomness setting
            temperature = float(request.form.get('temperature', 0.7))
            # Get max length for LLM response
            max_tokens = int(request.form.get('max_tokens', 4096))
            # Basic checks for valid ranges
            if not (0.0 <= temperature <= 1.0): raise ValueError("Temperature must be between 0.0 and 1.0.")
            if max_tokens < 50: raise ValueError("Max tokens must be at least 50.")
            app.logger.info(f"Parameters: Provider={provider}, Model={model}, Format={export_format}, Temp={temperature}, Tokens={max_tokens}")
        except ValueError as e:
            flash(f'Invalid generation parameter: {e}. Please enter valid numbers.', 'error')
            app.logger.error(f"Invalid parameter value: {e}")
            return redirect(request.url)

        # --- Process Uploaded Files ---
        all_text = [] # To store text extracted from all valid files
        valid_files_processed = [] # Keep track of files we successfully processed
        temp_paths = [] # Store paths to temporary files for later cleanup
        first_filename_base = None # Used for default output name if user doesn't provide one
        allowed_extensions = ('.docx', '.pptx', '.pdf') # Now includes PDF

        try:
            # Loop through each uploaded file
            for idx, file in enumerate(uploaded_files):
                if file and file.filename:
                    filename_lower = file.filename.lower()
                    app.logger.info(f"Processing file {idx+1}: {file.filename}")

                    # Check if the file extension is one we support
                    if not filename_lower.endswith(allowed_extensions):
                         flash(f'Skipping unsupported file type: {file.filename}. Only {", ".join(allowed_extensions)} are supported.', 'warning')
                         app.logger.warning(f"Skipping unsupported file type: {file.filename}")
                         continue # Skip to the next file

                    # Grab the name of the first valid file (without extension)
                    if first_filename_base is None:
                        first_filename_base = os.path.splitext(file.filename)[0]

                    # Save the uploaded file safely to a temporary location
                    # Give it the correct suffix just for clarity when debugging
                    suffix = '.pdf' if filename_lower.endswith('.pdf') else ('.pptx' if filename_lower.endswith('.pptx') else '.docx')
                    # 'delete=False' means we have to manually delete it later
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                        file.save(temp.name) # Save content to the temp file
                        temp_path = temp.name
                        temp_paths.append(temp_path) # Remember path for cleanup
                        app.logger.info(f"Saved uploaded file to temp path: {temp_path}")

                    # Extract text based on the file type using functions from utils.py
                    document_text = None
                    try:
                        if filename_lower.endswith('.docx'):
                            document_text = extract_text_from_docx(temp_path)
                            file_type = "DOCX"
                        elif filename_lower.endswith('.pptx'):
                            document_text = extract_text_from_pptx(temp_path)
                            file_type = "PPTX"
                        elif filename_lower.endswith('.pdf'): # Handle PDF files
                            document_text = extract_text_from_pdf(temp_path)
                            file_type = "PDF"
                        # We shouldn't get here because of the earlier check, but just in case
                        else:
                            app.logger.error(f"Internal logic error: File {file.filename} passed extension check but is not handled.")
                            continue

                        # If we got text, add it to our collection
                        if document_text is not None and document_text.strip():
                            all_text.append(document_text)
                            valid_files_processed.append(file.filename)
                            app.logger.info(f"Extracted {len(document_text)} characters from {file_type} file '{file.filename}'.")
                        else:
                             app.logger.warning(f"Text extraction returned little or no content for '{file.filename}'. Might be empty or problematic.")

                    # Handle errors during text extraction for a specific file
                    except IOError as e:
                         flash(f"Error extracting text from '{file.filename}': {e}", 'error')
                         app.logger.error(f"Extraction error for {file.filename}: {e}")
                         continue # Skip this file, try the next one

            # If no files were successfully processed after the loop
            if not valid_files_processed:
                 flash(f'No valid {" or ".join(allowed_extensions)} files were processed. Please upload at least one valid file.', 'error')
                 app.logger.error(f"No valid {allowed_extensions} files found after processing uploads.")
                 # Important: Clean up any temp files created before returning
                 # (The finally block below handles this)
                 return redirect(request.url)

            # Combine text from all files, separated by a marker
            combined_text = "\n\n--- End of Document / Start of Next ---\n\n".join(all_text)
            app.logger.info(f"Total combined text length: {len(combined_text)} characters from {len(valid_files_processed)} file(s).")

            # Figure out the name for the deck/export file
            current_deck_name = output_name if output_name else first_filename_base
            if not current_deck_name: current_deck_name = "Generated_Flashcards" # Default fallback
            app.logger.info(f"Using output name: {current_deck_name}")

            # Store the chosen export format for the results page
            current_export_format = export_format
            app.logger.info(f"Selected export format: {current_export_format}")

            # --- Select and Use LLM Client ---
            try:
                app.logger.info(f"Initializing LLM client: Provider={provider}, Model={model}")
                # Create the correct client object based on user selection
                if provider == 'claude':
                    client = ClaudeClient(api_key, model_name=model)
                elif provider == 'gemini':
                    client = GeminiClient(api_key, model_name=model)
                elif provider == 'openai':
                    client = OpenAIClient(api_key, model_name=model)
                else:
                    flash('Invalid LLM provider selected.', 'error')
                    app.logger.error(f"Invalid provider '{provider}' received.")
                    return redirect(request.url)

                # Store which provider/model was actually used (client might use a default)
                current_provider_used = provider
                current_model_used = client.model_name

                # Let the user know we're starting the potentially long process
                flash(f'Generating flashcards with {current_provider_used} ({current_model_used}). This might take a minute...', 'info')
                app.logger.info(f"Calling generate_flashcards for {current_model_used}...")

                # THE MAIN CALL: Ask the LLM to generate flashcards from the combined text
                generated_cards = client.generate_flashcards(
                    content=combined_text,
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
                current_flashcards = generated_cards # Store results globally
                app.logger.info(f"LLM generated {len(current_flashcards)} flashcards.")

                # Success! Tell the user and send them to the results page
                flash(f'Successfully generated {len(current_flashcards)} flashcards! Ready for review/export.', 'success')
                return redirect(url_for('results')) # Go to the '/results' page

            # Handle errors during the LLM API call or response parsing
            except (requests.exceptions.RequestException, RuntimeError, ValueError, IOError) as e:
                app.logger.error(f"Error during LLM call or processing: {e}", exc_info=False)
                flash(f'API or Processing Error: {str(e)}', 'error')
                return redirect(request.url) # Stay on index page on error

        # Catch any other unexpected problems during the whole process
        except Exception as e:
            app.logger.error(f"An unexpected error occurred in index route: {e}", exc_info=True) # Log full traceback
            flash('An unexpected server error occurred. Please check logs or try again later.', 'error')
            return redirect(request.url)
        # This 'finally' block runs *always*, whether there was an error or not
        finally:
            # IMPORTANT: Clean up all temporary files we created
            app.logger.info(f"Cleaning up {len(temp_paths)} temporary file(s).")
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.unlink(path) # Delete the temp file
                        app.logger.info(f"Removed temp file: {path}")
                    except OSError as e:
                         # Log if cleanup fails, but don't crash the app
                         app.logger.error(f"Error removing temporary file {path}: {e}")

    # This runs if it's a GET request (user just navigating to the page)
    app.logger.info("Serving GET request for / - Displaying upload form")
    return render_template('index.html') # Show the main upload form


# This is the results page '/results'
@app.route('/results', methods=['GET', 'POST'])
def results():
    """Displays generated flashcards, allows editing, and handles export."""
    # Make sure we can access the global variables storing the cards etc.
    global current_flashcards, current_deck_name, current_export_format

    # This block runs when the user clicks 'Update Cards' or 'Export' on the results page
    if request.method == 'POST':
        app.logger.info("Received POST request on /results")

        # --- Handle Flashcard Updates (Editing/Deleting/Adding) ---
        updated_cards = []
        # Find out how many cards were originally displayed on the form
        form_card_count = int(request.form.get('card_count', 0))
        app.logger.info(f"Processing updates for {form_card_count} cards shown.")

        # Loop through the cards that were on the page
        for i in range(form_card_count):
            # Check if the 'delete' checkbox for this card was checked
            if request.form.get(f'delete_{i}'):
                 app.logger.info(f"Deleting card index {i}")
                 continue # Skip this card, don't add it to updated_cards

            # Get the potentially edited text from the front/back textareas
            front = request.form.get(f'front_{i}', '').strip()
            back = request.form.get(f'back_{i}', '').strip()
            # Only keep the card if it still has some content on front or back
            if front or back:
                 updated_cards.append({"front": front, "back": back})
            else:
                app.logger.info(f"Skipping empty card index {i} after edit.")

        # Check if the user added a new card in the "Add New Card" section
        new_front = request.form.get('new_front', '').strip()
        new_back = request.form.get('new_back', '').strip()
        if new_front or new_back:
             app.logger.info("Adding new card provided by user.")
             updated_cards.append({"front": new_front, "back": new_back})

        # Update the global list *only* if something actually changed
        if len(current_flashcards) != len(updated_cards) or current_flashcards != updated_cards:
             current_flashcards = updated_cards
             app.logger.info(f"Flashcards updated in memory. New count: {len(current_flashcards)}")
             # Show a confirmation message only if they hit the 'Update' button
             if 'update' in request.form:
                 flash('Flashcards updated.', 'info')
        else:
             app.logger.info("No changes detected in flashcards after POST.")


        # --- Handle Export ---
        # Check if the 'Export' button was clicked
        if 'export' in request.form:
            app.logger.info(f"Export button pressed. Format: {current_export_format}, Name: {current_deck_name}")
            # Can't export if there are no cards left
            if not current_flashcards:
                 flash('No flashcards to export.', 'warning')
                 app.logger.warning("Export requested but no flashcards available.")
                 # Re-render the page showing the empty state
                 return render_template('results.html', flashcards=current_flashcards, deck_name=current_deck_name, export_format=current_export_format)

            # Use the name and format stored from the initial generation step
            export_name = current_deck_name if current_deck_name else "Generated_Export"
            export_format = current_export_format if current_export_format else 'apkg' # Default to apkg

            temp_export_path = None # Will hold path only for file-based exports like .apkg

            try:
                # --- APKG Export ---
                if export_format == 'apkg':
                    download_filename = f"{export_name}.apkg"
                    mimetype = 'application/octet-stream' # Generic type for packages/binary data
                    # genanki needs to write to a file path, so create a temporary one
                    # 'wb' mode = write binary. 'delete=False' needed again.
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.apkg') as temp_file:
                         temp_export_path = temp_file.name
                         # Call the utility function to create the .apkg file
                         save_flashcards_as_apkg(current_flashcards, export_name, temp_export_path)
                         app.logger.info(f"Generated APKG file at temporary path: {temp_export_path}")
                    # Send the created file to the user's browser for download
                    return send_file(temp_export_path, as_attachment=True, download_name=download_filename, mimetype=mimetype)

                # --- TSV (Text) Export ---
                elif export_format == 'tsv':
                    download_filename = f"{export_name}.txt" # Use .txt extension for TSV
                    mimetype = 'text/tab-separated-values; charset=utf-8'
                    # Generate the TSV content directly into a string variable
                    tsv_content = generate_tsv_content(current_flashcards)
                    app.logger.info(f"Generated TSV content for download: {download_filename}")
                    # Use Flask's Response object to send the string directly
                    # Set headers to tell the browser it's an attachment with the right name/type
                    return Response(
                        tsv_content, mimetype=mimetype,
                        headers={"Content-Disposition": f"attachment;filename=\"{download_filename}\""}
                    )

                # --- CSV Export ---
                elif export_format == 'csv':
                    download_filename = f"{export_name}.csv"
                    mimetype = 'text/csv; charset=utf-8'
                    # Generate the CSV content directly into a string variable
                    csv_content = generate_csv_content(current_flashcards)
                    app.logger.info(f"Generated CSV content for download: {download_filename}")
                    # Send the string using Flask's Response object
                    return Response(
                        csv_content, mimetype=mimetype,
                        headers={"Content-Disposition": f"attachment;filename=\"{download_filename}\""}
                    )
                else:
                    # Should not happen if the HTML select dropdown is correct
                    flash(f'Unknown export format selected: {export_format}', 'error')
                    app.logger.error(f"Invalid export format '{export_format}' encountered.")

            # Catch errors during file generation or sending
            except (IOError, Exception) as e:
                 app.logger.error(f"Error during export (format: {export_format}): {e}", exc_info=True)
                 flash(f'Error exporting file: {str(e)}', 'error')
                 # Fall through to re-render the results page if export fails
            finally:
                # Clean up the temporary .apkg file IF it was created
                 if temp_export_path and os.path.exists(temp_export_path):
                    try:
                        os.unlink(temp_export_path)
                        app.logger.info(f"Cleaned up temporary export file: {temp_export_path}")
                    except OSError as e:
                        app.logger.error(f"Error removing temporary export file {temp_export_path}: {e}")

    # This runs for a GET request to /results, or after an 'Update' POST action
    app.logger.info(f"Serving GET request for /results. Cards: {len(current_flashcards)}, Deck: {current_deck_name}, Format: {current_export_format}")
    # Render the results page template, passing the current state to it
    return render_template('results.html',
                           flashcards=current_flashcards,
                           deck_name=current_deck_name,
                           export_format=current_export_format)

@app.route('/list-models', methods=['GET', 'POST'])
def list_models():
    """Handles listing available models for a provider directly from APIs."""
    from LLM_webapp.model_listing import get_available_models
    
    app.logger.info("Handling list-models request")
    provider = request.args.get('provider', 'gemini')
    models = []
    error = None
    
    if request.method == 'POST':
        provider = request.form.get('provider', 'gemini')
        api_key = request.form.get('api_key', '').strip()
        
        if not api_key:
            flash('API Key is required.', 'error')
            app.logger.warning("API Key was missing in request.")
            return render_template('list_models.html', provider=provider, models=[], error=None)
        
        try:
            app.logger.info(f"Fetching models for provider: {provider}")
            
            # Fetch models directly from the API
            models = get_available_models(provider, api_key)
            
            if not models:
                flash(f'No models found for {provider} or API error occurred.', 'warning')
            else:
                flash(f'Found {len(models)} models for {provider}.', 'success')
                
        except Exception as e:
            error = str(e)
            app.logger.error(f"Error listing models: {e}")
            flash(f'Error listing models: {error}', 'error')
    
    return render_template('list_models.html', provider=provider, models=models, error=error)

# --- Function to Run the Web App ---
def run_webapp(host='127.0.0.1', port=5000):
    print(f" * Starting Flask web application on http://{host}:{port}/")
    print(" * Access the app in your web browser.")
    # debug=True enables auto-reloading when code changes and shows detailed errors in browser.
    # **IMPORTANT**: Set debug=False in a real production environment!
    app.run(host=host, port=port, debug=True)

# --- Main Execution Block ---
# This runs only when the script is executed directly (e.g., `python app.py`)
if __name__ == '__main__':
    setup_logging() # Turn on logging

    # Quick check to make sure the HTML template files exist where Flask expects them
    if not os.path.exists('templates') or \
       not os.path.exists(os.path.join('templates', 'index.html')) or \
       not os.path.exists(os.path.join('templates', 'results.html')):
        app.logger.error("templates directory or required HTML files (index.html, results.html) not found.")
        sys.exit("Error: Missing 'templates' directory or required HTML files. Please ensure they exist.")

    # NOTE: Basic CLI mode detection is removed here for simplicity.
    # To run CLI, use `python cli.py ...` directly.
    # This script will now always attempt to start the web app.

    run_webapp() # Start the Flask development server