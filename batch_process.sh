#!/bin/bash

# Exit script immediately if any command fails
# set -e # Keep disabled to allow skipping and error reporting per file

# --- Configuration ---
INPUT_DIR="Powerpoints" # Adjust if needed
OUTPUT_DIR="./anki_decks" # Creates this directory relative to where script is run
# Assuming cli.py is now the entry point for CLI operations
APP_SCRIPT_PATH="./cli.py" # Use the updated cli.py script
LLM_PROVIDER="gemini"
# LLM_MODEL="gemini-1.5-pro-preview-0409" # Example model, adjust as needed
LLM_MODEL="gemini-2.5-pro-preview-03-25" # Explicitly request 2.5 Pro for this run
# Add generation parameters if desired (matching cli.py arguments)
# TEMPERATURE="0.7"
# MAX_TOKENS="8190"
# --- End Configuration ---

# Check API Key environment variable
if [ -z "$LLM_API_KEY" ]; then
  echo "Error: LLM_API_KEY environment variable is not set."
  echo "Please set it before running: export LLM_API_KEY='your_actual_api_key'"
  exit 1
fi

# Optional: Check if python is available
if ! command -v python &> /dev/null; then
    echo "Error: 'python' command not found. Please ensure Python is installed and in your PATH."
    exit 1
fi

# Check if the python script exists
if [ ! -f "$APP_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at '$APP_SCRIPT_PATH'"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting batch processing..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Using Provider: $LLM_PROVIDER"
echo "Requesting Model: $LLM_MODEL"
# Add echo for temp/tokens if using them
# if [ -n "$TEMPERATURE" ]; then echo "Temperature: $TEMPERATURE"; fi
# if [ -n "$MAX_TOKENS" ]; then echo "Max Tokens: $MAX_TOKENS"; fi
echo "---"

# Overall success tracker
final_exit_code=0

# Find all .docx, .pptx, OR .pdf files and loop
# Use -iname for case-insensitive matching
find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.docx' -o -iname '*.pptx' -o -iname '*.pdf' \) -print0 | while IFS= read -r -d $'\0' input_file; do

    filename=$(basename "$input_file")
    file_ext_lower=$(echo "${filename##*.}" | tr '[:upper:]' '[:lower:]') # Get lower-case extension
    echo "Checking: $filename"

    deck_title=""
    # --- Title Extraction Attempt (Only for DOCX) ---
    # Try using docx2txt only if it's a docx file and the command exists
    if [ "$file_ext_lower" == "docx" ] && command -v docx2txt &> /dev/null; then
        # Find first line matching pattern: optional leading space, "Chapter", space, digit(s), colon (case-insensitive)
        deck_title=$(docx2txt "$input_file" | awk '
            # Match pattern and capture the line
            tolower($0) ~ /^[[:space:]]*chapter[[:space:]]+[0-9]+:/ {
                # Trim whitespace and carriage return from the matched line
                gsub(/^[ \t]+|[ \t]+$|\r$/, "", $0);
                print $0; # Print the matched line
                exit; # Stop after the first match
            }
        ')
        if [ -n "$deck_title" ]; then
            echo "  Deck Title (from DOCX Chapter line): [$deck_title]"
        fi
    elif [ "$file_ext_lower" == "docx" ]; then
        echo "  Info: 'docx2txt' command not found, cannot extract title from DOCX content."
    else
        # Updated message to include PDF
        echo "  Info: Title extraction from content not attempted for non-DOCX files (e.g., PPTX, PDF)."
    fi
    # --- End Title Extraction ---


    # Fallback if title extraction failed or wasn't attempted
    if [ -z "$deck_title" ]; then
        deck_title=$(basename "$filename" ."$file_ext_lower") # Use filename without extension
        echo "  Deck Title (using filename fallback): [$deck_title]"
    fi

    # --- Determine Output Path & Check if Exists ---
    # Sanitize deck title for use as a filename
    safe_filename=$(echo "$deck_title" | sed 's/[^a-zA-Z0-9_.-]/-/g' | sed 's/--\+/-/g')
    # Add fallback for completely sanitized away names
    if [ -z "$safe_filename" ]; then safe_filename=$(basename "$filename" ."$file_ext_lower"); fi
    output_apkg_path="${OUTPUT_DIR}/${safe_filename}.apkg"

    if [ -f "$output_apkg_path" ]; then
        echo "  Skipping: Output file '$output_apkg_path' already exists."
        echo "---"
        continue # Move to the next file
    fi
    # --- End Check ---

    # --- Process if Output Doesn't Exist ---
    echo "  Processing Required: $filename"
    echo "  Output Path: $output_apkg_path"
    echo "  Executing Python CLI script..."

    # Prepare optional arguments
    declare -a python_args
    # Uncomment and adjust if using these parameters
    # if [ -n "$TEMPERATURE" ]; then
    #   python_args+=( "--temperature" "$TEMPERATURE" )
    # fi
    # if [ -n "$MAX_TOKENS" ]; then
    #   python_args+=( "--max-tokens" "$MAX_TOKENS" )
    # fi

    # Call the Python CLI script (cli.py), passing the specified model and file
    # API Key is handled by environment variable within cli.py
    python "$APP_SCRIPT_PATH" "$input_file" \
        --provider "$LLM_PROVIDER" \
        --model "$LLM_MODEL" \
        --deck-name "$deck_title" \
        --output "$output_apkg_path" \
        "${python_args[@]}" # Add optional args here

    # Check exit status
    script_exit_code=$?
    if [ $script_exit_code -ne 0 ]; then
        echo "  Error processing $filename (Python script exited with code $script_exit_code). Check logs above."
        final_exit_code=1 # Record error occurred
    else
        echo "  Successfully processed $filename."
    fi
    echo "---"

done

echo "Batch processing finished!"
exit $final_exit_code # Exit with 0 if all successful, 1 if any error occurred
