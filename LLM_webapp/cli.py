"""Enhanced command-line interface with progress tracking."""
import argparse
import sys
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm

from LLM_webapp.clients import ClaudeClient, GeminiClient, OpenAIClient
from LLM_webapp.utils import (
    extract_text_from_docx,
    extract_text_from_pptx,
    extract_text_from_pdf,
    save_flashcards_as_apkg,
    save_flashcards_as_tsv,
    save_flashcards_as_csv,
    filter_low_quality_cards
)

def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate Anki decks from documents using LLMs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('files', nargs='+', help='Path(s) to DOCX, PPTX, or PDF files')
    
    # LLM configuration
    llm_group = parser.add_argument_group('LLM Configuration')
    llm_group.add_argument('--provider', choices=['claude', 'gemini', 'openai'], 
                          default='gemini', help='LLM provider')
    llm_group.add_argument('--model', help='Model name (provider-specific)')
    llm_group.add_argument('--api-key', help='API key (uses LLM_API_KEY env var if not specified)')
    llm_group.add_argument('--temperature', type=float, default=0.7, 
                          help='Generation temperature (0.0-1.0)')
    llm_group.add_argument('--max-tokens', type=int, default=4096, 
                          help='Max output tokens')
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--deck-name', help='Name for Anki deck (defaults to filename)')
    output_group.add_argument('--output', default='anki_deck.apkg', 
                             help='Output file path')
    output_group.add_argument('--export-format', choices=['apkg', 'csv', 'tsv'], 
                             default='apkg', help='Export format')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--filter-cards', action='store_true', 
                           help='Filter low-quality cards')
    proc_group.add_argument('--chunk-size', type=int, default=0, 
                           help='Force chunking at specified token size (0=auto)')
    
    return parser

def validate_args(args: argparse.Namespace) -> bool:
    """Validate command-line arguments."""
    if not (0.0 <= args.temperature <= 1.0):
        print("Error: Temperature must be between 0.0 and 1.0", file=sys.stderr)
        return False
        
    if args.max_tokens < 50:
        print("Error: Max tokens must be at least 50", file=sys.stderr)
        return False
        
    # Check API key
    api_key = args.api_key or os.environ.get('LLM_API_KEY')
    if not api_key:
        print("Error: API Key not provided via --api-key or LLM_API_KEY environment variable", 
              file=sys.stderr)
        return False
        
    return True

def process_files(file_paths: List[str]) -> tuple:
    """Extract text from all input files."""
    start_time = time.time()
    text_parts = []
    valid_files = []
    first_filename = None
    
    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            filename_lower = file_path.lower()
            
            # Skip invalid files
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                continue
                
            # Process by file type
            if filename_lower.endswith('.docx'):
                text = extract_text_from_docx(file_path)
                file_type = "DOCX"
            elif filename_lower.endswith('.pptx'):
                text = extract_text_from_pptx(file_path)
                file_type = "PPTX"
            elif filename_lower.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
                file_type = "PDF"
            else:
                print(f"Error: Unsupported file type: {file_path}", file=sys.stderr)
                continue
                
            # Store results
            if text and text.strip():
                if first_filename is None:
                    first_filename = os.path.splitext(os.path.basename(file_path))[0]
                valid_files.append(file_path)
                text_parts.append(text)
                print(f"Extracted {len(text):,} chars from {file_type}: {file_path}")
            else:
                print(f"Warning: No text extracted from {file_path}", file=sys.stderr)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    elapsed = time.time() - start_time
    combined_text = "\n\n--- End of Document / Start of Next ---\n\n".join(text_parts)
    
    print(f"Processed {len(valid_files)} files ({len(combined_text):,} chars) in {elapsed:.1f}s")
    return combined_text, valid_files, first_filename

def get_llm_client(provider: str, api_key: str, model: str = None):
    """Create the appropriate LLM client."""
    if provider == 'claude':
        return ClaudeClient(api_key, model_name=model)
    elif provider == 'gemini':
        return GeminiClient(api_key, model_name=model)
    elif provider == 'openai':
        return OpenAIClient(api_key, model_name=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def save_output(cards: List[Dict[str, Any]], args: argparse.Namespace, deck_name: str) -> bool:
    """Save flashcards in the requested format."""
    try:
        if args.export_format == 'apkg':
            save_flashcards_as_apkg(cards, deck_name, args.output)
        elif args.export_format == 'csv':
            output_path = args.output if args.output.endswith('.csv') else f"{args.output}.csv"
            save_flashcards_as_csv(cards, output_path)
        elif args.export_format == 'tsv':
            output_path = args.output if args.output.endswith('.txt') else f"{args.output}.txt"
            save_flashcards_as_tsv(cards, output_path)
            
        print(f"Saved {len(cards)} cards to {args.output}")
        return True
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        return False

def run_cli() -> int:
    """Main CLI execution function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not validate_args(args):
        return 1
        
    # Process files
    try:
        combined_text, valid_files, first_filename = process_files(args.files)
        if not valid_files:
            print("Error: No valid files were processed", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error processing files: {e}", file=sys.stderr)
        return 1
        
    # Determine output name
    deck_name = args.deck_name or first_filename or "Generated_Flashcards"
    
    # Create LLM client
    try:
        api_key = args.api_key or os.environ.get('LLM_API_KEY')
        client = get_llm_client(args.provider, api_key, args.model)
        print(f"Using {args.provider} model: {client.model_name}")
    except Exception as e:
        print(f"Error initializing LLM client: {e}", file=sys.stderr)
        return 1
        
    # Generate flashcards
    try:
        print("Generating flashcards...")
        start_time = time.time()
        
        # Simple progress indicator for long-running operation
        spinner = "|/-\\"
        spinner_idx = 0
        
        generation_thread = None
        cards = []
        
        # Setup progress indication
        if sys.stdout.isatty():
            import threading
            
            # Flag for thread communication
            running = True
            
            def progress_indicator():
                nonlocal spinner_idx
                while running:
                    print(f"\rGenerating cards... {spinner[spinner_idx % len(spinner)]}", end="")
                    spinner_idx += 1
                    time.sleep(0.2)
            
            # Start progress thread
            generation_thread = threading.Thread(target=progress_indicator)
            generation_thread.daemon = True
            generation_thread.start()
        
        try:
            # Generate cards
            cards = client.generate_flashcards(
                combined_text,
                temperature=args.temperature,
                max_output_tokens=args.max_tokens
            )
        finally:
            # Stop progress indicator
            if generation_thread:
                running = False
                generation_thread.join(1.0)
                print("\r" + " " * 30 + "\r", end="")  # Clear progress line
        
        elapsed = time.time() - start_time
        print(f"Generated {len(cards)} cards in {elapsed:.1f}s")
        
        # Filter cards if requested
        if args.filter_cards and cards:
            original_count = len(cards)
            cards = filter_low_quality_cards(cards)
            if len(cards) != original_count:
                print(f"Filtered out {original_count - len(cards)} low-quality cards")
        
        # Save output
        if cards:
            if save_output(cards, args, deck_name):
                return 0
            return 1
        else:
            print("Warning: No flashcards were generated", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"Error generating flashcards: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(run_cli())