# LLM_webapp/utils.py


import docx2txt
import genanki
import random
import csv # Needed for TSV/CSV export
import io # Needed for TSV/CSV export in memory
from typing import List, Dict, Optional
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pypdf import PdfReader
import markdown 


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract text from PDF using pypdf."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text_parts = []
            for page in reader.pages:
                 page_text = page.extract_text()
                 if page_text: # Add text only if extraction was successful for the page
                     text_parts.append(page_text)
            full_text = "\n".join(text_parts).strip()
            return full_text if full_text else None # Return None if no text was extracted
    except FileNotFoundError:
        # Re-raise FileNotFoundError to be handled by the caller
        raise
    except Exception as e:
        # Print an error message and return None for other pypdf errors
        print(f"Extraction failed using pypdf for {pdf_path}: {e}")
        return None

def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a DOCX file"""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        raise IOError(f"Failed to extract text from DOCX file: {file_path}. Error: {e}") from e

def extract_text_from_pptx(file_path: str) -> str:
    """Extract text content from a PPTX file"""
    try:
        prs = Presentation(file_path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                # Basic text extraction from text frames
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            text_runs.append(run.text)
                # Attempt to extract text from tables
                if shape.has_table:
                    for row in shape.table.rows:
                        for cell in row.cells:
                             # Check if cell has text_frame (safer)
                             if cell.text_frame:
                                text_runs.append(cell.text_frame.text)
                # Attempt extraction from chart titles/data (more complex, basic example)
                # Note: Extracting detailed chart data requires more specific handling
                if shape.has_chart and shape.shape_type == MSO_SHAPE_TYPE.CHART:
                     if shape.chart.has_title:
                         text_runs.append(shape.chart.chart_title.text_frame.text)

        return "\n".join(text_runs).strip() # Join text with newlines
    except Exception as e:
        raise IOError(f"Failed to extract text from PPTX file: {file_path}. Error: {e}") from e

# --- Anki Package (.apkg) Exporter ---
def save_flashcards_as_apkg(flashcards: List[Dict[str, str]], deck_name: str, output_path: str) -> None:
    """
    Generates an Anki .apkg file from a list of flashcards, converting Markdown lists to HTML.
    """
    if not deck_name:
        deck_name = "Generated Deck" # Default deck name if none provided

    # Generate unique IDs for model and deck
    model_id = random.randrange(1 << 30, 1 << 31)
    deck_id = random.randrange(1 << 30, 1 << 31)

    # Define the Anki model (defines card structure and appearance)
    anki_model = genanki.Model(
        model_id,
        f'Simple Model (Genanki {model_id})',
        fields=[
            {'name': 'Front'},
            {'name': 'Back'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Front}}', # Question format - Now expects HTML
                'afmt': '{{FrontSide}}\n\n<hr id="answer">\n\n{{Back}}', # Answer format - Now expects HTML
            },
        ],
        # Basic CSS styling (optional) - Anki's default list styling might apply
        css='''
            .card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }
            hr { border-top: 1px solid #ccc; }
            /* You might add specific list styling here if needed */
            ul, ol { text-align: left; display: inline-block; } /* Example */
            '''
    )

    # Create the Anki deck
    anki_deck = genanki.Deck(deck_id, deck_name)

    # Create Anki notes from the flashcards
    for card in flashcards:
        front_text = card.get('front', '').strip()
        back_text = card.get('back', '').strip()

        # Convert potential Markdown in front and back fields to HTML
        # Using 'nl2br' extension converts newlines to <br> tags, which is often desired in Anki
        # Using 'fenced_code' allows for code blocks if the LLM generates them
        front_html = markdown.markdown(front_text, extensions=['nl2br', 'fenced_code']) if front_text else ''
        back_html = markdown.markdown(back_text, extensions=['nl2br', 'fenced_code']) if back_text else ''

        # Only add cards that have some content on the front (after potential HTML conversion)
        if front_html: # Check the HTML version
            anki_note = genanki.Note(
                model=anki_model,
                fields=[front_html, back_html] # Use the HTML versions
            )
            anki_deck.add_note(anki_note)

    # Create the package and save to file
    try:
        anki_package = genanki.Package(anki_deck)
        anki_package.write_to_file(output_path)
    except Exception as e:
        raise IOError(f"Failed to write Anki deck to '{output_path}'. Error: {e}") from e
    
# --- Text File (.txt / TSV) Exporter ---
def save_flashcards_as_tsv(flashcards: List[Dict[str, str]], output_path: str) -> None:
    """
    Saves flashcards as a Tab-Separated Value (TSV) text file.
    Each line represents a card: Front<TAB>Back.
    """
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            for card in flashcards:
                front = card.get('front', '').strip()
                back = card.get('back', '').strip()
                # Replace newlines within fields with spaces or HTML breaks for Anki import
                front = front.replace('\n', ' ')
                back = back.replace('\n', '<br>') # Anki understands <br> in TSV
                writer.writerow([front, back])
    except Exception as e:
        raise IOError(f"Failed to write TSV file to '{output_path}'. Error: {e}") from e

def generate_tsv_content(flashcards: List[Dict[str, str]]) -> str:
    """
    Generates TSV content as a string (useful for direct download).
    """
    output = io.StringIO()
    writer = csv.writer(output, delimiter='\t', lineterminator='\n')
    for card in flashcards:
        front = card.get('front', '').strip().replace('\n', ' ')
        back = card.get('back', '').strip().replace('\n', '<br>')
        writer.writerow([front, back])
    return output.getvalue()

# --- CSV Exporter Functions ---
def save_flashcards_as_csv(flashcards: List[Dict[str, str]], output_path: str) -> None:
    """
    Saves flashcards as a Comma Separated Value (CSV) file.
    Includes a header row: Front,Back.
    Standard CSV quoting applied automatically for fields containing commas or newlines.
    """
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Use default comma delimiter
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Write header
            writer.writerow(['Front', 'Back'])
            # Write card data
            for card in flashcards:
                front = card.get('front', '').strip()
                back = card.get('back', '').strip()
                writer.writerow([front, back])
    except Exception as e:
        raise IOError(f"Failed to write CSV file to '{output_path}'. Error: {e}") from e

def generate_csv_content(flashcards: List[Dict[str, str]]) -> str:
    """
    Generates CSV content as a string, including a header row.
    """
    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Front', 'Back']) # Header row
    for card in flashcards:
        front = card.get('front', '').strip()
        back = card.get('back', '').strip()
        writer.writerow([front, back])
    return output.getvalue()