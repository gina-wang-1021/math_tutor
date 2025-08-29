from argparse import ZERO_OR_MORE
import pymupdf
import re
import os
import time
import logging
import datetime

# Set up logging
log_dir = "/Users/wangyichi/Documents/Projects/math_tutor/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pdf_extractor_{datetime.datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('pdf_extractor')

SOURCE_DIR = "/Users/wangyichi/LocalData/chromadb/all_docs"
LEVEL0_OUTPUT_DIR = "/Users/wangyichi/LocalData/chromadb/processed_docs/level0"
LEVEL1_OUTPUT_DIR = "/Users/wangyichi/LocalData/chromadb/processed_docs/level1"
LEVEL2_OUTPUT_DIR = "/Users/wangyichi/LocalData/chromadb/processed_docs/level2"

def math_level_zero_process_pdf(pdf_path, output_path):
    """Process a single PDF file, strip the header and footer, and save the filtered content to output path"""
    try:
        doc = pymupdf.open(pdf_path)
        with open(output_path, "w", encoding="utf-8") as output:
            for page in doc:
                text = page.get_text()
                lines = text.splitlines()
                
                if len(lines) > 3:
                    for line in lines[2:-1]:
                        output.write(line + '\n')
                    output.write("\n")
        return True
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False

def math_level_one_process_pdf(pdf_path, output_path):
    """Process a single PDF file, strip lines with only a number, and save the filtered content to output path"""
    try:
        doc = pymupdf.open(pdf_path)
        with open(output_path, "w", encoding="utf-8") as output:
            for page in doc:
                text = page.get_text()
                lines = text.splitlines()
                
                if len(lines) > 3:
                    for line in lines[2:-1]:
                        if not re.match(r'^-?\d+$', line.strip()):
                            output.write(line + '\n')
                    output.write("\n")
        return True
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False

def math_level_two_process_pdf(pdf_path, output_path):
    """Process a single PDF file and save the filtered content to output path"""
    try:
        doc = pymupdf.open(pdf_path)
        with open(output_path, "w", encoding="utf-8") as output:
            for page in doc:
                text = page.get_text()
                lines = text.splitlines()
                
                if len(lines) > 3:
                    for line in lines[2:-1]:
                        cleaned_line = line.strip()
                        if not cleaned_line.startswith('Fig.') and len(cleaned_line) > 1:
                            if re.search('[a-zA-Z]', cleaned_line) or re.match(r'^\d+\.', cleaned_line):
                                output.write(line + '\n')
                    output.write("\n")
        return True
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False

def math_level_extractor():
    """Process all PDF files in subdirectories of SOURCE_DIR"""
    pdf_count = 0
    zero_success_count = 0
    one_success_count = 0
    two_success_count = 0
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        logger.info(f"Processing directory: {root}")
        for file in files:
            logger.info(f"Processing file: {file}")
            if file.lower().endswith('.pdf'):
                logger.info(f"Processing PDF: {file}")
                pdf_count += 1
                pdf_path = os.path.join(root, file)
                level_zero_path = os.path.join(LEVEL0_OUTPUT_DIR, file.replace('.pdf', '.txt'))
                level_one_path = os.path.join(LEVEL1_OUTPUT_DIR, file.replace('.pdf', '.txt'))
                level_two_path = os.path.join(LEVEL2_OUTPUT_DIR, file.replace('.pdf', '.txt'))

                if math_level_zero_process_pdf(pdf_path, level_zero_path):
                    zero_success_count += 1
                if math_level_one_process_pdf(pdf_path, level_one_path):
                    one_success_count += 1
                if math_level_two_process_pdf(pdf_path, level_two_path):
                    two_success_count += 1
            time.sleep(1)
        
    logger.info(f"For level 0, Processed {zero_success_count} of {pdf_count} PDF files successfully.")
    logger.info(f"For level 1, Processed {one_success_count} of {pdf_count} PDF files successfully.")
    logger.info(f"For level 2, Processed {two_success_count} of {pdf_count} PDF files successfully.")

def parse_pdf_to_text(pdf_path, output_path):
    """
    Parse a PDF file into a text file in the root folder.
    Extracts text in proper left-to-right reading order for multi-column layouts.

    Args:
        pdf_path (str): The path to the PDF file.
        output_path (str): The path to the output text file.

    Returns:
        bool: True if the PDF file was successfully parsed, False otherwise.
    """
    try:
        doc = pymupdf.open(pdf_path)
        with open(output_path, "w", encoding="utf-8") as output:
            for page in doc:
                # Get text blocks with position information
                blocks = page.get_text("dict")["blocks"]
                
                # Filter text blocks and get their positions
                text_blocks = []
                for block in blocks:
                    if "lines" in block:  # Text block
                        x0, y0, x1, y1 = block["bbox"]
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                            block_text += "\n"
                        text_content = block_text.strip()
                        if text_content:  # Only add non-empty blocks
                            text_blocks.append((y0, x0, text_content))
                
                if not text_blocks:
                    continue
                
                # Determine page width to split left/right columns
                page_rect = page.rect
                page_width = page_rect.width
                column_boundary = page_width / 2
                
                # Separate left and right column blocks
                left_blocks = []
                right_blocks = []
                
                for y, x, text in text_blocks:
                    
                    # Skip blocks that start with "Reprint"
                    if text.strip().startswith("Reprint"):
                        continue
                    
                    if x < column_boundary:
                        left_blocks.append((y, x, text))
                    else:
                        right_blocks.append((y, x, text))
                
                # Sort each column by y-coordinate (top to bottom)
                left_blocks.sort(key=lambda x: x[0])
                right_blocks.sort(key=lambda x: x[0])
                
                # Write left column first (skip first 2 lines)
                left_all_text = ""
                for _, _, text in left_blocks:
                    left_all_text += text + "\n"
                
                left_lines = left_all_text.splitlines()
                if len(left_lines) > 2:
                    for line in left_lines[2:]:  # Skip first 2 lines
                        output.write(line + "\n")
                    output.write("\n")
                
                # Write right column (keep all lines)
                for _, _, text in right_blocks:
                    output.write(text)
                    output.write("\n\n")
        return True
    except Exception as e:
        print(f"Error parsing PDF file {pdf_path}: {e}")
        return False

def parse_pdfs_in_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, file.replace('.pdf', '.txt'))
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                if parse_pdf_to_text(pdf_path, output_file_path):
                    print(f"Successfully parsed {pdf_path} and saved to {output_file_path}")
                else:
                    print(f"Failed to parse {pdf_path}")


def normal_parsing(input_file, output_file):
    """Parse a PDF file into a text file using standard text extraction.
    
    Args:
        input_file (str): Path to the input PDF file
        output_file (str): Path to the output text file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        doc = pymupdf.open(input_file)
        with open(output_file, "w", encoding="utf-8") as output:
            for page in doc:
                # Simply extract all text from the page
                text = page.get_text()
                
                # Split into lines and skip first 2 and last 1 lines
                lines = text.splitlines()
                if len(lines) > 3:  # Only skip if we have more than 3 lines
                    filtered_lines = lines[2:-1]  # Skip first 2 and last 1
                    
                    # Filter out lines with only one character (excluding spaces)
                    final_lines = []
                    for line in filtered_lines:
                        stripped_line = line.strip()
                        if len(stripped_line) > 1:  # Keep lines with more than 1 character
                            final_lines.append(line)
                    
                    filtered_text = "\n".join(final_lines)
                    output.write(filtered_text)
                else:
                    # If 3 or fewer lines, write as is (or skip entirely)
                    output.write(text)
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"Error parsing PDF file {input_file}: {e}")
        return False
    

if __name__ == "__main__":
    # parse_pdfs_in_folder("/Users/wangyichi/LocalData/chromadb/all_docs/XI Econ II - kest1dd", "/Users/wangyichi/LocalData/chromadb/processed_docs/XI Econ II - kest1dd")
    normal_parsing("/Users/wangyichi/LocalData/chromadb/all_docs/XII Econ II - leec2dd/leec205.pdf", "/Users/wangyichi/LocalData/chromadb/all_docs/XII Econ II - leec2dd/leec205.txt")