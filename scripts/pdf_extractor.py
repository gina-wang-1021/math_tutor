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

def level_zero_process_pdf(pdf_path, output_path):
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

def level_one_process_pdf(pdf_path, output_path):
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

def level_two_process_pdf(pdf_path, output_path):
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

def main():
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

                if level_zero_process_pdf(pdf_path, level_zero_path):
                    zero_success_count += 1
                if level_one_process_pdf(pdf_path, level_one_path):
                    one_success_count += 1
                if level_two_process_pdf(pdf_path, level_two_path):
                    two_success_count += 1
            time.sleep(1)
        
    logger.info(f"For level 0, Processed {zero_success_count} of {pdf_count} PDF files successfully.")
    logger.info(f"For level 1, Processed {one_success_count} of {pdf_count} PDF files successfully.")
    logger.info(f"For level 2, Processed {two_success_count} of {pdf_count} PDF files successfully.")

if __name__ == "__main__":
    main()