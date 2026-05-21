import os
import logging
from src.data_ingestion.pdf_parser import PDFParser

logger = logging.getLogger("SanskritRAG")


def load_document(file_path):
    """
    Load Sanskrit document from PDF or TXT file.
    Raises FileNotFoundError or ValueError for bad inputs.
    """

    # [CHANGED] validate input path with try/except
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading document: {file_path} (type: {file_extension})")

        # PDF Loading
        if file_extension == ".pdf":
            try:
                parser = PDFParser(file_path)
                text = parser.extract_text()
            except Exception as e:
                logger.error(f"PDF loading failed for '{file_path}': {e}")
                raise

        # TXT Loading
        elif file_extension == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
            except UnicodeDecodeError:
                logger.warning("UTF-8 decode failed. Retrying with latin-1.")  # [CHANGED] fallback encoding
                with open(file_path, "r", encoding="latin-1") as file:
                    text = file.read()

        else:
            raise ValueError(f"Unsupported file format: '{file_extension}'. Use .pdf or .txt")

        if not text.strip():
            logger.warning(f"Loaded document is empty: {file_path}")

        logger.info(f"Document loaded successfully. Length: {len(text)} characters.")
        return text

    except (FileNotFoundError, ValueError):
        raise  # re-raise known errors as-is

    except Exception as e:
        logger.error(f"Unexpected error loading document: {e}")
        raise RuntimeError(f"Document loading failed: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    file_path = "../../data/raw/sanskrit_doc.pdf"

    try:
        document_text = load_document(file_path)
        print("\n===== DOCUMENT PREVIEW =====\n")
        print(document_text[:1000])
        print("\n===== DOCUMENT LOADED SUCCESSFULLY =====")

    except Exception as e:
        print(f"Error: {e}")