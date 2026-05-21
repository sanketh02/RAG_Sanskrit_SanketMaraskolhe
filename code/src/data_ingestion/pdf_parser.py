import logging
from pypdf import PdfReader

logger = logging.getLogger("SanskritRAG")


class PDFParser:
    """
    PDF Parser for Sanskrit documents.
    Extracts text page by page with error handling per page.
    """

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        try:
            self.reader = PdfReader(pdf_path)
            logger.info(f"PDFParser initialized for: {pdf_path}")

        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise

        except Exception as e:
            logger.error(f"Failed to open PDF '{pdf_path}': {e}")
            raise RuntimeError(f"Cannot open PDF: {e}") from e

    def extract_text(self):
        """
        Extract complete text from all PDF pages.
        Skips unreadable pages with a warning instead of crashing.
        """

        extracted_text = ""

        try:
            for page_number, page in enumerate(self.reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
                    else:
                        logger.warning(f"Page {page_number} returned empty text.")

                except Exception as e:
                    # [CHANGED] log per-page errors and continue instead of silently skipping
                    logger.warning(f"Error reading page {page_number}: {e}. Skipping page.")

            if not extracted_text.strip():
                logger.warning("No text extracted from PDF. Document may be image-only.")

            logger.info(f"Text extraction complete. Characters extracted: {len(extracted_text)}")
            return extracted_text

        except Exception as e:
            logger.error(f"Critical error during text extraction: {e}")
            raise RuntimeError(f"Text extraction failed: {e}") from e

    def get_total_pages(self):
        """Return total pages in PDF."""

        try:
            return len(self.reader.pages)
        except Exception as e:
            logger.error(f"Could not get page count: {e}")
            return 0

    def get_metadata(self):
        """Return PDF metadata."""

        try:
            return self.reader.metadata
        except Exception as e:
            logger.warning(f"Could not read metadata: {e}")
            return {}


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    pdf_path = "../../data/raw/sanskrit_doc.pdf"

    try:
        parser = PDFParser(pdf_path)
        print(f"\nTotal Pages: {parser.get_total_pages()}")
        print(f"\nMetadata: {parser.get_metadata()}")
        text = parser.extract_text()
        print(text[:1000])

    except Exception as e:
        print(f"Error: {e}")