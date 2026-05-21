import re
import logging
from unidecode import unidecode

logger = logging.getLogger("SanskritRAG")


def clean_text(text):
    """
    Clean Sanskrit text for RAG processing.

    Steps:
    1. Remove extra spaces/newlines
    2. Remove unwanted special characters
    3. Normalize unicode text via unidecode
    4. Preserve Sanskrit punctuation (। ॥)
    """

    # [CHANGED] added validation and try/except around each cleaning step
    try:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if not text.strip():
            logger.warning("clean_text received empty or whitespace-only input.")
            return ""

        # Step 1: Collapse multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)

        # Step 2: Remove unwanted special characters, keep Sanskrit punctuation
        text = re.sub(r'[^\w\s।॥]', '', text)

        # Step 3: Normalize unicode (converts Devanagari to Latin approximation)
        try:
            text = unidecode(text)
        except Exception as e:
            logger.warning(f"unidecode normalization failed: {e}. Using raw text.")  # [CHANGED] fallback if unidecode fails

        # Step 4: Strip leading/trailing whitespace
        text = text.strip()

        logger.info(f"Text cleaned. Output length: {len(text)} characters.")
        return text

    except TypeError:
        raise

    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        raise RuntimeError(f"clean_text error: {e}") from e


# Testing
if __name__ == "__main__":

    from logger import setup_logger
    setup_logger()

    sample_text = """
    अथ योगानुशासनम् ॥

    योगश्चित्तवृत्तिनिरोधः ॥
    """

    try:
        cleaned = clean_text(sample_text)
        print("\n===== CLEANED TEXT =====\n")
        print(cleaned)

    except Exception as e:
        print(f"Error: {e}")