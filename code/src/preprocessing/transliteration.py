import logging
from unidecode import unidecode

logger = logging.getLogger("SanskritRAG")


def transliterate_text(text):
    """
    Convert Sanskrit unicode (Devanagari) text into Roman transliterated text.
    Returns original text if transliteration fails.
    """

    # [CHANGED] added input validation and exception handling
    try:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if not text.strip():
            logger.warning("transliterate_text received empty input. Returning as-is.")
            return text

        transliterated = unidecode(text)

        logger.info(f"Transliteration complete. Output length: {len(transliterated)} characters.")
        return transliterated

    except TypeError:
        raise

    except Exception as e:
        logger.error(f"Transliteration failed: {e}. Returning original text.")
        return text  # [CHANGED] graceful fallback — return original rather than crashing pipeline


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    sample_text = "योगश्चित्तवृत्तिनिरोधः"

    try:
        result = transliterate_text(sample_text)
        print("\n===== TRANSLITERATED TEXT =====\n")
        print(result)

    except Exception as e:
        print(f"Error: {e}")