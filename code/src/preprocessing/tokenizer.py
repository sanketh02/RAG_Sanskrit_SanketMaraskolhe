import re
import logging

logger = logging.getLogger("SanskritRAG")


def tokenize_text(text):
    """
    Tokenize Sanskrit/transliterated text into word tokens.
    Strips empty tokens produced by extra whitespace.
    """

    # [CHANGED] added input validation and exception handling
    try:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if not text.strip():
            logger.warning("tokenize_text received empty input. Returning empty list.")
            return []

        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)

        # Split on whitespace and remove empty tokens
        tokens = [token.strip() for token in text.split(" ") if token.strip()]

        logger.info(f"Tokenization complete. Total tokens: {len(tokens)}")
        return tokens

    except TypeError:
        raise

    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise RuntimeError(f"tokenize_text error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    sample_text = """
    अथ योगानुशासनम् ।
    योगश्चित्तवृत्तिनिरोधः ॥
    """

    try:
        tokens = tokenize_text(sample_text)
        print("\n===== TOKENS =====\n")
        print(tokens)

    except Exception as e:
        print(f"Error: {e}")