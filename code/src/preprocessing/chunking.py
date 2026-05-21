import logging
from src.preprocessing.tokenizer import tokenize_text
from src.config import CHUNK_SIZE

logger = logging.getLogger("SanskritRAG")


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=20):
    """
    Create overlapping token-based chunks from text.

    Args:
        text (str): Input text (transliterated or raw)
        chunk_size (int): Number of tokens per chunk
        overlap (int): Number of overlapping tokens between chunks

    Returns:
        list[str]: List of text chunk strings
    """

    # [CHANGED] added parameter validation and exception handling
    try:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(f"overlap must be >= 0 and < chunk_size. Got overlap={overlap}, chunk_size={chunk_size}")

        tokens = tokenize_text(text)

        if not tokens:
            logger.warning("No tokens found in text. Returning empty chunks list.")
            return []

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(chunk_text)
            start += chunk_size - overlap

        logger.info(f"Chunking complete. Total chunks: {len(chunks)} (chunk_size={chunk_size}, overlap={overlap})")
        return chunks

    except (TypeError, ValueError):
        raise

    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise RuntimeError(f"create_chunks error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    sample_text = """
    अथ योगानुशासनम् ।
    योगश्चित्तवृत्तिनिरोधः ॥
    तदा द्रष्टुः स्वरूपेऽवस्थानम् ॥
    वृत्तिसारूप्यमितरत्र ॥
    """

    try:
        chunks = create_chunks(sample_text, chunk_size=5, overlap=2)
        print("\n===== CHUNKS =====\n")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:\n{chunk}")

    except Exception as e:
        print(f"Error: {e}")