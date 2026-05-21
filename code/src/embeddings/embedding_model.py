import logging
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger("SanskritRAG")


class EmbeddingModel:
    """
    Generates sentence embeddings using a multilingual transformer.
    Default model supports Sanskrit (Devanagari) and transliterated text.
    """

    def __init__(
        self,
        model_name=EMBEDDING_MODEL_NAME
    ):

        # [CHANGED] wrapped model loading in try/except
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(f"EmbeddingModel init failed: {e}") from e

    def generate_embeddings(self, texts):
        """
        Generate embedding vectors for a list of text chunks.

        Args:
            texts (list[str]): Text chunks to embed
        Returns:
            numpy.ndarray: Embedding matrix of shape (n, dim)
        """

        try:
            if not texts:
                raise ValueError("texts list is empty. Cannot generate embeddings.")

            if not isinstance(texts, list):
                raise TypeError(f"Expected list, got {type(texts).__name__}")

            embeddings = self.model.encode(texts, convert_to_numpy=True)

            logger.info(f"Generated embeddings for {len(texts)} chunks. Shape: {embeddings.shape}")
            return embeddings

        except (ValueError, TypeError):
            raise

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"generate_embeddings error: {e}") from e

    def generate_query_embedding(self, query):
        """
        Generate embedding vector for a single query string.
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            query_embedding = self.model.encode([query], convert_to_numpy=True)

            logger.info(f"Query embedding generated. Shape: {query_embedding.shape}")
            return query_embedding

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise RuntimeError(f"generate_query_embedding error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    sample_chunks = [
        "yogascittavrttinirodhah",
        "tada drastuh svarupe vasthanam"
    ]

    try:
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.generate_embeddings(sample_chunks)
        print("\n===== EMBEDDING SHAPE =====\n")
        print(embeddings.shape)

    except Exception as e:
        print(f"Error: {e}")