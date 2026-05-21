import logging
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore

logger = logging.getLogger("SanskritRAG")


class Retriever:
    """
    Semantic retriever: embeds a query and searches the FAISS vector store.
    """

    def __init__(self):

        # [CHANGED] wrapped component initialization in try/except
        try:
            logger.info("Initializing Retriever...")

            self.embedding_model = EmbeddingModel()

            self.vector_store = VectorStore()
            self.vector_store.load_index()

            logger.info("Retriever initialized successfully.")

        except FileNotFoundError as e:
            logger.error(f"FAISS index not found. Run the pipeline build step first. Details: {e}")
            raise

        except Exception as e:
            logger.error(f"Retriever initialization failed: {e}")
            raise RuntimeError(f"Retriever init error: {e}") from e

    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query (str): User's search query
            top_k (int): Number of chunks to retrieve
        Returns:
            list[str]: Retrieved text chunks
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            logger.info(f"Retrieving top-{top_k} chunks for query: '{query}'")

            query_embedding = self.embedding_model.generate_query_embedding(query)

            retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
            return retrieved_chunks

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            raise RuntimeError(f"retrieve error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    try:
        retriever = Retriever()
        results = retriever.retrieve(query="yoga", top_k=2)

        print("\n===== RETRIEVED RESULTS =====\n")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:\n{result}")

    except Exception as e:
        print(f"Error: {e}")