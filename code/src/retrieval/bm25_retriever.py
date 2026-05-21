import logging
from rank_bm25 import BM25Okapi
from src.embeddings.vector_store import VectorStore  # [CHANGED] fixed import path to use src.embeddings

logger = logging.getLogger("SanskritRAG")


class BM25Retriever:
    """
    Keyword-based retriever using BM25 (Okapi BM25 ranking).
    Loads stored text chunks from the FAISS vector store on init.
    """

    def __init__(self):

        # [CHANGED] wrapped all init steps in try/except
        try:
            logger.info("Initializing BM25Retriever...")

            self.vector_store = VectorStore()
            self.vector_store.load_index()
            self.text_chunks = self.vector_store.text_chunks

            if not self.text_chunks:
                raise ValueError("Loaded text chunks are empty. Rebuild the pipeline index.")

            # Tokenize each chunk for BM25
            self.tokenized_corpus = [
                chunk.split(" ") for chunk in self.text_chunks
            ]

            self.bm25 = BM25Okapi(self.tokenized_corpus)

            logger.info(f"BM25Retriever initialized with {len(self.text_chunks)} chunks.")

        except FileNotFoundError as e:
            logger.error(f"Index files not found. Run build_pipeline() first. Details: {e}")
            raise

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"BM25Retriever initialization failed: {e}")
            raise RuntimeError(f"BM25Retriever init error: {e}") from e

    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k relevant chunks using BM25 keyword scoring.

        Args:
            query (str): User query string
            top_k (int): Number of results to return
        Returns:
            list[str]: Top matching text chunks
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            # [CHANGED] guard top_k against corpus size
            actual_top_k = min(top_k, len(self.text_chunks))
            if actual_top_k < top_k:
                logger.warning(f"top_k={top_k} exceeds corpus size={len(self.text_chunks)}. Using {actual_top_k}.")

            tokenized_query = query.split(" ")
            results = self.bm25.get_top_n(tokenized_query, self.text_chunks, n=actual_top_k)

            logger.info(f"BM25 retrieval complete. Retrieved {len(results)} chunks.")
            return results

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"BM25 retrieval failed for query '{query}': {e}")
            raise RuntimeError(f"BM25Retriever.retrieve error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    try:
        retriever = BM25Retriever()
        results = retriever.retrieve(query="yoga", top_k=3)

        print("\n===== BM25 RETRIEVAL RESULTS =====\n")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:\n{result}")

    except Exception as e:
        print(f"Error: {e}")