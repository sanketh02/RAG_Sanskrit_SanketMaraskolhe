import logging
from rank_bm25 import BM25Okapi
from src.embeddings.embedding_model import EmbeddingModel  # [CHANGED] fixed import path to use src.embeddings
from src.embeddings.vector_store import VectorStore        # [CHANGED] fixed import path to use src.embeddings

logger = logging.getLogger("SanskritRAG")


class HybridSearch:
    """
    Hybrid Retrieval combining:
    - Semantic search via FAISS (dense embeddings)
    - Keyword search via BM25 (sparse term matching)

    Results are merged and deduplicated, prioritizing semantic hits.
    """

    def __init__(self):

        # [CHANGED] wrapped all init steps in try/except
        try:
            logger.info("Initializing HybridSearch...")

            self.embedding_model = EmbeddingModel()

            self.vector_store = VectorStore()
            self.vector_store.load_index()
            self.text_chunks = self.vector_store.text_chunks

            if not self.text_chunks:
                raise ValueError("Loaded text chunks are empty. Rebuild the pipeline index.")

            # Tokenize corpus for BM25
            tokenized_corpus = [chunk.split(" ") for chunk in self.text_chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)

            logger.info(f"HybridSearch initialized with {len(self.text_chunks)} chunks.")

        except FileNotFoundError as e:
            logger.error(f"Index files not found. Run build_pipeline() first. Details: {e}")
            raise

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"HybridSearch initialization failed: {e}")
            raise RuntimeError(f"HybridSearch init error: {e}") from e

    def semantic_search(self, query, top_k=3):
        """
        Dense semantic retrieval using FAISS vector store.

        Args:
            query (str): User query
            top_k (int): Number of results
        Returns:
            list[str]: Semantically relevant chunks
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            query_embedding = self.embedding_model.generate_query_embedding(query)
            results = self.vector_store.search(query_embedding, top_k=top_k)

            logger.info(f"Semantic search returned {len(results)} results.")
            return results

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise RuntimeError(f"semantic_search error: {e}") from e

    def keyword_search(self, query, top_k=3):
        """
        Sparse keyword retrieval using BM25.

        Args:
            query (str): User query
            top_k (int): Number of results
        Returns:
            list[str]: Keyword-matched chunks
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            # [CHANGED] guard top_k against corpus size
            actual_top_k = min(top_k, len(self.text_chunks))
            tokenized_query = query.split(" ")
            results = self.bm25.get_top_n(tokenized_query, self.text_chunks, n=actual_top_k)

            logger.info(f"Keyword search returned {len(results)} results.")
            return results

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise RuntimeError(f"keyword_search error: {e}") from e

    def hybrid_retrieve(self, query, top_k=3):
        """
        Merge semantic and keyword results, deduplicate, and return top_k.
        Semantic results appear first in the merged list.

        Args:
            query (str): User query
            top_k (int): Final number of results to return
        Returns:
            list[str]: Deduplicated hybrid results
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            logger.info(f"Running hybrid retrieval for query: '{query}'")

            semantic_results = self.semantic_search(query, top_k)
            keyword_results = self.keyword_search(query, top_k)

            # Merge and deduplicate while preserving order (semantic first)
            # [CHANGED] dict.fromkeys preserves insertion order in Python 3.7+
            combined_results = list(dict.fromkeys(semantic_results + keyword_results))

            final_results = combined_results[:top_k]
            logger.info(f"Hybrid retrieval complete. Returning {len(final_results)} results.")
            return final_results

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"Hybrid retrieval failed for query '{query}': {e}")
            raise RuntimeError(f"hybrid_retrieve error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    setup_logger()

    try:
        hybrid_search = HybridSearch()
        results = hybrid_search.hybrid_retrieve(query="yoga", top_k=3)

        print("\n===== HYBRID SEARCH RESULTS =====\n")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:\n{result}")

    except Exception as e:
        print(f"Error: {e}")