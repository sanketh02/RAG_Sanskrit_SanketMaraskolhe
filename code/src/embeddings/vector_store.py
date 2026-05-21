import faiss
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger("SanskritRAG")


class VectorStore:
    """
    FAISS-backed vector store for storing and searching chunk embeddings.
    Supports save/load of index and associated text chunks.
    """

    def __init__(self):
        self.index = None
        self.text_chunks = []

    def create_index(self, embeddings, text_chunks):
        """
        Build a FAISS flat L2 index from embedding matrix.

        Args:
            embeddings (numpy.ndarray): Shape (n, dim)
            text_chunks (list[str]): Corresponding text for each embedding
        """

        # [CHANGED] added input validation and exception handling
        try:
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Embeddings array is empty.")

            if len(embeddings) != len(text_chunks):
                raise ValueError(
                    f"Mismatch: {len(embeddings)} embeddings vs {len(text_chunks)} chunks."
                )

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype("float32"))
            self.text_chunks = text_chunks

            logger.info(f"FAISS index created. Vectors: {len(text_chunks)}, Dimension: {dimension}")

        except (ValueError, TypeError):
            raise

        except Exception as e:
            logger.error(f"FAISS index creation failed: {e}")
            raise RuntimeError(f"create_index error: {e}") from e

    def search(self, query_embedding, top_k=3):
        """
        Find top-k nearest chunks by L2 distance.

        Args:
            query_embedding (numpy.ndarray): Shape (1, dim)
            top_k (int): Number of results to return
        Returns:
            list[str]: Top matching text chunks
        """

        try:
            if self.index is None:
                raise RuntimeError("FAISS index not initialized. Call create_index() or load_index() first.")

            # [CHANGED] guard top_k against index size
            actual_top_k = min(top_k, self.index.ntotal)
            if actual_top_k < top_k:
                logger.warning(f"top_k={top_k} exceeds index size={self.index.ntotal}. Using {actual_top_k}.")

            distances, indices = self.index.search(
                np.array(query_embedding).astype("float32"),
                actual_top_k
            )

            retrieved_chunks = [
                self.text_chunks[idx]
                for idx in indices[0]
                if idx < len(self.text_chunks)
            ]

            logger.info(f"Search complete. Retrieved {len(retrieved_chunks)} chunks.")
            return retrieved_chunks

        except RuntimeError:
            raise

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise RuntimeError(f"search error: {e}") from e

    def save_index(self, save_path="../../artifacts/faiss_index"):
        """
        Persist FAISS index and chunk list to disk (artifact directory).
        """

        try:
            if self.index is None:
                raise RuntimeError("No index to save. Create an index first.")

            os.makedirs(save_path, exist_ok=True)  # [CHANGED] ensures artifact dir is created

            faiss.write_index(self.index, f"{save_path}/index.faiss")

            with open(f"{save_path}/chunks.pkl", "wb") as file:
                pickle.dump(self.text_chunks, file)

            logger.info(f"FAISS index saved to: {save_path}")

        except RuntimeError:
            raise

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise RuntimeError(f"save_index error: {e}") from e

    def load_index(self, load_path="artifacts/faiss_index"):
        """
        Load a previously saved FAISS index and chunk list from disk.
        """

        try:
            index_file = f"{load_path}/index.faiss"
            chunks_file = f"{load_path}/chunks.pkl"

            # [CHANGED] explicit file existence check with clear error messages
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")

            if not os.path.exists(chunks_file):
                raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

            self.index = faiss.read_index(index_file)

            with open(chunks_file, "rb") as file:
                self.text_chunks = pickle.load(file)

            logger.info(f"FAISS index loaded from: {load_path}. Total chunks: {len(self.text_chunks)}")

        except FileNotFoundError:
            raise

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise RuntimeError(f"load_index error: {e}") from e


# Testing
if __name__ == "__main__":

    from src.utils.logger import setup_logger
    from src.embeddings.embedding_model import EmbeddingModel

    setup_logger()

    sample_chunks = [
        "yogascittavrttinirodhah",
        "tada drastuh svarupe vasthanam",
        "vrttisarupyamitaratra"
    ]

    try:
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.generate_embeddings(sample_chunks)

        vector_store = VectorStore()
        vector_store.create_index(embeddings, sample_chunks)

        query_embedding = embedding_model.generate_query_embedding("yoga")
        results = vector_store.search(query_embedding, top_k=2)

        print("\n===== RETRIEVED CHUNKS =====\n")
        for result in results:
            print(result)

    except Exception as e:
        print(f"Error: {e}")