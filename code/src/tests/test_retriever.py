import logging
from src.retrieval.retriever import Retriever  # [CHANGED] fixed import path to use src.retrieval
from src.utils.logger import setup_logger

logger = logging.getLogger("SanskritRAG")


def test_retrieval():
    """
    Integration test for the Retriever component.
    Verifies that at least one chunk is returned for a sample query.
    """

    logger.info("===== TEST: Retrieval =====")

    # [CHANGED] wrapped test body in try/except for clean failure reporting
    try:
        retriever = Retriever()

        query = "yoga"

        results = retriever.retrieve(query=query, top_k=3)

        # Assertions
        assert isinstance(results, list), "Results must be a list."
        assert len(results) > 0, "No retrieval results found. Index may be empty."
        assert all(isinstance(r, str) for r in results), "Each result must be a string."

        logger.info(f"TEST PASSED: Retrieved {len(results)} chunks for query '{query}'.")
        print("\n===== TEST PASSED: Retrieval =====\n")

        for i, result in enumerate(results):
            print(f"Result {i+1}:\n{result}")
            print("\n" + "=" * 50 + "\n")

    except FileNotFoundError as e:
        logger.error(f"TEST FAILED — FAISS index not found: {e}. Run build_pipeline() first.")
        print(f"\nTEST FAILED (index not found): {e}")
        raise

    except AssertionError as e:
        logger.error(f"TEST FAILED (assertion): {e}")
        print(f"\nTEST FAILED: {e}")
        raise

    except Exception as e:
        logger.error(f"TEST FAILED (exception): {e}")
        print(f"\nTEST ERROR: {e}")
        raise


if __name__ == "__main__":

    setup_logger()
    test_retrieval()