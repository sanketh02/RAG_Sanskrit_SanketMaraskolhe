import logging
from src.pipeline.rag_pipeline import SanskritRAGPipeline  # [CHANGED] fixed import path to use src.pipeline
from src.utils.logger import setup_logger

logger = logging.getLogger("SanskritRAG")


def test_pipeline():
    """
    End-to-end integration test for SanskritRAGPipeline.
    Builds the index from a sample document and runs a test query.
    """

    logger.info("===== TEST: Complete RAG Pipeline =====")

    # [CHANGED] wrapped entire test in try/except for clean failure reporting
    try:
        pipeline = SanskritRAGPipeline()

        document_path = "data/raw/sanskrit_doc.pdf"

        # Build pipeline (index document)
        pipeline.build_pipeline(document_path)

        # Run a test query
        query = "yoga"
        result = pipeline.ask_question(query=query, top_k=3)

        # Assertions
        assert isinstance(result, dict), "Result must be a dict."
        assert "response" in result, "Result must contain 'response' key."
        assert "retrieved_chunks" in result, "Result must contain 'retrieved_chunks' key."
        assert len(result["retrieved_chunks"]) > 0, "At least one chunk must be retrieved."
        assert isinstance(result["response"], str) and result["response"].strip(), \
            "Response must be a non-empty string."

        logger.info("TEST PASSED: Full pipeline ran successfully.")
        print("\n===== TEST PASSED: Pipeline =====\n")
        print("Generated Response:\n")
        print(result["response"])

    except FileNotFoundError as e:
        logger.error(f"TEST FAILED — document not found: {e}")
        print(f"\nTEST FAILED (file not found): {e}")
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
    test_pipeline()