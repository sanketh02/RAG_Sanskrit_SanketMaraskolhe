import logging
from src.llm.response_generator import ResponseGenerator
from src.utils.logger import setup_logger  # [CHANGED] explicit logger setup for test runner

logger = logging.getLogger("SanskritRAG")


def test_response_generation():
    """
    Integration test for ResponseGenerator.
    Verifies that generate_answer returns a dict containing a 'response' key.
    """

    logger.info("===== TEST: Response Generation =====")

    # [CHANGED] wrapped test body in try/except for clean failure reporting
    try:
        generator = ResponseGenerator()

        query = "What is yoga?"  # [CHANGED] using transliterated query for portability

        result = generator.generate_answer(query=query, top_k=3)

        # Assertions
        assert isinstance(result, dict), "Result must be a dict."
        assert "response" in result, "Result must contain 'response' key."
        assert "retrieved_chunks" in result, "Result must contain 'retrieved_chunks' key."
        assert isinstance(result["response"], str) and result["response"].strip(), \
            "Response must be a non-empty string."

        logger.info("TEST PASSED: Response generation returned valid output.")
        print("\n===== TEST PASSED: Response Generation =====\n")
        print("Generated Response:\n")
        print(result["response"])

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
    test_response_generation()