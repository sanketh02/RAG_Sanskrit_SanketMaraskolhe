import logging
import os


def setup_logger(log_dir="../../artifacts/logs"):
    """
    Configure application logger with file and stream handlers.
    Artifacts directory is created automatically if it doesn't exist.
    """

    try:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "rag_pipeline.log")

        # Configure logging with both file and console output
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),  # [CHANGED] added encoding for Sanskrit text support
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger("SanskritRAG")
        logger.info(f"Logger initialized. Log file: {log_file}")

        return logger

    except OSError as e:
        # Fallback to console-only logging if file creation fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        logger = logging.getLogger("SanskritRAG")
        logger.warning(f"Could not create log file at {log_dir}: {e}. Logging to console only.")  # [CHANGED] fallback handler
        return logger

    except Exception as e:
        raise RuntimeError(f"Failed to setup logger: {e}") from e


# Testing
if __name__ == "__main__":

    logger = setup_logger()
    logger.info("Logger initialized successfully!")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")