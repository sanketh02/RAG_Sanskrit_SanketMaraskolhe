import time
import json
import os
import logging

logger = logging.getLogger("SanskritRAG")


class MetricsTracker:
    """
    Tracks pipeline performance metrics: latency and retrieval accuracy.
    Supports saving metrics as a JSON artifact for reporting.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        # [CHANGED] store all recorded metrics for artifact export
        self.metrics_log = []

    def start_timer(self):
        """Start the execution timer."""

        try:
            self.start_time = time.time()
            logger.info("Timer started.")

        except Exception as e:
            logger.error(f"start_timer failed: {e}")
            raise

    def stop_timer(self):
        """Stop the execution timer."""

        try:
            if self.start_time is None:
                raise RuntimeError("Timer was never started. Call start_timer() first.")

            self.end_time = time.time()
            logger.info("Timer stopped.")

        except RuntimeError:
            raise

        except Exception as e:
            logger.error(f"stop_timer failed: {e}")
            raise

    def get_latency(self):
        """
        Return elapsed time in seconds between start and stop.

        Returns:
            float | None: Latency in seconds, or None if timer incomplete
        """

        try:
            if self.start_time is None or self.end_time is None:
                logger.warning("Cannot compute latency: timer not fully started/stopped.")
                return None

            latency = round(self.end_time - self.start_time, 3)
            logger.info(f"Latency: {latency}s")
            return latency

        except Exception as e:
            logger.error(f"get_latency failed: {e}")
            return None

    @staticmethod
    def calculate_retrieval_accuracy(retrieved_chunks, expected_keyword):
        """
        Compute what fraction of retrieved chunks contain the expected keyword.

        Args:
            retrieved_chunks (list[str]): Chunks returned by retriever
            expected_keyword (str): Keyword that should appear in relevant chunks
        Returns:
            float: Accuracy score between 0.0 and 1.0
        """

        # [CHANGED] added input validation and division-by-zero guard
        try:
            if not isinstance(retrieved_chunks, list):
                raise TypeError(f"retrieved_chunks must be a list, got {type(retrieved_chunks).__name__}")

            if not retrieved_chunks:
                logger.warning("retrieved_chunks is empty. Accuracy is 0.0.")
                return 0.0

            if not isinstance(expected_keyword, str) or not expected_keyword.strip():
                raise ValueError("expected_keyword must be a non-empty string.")

            correct = sum(
                1 for chunk in retrieved_chunks
                if expected_keyword.lower() in chunk.lower()
            )

            accuracy = round(correct / len(retrieved_chunks), 2)
            logger.info(f"Retrieval accuracy for keyword '{expected_keyword}': {accuracy}")
            return accuracy

        except (TypeError, ValueError):
            raise

        except Exception as e:
            logger.error(f"calculate_retrieval_accuracy failed: {e}")
            raise RuntimeError(f"Metrics error: {e}") from e

    def record_run(self, query, latency, accuracy, num_chunks):
        """
        Append a metrics entry for a single pipeline run.

        Args:
            query (str): The query that was run
            latency (float): Time taken in seconds
            accuracy (float): Retrieval accuracy score
            num_chunks (int): Number of chunks retrieved
        """

        # [CHANGED] new method to accumulate per-run metrics for artifact export
        entry = {
            "query": query,
            "latency_seconds": latency,
            "retrieval_accuracy": accuracy,
            "chunks_retrieved": num_chunks,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.metrics_log.append(entry)
        logger.info(f"Metrics recorded: {entry}")

    def save_metrics_artifact(self, save_path="../../artifacts/metrics"):
        """
        Write all recorded metrics to a JSON file in the artifacts directory.

        Args:
            save_path (str): Directory path for the metrics artifact
        """

        # [CHANGED] new method — saves metrics as a JSON artifact file
        try:
            os.makedirs(save_path, exist_ok=True)
            artifact_file = os.path.join(save_path, "pipeline_metrics.json")

            with open(artifact_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics_log, f, indent=4, ensure_ascii=False)

            logger.info(f"Metrics artifact saved to: {artifact_file}")

        except Exception as e:
            logger.error(f"Failed to save metrics artifact: {e}")
            raise RuntimeError(f"save_metrics_artifact error: {e}") from e


# Testing
if __name__ == "__main__":

    from logger import setup_logger
    setup_logger()

    try:
        metrics = MetricsTracker()

        metrics.start_timer()
        time.sleep(1)
        metrics.stop_timer()

        latency = metrics.get_latency()
        print(f"\nLatency: {latency} seconds")

        retrieved_chunks = [
            "yogascittavrttinirodhah",
            "tada drastuh svarupe vasthanam"
        ]

        accuracy = metrics.calculate_retrieval_accuracy(
            retrieved_chunks, expected_keyword="yoga"
        )
        print(f"\nRetrieval Accuracy: {accuracy}")

        metrics.record_run(
            query="yoga",
            latency=latency,
            accuracy=accuracy,
            num_chunks=len(retrieved_chunks)
        )

        metrics.save_metrics_artifact()

    except Exception as e:
        print(f"Error: {e}")