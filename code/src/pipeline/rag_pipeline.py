import os
import json
import logging

from src.data_ingestion.loader import load_document
from src.data_ingestion.text_cleaner import clean_text
from src.preprocessing.transliteration import transliterate_text
from src.preprocessing.chunking import create_chunks
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.vector_store import VectorStore
from src.llm.response_generator import ResponseGenerator
from src.utils.logger import setup_logger        # [CHANGED] import logger setup
from src.utils.metrics import MetricsTracker     # [CHANGED] import metrics tracker

# Initialize module-level logger
logger = logging.getLogger("SanskritRAG")


class SanskritRAGPipeline:
    """
    End-to-End Sanskrit RAG Pipeline.

    build_pipeline()  — ingest a document, chunk it, embed it, index it.
    ask_question()    — query the indexed knowledge base and get an answer.

    Artifacts produced:
      artifacts/faiss_index/    — FAISS index + chunk pickle
      artifacts/logs/           — pipeline run log
      artifacts/metrics/        — JSON metrics per query
      artifacts/responses/      — JSON file with all query responses
    """

    # [CHANGED] centralized artifact paths as class constants
    ARTIFACT_DIR       = "artifacts"
    FAISS_PATH         = "artifacts/faiss_index"
    METRICS_PATH       = "artifacts/metrics"
    RESPONSES_PATH     = "artifacts/responses"
    LOG_PATH           = "artifacts/logs"

    def __init__(self):

        try:
            # Setup logger before anything else so all init messages are captured
            setup_logger(log_dir=self.LOG_PATH)  # [CHANGED] logger init moved into pipeline constructor

            logger.info("Initializing SanskritRAGPipeline...")

            self.embedding_model = EmbeddingModel()
            self.vector_store    = VectorStore()
            self.metrics         = MetricsTracker()  # [CHANGED] metrics tracker attached to pipeline

            # [CHANGED] ensure all artifact directories exist at startup
            for path in [self.FAISS_PATH, self.METRICS_PATH, self.RESPONSES_PATH, self.LOG_PATH]:
                os.makedirs(path, exist_ok=True)

            logger.info("SanskritRAGPipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise RuntimeError(f"SanskritRAGPipeline init error: {e}") from e

    # BUILD

    def build_pipeline(self, document_path):
        """
        Ingest a document and build the FAISS index.

        Steps:
            1. Load document (PDF or TXT)
            2. Clean text
            3. Transliterate Devanagari → Roman
            4. Chunk into overlapping token windows
            5. Generate embeddings
            6. Store in FAISS and save to artifacts/faiss_index/
        """

        logger.info("===== STARTING RAG PIPELINE BUILD =====")
        self.metrics.start_timer()  # [CHANGED] time the full build step

        try:
            # Step 1: Load
            logger.info(f"Step 1 — Loading document: {document_path}")
            raw_text = load_document(document_path)

            # Step 2: Clean
            logger.info("Step 2 — Cleaning text...")
            cleaned_text = clean_text(raw_text)

            # Step 3: Transliterate
            logger.info("Step 3 — Transliterating text...")
            transliterated_text = transliterate_text(cleaned_text)

            # Step 4: Chunk
            logger.info("Step 4 — Creating chunks...")
            chunks = create_chunks(transliterated_text, chunk_size=100, overlap=20)
            logger.info(f"Total chunks created: {len(chunks)}")

            if not chunks:
                raise ValueError("No chunks were created. Check document content and chunking config.")

            # Step 5: Embed
            logger.info("Step 5 — Generating embeddings...")
            embeddings = self.embedding_model.generate_embeddings(chunks)

            # Step 6: Index + save artifact
            logger.info("Step 6 — Building and saving FAISS index...")
            self.vector_store.create_index(embeddings, chunks)
            self.vector_store.save_index(save_path=self.FAISS_PATH)

            self.metrics.stop_timer()
            build_latency = self.metrics.get_latency()

            # [CHANGED] save a build summary artifact
            self._save_build_artifact(document_path, len(chunks), build_latency)

            logger.info(f"===== PIPELINE BUILD COMPLETE (latency: {build_latency}s) =====")

        except FileNotFoundError as e:
            logger.error(f"Document not found: {e}")
            raise

        except ValueError as e:
            logger.error(f"Pipeline build data error: {e}")
            raise

        except Exception as e:
            logger.error(f"Pipeline build failed unexpectedly: {e}")
            raise RuntimeError(f"build_pipeline error: {e}") from e

    # QUERY

    def ask_question(self, query, top_k=3):
        """
        Query the RAG system and return a generated answer.

        Args:
            query (str): User query in Sanskrit or English
            top_k (int): Number of context chunks to retrieve
        Returns:
            dict: {query, retrieved_chunks, response}
        """

        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            logger.info(f"Processing query: '{query}' | top_k={top_k}")

            self.metrics.start_timer()  # [CHANGED] time each query

            response_generator = ResponseGenerator()
            result = response_generator.generate_answer(query=query, top_k=top_k)

            self.metrics.stop_timer()
            latency = self.metrics.get_latency()

            # [CHANGED] compute retrieval accuracy using query as keyword proxy
            accuracy = MetricsTracker.calculate_retrieval_accuracy(
                result["retrieved_chunks"],
                expected_keyword=query.split()[0]  # use first word as a simple keyword signal
            )

            # [CHANGED] record this run in the metrics tracker
            self.metrics.record_run(
                query=query,
                latency=latency,
                accuracy=accuracy,
                num_chunks=len(result["retrieved_chunks"])
            )

            # [CHANGED] persist metrics artifact after every query
            self.metrics.save_metrics_artifact(save_path=self.METRICS_PATH)

            # [CHANGED] append response to the responses artifact file
            self._save_response_artifact(result)

            logger.info(f"Query complete. Latency: {latency}s | Accuracy: {accuracy}")
            return result

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"ask_question failed for query '{query}': {e}")
            raise RuntimeError(f"ask_question error: {e}") from e

    # ARTIFACT HELPERS

    def _save_build_artifact(self, document_path, num_chunks, latency):
        """
        Write a JSON summary of the pipeline build run to artifacts/metrics/.
        """

        # [CHANGED] new method — generates a build summary JSON artifact
        try:
            summary = {
                "document": document_path,
                "chunks_created": num_chunks,
                "build_latency_seconds": latency,
                "faiss_index_path": self.FAISS_PATH,
                "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")
            }

            artifact_file = os.path.join(self.METRICS_PATH, "build_summary.json")

            with open(artifact_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)

            logger.info(f"Build summary artifact saved: {artifact_file}")

        except Exception as e:
            logger.warning(f"Could not save build artifact: {e}")  # non-fatal warning

    def _save_response_artifact(self, result):
        """
        Append a query result entry to artifacts/responses/responses.json.
        Creates the file if it does not exist.
        """

        # [CHANGED] new method — accumulates all query responses into one JSON artifact
        try:
            artifact_file = os.path.join(self.RESPONSES_PATH, "responses.json")

            # Load existing responses if file already exists
            if os.path.exists(artifact_file):
                with open(artifact_file, "r", encoding="utf-8") as f:
                    all_responses = json.load(f)
            else:
                all_responses = []

            # Append new entry
            all_responses.append({
                "query": result["query"],
                "retrieved_chunks": result["retrieved_chunks"],
                "response": result["response"],
                "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")
            })

            with open(artifact_file, "w", encoding="utf-8") as f:
                json.dump(all_responses, f, indent=4, ensure_ascii=False)

            logger.info(f"Response artifact updated: {artifact_file}")

        except Exception as e:
            logger.warning(f"Could not save response artifact: {e}")  # non-fatal warning


# Entry Point

if __name__ == "__main__":

    document_path = "data/raw/Sanskrit_doc.pdf"

    try:
        rag_pipeline = SanskritRAGPipeline()

        rag_pipeline.build_pipeline(document_path)

        while True:
            print("\n====================================")
            query = input("\nEnter your Sanskrit query (or type 'exit'): ").strip()

            if query.lower() == "exit":
                logger.info("User exited the query loop.")
                break

            if not query:
                print("Please enter a valid query.")
                continue

            try:
                result = rag_pipeline.ask_question(query=query, top_k=3)
                print("\n===== GENERATED RESPONSE =====\n")
                print(result["response"])

            except Exception as e:
                print(f"Error processing query: {e}")
                logger.error(f"Query loop error: {e}")

    except Exception as e:
        print(f"Fatal pipeline error: {e}")