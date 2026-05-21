import logging
from transformers import pipeline
from src.config import LLM_MODEL_NAME

logger = logging.getLogger("SanskritRAG")


class LLMModel:
    """
    CPU-based LLM loader using HuggingFace transformers pipeline.
    Generates text responses from a prompt string.
    """

    def __init__(self, model_name=LLM_MODEL_NAME):

        # [CHANGED] wrapped model loading in try/except with descriptive errors
        try:
            logger.info(f"Loading LLM model: {model_name}")

            self.generator = pipeline(
                task="text-generation",
                model=model_name,
                device=-1  # CPU only
            )

            logger.info("LLM model loaded successfully.")

        except OSError as e:
            logger.error(f"Model not found or inaccessible: '{model_name}'. Details: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to load LLM model '{model_name}': {e}")
            raise RuntimeError(f"LLMModel init error: {e}") from e

    def generate_response(self, prompt, max_new_tokens=200):
        """
        Generate a text response for a given prompt.

        Args:
            prompt (str): Full formatted prompt string
            max_new_tokens (int): Token budget for the generated response
        Returns:
            str: Generated text (full prompt + response)
        """

        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string.")

            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")

            logger.info(f"Generating response. Prompt length: {len(prompt)} chars, max_new_tokens: {max_new_tokens}")

            response = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )

            generated_text = response[0]["generated_text"]

            logger.info(f"Response generated. Output length: {len(generated_text)} chars.")
            return generated_text

        except ValueError:
            raise

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"generate_response error: {e}") from e