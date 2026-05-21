import logging

logger = logging.getLogger("SanskritRAG")


class PromptTemplate:
    """
    Builds the final RAG prompt by injecting retrieved context chunks
    and the user query into a structured template.
    """

    @staticmethod
    def build_prompt(query, context_chunks):
        """
        Assemble the prompt from context chunks and user query.

        Args:
            query (str): User's question
            context_chunks (list[str]): Retrieved text passages
        Returns:
            str: Formatted prompt ready for the LLM
        """

        # [CHANGED] added input validation and exception handling
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string.")

            if not isinstance(context_chunks, list):
                raise TypeError(f"context_chunks must be a list, got {type(context_chunks).__name__}")

            # [CHANGED] filter out empty/whitespace-only chunks before joining
            valid_chunks = [c for c in context_chunks if isinstance(c, str) and c.strip()]

            if not valid_chunks:
                logger.warning("All context chunks are empty. LLM will have no context to answer from.")

            context = "\n\n".join(valid_chunks)

            prompt = f"""You are an expert Sanskrit assistant.

Answer the user question only from the provided context.

If answer is not available in context, say:
"I could not find the answer in the provided Sanskrit documents."

================ CONTEXT ================

{context}

=========================================

Question:
{query}

Answer:"""

            logger.info(f"Prompt built. Context chunks: {len(valid_chunks)}, Prompt length: {len(prompt)} chars.")
            return prompt

        except (ValueError, TypeError):
            raise

        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            raise RuntimeError(f"build_prompt error: {e}") from e