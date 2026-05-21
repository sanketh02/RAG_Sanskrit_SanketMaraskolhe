import os
import json
import logging

logger = logging.getLogger("SanskritRAG")


def create_directory(directory_path):
    """
    Create directory (and any missing parents) if it does not already exist.
    """

    # [CHANGED] added input validation and exception handling
    try:
        if not isinstance(directory_path, str) or not directory_path.strip():
            raise ValueError("directory_path must be a non-empty string.")

        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory ensured: {directory_path}")

    except ValueError:
        raise

    except OSError as e:
        logger.error(f"Failed to create directory '{directory_path}': {e}")
        raise RuntimeError(f"create_directory error: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error creating directory '{directory_path}': {e}")
        raise RuntimeError(f"create_directory error: {e}") from e


def save_json(data, file_path):
    """
    Serialize a dictionary to a JSON file with UTF-8 encoding.
    Creates parent directories automatically if they don't exist.
    """

    # [CHANGED] added type check, auto-mkdir for parent dirs, and exception handling
    try:
        if not isinstance(data, (dict, list)):
            raise TypeError(f"data must be a dict or list, got {type(data).__name__}")

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string.")

        # Ensure parent directory exists before writing
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)  # [CHANGED] auto-create parent dirs

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        logger.info(f"JSON saved: {file_path}")

    except (TypeError, ValueError):
        raise

    except OSError as e:
        logger.error(f"Failed to write JSON to '{file_path}': {e}")
        raise RuntimeError(f"save_json error: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error saving JSON to '{file_path}': {e}")
        raise RuntimeError(f"save_json error: {e}") from e


def load_json(file_path):
    """
    Load and parse a JSON file. Returns the deserialized Python object.
    """

    # [CHANGED] added file existence check, JSON decode error handling
    try:
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        logger.info(f"JSON loaded: {file_path}")
        return data

    except (ValueError, FileNotFoundError):
        raise

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file '{file_path}': {e}")
        raise RuntimeError(f"load_json decode error: {e}") from e  # [CHANGED] explicit decode error

    except OSError as e:
        logger.error(f"Failed to read JSON from '{file_path}': {e}")
        raise RuntimeError(f"load_json error: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error loading JSON from '{file_path}': {e}")
        raise RuntimeError(f"load_json error: {e}") from e


def print_separator(label=None):
    """
    Print a console separator line, with an optional centered label.

    Args:
        label (str | None): Optional text to display inside the separator
    """

    # [CHANGED] added optional label parameter and exception handling
    try:
        if label:
            if not isinstance(label, str):
                raise TypeError(f"label must be a string, got {type(label).__name__}")
            line = f"\n{'=' * 20} {label.strip()} {'=' * 20}\n"
        else:
            line = "\n" + "=" * 50 + "\n"

        print(line)

    except TypeError:
        raise

    except Exception as e:
        logger.warning(f"print_separator failed: {e}")  # non-fatal — just warn


# Testing
if __name__ == "__main__":

    from logger import setup_logger
    setup_logger()

    try:
        # Test directory creation
        create_directory("../../artifacts/test")

        # Test JSON save
        sample_data = {
            "project": "Sanskrit RAG",
            "status": "working"
        }

        save_json(sample_data, "../../artifacts/test/sample.json")

        # Test JSON load
        loaded_data = load_json("../../artifacts/test/sample.json")

        print_separator(label="Loaded JSON")
        print(loaded_data)

        # Test separator without label
        print_separator()

    except Exception as e:
        print(f"Error: {e}")