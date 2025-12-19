import pickle


def load_embeddings(file_path: str) -> dict:
    """
    Load embeddings from a pickle file.
    
    Args:
        file_path: Path to the pickle file containing the embeddings dictionary
    
    Returns:
        A dictionary with show titles as keys
    
    Raises:
        FileNotFoundError: If the file does not exist
        EOFError: If the pickle file is corrupted or empty
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_embeddings(data: dict, file_path: str) -> None:
    """
    Save embeddings to a pickle file.
    
    Args:
        data: Dictionary with show titles as keys
        file_path: Path where the pickle file will be saved
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def get_all_titles(data: dict) -> list[str]:
    """
    Extract all show titles (keys) from the embeddings dictionary.
    
    Args:
        data: Dictionary with show titles as keys
    
    Returns:
        A list of all show titles (keys) from the dictionary
    """
    return list(data.keys())
