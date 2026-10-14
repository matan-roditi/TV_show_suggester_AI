import numpy as np

from src.vector_db import VectorDB


def calculate_average_vector(vectors: list[np.ndarray]) -> np.ndarray:
    """
    Calculate the element-wise mean (average) of a list of numpy arrays.
    
    Args:
        vectors: A list of numpy arrays
    
    Returns:
        A single numpy array representing the element-wise mean of the input vectors
    
    Raises:
        ValueError: If the input list is empty or contains vectors of different dimensions
        TypeError: If the list contains None or non-array items
    """
    # Check if the input list is empty
    if not vectors:
        raise ValueError("Cannot calculate average of an empty list")
    
    # Check for None or invalid items and get the reference shape
    reference_shape = None
    for i, vec in enumerate(vectors):
        # Check if item is None or not a numpy array
        if vec is None:
            raise TypeError(f"Item at index {i} is None, expected numpy array")
        
        try:
            # Ensure it's a numpy array
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"Item at index {i} is not a numpy array, got {type(vec)}")
        except (AttributeError, TypeError):
            raise TypeError(f"Item at index {i} is not a valid numpy array")
        
        # Check dimensions match
        if reference_shape is None:
            reference_shape = vec.shape
        elif vec.shape != reference_shape:
            raise ValueError(
                f"Vector at index {i} has shape {vec.shape}, "
                f"but expected shape {reference_shape}. "
                f"All vectors must have the same dimensions."
            )
    
    return np.mean(vectors, axis=0)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns a value in [-1, 1] (1 = identical, -1 = opposite).
    """
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return float(np.clip(sim, -1.0, 1.0))


def similarity_to_percentage(sim: float) -> int:
    """
    Map cosine similarity [-1, 1] to percentage [0, 100].
    -1 -> 0, 0 -> 50, 1 -> 100
    """
    pct = (sim + 1.0) / 2.0 * 100.0
    return int(round(np.clip(pct, 0.0, 100.0)))


def get_recommendations(
    user_vector: np.ndarray,
    db_vectors: dict[str, np.ndarray],
    limit: int = 5,
    input_shows: list[str] | None = None
) -> list[dict[str, int | str]]:
    """
    Get recommendations based on user's taste vector.

    Returns:
        A list of dictionaries with keys 'title' and 'percentage',
        sorted by best match first.
    """
    if input_shows is None:
        input_shows = []

    recommendations: list[dict[str, float | int | str]] = []

    for title, show_vector in db_vectors.items():
        if title in input_shows:
            continue

        sim = cosine_similarity(user_vector, show_vector)
        percentage = similarity_to_percentage(sim)

        recommendations.append({
            "title": title,
            "percentage": percentage,
            "_sim": sim  # internal sorting key
        })

    recommendations.sort(key=lambda x: (x["percentage"], x["_sim"]), reverse=True)
    top = recommendations[:limit]

    return [{"title": r["title"], "percentage": int(r["percentage"])} for r in top]


def get_recommendations_vectordb(
    user_vector: np.ndarray,
    vector_db: VectorDB,
    id_to_title: dict[int, str],
    vectors_by_title: dict[str, np.ndarray],
    limit: int = 5,
    input_shows: list[str] | None = None
) -> list[dict[str, int | str]]:
    """
    Get recommendations using an ANN index (VectorDB) instead of a brute-force scan.

    Shortlists nearest neighbors via usearch, then rescoring the shortlist with
    exact cosine similarity (ANN search is approximate; scores shown to the user
    should be exact). vectors_by_title is used for rescoring rather than
    vector_db.id_to_vector, since a VectorDB loaded from disk via load_index()
    does not repopulate that in-memory cache.

    Returns:
        A list of dictionaries with keys 'title' and 'percentage',
        sorted by best match first.
    """
    if input_shows is None:
        input_shows = []
    excluded_ids = {
        id_ for id_, title in id_to_title.items() if title in input_shows
    }

    # Over-fetch so that excluded (input) shows don't shrink the final result below `limit`.
    fetch_count = limit + len(excluded_ids)
    results = vector_db.search(user_vector, limit=fetch_count)

    recommendations: list[dict[str, float | int | str]] = []
    for id_, _distance in results:
        if id_ in excluded_ids:
            continue

        title = id_to_title[id_]
        show_vector = vectors_by_title[title]
        sim = cosine_similarity(user_vector, show_vector)
        percentage = similarity_to_percentage(sim)

        recommendations.append({
            "title": title,
            "percentage": percentage,
            "_sim": sim
        })

    recommendations.sort(key=lambda x: (x["percentage"], x["_sim"]), reverse=True)
    top = recommendations[:limit]

    return [{"title": r["title"], "percentage": int(r["percentage"])} for r in top]
