import os
from typing import List
import numpy as np  # <-- add this

from src.data_handler import load_embeddings, get_all_titles
from src.matcher import parse_user_input, fuzzy_match_shows
from src.recommender import calculate_average_vector, get_recommendations_vectordb
from src.vector_db import VectorDB

EMBEDDINGS_PATH = os.path.join("data", "embeddings.pkl")
VECTOR_INDEX_PATH = os.path.join("data", "vector_index.usearch")

DEFAULT_THRESHOLD = 70
DEFAULT_LIMIT = 5


def _ask_for_liked_shows() -> List[str]:
    print("Which TV shows did you really like watching? Separate them by a")
    print("comma. Make sure to enter more than 1 show")
    raw = input().strip()
    return parse_user_input(raw)


def _retry_message() -> None:
    print("Sorry about that. Lets try again, please make sure to write the names of")
    print("the tv shows correctly")


def run_cli(limit: int = DEFAULT_LIMIT, threshold: int = DEFAULT_THRESHOLD) -> None:
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    titles = get_all_titles(embeddings)

    id_to_title = dict(enumerate(titles))
    dimension = len(next(iter(embeddings.values())))
    vector_db = VectorDB(dimension=dimension)

    # The on-disk index only matches id_to_title as long as embeddings.pkl hasn't
    # changed since it was saved, so treat it as stale if the pickle is newer.
    index_is_fresh = (
        os.path.exists(VECTOR_INDEX_PATH)
        and os.path.getmtime(VECTOR_INDEX_PATH) >= os.path.getmtime(EMBEDDINGS_PATH)
    )

    if index_is_fresh:
        vector_db.load_index(VECTOR_INDEX_PATH)
    else:
        vector_db.build_index({
            id_: np.asarray(embeddings[title], dtype=np.float32)
            for id_, title in id_to_title.items()
        })
        vector_db.save_index(VECTOR_INDEX_PATH)

    while True:
        user_shows = _ask_for_liked_shows()
        if len(user_shows) < 2:
            _retry_message()
            continue

        matched = fuzzy_match_shows(user_shows, titles, threshold=threshold)
        if len(matched) < 2:
            _retry_message()
            continue

        joined = ", ".join(matched)
        confirm = input(f"Making sure, do you mean {joined}?(y/n) ").strip().lower()
        if confirm == "y":
            confirmed = matched
            break

        _retry_message()

    print("Great! Generating recommendations now…")

    # Convert list embeddings -> numpy arrays
    vectors = [np.asarray(embeddings[t], dtype=float) for t in confirmed if t in embeddings]
    if len(vectors) < 1:
        _retry_message()
        return

    user_vector = calculate_average_vector(vectors)

    recs = get_recommendations_vectordb(
        user_vector=user_vector,
        vector_db=vector_db,
        id_to_title=id_to_title,
        vectors_by_title={k: np.asarray(v, dtype=float) for k, v in embeddings.items()},
        limit=limit,
        input_shows=confirmed,
    )

    print("Here are the tv shows that i think you would love:")
    for r in recs:
        print(f'{r["title"]} ({r["percentage"]}%)')


if __name__ == "__main__":
    run_cli()
