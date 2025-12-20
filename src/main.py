import os
from typing import List
import numpy as np

from src.data_handler import load_embeddings, get_all_titles
from src.matcher import parse_user_input, fuzzy_match_shows
from src.recommender import calculate_average_vector, get_recommendations

EMBEDDINGS_PATH = os.path.join("data", "embeddings.pkl")

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

    recs = get_recommendations(
        user_vector=user_vector,
        db_vectors={k: np.asarray(v, dtype=float) for k, v in embeddings.items()},  # ensure db vectors also arrays
        limit=limit,
        input_shows=confirmed,
    )

    print("Here are the tv shows that i think you would love:")
    for r in recs:
        print(f'{r["title"]} ({r["percentage"]}%)')

    print(
        "I have also created just for you two shows which I think you would love.\n"
        "Show #1 is based on the fact that you loved the input shows that you\n"
        "gave me. Its name is TBD Show 1 and it is about TBD.\n"
        "Show #2 is based on the shows that I recommended for you. Its name is\n"
        "TBD Show 2 and it is about TBD.\n"
        "Here are also the 2 tv show ads. Hope you like them!"
    )


if __name__ == "__main__":
    run_cli()
