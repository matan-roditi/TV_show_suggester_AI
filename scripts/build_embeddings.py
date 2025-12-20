import os
import time
import pickle
import pandas as pd
from google import genai
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

CSV_PATH = os.path.join("data", "tv_shows.csv")
OUT_PATH = os.path.join("data", "embeddings.pkl")
MODEL = "gemini-embedding-001"
BATCH_SIZE = 100


def _extract_vector(emb_obj) -> list[float]:
    # Most common shape in the SDK: emb_obj.values
    if hasattr(emb_obj, "values"):
        return list(emb_obj.values)

    # Fallback: dict-like shapes
    if isinstance(emb_obj, dict):
        if "values" in emb_obj:
            return list(emb_obj["values"])
        if "embedding" in emb_obj:
            return list(emb_obj["embedding"])

    raise TypeError(f"Unknown embedding object shape: {type(emb_obj)}")


def main() -> None:
    os.makedirs("data", exist_ok=True)

    if os.path.exists(OUT_PATH):
        print(f"Found existing {OUT_PATH}. Delete it if you want to rebuild.")
        return

    df = pd.read_csv(CSV_PATH)

    if "Title" not in df.columns or "Description" not in df.columns:
        raise ValueError("CSV must contain columns: Title, Description")

    titles = df["Title"].astype(str).tolist()
    descriptions = df["Description"].fillna("").astype(str).tolist()
    descriptions = [d if d.strip() else " " for d in descriptions]

    client = genai.Client()
    embeddings: dict[str, list[float]] = {}

    total = len(descriptions)

    for start in range(0, total, BATCH_SIZE):
        batch_titles = titles[start:start + BATCH_SIZE]
        batch_desc = descriptions[start:start + BATCH_SIZE]

        while True:
            try:
                result = client.models.embed_content(
                    model=MODEL,
                    contents=batch_desc,
                )
                break  # success → exit retry loop

            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    print("Quota hit. Waiting 40 seconds before retrying...")
                    time.sleep(40)
                else:
                    raise

        if len(result.embeddings) != len(batch_titles):
            raise RuntimeError("Embedding count mismatch with input batch size")

        for i, emb in enumerate(result.embeddings):
            embeddings[batch_titles[i]] = _extract_vector(emb)

        print(f"Embedded {min(start + BATCH_SIZE, total)}/{total}")

        # Respect free-tier rate limit
        time.sleep(40)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings for {len(embeddings)} shows to {OUT_PATH}")
if __name__ == "__main__":
    main()
