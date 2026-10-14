# TV Show Suggester AI

A command-line tool that recommends TV shows based on semantic similarity. You enter some of your favorite tv shows and the program uses Google Gemini embeddings + cosine similarity (accelerated by an approximate nearest-neighbor index) to find other shows that match your taste.

## How it works

1. **Offline step — build embeddings.** Each show's description in `data/tv_shows.csv` is embedded with Gemini's `gemini-embedding-001` model and cached to `data/embeddings.pkl`, so the embedding API is only called once per show.
2. **Fuzzy matching.** When you type show names, they're fuzzy-matched (via [`thefuzz`](https://github.com/seatgeek/thefuzz)) against the show catalog, so typos or partial names ("Thrones" → "Game of Thrones") still resolve correctly.
3. **Taste vector.** Your matched shows' embeddings are averaged into a single "taste vector."
4. **Recommendation.** A [usearch](https://github.com/unum-cloud/usearch) HNSW index shortlists the nearest shows to your taste vector, which are then rescored with exact cosine similarity and returned as percentage match scores.

```
Offline (one-time):
  data/tv_shows.csv --> build_embeddings.py --> data/embeddings.pkl
                            (Gemini API)

Online (each CLI run):
  user input --> fuzzy match --> taste vector (avg of matched embeddings)
                                        |
                                        v
                      VectorDB (usearch ANN index) --> shortlist of candidates
                                        |
                                        v
                       exact cosine similarity rescoring
                                        |
                                        v
                         top-N shows with match %
```

## Project structure

```
src/
  main.py          CLI entry point
  data_handler.py  Load/save the embeddings pickle
  matcher.py       Parse user input, fuzzy-match against the catalog
  recommender.py   Vector math, cosine similarity, recommendation logic
  vector_db.py      usearch-based nearest-neighbor index wrapper
  creative.py      Gemini-powered creative show-idea generator (standalone, not yet wired into the CLI)
scripts/
  build_embeddings.py   One-time script to generate data/embeddings.pkl from the CSV
data/
  tv_shows.csv     Show catalog (title, description, genres, actors, rating, etc.)
tests/
  test_project.py  Unit tests covering every module above
```

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://ai.google.dev/)

### 2. Clone and create a virtual environment (activate it)

```bash
git clone https://github.com/matan-roditi/TV_show_suggester_AI.git
cd TV_show_suggester_AI
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

### 5. Build the embeddings (one-time)

```bash
python -m scripts.build_embeddings
```

This reads `data/tv_shows.csv` and writes `data/embeddings.pkl`. It batches requests (100 shows at a time) and sleeps between batches to respect Gemini's free-tier rate limits, so it can take a few minutes on the full catalog. If `data/embeddings.pkl` already exists, the script does nothing — delete it first if you want to regenerate.

## Usage

```bash
python -m src.main
```

You'll be prompted for two or more shows you liked:

```
Which TV shows did you really like watching? Separate them by a
comma. Make sure to enter more than 1 show
ozark, game of thornes, mad men
Making sure, do you mean Ozark, Game of Thrones, Mad Men?(y/n) y
Great! Generating recommendations now…
Here are the tv shows that i think you would love:
Dark (87%)
Breaking Bad (86%)
Peaky Blinders (86%)
Lost in Space (86%)
The Originals (86%)
```

The first run also builds and saves a usearch index to `data/vector_index.usearch`, so later runs load it instead of rebuilding — unless `data/embeddings.pkl` has changed since, in which case it's rebuilt automatically.

## Running tests

```bash
python -m pytest tests/test_project.py -q
```

The suite covers input parsing, fuzzy matching, embedding load/save, vector math, cosine similarity, the usearch-backed `VectorDB`, and the Gemini-based creative generator (mocked, no API calls made during tests).

## Notes

- `data/embeddings.pkl` and `data/vector_index.usearch` are build artifacts (gitignored) — generate them locally via the steps above rather than committing them.
- `src/creative.py` can generate an original show idea from a list of shows (`generate_new_show_idea`), but isn't currently called from the CLI — it's a standalone, tested module you can build on.
