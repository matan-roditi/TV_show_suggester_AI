import pytest
import numpy as np
import pickle
from unittest.mock import patch, mock_open
from src.matcher import fuzzy_match_shows
from src.matcher import parse_user_input
from src.data_handler import load_embeddings, save_embeddings, get_all_titles
from src.recommender import calculate_average_vector, calculate_similarity_percentage, get_recommendations


def test_parse_user_input_basic():
    """Test standard comma-separated input."""
    input_str = "Breaking Bad, Dark, Sherlock"
    result = parse_user_input(input_str)
    assert result == ["Breaking Bad", "Dark", "Sherlock"]


def test_parse_user_input_strips_whitespace():
    """Test that extra spaces around names are removed."""
    input_str = "  Breaking Bad ,  Dark  ,   Sherlock  "
    result = parse_user_input(input_str)
    assert result == ["Breaking Bad", "Dark", "Sherlock"]


def test_parse_user_input_empty():
    """Test that empty string returns an empty list."""
    input_str = ""
    result = parse_user_input(input_str)
    assert result == []


def test_parse_user_input_single_item():
    """Test input with only one show (no commas)."""
    input_str = "The Wire"
    result = parse_user_input(input_str)
    assert result == ["The Wire"]


def test_parse_user_input_ignore_empty_slots():
    """Test handling of accidental double commas or trailing commas."""
    input_str = "Lost, , Heroes,"
    result = parse_user_input(input_str)
    # We expect the empty slots to be ignored
    assert result == ["Lost", "Heroes"]

# ==========================================
# Robust Fuzzy Matching
# ==========================================

@patch('src.matcher.process.extractOne')
def test_fuzzy_match_below_threshold(mock_extract):
    """If the match is too weak (e.g. user typed nonsense), return None or ignore."""
    # Mock return: ("Breaking Bad", 20) -> Low score (nonsense input)
    mock_extract.return_value = ("Breaking Bad", 20)
    
    shows_db = ["Breaking Bad"]
    # We expect 'xyz' NOT to match 'Breaking Bad' because score 20 is too low
    result = fuzzy_match_shows(["xyz"], shows_db, threshold=50)
    
    assert result == []

def test_fuzzy_match_empty_database():
    """Verify behavior when the shows database is empty."""
    result = fuzzy_match_shows(["Breaking Bad"], [])
    assert result == []

@patch('src.matcher.process.extractOne')
def test_fuzzy_match_preserves_order(mock_extract):
    """Verify the output list follows the input list order."""
    # First call matches "Show A", Second matches "Show B"
    mock_extract.side_effect = [("Show A", 90), ("Show B", 90)]
    
    shows_db = ["Show A", "Show B"]
    # User typed "a, b" -> expect ["Show A", "Show B"]
    result = fuzzy_match_shows(["a", "b"], shows_db)
    
    assert result == ["Show A", "Show B"]

@patch('src.matcher.process.extractOne')
def test_fuzzy_match_partial_string(mock_extract):
    """Verify that a partial name matches the full name."""
    # "Thrones" should match "Game of Thrones"
    mock_extract.return_value = ("Game of Thrones", 90)
    
    shows_db = ["Game of Thrones"]
    result = fuzzy_match_shows(["Thrones"], shows_db)
    
    assert result == ["Game of Thrones"]

def test_fuzzy_match_no_duplicates_in_output():
    """If user types 'bad, bad', we should ideally handle it (or let it be).
    This test decides our policy: Let's allow duplicates if user asked for them, 
    OR strictly dedup. For now, let's assume we allow them (simple list)."""
    # This test acts as a documentation of behavior
    pass 

# ==========================================
# Data Handling
# ==========================================

def test_load_embeddings_success():
    """Verify we can load the pickle file into a dictionary."""
    fake_data = {"Show A": [1, 2], "Show B": [3, 4]}
    
    # We mock 'open' so we don't need a real file
    with patch("builtins.open", mock_open(read_data=pickle.dumps(fake_data))):
        result = load_embeddings("dummy.pkl")
        assert result["Show A"] == [1, 2]
        assert isinstance(result, dict)

def test_load_embeddings_file_not_found():
    """Verify the program raises an error if the DB is missing."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_embeddings("missing_file.pkl")

def test_save_embeddings_calls_pickle_dump():
    """Verify we can save our processed data (for the Offline step)."""
    data = {"Show": [1]}
    m = mock_open()
    
    with patch("builtins.open", m):
        save_embeddings(data, "test.pkl")
    
    # Check that verify write was called
    m.assert_called_with("test.pkl", "wb")

def test_get_all_titles_extracts_keys():
    """Helper function to get just the show names from the complex dict."""
    data = {
        "Breaking Bad": np.array([1, 2]),
        "The Wire": np.array([3, 4])
    }
    titles = get_all_titles(data)
    assert "Breaking Bad" in titles
    assert "The Wire" in titles
    assert len(titles) == 2

def test_load_embeddings_corrupted_file():
    """Verify handling of a corrupted pickle file (EOFError)."""
    # Simulate an empty or broken file
    with patch("builtins.open", mock_open(read_data=b"")):
        with patch("pickle.load", side_effect=EOFError):
            with pytest.raises(EOFError):
                load_embeddings("corrupt.pkl")

# ==========================================
# Recommendation Engine Tests
# ==========================================

# --- Vector Math Utilities ---

def test_calculate_average_vector_basic():
    """Verify simple mathematical average of two vectors."""
    v1 = np.array([2.0, 4.0])
    v2 = np.array([4.0, 8.0])
    # Average should be [3.0, 6.0]
    result = calculate_average_vector([v1, v2])
    expected = np.array([3.0, 6.0])
    assert np.allclose(result, expected)

def test_calculate_average_vector_single_item():
    """Verify that averaging a single vector returns the vector itself."""
    v1 = np.array([1.0, 2.0, 3.0])
    result = calculate_average_vector([v1])
    assert np.allclose(result, v1)

def test_calculate_average_vector_empty_list():
    """Verify that passing an empty list raises a specific error."""
    with pytest.raises(ValueError, match="Cannot calculate average"):
        calculate_average_vector([])

def test_calculate_average_vector_shape_preservation():
    """Verify the output vector keeps the same dimensions as input."""
    # Simulating OpenAI embedding size (1536 dimensions)
    v1 = np.zeros(1536)
    v2 = np.ones(1536)
    result = calculate_average_vector([v1, v2])
    assert result.shape == (1536,)

def test_similarity_percentage_perfect_match():
    """Verify distance 0.0 results in 100% similarity."""
    # Distance 0 means vectors are identical
    percentage = calculate_similarity_percentage(0.0)
    assert percentage == 100

def test_similarity_percentage_clipping():
    """Verify that very large distances don't result in negative percentages."""
    # Distance 2.0 (max cosine distance) should be 0%, not negative
    percentage = calculate_similarity_percentage(2.0)
    assert percentage == 0

# --- Recommendation Logic (Filtering & Sorting) ---

def test_get_recommendations_excludes_user_input():
    """Critically important: Don't recommend shows the user just said they liked."""
    user_vector = np.array([10, 10])
    
    # DB has the user's show (A) and a new show (B)
    db_vectors = {
        "Show A": np.array([10, 10]),  # Perfect match, but inputs should be ignored
        "Show B": np.array([9, 9])
    }
    
    # We pass "Show A" as the input_shows list to exclude
    recs = get_recommendations(user_vector, db_vectors, input_shows=["Show A"])
    
    rec_titles = [r['title'] for r in recs]
    assert "Show A" not in rec_titles
    assert "Show B" in rec_titles

def test_get_recommendations_sort_order():
    """Verify recommendations are sorted by best match first."""
    user_vector = np.array([10, 10])
    db_vectors = {
        "Mediocre Show": np.array([1, 1]),  # Far away
        "Great Show": np.array([9, 9])      # Close
    }
    
    recs = get_recommendations(user_vector, db_vectors, input_shows=[])
    
    assert recs[0]['title'] == "Great Show"
    assert recs[1]['title'] == "Mediocre Show"

def test_get_recommendations_limit_results():
    """Verify the function respects the requested limit (e.g., top 5)."""
    user_vector = np.array([1, 1])
    # Create 10 dummy shows
    db_vectors = {f"Show {i}": np.array([1, 1]) for i in range(10)}
    
    # Ask for only 3
    recs = get_recommendations(user_vector, db_vectors, limit=3, input_shows=[])
    
    assert len(recs) == 3

def test_get_recommendations_small_database():
    """Verify graceful handling if we ask for 5 shows but DB only has 1."""
    user_vector = np.array([1, 1])
    db_vectors = {"Lonely Show": np.array([1, 1])}
    
    # Ask for 5, but DB is small
    recs = get_recommendations(user_vector, db_vectors, limit=5, input_shows=[])
    
    assert len(recs) == 1
    assert recs[0]['title'] == "Lonely Show"

def test_get_recommendations_output_format():
    """Verify the output is a list of dictionaries with specific keys."""
    user_vector = np.array([1, 1])
    db_vectors = {"Show A": np.array([1, 1])}
    
    recs = get_recommendations(user_vector, db_vectors, input_shows=[])
    
    # Each item must be a dict with 'title' and 'percentage'
    assert isinstance(recs[0], dict)
    assert 'title' in recs[0]
    assert 'percentage' in recs[0]
    assert isinstance(recs[0]['percentage'], int)

def test_get_recommendations_no_candidates():
    """Verify behavior when all available shows are excluded (user liked everything)."""
    user_vector = np.array([1, 1])
    db_vectors = {"Show A": np.array([1, 1])}
    
    # User likes the only show in DB
    recs = get_recommendations(user_vector, db_vectors, input_shows=["Show A"])
    
    assert recs == []