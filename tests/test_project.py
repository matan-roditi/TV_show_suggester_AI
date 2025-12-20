import pytest
import numpy as np
import pickle
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
from src.matcher import fuzzy_match_shows
from src.matcher import parse_user_input
from src.data_handler import load_embeddings, save_embeddings, get_all_titles
from src.recommender import calculate_average_vector, similarity_to_percentage, cosine_similarity, get_recommendations
import os
from src.vector_db import VectorDB
from src.creative import generate_new_show_idea


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
    """Verify identical vectors result in 100% similarity."""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sim = cosine_similarity(v, v)
    percentage = similarity_to_percentage(sim)
    assert percentage == 100


def test_similarity_percentage_opposite_vectors():
    """Verify opposite vectors result in 0% similarity."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    sim = cosine_similarity(v1, v2)
    percentage = similarity_to_percentage(sim)
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
        # This vector points in a different direction (mostly X, little Y)
        # So it will have a large distance from [10, 10]
        "Mediocre Show": np.array([10, 1]), 
        
        # This vector points in the same direction as user, so distance is small
        "Great Show": np.array([9, 9])
    }
    
    recs = get_recommendations(user_vector, db_vectors, input_shows=[])
    
    assert recs[0]['title'] == "Great Show"
    assert recs[1]['title'] == "Mediocre Show"
    
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

def test_calculate_average_vector_negatives():
    """Verify math works correctly with negative numbers (common in embeddings)."""
    # Vector 1: [1.0, -1.0]
    # Vector 2: [-1.0, 1.0]
    # Average should be [0.0, 0.0]
    v1 = np.array([1.0, -1.0])
    v2 = np.array([-1.0, 1.0])
    
    result = calculate_average_vector([v1, v2])
    expected = np.array([0.0, 0.0])
    assert np.allclose(result, expected)

def test_calculate_average_vector_mismatched_dimensions():
    """Critical: Verify that we cannot average vectors of different lengths."""
    # OpenAI vector is length 1536. If we accidentally mix in a length 100 vector,
    # the math will break or be meaningless. We expect an error.
    v1 = np.array([1.0, 2.0])
    v2 = np.array([1.0, 2.0, 3.0]) # Different length!
    
    # NumPy usually raises a ValueError here, or we should catch it.
    with pytest.raises(ValueError):
        calculate_average_vector([v1, v2])

def test_calculate_average_vector_large_variance():
    """Verify average works when vectors are very far apart."""
    v1 = np.array([1000.0, 1000.0])
    v2 = np.array([-1000.0, -1000.0])
    # Average should still be exactly 0
    result = calculate_average_vector([v1, v2])
    assert np.allclose(result, np.array([0.0, 0.0]))

def test_calculate_average_vector_many_vectors():
    """Verify performance/logic on a larger list of vectors (e.g. user liked 100 shows)."""
    # Create 100 vectors, all equal to [1, 1]
    vectors = [np.array([1.0, 1.0]) for _ in range(100)]
    
    result = calculate_average_vector(vectors)
    # Average of identical numbers is the number itself
    assert np.allclose(result, np.array([1.0, 1.0]))

def test_calculate_average_vector_invalid_types():
    """Verify that the function fails if the list contains None or non-arrays."""
    vectors = [np.array([1.0, 2.0]), None] # Invalid input
    
    with pytest.raises((TypeError, ValueError, AttributeError)):
        calculate_average_vector(vectors)

# ==========================================
# Efficient Vector Search (usearch)
# ==========================================

def test_vector_db_build_and_search():
    """Verify we can add vectors and find the nearest neighbor accurately."""
    # Create DB for 2D vectors
    db = VectorDB(dimension=2)
    
    # Add 3 items with specific integer IDs
    # 1: [1, 0], 2: [0, 1], 3: [-1, 0]
    data = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([-1.0, 0.0], dtype=np.float32)
    }
    db.build_index(data)
    
    # Query with [1, 0] -> Should match ID 1 perfectly
    query = np.array([1.0, 0.0], dtype=np.float32)
    results = db.search(query, limit=1)
    
    # Results is expected to be a list of (id, distance) tuples
    assert results[0][0] == 1
    # Distance should be very close to 0 (floating point epsilon)
    assert results[0][1] < 0.001

def test_vector_db_cosine_sorting():
    """Verify that results are sorted by cosine distance (closest first)."""
    db = VectorDB(dimension=2)
    data = {
        10: np.array([1.0, 0.1], dtype=np.float32), # Very close
        20: np.array([0.0, 1.0], dtype=np.float32), # Far (Orthogonal)
        30: np.array([0.9, 0.1], dtype=np.float32)  # Pretty close
    }
    db.build_index(data)
    
    query = np.array([1.0, 0.0], dtype=np.float32)
    results = db.search(query, limit=3)
    
    # Order should be: 10 (closest), 30 (next), 20 (farthest)
    ids = [r[0] for r in results]
    assert ids == [10, 30, 20]

def test_vector_db_save_and_load(tmp_path):
    """Verify we can save the index to disk and load it back (Offline/Online split)."""
    # tmp_path is a built-in pytest fixture for creating temp files
    index_path = str(tmp_path / "test_index.usearch")
    
    # 1. Create and Save
    db_orig = VectorDB(dimension=2)
    db_orig.build_index({5: np.array([1.0, 1.0], dtype=np.float32)})
    db_orig.save_index(index_path)
    
    assert os.path.exists(index_path)
    
    # 2. Load into new instance
    db_new = VectorDB(dimension=2)
    db_new.load_index(index_path)
    
    # 3. Search should still work
    query = np.array([1.0, 1.0], dtype=np.float32)
    results = db_new.search(query, limit=1)
    assert results[0][0] == 5

def test_vector_db_dimension_mismatch_error():
    """Verify we get an error if we try to add vectors of wrong size."""
    db = VectorDB(dimension=3) # Index expects size 3
    
    bad_data = {
        1: np.array([1.0, 2.0], dtype=np.float32) # Size 2!
    }
    
    # Should raise ValueError (or RuntimeError from usearch)
    with pytest.raises(ValueError):
        db.build_index(bad_data)

def test_vector_db_search_limit():
    """Verify that the 'limit' parameter is respected."""
    db = VectorDB(dimension=2)
    # Add 10 items
    data = {i: np.array([1.0, 1.0], dtype=np.float32) for i in range(10)}
    db.build_index(data)
    
    query = np.array([1.0, 1.0], dtype=np.float32)
    results = db.search(query, limit=4)
    
    assert len(results) == 4

# ==========================================
# Creative AI Tests (Mocking Google Gemini)
# ==========================================

@patch('src.creative._get_model')
def test_generate_idea_input_mode_success(mock_get_model):
    """Verify standard success flow for input-based generation."""
    # Setup Mock
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Show Idea: The Code Wizard."
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model

    # Action
    result = generate_new_show_idea(["Breaking Bad"], mode="input_based")

    # Assert
    assert result == "Show Idea: The Code Wizard."
    mock_model.generate_content.assert_called_once()

@patch('src.creative._get_model')
def test_generate_idea_rec_mode_success(mock_get_model):
    """Verify standard success flow for recommendation-based generation."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Show Idea: The Algorithm."
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model

    result = generate_new_show_idea(["The Wire"], mode="recommendation_based")

    assert result == "Show Idea: The Algorithm."

@patch('src.creative._get_model')
def test_prompt_includes_show_names(mock_get_model):
    """Verify that the user's specific show names actually end up in the prompt sent to Google."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "..."
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model

    shows = ["Show A", "Show B"]
    generate_new_show_idea(shows, "input_based")

    # Dig into the call arguments
    args, _ = mock_model.generate_content.call_args
    prompt_sent = args[0]

    assert "Show A" in prompt_sent
    assert "Show B" in prompt_sent

@patch('src.creative._get_model')
def test_prompt_varies_by_mode(mock_get_model):
    """Verify that 'input_based' and 'recommendation_based' send DIFFERENT prompts."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "..."
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model

    # Run twice
    generate_new_show_idea(["Show"], "input_based")
    args_input, _ = mock_model.generate_content.call_args
    prompt_input = args_input[0]

    generate_new_show_idea(["Show"], "recommendation_based")
    args_rec, _ = mock_model.generate_content.call_args
    prompt_rec = args_rec[0]

    # The prompts should not be identical
    assert prompt_input != prompt_rec
    # Specific keywords expected
    assert "loved" in prompt_input.lower() or "liked" in prompt_input.lower()
    assert "recommended" in prompt_rec.lower()

@patch('src.creative._get_model')
def test_returns_clean_text(mock_get_model):
    """Verify that extra whitespace or newlines from the LLM are trimmed."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    # LLMs often return text with leading/trailing newlines
    mock_response.text = "\n\n   A creative show description.   \n"
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model

    result = generate_new_show_idea(["Show A"], "input_based")

    assert result == "A creative show description."

def test_raises_error_on_empty_list():
    """Verify validtion: Logic should prevent wasting API calls on empty lists."""
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_new_show_idea([], "input_based")

def test_raises_error_on_invalid_mode():
    """Verify validtion: Should catch typos in the mode argument."""
    with pytest.raises(ValueError, match="Invalid mode"):
        generate_new_show_idea(["Show A"], "random_mode")

@patch('src.creative._get_model')
def test_handles_api_exception(mock_get_model):
    """Verify behavior when Google API crashes (e.g. 500 error or No Internet)."""
    mock_model = MagicMock()
    # Simulate a crash
    mock_model.generate_content.side_effect = Exception("Connection Refused")
    mock_get_model.return_value = mock_model

    with pytest.raises(Exception, match="Connection Refused"):
        generate_new_show_idea(["Show A"], "input_based")

@patch('src.creative._get_model')
def test_handles_blocked_content(mock_get_model):
    """Verify behavior when AI returns no text (e.g. safety filter triggered)."""
    mock_model = MagicMock()
    # Simulate a response object that exists but has no text (or raises error on .text access)
    mock_response = MagicMock()
    # Mocking behavior where .text might be empty or unavailable
    type(mock_response).text = PropertyMock(return_value=None) 
    # Or simplified: if .text returns empty string
    mock_response.text = "" 
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model
    
    # We might expect it to return a default message or raise an error
    # Let's assume we want a fallback string
    result = generate_new_show_idea(["Show A"], "input_based")
    assert result == "Description unavailable." or result == ""

@patch('src.creative._get_model')
def test_special_characters_handling(mock_get_model):
    """Verify that weird characters in show names don't break the string interpolation."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Ok"
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model

    weird_shows = ["Show & Name", "Show with \"Quotes\"", "Show\nNew line"]
    
    # Should not raise error
    generate_new_show_idea(weird_shows, "input_based")
    
    args, _ = mock_model.generate_content.call_args
    prompt = args[0]
    
    # Ensure they are in the prompt correctly
    assert "Show & Name" in prompt
    assert "Show with \"Quotes\"" in prompt