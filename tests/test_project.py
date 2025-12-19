import pytest
import numpy as np
import pickle
from unittest.mock import patch, mock_open
from src.matcher import fuzzy_match_shows
from src.matcher import parse_user_input
from src.data_handler import load_embeddings, save_embeddings, get_all_titles


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