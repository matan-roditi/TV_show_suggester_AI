import pytest
from src.matcher import parse_user_input


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