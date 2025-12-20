from thefuzz import process


def parse_user_input(user_input: str) -> list[str]:
    """
    Parse a comma-separated string of TV show names and return a clean list.

    Args:
        user_input: A string with TV show names separated by commas

    Returns:
        A list of cleaned show names with whitespace stripped and empty slots removed
    """
    if not user_input:
        return []

    # Split by comma and strip whitespace from each item
    shows = [show.strip() for show in user_input.split(",")]

    # Filter out empty strings
    return [show for show in shows if show]


def fuzzy_match_shows(user_shows: list[str], shows_db: list[str], threshold: int = 70) -> list[str]:
    """
    Fuzzy match user input show names against a database of shows.
    
    Args:
        user_shows: List of show names provided by the user
        shows_db: List of show names from the database
        threshold: Minimum match score to accept a match (0-100). Defaults to 70.
    
    Returns:
        A list of matched show names from the database in the same order as user input
    """
    if not user_shows or not shows_db:
        return []

    matched_shows: list[str] = []

    for user_show in user_shows:
        match = process.extractOne(user_show, shows_db)
        if match and match[1] >= threshold:
            matched_shows.append(match[0])

    return matched_shows
