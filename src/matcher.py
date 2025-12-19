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
