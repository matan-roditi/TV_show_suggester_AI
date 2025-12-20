import google.genai as genai
import os

# Get API key from environment
api_key = os.environ.get("GEMINI_API_KEY")

# Lazy initialization - model will be created only when needed
_model = None


def _get_model():
    """Get or initialize the Gemini model."""
    global _model
    if _model is None:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
        _model = client.models.get("gemini-1.5-flash")
    return _model


def generate_new_show_idea(source_shows: list[str], mode: str) -> str:
    """
    Generate a creative TV show idea based on source shows using Google Gemini API.
    
    Args:
        source_shows: List of TV show names to base the idea on
        mode: Either "input_based" (user liked these) or "recommendation_based" (system recommended)
    
    Returns:
        A string containing the generated show idea
    
    Raises:
        ValueError: If source_shows is empty or mode is invalid
        Exception: If the API call fails (network errors, etc.)
    """
    # Input Validation
    if not source_shows:
        raise ValueError("Source shows list cannot be empty")
    
    if mode not in ("input_based", "recommendation_based"):
        raise ValueError("Invalid mode")
    
    # Construct the prompt based on mode
    shows_list = ", ".join(source_shows)
    
    if mode == "input_based":
        prompt = (
            f"The user loved these TV shows: {shows_list}. "
            f"Based on their taste, generate a creative and unique TV show idea that would appeal to them. "
            f"Provide the show name, genre, and a brief description."
        )
    else:  # recommendation_based
        prompt = (
            f"These TV shows were recommended to the user: {shows_list}. "
            f"Based on these recommended shows, generate a creative and unique TV show idea that would be another great recommendation for them. "
            f"Provide the show name, genre, and a brief description."
        )
    
    # Call the API
    try:
        model = _get_model()
        response = model.generate_content(prompt)
    except Exception:
        # Let API exceptions bubble up (network errors, etc.)
        raise
    
    # Handle safety filters or empty responses
    try:
        result_text = response.text
    except (AttributeError, ValueError):
        # response.text might not be available due to safety filters
        return "Description unavailable."
    
    # Check if response is empty or None
    if not result_text:
        return "Description unavailable."
    
    # Clean up and return
    return result_text.strip()
