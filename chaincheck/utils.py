"""
Small utility: truncate a string to max_len chars with an ellipsis.
Added to support the ChainCheck action comment formatter.
"""

def truncate(text: str, max_len: int = 70) -> str:
    """Return text[:max_len] + '…' if text is longer than max_len."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"
