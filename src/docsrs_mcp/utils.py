"""Shared utilities for docsrs-mcp server."""

import re


def extract_smart_snippet(
    content: str,
    target_length: int = 300,
    min_length: int = 200,
    max_length: int = 400,
) -> str:
    """
    Extract intelligent snippet with context preservation.

    Progressive fallback strategy:
    1. Try sentence boundaries (optimal)
    2. Fall back to word boundaries
    3. Fall back to character truncation (last resort)
    """
    if not content:
        return ""

    # If content is already short, return as-is
    if len(content) <= min_length:
        return content

    # If content is just slightly over, return with ellipsis
    if len(content) <= max_length:
        return content

    try:
        # Try to find sentence boundaries
        # Look for sentence endings: period, exclamation, question mark followed by space
        sentence_pattern = r"[.!?]\s+"
        sentences = re.split(sentence_pattern, content)

        # Build snippet from complete sentences
        snippet = ""
        for i, sentence in enumerate(sentences):
            # Add sentence with its ending punctuation back
            if i < len(sentences) - 1:
                # Not the last sentence, so it had punctuation
                test_snippet = snippet + sentence + ". "
            else:
                # Last sentence fragment
                test_snippet = snippet + sentence

            # Check if adding this sentence keeps us in range
            if len(test_snippet) >= min_length and len(test_snippet) <= max_length:
                return test_snippet.strip()
            elif len(test_snippet) > max_length:
                # This sentence makes it too long
                if snippet and len(snippet) >= min_length:
                    # We have enough content already
                    return snippet.strip() + "..."
                else:
                    # Need to truncate this sentence
                    break
            else:
                # Keep building
                snippet = test_snippet

        # If we have a good snippet from sentences, return it
        if snippet and len(snippet) >= min_length:
            return snippet.strip() + "..."
    except Exception:
        # Sentence detection failed, continue to word boundaries
        pass

    try:
        # Fall back to word boundaries
        words = content.split()
        snippet = ""

        for word in words:
            test_snippet = snippet + (" " if snippet else "") + word

            # Check if adding this word would exceed our target
            if len(test_snippet) >= target_length:
                # Check if we can include this word without exceeding max_length
                if len(test_snippet) + 3 <= max_length:  # +3 for "..."
                    return test_snippet + "..."
                elif snippet:
                    # Previous iteration was good, use that
                    return snippet + "..."
                else:
                    # Single word is too long, need character truncation
                    break
            snippet = test_snippet

        # If we processed all words but didn't hit target
        if snippet and len(snippet) >= min_length:
            return snippet
    except Exception:
        # Word boundary detection failed, continue to character truncation
        pass

    # Last resort: character truncation
    if len(content) > max_length:
        return content[:max_length] + "..."
    else:
        return content
