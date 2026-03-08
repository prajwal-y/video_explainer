"""Audio delivery tags for voiceover guidance.

This module adds delivery tags like [thoughtful], [puzzled], [excited] to narration
text to guide voice actor delivery or TTS systems.

Uses an LLM to analyze narration context and add appropriate tags.
"""

from pathlib import Path
from typing import Protocol


class LLMProvider(Protocol):
    """Protocol for LLM providers (for dependency injection)."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response from the LLM."""
        ...


# Available delivery tags
DELIVERY_TAGS = [
    # Core emotional tones
    "thoughtful",      # Reflective, contemplative
    "puzzled",         # Questions, uncertainty
    "excited",         # Impressive facts, revelations
    "serious",         # Important technical information
    "wonder",          # Awe-inspiring moments
    "matter-of-fact",  # Straightforward statements
    "dramatic",        # Building tension
    "warm",            # Human, personal moments
    # Additional emotional tones
    "curious",         # Inquisitive, exploring
    "confident",       # Assured, authoritative
    "playful",         # Light, fun tone
    "reverent",        # Respectful awe, profound moments
    "urgent",          # Time pressure, high stakes
    "satisfied",       # Resolution, conclusion
    "intrigued",       # Mystery, hook
]

SYSTEM_PROMPT = """You are an expert voice director adding delivery tags to narration scripts for an AI voiceover generator.

Your job is to add delivery tags like [thoughtful], [puzzled], [excited] to guide emotional delivery.

Available tags:
- [thoughtful] - Reflective, contemplative moments
- [puzzled] - Questions, uncertainty, confusion
- [excited] - Impressive facts, revelations, achievements
- [serious] - Important technical information, gravity
- [wonder] - Awe-inspiring moments, scale, beauty
- [matter-of-fact] - Straightforward statements, neutral
- [dramatic] - Building tension, pivots, key reveals
- [warm] - Human elements, personal moments, connection
- [curious] - Inquisitive, exploring, discovery
- [confident] - Assured, authoritative, certain
- [playful] - Light, fun, slightly humorous
- [reverent] - Respectful awe, profound moments
- [urgent] - Time pressure, high stakes, importance
- [satisfied] - Resolution, conclusion, contentment
- [intrigued] - Mystery, hook, drawing listener in

Rules:
1. Place tags at the START of sentences or phrases where the tone should shift
2. Be generous with tags - they help the AI voice generator deliver better results
3. Group consecutive sentences with the same tone under one tag
4. Match tags to content: questions often get [puzzled] or [curious], technical explanations get [serious], impressive numbers get [excited] or [wonder]
5. Use [intrigued] for hooks and mysteries, [dramatic] for reveals and pivots
6. Use [satisfied] for conclusions and resolutions
7. Preserve the exact original text - only add tags, don't change words

Output ONLY the tagged narration text, nothing else."""


def add_delivery_tags(
    narration: str,
    llm: LLMProvider | None = None,
    working_dir: Path | None = None,
) -> str:
    """Add delivery tags to a narration using LLM analysis.

    Args:
        narration: The narration text to tag
        llm: Optional LLM provider (creates default if not provided)
        working_dir: Working directory for LLM provider

    Returns:
        The narration with delivery tags added
    """
    if not narration.strip():
        return narration

    # Create default LLM provider if not provided
    if llm is None:
        from ..understanding.llm_provider import get_llm_provider
        from ..config import load_config

        llm = get_llm_provider(load_config())

    prompt = f"""Add delivery tags to this narration:

---
{narration}
---

Remember: Output ONLY the tagged narration, preserving the exact original text."""

    try:
        result = llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        # Clean up any markdown formatting that might have been added
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return result.strip()
    except Exception as e:
        # If LLM fails, return original narration without tags
        print(f"Warning: Failed to add delivery tags: {e}")
        return narration


def format_narration_for_recording(
    narration: str,
    include_tags: bool = True,
    llm: LLMProvider | None = None,
    working_dir: Path | None = None,
) -> str:
    """Format a narration for voice recording with optional delivery tags.

    Args:
        narration: The narration text
        include_tags: Whether to include delivery tags
        llm: Optional LLM provider for tagging
        working_dir: Working directory for LLM provider

    Returns:
        Formatted narration text
    """
    if include_tags:
        return add_delivery_tags(narration, llm=llm, working_dir=working_dir)
    return narration
