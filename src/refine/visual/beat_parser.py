"""
AI-powered narration beat parser.

Parses narration text into visual "beats" - key phrases that should trigger
specific visual changes at specific timestamps.
"""

import json
from pathlib import Path
from typing import Optional

from ...config import LLMConfig
from ...understanding.llm_provider import (
    ClaudeCodeLLMProvider,
    LLMProvider,
    MockLLMProvider,
    get_llm_provider,
)
from ..models import Beat


BEAT_PARSER_SYSTEM_PROMPT = """You are an expert video editor analyzing narration scripts.

Your task is to identify "visual beats" in narration text - key phrases that should trigger
specific visual changes in an educational video.

A beat is:
- A distinct phrase or sentence that conveys a specific piece of information
- Something that warrants its own visual representation
- Usually 2-6 seconds long (varies with content)

Good beats are:
- Numbers and statistics ("83.3% versus 13.4%")
- Comparisons ("OpenAI's o1 against GPT-4")
- Key insights ("a six-fold explosion")
- Transitions ("September 2024 changed everything")
- Process steps ("the model first analyzes...")
- Conclusions ("this is how reasoning emerged")

Bad beats are:
- Too granular (every word)
- Too vague (entire paragraphs)
- Filler phrases that don't need visuals

For each beat, also describe what visual should ideally appear.
"""

BEAT_PARSER_PROMPT_TEMPLATE = """Parse the following narration into visual beats.

Narration (duration: {duration_seconds} seconds):
"{narration_text}"

Analyze this narration and identify 3-8 distinct visual beats. For each beat:
1. Identify the key phrase
2. Estimate its start and end time (must fit within {duration_seconds}s total)
3. Describe what visual should appear during this beat

Respond with JSON in this exact format:
{{
    "beats": [
        {{
            "index": 0,
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "text": "The key phrase from narration",
            "expected_visual": "Description of what visual should appear"
        }},
        ...
    ]
}}

Important:
- Beats should cover the entire duration without gaps
- Each beat's end_seconds should equal the next beat's start_seconds
- The last beat should end at {duration_seconds}
- Keep beat text concise (the actual phrase, not full sentences)
"""


class BeatParser:
    """Parses narration text into visual beats using AI."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        working_dir: Optional[Path] = None,
    ):
        """
        Initialize the beat parser.

        Args:
            llm_provider: LLM provider to use. If None, creates ClaudeCodeLLMProvider.
            working_dir: Working directory for file operations.
        """
        if llm_provider is None:
            from ...config import load_config
            self.llm = get_llm_provider(load_config())
        else:
            self.llm = llm_provider

    def parse(self, narration_text: str, duration_seconds: float) -> list[Beat]:
        """
        Parse narration text into visual beats.

        Args:
            narration_text: The narration text to parse.
            duration_seconds: Total duration of the narration in seconds.

        Returns:
            List of Beat objects representing visual beats.
        """
        prompt = BEAT_PARSER_PROMPT_TEMPLATE.format(
            narration_text=narration_text,
            duration_seconds=duration_seconds,
        )

        try:
            response = self.llm.generate_json(prompt, BEAT_PARSER_SYSTEM_PROMPT)
            return self._parse_response(response, duration_seconds)
        except Exception as e:
            # If LLM fails, fall back to simple heuristic parsing
            print(f"Warning: AI beat parsing failed ({e}), using fallback")
            return self._fallback_parse(narration_text, duration_seconds)

    def _parse_response(self, response: dict, duration_seconds: float) -> list[Beat]:
        """
        Parse the LLM response into Beat objects.

        Args:
            response: Parsed JSON response from LLM.
            duration_seconds: Total duration for validation.

        Returns:
            List of Beat objects.
        """
        beats = []
        raw_beats = response.get("beats", [])

        for i, beat_data in enumerate(raw_beats):
            beat = Beat(
                index=beat_data.get("index", i),
                start_seconds=float(beat_data.get("start_seconds", 0)),
                end_seconds=float(beat_data.get("end_seconds", duration_seconds)),
                text=beat_data.get("text", ""),
                expected_visual=beat_data.get("expected_visual", ""),
            )
            beats.append(beat)

        # Validate and fix beat timing issues
        beats = self._validate_and_fix_beats(beats, duration_seconds)

        return beats

    def _validate_and_fix_beats(
        self, beats: list[Beat], duration_seconds: float
    ) -> list[Beat]:
        """
        Validate and fix beat timing issues.

        Ensures:
        - Beats are in order by start time
        - No gaps between beats
        - Last beat ends at duration_seconds
        - Beat indices are sequential

        Args:
            beats: List of beats to validate.
            duration_seconds: Total duration.

        Returns:
            Fixed list of beats.
        """
        if not beats:
            return []

        # Sort by start time
        beats = sorted(beats, key=lambda b: b.start_seconds)

        # Fix indices
        for i, beat in enumerate(beats):
            beat.index = i

        # Ensure no gaps and proper ending
        for i in range(len(beats) - 1):
            # Make each beat's end equal to next beat's start
            beats[i].end_seconds = beats[i + 1].start_seconds

        # Ensure last beat ends at duration
        beats[-1].end_seconds = duration_seconds

        # Ensure first beat starts at 0
        if beats[0].start_seconds > 0:
            beats[0].start_seconds = 0

        return beats

    def _fallback_parse(
        self, narration_text: str, duration_seconds: float
    ) -> list[Beat]:
        """
        Simple fallback parsing when AI is unavailable.

        Splits narration into sentences and distributes time evenly.

        Args:
            narration_text: The narration text.
            duration_seconds: Total duration.

        Returns:
            List of Beat objects.
        """
        # Split by sentence-ending punctuation
        import re

        sentences = re.split(r"(?<=[.!?])\s+", narration_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            # If no sentences found, treat whole text as one beat
            return [
                Beat(
                    index=0,
                    start_seconds=0,
                    end_seconds=duration_seconds,
                    text=narration_text[:50] + "..." if len(narration_text) > 50 else narration_text,
                    expected_visual="Main content",
                )
            ]

        # Distribute time based on sentence length (word count)
        word_counts = [len(s.split()) for s in sentences]
        total_words = sum(word_counts)

        if total_words == 0:
            # Equal distribution
            time_per_beat = duration_seconds / len(sentences)
            word_counts = [1] * len(sentences)
            total_words = len(sentences)

        beats = []
        current_time = 0.0

        for i, sentence in enumerate(sentences):
            # Time proportional to word count
            proportion = word_counts[i] / total_words
            beat_duration = proportion * duration_seconds
            end_time = min(current_time + beat_duration, duration_seconds)

            # Extract first few words as the beat text
            words = sentence.split()
            beat_text = " ".join(words[:8])
            if len(words) > 8:
                beat_text += "..."

            beats.append(
                Beat(
                    index=i,
                    start_seconds=current_time,
                    end_seconds=end_time,
                    text=beat_text,
                    expected_visual=f"Visual for: {beat_text[:30]}...",
                )
            )

            current_time = end_time

        # Ensure last beat ends at duration
        if beats:
            beats[-1].end_seconds = duration_seconds

        return beats


def parse_narration_to_beats(
    narration_text: str,
    duration_seconds: float,
    llm_provider: Optional[LLMProvider] = None,
) -> list[Beat]:
    """
    Convenience function to parse narration into beats.

    Args:
        narration_text: The narration text to parse.
        duration_seconds: Total duration in seconds.
        llm_provider: Optional LLM provider to use.

    Returns:
        List of Beat objects.
    """
    parser = BeatParser(llm_provider=llm_provider)
    return parser.parse(narration_text, duration_seconds)


class MockBeatParser(BeatParser):
    """Mock beat parser for testing."""

    def __init__(self):
        """Initialize with mock LLM provider."""
        config = LLMConfig(provider="mock")
        super().__init__(llm_provider=MockLLMProvider(config))

    def parse(self, narration_text: str, duration_seconds: float) -> list[Beat]:
        """Return predictable mock beats for testing."""
        # Create 4 evenly spaced beats
        num_beats = 4
        beat_duration = duration_seconds / num_beats

        beats = []
        for i in range(num_beats):
            beats.append(
                Beat(
                    index=i,
                    start_seconds=i * beat_duration,
                    end_seconds=(i + 1) * beat_duration,
                    text=f"Mock beat {i + 1}",
                    expected_visual=f"Mock visual for beat {i + 1}",
                )
            )

        return beats
