"""Feedback parser - analyzes feedback to determine intent and targets."""

import json
from typing import Any

from ...project import Project
from ...understanding.llm_provider import LLMProvider, ClaudeCodeLLMProvider, get_llm_provider
from .models import (
    FeedbackIntent,
    FeedbackItem,
    FeedbackScope,
    FeedbackStatus,
    FeedbackTarget,
)
from .prompts import PARSE_FEEDBACK_PROMPT, PARSE_FEEDBACK_SYSTEM_PROMPT


class FeedbackParser:
    """Parses user feedback into structured intent and targets."""

    def __init__(
        self,
        project: Project,
        llm_provider: LLMProvider | None = None,
        verbose: bool = True,
    ):
        """Initialize the feedback parser.

        Args:
            project: The project to parse feedback for.
            llm_provider: LLM provider for analysis. If None, creates default.
            verbose: Whether to print progress.
        """
        self.project = project
        self.verbose = verbose

        if llm_provider is None:
            from ...config import load_config
            self.llm = get_llm_provider(load_config())
        else:
            self.llm = llm_provider

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _get_scene_list(self) -> str:
        """Get formatted list of scenes from script.json.

        Returns:
            Formatted string listing all scenes.
        """
        script_path = self.project.root_dir / "script" / "script.json"
        if not script_path.exists():
            return "(No scenes found - script.json missing)"

        try:
            with open(script_path) as f:
                script = json.load(f)

            scenes = script.get("scenes", [])
            if not scenes:
                return "(No scenes in script)"

            lines = []
            for i, scene in enumerate(scenes):
                scene_id = scene.get("scene_id", f"scene_{i + 1}")
                title = scene.get("title", "Untitled")
                scene_type = scene.get("scene_type", "unknown")
                duration = scene.get("duration_seconds", 0)
                lines.append(f"{i + 1}. [{scene_id}] {title} ({scene_type}, {duration}s)")

            return "\n".join(lines)
        except (json.JSONDecodeError, IOError):
            return "(Error reading script.json)"

    def parse(self, item: FeedbackItem) -> FeedbackItem:
        """Parse feedback text into structured intent and targets.

        Args:
            item: The feedback item to parse.

        Returns:
            Updated FeedbackItem with intent, target, and interpretation.
        """
        self._log(f"\nParsing feedback: {item.feedback_text[:50]}...")

        item.status = FeedbackStatus.ANALYZING

        # Build prompt
        scene_list = self._get_scene_list()
        prompt = PARSE_FEEDBACK_PROMPT.format(
            project_id=self.project.id,
            scene_list=scene_list,
            feedback_text=item.feedback_text,
        )

        try:
            # Call LLM to analyze feedback
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=PARSE_FEEDBACK_SYSTEM_PROMPT,
            )

            if result is None:
                item.status = FeedbackStatus.FAILED
                item.error_message = "LLM returned no response"
                return item

            # Extract intent
            intent_str = result.get("intent", "visual_impl")
            try:
                item.intent = FeedbackIntent(intent_str)
            except ValueError:
                item.intent = FeedbackIntent.VISUAL_IMPLEMENTATION

            # Extract sub-intents if mixed
            if item.intent == FeedbackIntent.MIXED:
                sub_intents = result.get("sub_intents", [])
                item.sub_intents = [
                    FeedbackIntent(s) for s in sub_intents
                    if s in [e.value for e in FeedbackIntent]
                ]

            # Extract target
            affected_scenes = result.get("affected_scene_ids", [])
            scope_str = result.get("scope", "scene")
            try:
                scope = FeedbackScope(scope_str)
            except ValueError:
                scope = FeedbackScope.SCENE

            item.target = FeedbackTarget(
                scene_ids=affected_scenes,
                scope=scope,
            )

            # Extract interpretation
            item.interpretation = result.get("interpretation", "")

            self._log(f"  Intent: {item.intent.value}")
            self._log(f"  Scope: {scope.value}")
            self._log(f"  Scenes: {affected_scenes}")
            self._log(f"  Interpretation: {item.interpretation[:80]}...")

            return item

        except Exception as e:
            item.status = FeedbackStatus.FAILED
            item.error_message = f"Parse error: {str(e)}"
            self._log(f"  Error: {item.error_message}")
            return item
