"""Patch generator - generates patches from parsed feedback.

The generator creates patches that modify project files. For visual changes,
it generates visual_cue patches that update script.json, and marks them for
scene refinement which uses ClaudeCodeVisualInspector for proper implementation.
"""

import json
from typing import Any

from ...project import Project
from ...understanding.llm_provider import LLMProvider, ClaudeCodeLLMProvider, get_llm_provider
from ..models import (
    ModifyScenePatch,
    UpdateVisualCuePatch,
    AddScenePatch,
    ScriptPatchType,
)
from .models import (
    FeedbackIntent,
    FeedbackItem,
    FeedbackStatus,
)
from .prompts import (
    GENERATE_SCRIPT_PATCH_PROMPT,
    GENERATE_VISUAL_CUE_PATCH_PROMPT,
    GENERATE_STRUCTURE_PATCH_PROMPT,
)


class PatchGenerator:
    """Generates patches from parsed feedback."""

    def __init__(
        self,
        project: Project,
        llm_provider: LLMProvider | None = None,
        verbose: bool = True,
    ):
        """Initialize the patch generator.

        Args:
            project: The project to generate patches for.
            llm_provider: LLM provider for generation. If None, creates default.
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

    def _load_script(self) -> dict[str, Any] | None:
        """Load script.json."""
        script_path = self.project.root_dir / "script" / "script.json"
        if not script_path.exists():
            return None
        try:
            with open(script_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _get_scene_by_id(self, scene_id: str, script: dict) -> dict | None:
        """Get a scene from script by ID or by matching title.

        Handles both numeric IDs (1, 2, 3) and slug IDs (the_impossible_leap).
        """
        for scene in script.get("scenes", []):
            # Direct ID match (handles both string and int IDs)
            if str(scene.get("scene_id")) == str(scene_id):
                return scene

            # Try matching by title (convert slug to title format)
            # e.g., "beyond_linear_thinking" -> "Beyond Linear Thinking"
            title = scene.get("title", "")
            title_slug = self._slugify(title)
            if title_slug == scene_id:
                return scene

        return None

    def generate(self, item: FeedbackItem) -> FeedbackItem:
        """Generate patches based on feedback intent.

        For visual changes (VISUAL_CUE, VISUAL_IMPLEMENTATION, STYLE), we generate
        visual_cue patches that update script.json. These patches are marked with
        `trigger_scene_refinement=True` so that after applying, the scene refinement
        system (ClaudeCodeVisualInspector) is triggered to implement the changes
        with proper visual verification.

        Args:
            item: The parsed feedback item.

        Returns:
            Updated FeedbackItem with patches.
        """
        if not item.intent or not item.target:
            item.status = FeedbackStatus.FAILED
            item.error_message = "Feedback not parsed (missing intent or target)"
            return item

        self._log(f"\nGenerating patches for intent: {item.intent.value}")
        item.status = FeedbackStatus.GENERATING

        try:
            # Route to appropriate generator based on intent
            if item.intent == FeedbackIntent.SCRIPT_CONTENT:
                patches = self._generate_script_content_patches(item)
            elif item.intent == FeedbackIntent.VISUAL_CUE:
                # Visual cue updates - update spec and trigger scene refinement
                patches = self._generate_visual_cue_patches(item, trigger_refinement=True)
            elif item.intent == FeedbackIntent.VISUAL_IMPLEMENTATION:
                # Visual implementation changes are now handled via visual_cue + scene refinement
                # This ensures proper use of /remotion skill, visual verification, and 13 principles
                self._log("    -> Routing visual_impl to visual_cue + scene refinement")
                patches = self._generate_visual_cue_patches(item, trigger_refinement=True)
            elif item.intent == FeedbackIntent.SCRIPT_STRUCTURE:
                patches = self._generate_structure_patches(item)
            elif item.intent == FeedbackIntent.TIMING:
                patches = self._generate_timing_patches(item)
            elif item.intent == FeedbackIntent.STYLE:
                # Style changes are handled via visual_cue + scene refinement
                self._log("    -> Routing style to visual_cue + scene refinement")
                patches = self._generate_visual_cue_patches(item, trigger_refinement=True)
            elif item.intent == FeedbackIntent.MIXED:
                patches = self._generate_mixed_patches(item)
            else:
                patches = []

            item.patches = [p if isinstance(p, dict) else p.to_dict() for p in patches]
            self._log(f"  Generated {len(item.patches)} patches")

            return item

        except Exception as e:
            item.status = FeedbackStatus.FAILED
            item.error_message = f"Generation error: {str(e)}"
            self._log(f"  Error: {item.error_message}")
            return item

    def _generate_script_content_patches(self, item: FeedbackItem) -> list:
        """Generate patches for narration/script content changes."""
        patches = []
        script = self._load_script()
        if not script:
            return patches

        for scene_id in item.target.scene_ids:
            scene = self._get_scene_by_id(scene_id, script)
            if not scene:
                continue

            prompt = GENERATE_SCRIPT_PATCH_PROMPT.format(
                scene_id=scene_id,
                scene_title=scene.get("title", ""),
                current_narration=scene.get("voiceover", ""),
                feedback_text=item.feedback_text,
                interpretation=item.interpretation,
            )

            result = self.llm.generate_json(prompt=prompt)
            if not result:
                continue

            for change in result.get("changes", []):
                if change.get("field") == "voiceover" and change.get("new_text"):
                    patch = ModifyScenePatch(
                        reason=change.get("reason", "User feedback"),
                        scene_id=scene_id,
                        field_name="voiceover",
                        old_value=scene.get("voiceover", ""),
                        new_value=change["new_text"],
                    )
                    patches.append(patch)
                    self._log(f"    -> Modify voiceover for {scene_id}")

        return patches

    def _generate_visual_cue_patches(
        self, item: FeedbackItem, trigger_refinement: bool = False
    ) -> list:
        """Generate patches for visual_cue specification changes.

        Args:
            item: The feedback item with target scenes and feedback text.
            trigger_refinement: If True, mark patches to trigger scene refinement
                after application. This runs ClaudeCodeVisualInspector which uses
                the /remotion skill and visual verification to implement changes.

        Returns:
            List of UpdateVisualCuePatch objects (or dicts with trigger_scene_refinement).
        """
        patches = []
        script = self._load_script()
        if not script:
            return patches

        for scene_id in item.target.scene_ids:
            scene = self._get_scene_by_id(scene_id, script)
            if not scene:
                self._log(f"    -> Scene not found: {scene_id}")
                continue

            current_cue = scene.get("visual_cue", {})
            prompt = GENERATE_VISUAL_CUE_PATCH_PROMPT.format(
                scene_id=scene_id,
                scene_title=scene.get("title", ""),
                scene_type=scene.get("scene_type", ""),
                narration=scene.get("voiceover", "")[:500],
                current_visual_cue=json.dumps(current_cue, indent=2),
                feedback_text=item.feedback_text,
                interpretation=item.interpretation,
                duration=scene.get("duration_seconds", 25),
            )

            result = self.llm.generate_json(prompt=prompt)
            if not result or not result.get("needs_update"):
                continue

            new_cue = result.get("new_visual_cue", {})
            if new_cue:
                # Create the patch as a dict so we can add the trigger flag
                patch_dict = {
                    "patch_type": "update_visual_cue",
                    "reason": result.get("reason", "User feedback"),
                    "scene_id": scene_id,
                    "scene_title": scene.get("title", ""),
                    "current_visual_cue": current_cue,
                    "new_visual_cue": new_cue,
                    "trigger_scene_refinement": trigger_refinement,
                }
                patches.append(patch_dict)
                if trigger_refinement:
                    self._log(f"    -> Update visual_cue for {scene_id} (+ scene refinement)")
                else:
                    self._log(f"    -> Update visual_cue for {scene_id}")

        return patches

    def _generate_structure_patches(self, item: FeedbackItem) -> list:
        """Generate patches for scene structure changes (add/remove/reorder)."""
        patches = []
        script = self._load_script()
        if not script:
            return patches

        # Format scene list for prompt
        scene_list = "\n".join(
            f"- {s.get('scene_id')}: {s.get('title')} ({s.get('scene_type')})"
            for s in script.get("scenes", [])
        )

        prompt = GENERATE_STRUCTURE_PATCH_PROMPT.format(
            scene_list=scene_list,
            feedback_text=item.feedback_text,
            interpretation=item.interpretation,
        )

        result = self.llm.generate_json(prompt=prompt)
        if not result:
            return patches

        action = result.get("action")
        details = result.get("details", {})

        if action == "add" and details.get("new_scene"):
            new_scene = details["new_scene"]
            patch = AddScenePatch(
                reason=result.get("reason", "User feedback"),
                insert_after_scene_id=details.get("insert_after"),
                new_scene_id=self._slugify(new_scene.get("title", "new_scene")),
                title=new_scene.get("title", "New Scene"),
                narration=new_scene.get("narration", ""),
                visual_description=new_scene.get("visual_description", ""),
                duration_seconds=new_scene.get("duration_seconds", 25),
                concepts_addressed=[],
            )
            patches.append(patch)
            self._log(f"    -> Add scene: {patch.title}")

        elif action == "remove":
            # Store as dict - removal patches need special handling
            patch = {
                "patch_type": "remove_scene",
                "scene_id": details.get("scene_id"),
                "reason": result.get("reason", "User feedback"),
            }
            patches.append(patch)
            self._log(f"    -> Remove scene: {details.get('scene_id')}")

        elif action == "reorder":
            patch = {
                "patch_type": "reorder_scenes",
                "new_order": details.get("new_order", []),
                "reason": result.get("reason", "User feedback"),
            }
            patches.append(patch)
            self._log(f"    -> Reorder scenes")

        return patches

    def _generate_timing_patches(self, item: FeedbackItem) -> list:
        """Generate patches for timing/duration changes."""
        patches = []
        script = self._load_script()
        if not script:
            return patches

        # For timing changes, we modify duration_seconds in script.json
        for scene_id in item.target.scene_ids:
            scene = self._get_scene_by_id(scene_id, script)
            if not scene:
                continue

            # Use LLM to determine new timing
            # For now, create a simple modify patch
            patch = {
                "patch_type": "modify_timing",
                "scene_id": scene_id,
                "current_duration": scene.get("duration_seconds"),
                "feedback": item.feedback_text,
                "interpretation": item.interpretation,
                "reason": "User feedback",
            }
            patches.append(patch)
            self._log(f"    -> Timing change for {scene_id}")

        return patches

    def _generate_mixed_patches(self, item: FeedbackItem) -> list:
        """Generate patches for mixed-intent feedback."""
        patches = []

        # Process each sub-intent
        for sub_intent in item.sub_intents:
            # Create a temporary item with the sub-intent
            sub_item = FeedbackItem(
                id=item.id,
                timestamp=item.timestamp,
                feedback_text=item.feedback_text,
                intent=sub_intent,
                target=item.target,
                interpretation=item.interpretation,
            )

            # Route to appropriate generator
            if sub_intent == FeedbackIntent.SCRIPT_CONTENT:
                patches.extend(self._generate_script_content_patches(sub_item))
            elif sub_intent == FeedbackIntent.VISUAL_CUE:
                patches.extend(self._generate_visual_cue_patches(sub_item, trigger_refinement=True))
            elif sub_intent == FeedbackIntent.VISUAL_IMPLEMENTATION:
                # Route to visual_cue + scene refinement
                patches.extend(self._generate_visual_cue_patches(sub_item, trigger_refinement=True))
            elif sub_intent == FeedbackIntent.STYLE:
                # Route to visual_cue + scene refinement
                patches.extend(self._generate_visual_cue_patches(sub_item, trigger_refinement=True))
            elif sub_intent == FeedbackIntent.SCRIPT_STRUCTURE:
                patches.extend(self._generate_structure_patches(sub_item))
            elif sub_intent == FeedbackIntent.TIMING:
                patches.extend(self._generate_timing_patches(sub_item))

        return patches

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        import re
        slug = text.lower()
        slug = re.sub(r'[\s\-]+', '_', slug)
        slug = re.sub(r'[^a-z0-9_]', '', slug)
        slug = re.sub(r'_+', '_', slug)
        return slug.strip('_')
