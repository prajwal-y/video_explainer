"""Patch applicator - applies patches and verifies changes.

After applying visual_cue patches, this module can trigger scene refinement
using ClaudeCodeVisualInspector which:
1. Loads the /remotion skill for best practices
2. Reads the updated visual_cue from script.json
3. Takes screenshots at key frames
4. Evaluates against 13 guiding principles (including visual_spec_match)
5. Makes targeted fixes to implement the visual specification
6. Verifies improvements
"""

import json
from pathlib import Path
from typing import Any

from ...project import Project
from ...understanding.llm_provider import LLMProvider, ClaudeCodeLLMProvider, get_llm_provider
from ..models import (
    ModifyScenePatch,
    UpdateVisualCuePatch,
    AddScenePatch,
    ScriptPatchType,
)
from ..visual import ClaudeCodeVisualInspector
from ..validation import ProjectValidator
from .models import (
    FeedbackItem,
    FeedbackStatus,
)


class PatchApplicator:
    """Applies patches to project files and verifies changes."""

    def __init__(
        self,
        project: Project,
        llm_provider: LLMProvider | None = None,
        verbose: bool = True,
        live_output: bool = False,
    ):
        """Initialize the patch applicator.

        Args:
            project: The project to apply patches to.
            llm_provider: LLM provider for code changes. If None, creates default.
            verbose: Whether to print progress.
            live_output: Whether to stream Claude Code output in real-time.
        """
        self.project = project
        self.verbose = verbose
        self.live_output = live_output

        if llm_provider is None:
            from ...config import load_config
            self.llm = get_llm_provider(load_config())
        else:
            self.llm = llm_provider

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        import re
        slug = text.lower()
        slug = re.sub(r'[\s\-]+', '_', slug)
        slug = re.sub(r'[^a-z0-9_]', '', slug)
        slug = re.sub(r'_+', '_', slug)
        return slug.strip('_')

    def _match_scene_id(self, scene: dict, target_id: str) -> bool:
        """Check if a scene matches the target ID.

        Handles both numeric IDs (1, 2, 3) and slug IDs (the_impossible_leap).
        """
        scene_id = scene.get("scene_id")
        # Direct match
        if str(scene_id) == str(target_id):
            return True
        # Match by title slug
        title = scene.get("title", "")
        if self._slugify(title) == target_id:
            return True
        return False

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

    def _save_script(self, script: dict[str, Any]) -> bool:
        """Save script.json."""
        script_path = self.project.root_dir / "script" / "script.json"
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(script, f, indent=2, ensure_ascii=False)
            return True
        except IOError:
            return False

    def _load_narrations(self) -> dict[str, Any] | None:
        """Load narrations.json."""
        narrations_path = self.project.root_dir / "narration" / "narrations.json"
        if not narrations_path.exists():
            return None
        try:
            with open(narrations_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_narrations(self, narrations: dict[str, Any]) -> bool:
        """Save narrations.json."""
        narrations_path = self.project.root_dir / "narration" / "narrations.json"
        try:
            with open(narrations_path, "w", encoding="utf-8") as f:
                json.dump(narrations, f, indent=2, ensure_ascii=False)
            return True
        except IOError:
            return False

    def apply(self, item: FeedbackItem, verify: bool = True) -> FeedbackItem:
        """Apply patches from a feedback item.

        For visual_cue patches with trigger_scene_refinement=True, this will:
        1. First update script.json with the new visual_cue
        2. Then run ClaudeCodeVisualInspector to implement the changes with
           proper visual verification using the /remotion skill

        Args:
            item: The feedback item with patches to apply.
            verify: Whether to verify changes after applying.

        Returns:
            Updated FeedbackItem with application results.
        """
        if not item.patches:
            item.status = FeedbackStatus.FAILED
            item.error_message = "No patches to apply"
            return item

        self._log(f"\nApplying {len(item.patches)} patches...")
        item.status = FeedbackStatus.APPLYING
        files_modified = []
        scenes_to_refine = []  # Track scenes that need visual refinement

        try:
            for patch_data in item.patches:
                # Convert dict to patch object if needed
                if isinstance(patch_data, dict):
                    patch_type = patch_data.get("patch_type")
                else:
                    patch_type = patch_data.patch_type.value if hasattr(patch_data, 'patch_type') else None

                # Route to appropriate apply method
                if patch_type == "modify_scene" or patch_type == ScriptPatchType.MODIFY_SCENE.value:
                    modified = self._apply_modify_scene_patch(patch_data)
                elif patch_type == "update_visual_cue" or patch_type == ScriptPatchType.UPDATE_VISUAL_CUE.value:
                    modified, scene_id = self._apply_visual_cue_patch(patch_data)
                    # Check if this patch triggers scene refinement
                    if isinstance(patch_data, dict) and patch_data.get("trigger_scene_refinement"):
                        if scene_id:
                            scenes_to_refine.append(scene_id)
                elif patch_type == "add_scene" or patch_type == ScriptPatchType.ADD_SCENE.value:
                    modified = self._apply_add_scene_patch(patch_data)
                elif patch_type == "remove_scene":
                    modified = self._apply_remove_scene_patch(patch_data)
                elif patch_type == "reorder_scenes":
                    modified = self._apply_reorder_patch(patch_data)
                elif patch_type == "modify_timing":
                    modified = self._apply_timing_patch(patch_data)
                else:
                    self._log(f"  Unknown patch type: {patch_type}")
                    modified = []

                files_modified.extend(modified)

            item.files_modified = list(set(files_modified))  # Deduplicate
            self._log(f"  Modified files: {item.files_modified}")

            # Run scene refinement for scenes that need it
            if scenes_to_refine:
                self._log(f"\n  Running scene refinement for {len(scenes_to_refine)} scene(s)...")
                refinement_results = self._run_scene_refinement(scenes_to_refine)

                # Add any modified files from refinement
                for result in refinement_results:
                    if result.get("scene_file"):
                        item.files_modified.append(str(result["scene_file"]))
                    if result.get("verification_passed"):
                        self._log(f"    ✅ {result.get('scene_title', 'Scene')}: refinement passed")
                    else:
                        self._log(f"    ⚠️ {result.get('scene_title', 'Scene')}: refinement incomplete")

            # Verify if requested
            if verify and files_modified:
                item.status = FeedbackStatus.VERIFYING
                item.verification_passed = self._verify_changes(item)
                if item.verification_passed:
                    self._log("  Verification passed")
                else:
                    self._log("  Verification failed")

            item.status = FeedbackStatus.APPLIED
            return item

        except Exception as e:
            item.status = FeedbackStatus.FAILED
            item.error_message = f"Application error: {str(e)}"
            self._log(f"  Error: {item.error_message}")
            return item

    def _apply_modify_scene_patch(self, patch_data: dict | ModifyScenePatch) -> list[str]:
        """Apply a modify scene patch (narration/title changes)."""
        modified = []

        if isinstance(patch_data, dict):
            scene_id = patch_data.get("scene_id")
            field_name = patch_data.get("field_name")
            new_value = patch_data.get("new_value")
        else:
            scene_id = patch_data.scene_id
            field_name = patch_data.field_name
            new_value = patch_data.new_value

        if not scene_id or not field_name or not new_value:
            return modified

        # Update script.json
        script = self._load_script()
        if script:
            for scene in script.get("scenes", []):
                if self._match_scene_id(scene, scene_id):
                    scene[field_name] = new_value
                    break
            if self._save_script(script):
                modified.append("script/script.json")
                self._log(f"    -> Updated {field_name} in script.json for {scene_id}")

        # Also update narrations.json if it's voiceover
        if field_name == "voiceover":
            narrations = self._load_narrations()
            if narrations:
                for scene in narrations.get("scenes", []):
                    if self._match_scene_id(scene, scene_id):
                        scene["narration"] = new_value
                        break
                if self._save_narrations(narrations):
                    modified.append("narration/narrations.json")
                    self._log(f"    -> Updated narration in narrations.json for {scene_id}")

        return modified

    def _apply_visual_cue_patch(
        self, patch_data: dict | UpdateVisualCuePatch
    ) -> tuple[list[str], str | None]:
        """Apply a visual_cue update patch.

        Returns:
            Tuple of (list of modified files, scene_id that was modified)
        """
        modified = []
        applied_scene_id = None

        if isinstance(patch_data, dict):
            scene_id = patch_data.get("scene_id")
            new_visual_cue = patch_data.get("new_visual_cue")
        else:
            scene_id = patch_data.scene_id
            new_visual_cue = patch_data.new_visual_cue

        if not scene_id or not new_visual_cue:
            return modified, None

        script = self._load_script()
        if not script:
            return modified, None

        for scene in script.get("scenes", []):
            if self._match_scene_id(scene, scene_id):
                scene["visual_cue"] = new_visual_cue
                applied_scene_id = scene_id
                break

        if self._save_script(script):
            modified.append("script/script.json")
            self._log(f"    -> Updated visual_cue in script.json for {scene_id}")

        return modified, applied_scene_id

    def _run_scene_refinement(self, scene_ids: list[str]) -> list[dict]:
        """Run visual refinement for the specified scenes.

        This triggers ClaudeCodeVisualInspector which:
        1. Loads the /remotion skill for best practices
        2. Reads the updated visual_cue from script.json
        3. Takes screenshots at key frames
        4. Evaluates against 13 guiding principles
        5. Makes targeted fixes to implement the visual specification
        6. Verifies improvements

        Args:
            scene_ids: List of scene IDs (slug format) to refine.

        Returns:
            List of result dicts with scene_title, verification_passed, etc.
        """
        results = []

        try:
            # Create the inspector
            inspector = ClaudeCodeVisualInspector(
                project=self.project,
                verbose=self.verbose,
                live_output=self.live_output,
            )

            # Get scene indices from script.json
            validator = ProjectValidator(self.project)
            storyboard = self.project.load_storyboard()
            scenes = storyboard.get("scenes", [])

            for scene_id in scene_ids:
                # Find scene index
                scene_index = None
                for i, scene in enumerate(scenes):
                    if self._match_scene_id(scene, scene_id):
                        scene_index = i
                        break

                if scene_index is None:
                    self._log(f"    -> Scene not found in storyboard: {scene_id}")
                    results.append({
                        "scene_id": scene_id,
                        "scene_title": scene_id,
                        "verification_passed": False,
                        "error": "Scene not found in storyboard",
                    })
                    continue

                # Run refinement
                self._log(f"    -> Refining scene {scene_index + 1}: {scene_id}...")
                result = inspector.refine_scene(scene_index)

                results.append({
                    "scene_id": result.scene_id,
                    "scene_title": result.scene_title,
                    "scene_file": str(result.scene_file) if result.scene_file else None,
                    "verification_passed": result.verification_passed,
                    "issues_found": len(result.issues_found),
                    "fixes_applied": len(result.fixes_applied),
                    "error": result.error_message,
                })

        except Exception as e:
            self._log(f"    -> Scene refinement error: {e}")
            for scene_id in scene_ids:
                results.append({
                    "scene_id": scene_id,
                    "scene_title": scene_id,
                    "verification_passed": False,
                    "error": str(e),
                })

        return results

    def _apply_add_scene_patch(self, patch_data: dict | AddScenePatch) -> list[str]:
        """Apply an add scene patch."""
        modified = []

        if isinstance(patch_data, dict):
            insert_after = patch_data.get("insert_after_scene_id")
            new_scene_id = patch_data.get("new_scene_id")
            title = patch_data.get("title")
            narration = patch_data.get("narration")
            visual_desc = patch_data.get("visual_description")
            duration = patch_data.get("duration_seconds", 25)
        else:
            insert_after = patch_data.insert_after_scene_id
            new_scene_id = patch_data.new_scene_id
            title = patch_data.title
            narration = patch_data.narration
            visual_desc = patch_data.visual_description
            duration = patch_data.duration_seconds

        if not new_scene_id or not title:
            return modified

        # Update script.json
        script = self._load_script()
        if script:
            new_scene = {
                "scene_id": new_scene_id,
                "scene_type": "explanation",
                "title": title,
                "voiceover": narration or "",
                "visual_cue": {
                    "description": visual_desc or "",
                    "visual_type": "animation",
                    "elements": [],
                    "duration_seconds": duration,
                },
                "duration_seconds": duration,
            }

            scenes = script.get("scenes", [])
            if insert_after:
                # Find insert position using flexible matching
                insert_idx = len(scenes)  # Default to end if not found
                for i, s in enumerate(scenes):
                    if self._match_scene_id(s, insert_after):
                        insert_idx = i + 1
                        break
                scenes.insert(insert_idx, new_scene)
            else:
                scenes.insert(0, new_scene)

            script["scenes"] = scenes
            # Update total duration
            script["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in scenes
            )

            if self._save_script(script):
                modified.append("script/script.json")
                self._log(f"    -> Added scene {new_scene_id} to script.json")

        # Update narrations.json
        narrations = self._load_narrations()
        if narrations:
            new_narration = {
                "scene_id": new_scene_id,
                "title": title,
                "duration_seconds": duration,
                "narration": narration or "",
            }

            scenes = narrations.get("scenes", [])
            if insert_after:
                # Find insert position using flexible matching
                insert_idx = len(scenes)  # Default to end if not found
                for i, s in enumerate(scenes):
                    if self._match_scene_id(s, insert_after):
                        insert_idx = i + 1
                        break
                scenes.insert(insert_idx, new_narration)
            else:
                scenes.insert(0, new_narration)

            narrations["scenes"] = scenes
            narrations["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in scenes
            )

            if self._save_narrations(narrations):
                modified.append("narration/narrations.json")
                self._log(f"    -> Added scene {new_scene_id} to narrations.json")

        return modified

    def _apply_remove_scene_patch(self, patch_data: dict) -> list[str]:
        """Apply a remove scene patch."""
        modified = []
        scene_id = patch_data.get("scene_id")

        if not scene_id:
            return modified

        # Update script.json
        script = self._load_script()
        if script:
            # Use _match_scene_id for flexible matching
            script["scenes"] = [
                s for s in script.get("scenes", [])
                if not self._match_scene_id(s, scene_id)
            ]
            script["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in script["scenes"]
            )

            if self._save_script(script):
                modified.append("script/script.json")
                self._log(f"    -> Removed scene {scene_id} from script.json")

        # Update narrations.json
        narrations = self._load_narrations()
        if narrations:
            # Use _match_scene_id for flexible matching
            narrations["scenes"] = [
                s for s in narrations.get("scenes", [])
                if not self._match_scene_id(s, scene_id)
            ]
            narrations["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in narrations["scenes"]
            )

            if self._save_narrations(narrations):
                modified.append("narration/narrations.json")
                self._log(f"    -> Removed scene {scene_id} from narrations.json")

        return modified

    def _apply_reorder_patch(self, patch_data: dict) -> list[str]:
        """Apply a reorder scenes patch."""
        modified = []
        new_order = patch_data.get("new_order", [])

        if not new_order:
            return modified

        def reorder_scenes(scenes: list, order: list) -> list:
            """Reorder scenes using flexible ID matching."""
            reordered = []
            used_indices = set()

            # First, add scenes in the requested order
            for target_id in order:
                for i, scene in enumerate(scenes):
                    if i not in used_indices and self._match_scene_id(scene, target_id):
                        reordered.append(scene)
                        used_indices.add(i)
                        break

            # Add any remaining scenes not in the new order at the end
            for i, scene in enumerate(scenes):
                if i not in used_indices:
                    reordered.append(scene)

            return reordered

        # Update script.json
        script = self._load_script()
        if script:
            script["scenes"] = reorder_scenes(script.get("scenes", []), new_order)

            if self._save_script(script):
                modified.append("script/script.json")
                self._log(f"    -> Reordered scenes in script.json")

        # Update narrations.json
        narrations = self._load_narrations()
        if narrations:
            narrations["scenes"] = reorder_scenes(narrations.get("scenes", []), new_order)

            if self._save_narrations(narrations):
                modified.append("narration/narrations.json")
                self._log(f"    -> Reordered scenes in narrations.json")

        return modified

    def _apply_timing_patch(self, patch_data: dict) -> list[str]:
        """Apply a timing change patch."""
        modified = []
        scene_id = patch_data.get("scene_id")
        new_duration = patch_data.get("new_duration")

        if not scene_id or new_duration is None:
            self._log(f"    -> Timing patch missing required fields: {patch_data}")
            return modified

        # Update script.json
        script = self._load_script()
        if script:
            for scene in script.get("scenes", []):
                if self._match_scene_id(scene, scene_id):
                    scene["duration_seconds"] = new_duration
                    if "visual_cue" in scene and isinstance(scene["visual_cue"], dict):
                        scene["visual_cue"]["duration_seconds"] = new_duration
                    break

            # Recalculate total duration
            script["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in script.get("scenes", [])
            )

            if self._save_script(script):
                modified.append("script/script.json")
                self._log(f"    -> Updated duration for {scene_id} to {new_duration}s")

        # Update narrations.json
        narrations = self._load_narrations()
        if narrations:
            for scene in narrations.get("scenes", []):
                if self._match_scene_id(scene, scene_id):
                    scene["duration_seconds"] = new_duration
                    break

            narrations["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in narrations.get("scenes", [])
            )

            if self._save_narrations(narrations):
                modified.append("narration/narrations.json")

        return modified

    def _verify_changes(self, item: FeedbackItem) -> bool:
        """Verify that changes were applied correctly.

        For visual changes, this should take screenshots and verify.
        For now, we do basic file validation.
        """
        # Basic verification: check that modified files are valid JSON
        for file_path in item.files_modified:
            full_path = self.project.root_dir.parent.parent / file_path
            if full_path.suffix == ".json":
                try:
                    with open(full_path) as f:
                        json.load(f)
                except (json.JSONDecodeError, IOError):
                    return False

        return True
