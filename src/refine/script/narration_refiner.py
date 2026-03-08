"""
Script Refiner - Phase 2: Apply Patches and Refine Storytelling.

This module:
1. Loads patches from Phase 1 (gap_analysis.json)
2. Generates additional storytelling refinement patches
3. Presents all patches for user approval
4. Applies approved patches to script.json and narrations.json
"""

import json
import re
from pathlib import Path
from typing import Optional

from ...project import Project
from ...understanding.llm_provider import ClaudeCodeLLMProvider, LLMProvider, get_llm_provider
from ..models import (
    AddBridgePatch,
    AddScenePatch,
    ExpandScenePatch,
    GapAnalysisResult,
    ModifyScenePatch,
    NarrationIssue,
    NarrationIssueType,
    NarrationRefinementResult,
    NarrationScores,
    SceneNarrationAnalysis,
    ScriptPatch,
    ScriptPatchType,
)
from ..narration_principles import format_principles_for_prompt


# =============================================================================
# Prompts for Narration Refinement
# =============================================================================

NARRATION_ANALYSIS_SYSTEM_PROMPT = """You are an expert video scriptwriter improving narrations for professional technical explainer videos.

Your task is to evaluate narrations against storytelling principles and generate PATCHES to fix identified issues.

Be specific and actionable. Each issue should result in a concrete patch that can be applied."""

SINGLE_SCENE_ANALYSIS_PROMPT = """Evaluate this scene's narration against storytelling principles and generate patches to fix issues.

## Scene Information
- Scene ID: {scene_id}
- Title: {scene_title}
- Duration: {duration_seconds} seconds
- Expected word count: ~{expected_words} words (at 150 wpm)
- Actual word count: {actual_words} words

## Context
- Previous scene ended with: "{prev_ending}"
- Next scene starts with: "{next_start}"

## Gap Analysis Context (from Phase 1)
{gap_context}

## Current Narration
"{narration}"

## Narration Principles
{principles}

## Instructions
1. Score each aspect of the narration (0-10):
   - hook: Does it grab attention in the first sentence?
   - flow: Does it connect smoothly to the previous scene?
   - tension: Does it build anticipation before the insight?
   - insight: Is there one clear takeaway?
   - engagement: Does it use questions, analogies, surprises?
   - accuracy: Is it technically correct?
   - length: Is the word count appropriate for the duration?
   - specificity: Does it use SPECIFIC NUMBERS instead of vague qualifiers? (e.g., "17% to 78%" not "improved dramatically")
   - mechanism: Does it explain HOW things work step-by-step, not just THAT they work? (e.g., "generate candidates, evaluate each, expand best one" not "explores multiple paths")

2. Identify specific issues (if any)

3. Generate a MODIFY_SCENE patch if the narration needs revision

Respond with JSON in this exact format:
{{
    "scores": {{
        "hook": 8,
        "flow": 7,
        "tension": 6,
        "insight": 9,
        "engagement": 7,
        "accuracy": 10,
        "length": 8,
        "specificity": 7,
        "mechanism": 6
    }},
    "issues": [
        {{
            "issue_type": "weak_hook|poor_transition|missing_tension|no_key_insight|lacks_analogy|no_emotional_beat|wrong_length|technical_inaccuracy|redundant_text|lacks_specificity|lacks_mechanism|other",
            "description": "Clear description of the issue",
            "current_text": "The specific problematic text",
            "severity": "low|medium|high",
            "suggested_fix": "How to fix this specific issue"
        }}
    ],
    "patch": {{
        "patch_type": "modify_scene",
        "priority": "low|medium|high",
        "reason": "Why this revision improves the narration",
        "scene_id": "{scene_id}",
        "field_name": "narration",
        "old_value": "Current narration text",
        "new_value": "Revised narration text"
    }},
    "analysis_notes": "Brief overall assessment"
}}

If no revision is needed, set "patch" to null.
"""


class ScriptRefiner:
    """Applies patches to refine scripts and narrations."""

    def __init__(
        self,
        project: Project,
        llm_provider: Optional[LLMProvider] = None,
        verbose: bool = True,
    ):
        """Initialize the script refiner.

        Args:
            project: The project to refine
            llm_provider: LLM provider to use (defaults to ClaudeCodeLLMProvider)
            verbose: Whether to print progress messages
        """
        self.project = project
        self.verbose = verbose

        # Use ClaudeCodeLLMProvider by default
        if llm_provider is None:
            from ...config import load_config
            self.llm = get_llm_provider(load_config())
        else:
            self.llm = llm_provider

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"   {message}")

    def _slugify(self, text: str) -> str:
        """Convert text to slug format for scene ID matching.

        Args:
            text: Text to convert (e.g., scene title)

        Returns:
            Slug version (e.g., "The Imitation Trap" -> "the_imitation_trap")
        """
        slug = text.lower()
        slug = re.sub(r"[\s\-]+", "_", slug)
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        slug = re.sub(r"_+", "_", slug)
        return slug.strip("_")

    def _match_scene_id(self, scene: dict, target_id: str) -> bool:
        """Check if a scene matches the target ID using flexible matching.

        Supports:
        - Direct ID match (string comparison of scene_id)
        - Slug match via title (converts title to slug and compares)

        Args:
            scene: Scene dictionary with scene_id and optionally title
            target_id: Target scene ID to match (can be numeric string or slug)

        Returns:
            True if the scene matches the target ID
        """
        scene_id = scene.get("scene_id")

        # Direct match (convert both to string for comparison)
        if str(scene_id) == str(target_id):
            return True

        # Match by title slug
        title = scene.get("title", "")
        if title and self._slugify(title) == target_id:
            return True

        return False

    def load_gap_analysis(self) -> Optional[GapAnalysisResult]:
        """Load gap analysis result from Phase 1.

        Returns:
            GapAnalysisResult if found, None otherwise
        """
        gap_analysis_path = self.project.root_dir / "refinement" / "gap_analysis.json"
        if not gap_analysis_path.exists():
            self._log("No gap_analysis.json found. Run Phase 1 (--phase analyze) first.")
            return None

        with open(gap_analysis_path) as f:
            data = json.load(f)

        return GapAnalysisResult.from_dict(data)

    def refine(self) -> tuple[list[ScriptPatch], NarrationRefinementResult]:
        """Run full refinement: load Phase 1 patches + generate storytelling patches.

        Returns:
            Tuple of (all_patches, narration_analysis_result)
        """
        self._log("Starting script refinement (Phase 2)...")

        # Step 1: Load patches from Phase 1
        gap_analysis = self.load_gap_analysis()
        phase1_patches: list[ScriptPatch] = []
        gap_context = "No gap analysis available."

        if gap_analysis:
            phase1_patches = gap_analysis.patches
            self._log(f"Loaded {len(phase1_patches)} patches from Phase 1")
            gap_context = self._format_gap_context(gap_analysis)
        else:
            self._log("No Phase 1 patches found, proceeding with storytelling refinement only")

        # Step 2: Load narrations
        scenes = self._load_narrations()
        if not scenes:
            return phase1_patches, NarrationRefinementResult(
                project_id=self.project.id,
                overall_storytelling_score=0.0,
                total_issues_found=0,
            )

        self._log(f"Analyzing {len(scenes)} scenes for storytelling improvements...")

        # Step 3: Analyze each scene and generate storytelling patches
        scene_analyses = []
        storytelling_patches: list[ScriptPatch] = []
        total_issues = 0

        for i, scene in enumerate(scenes):
            prev_ending = self._get_scene_ending(scenes[i - 1] if i > 0 else None)
            next_start = self._get_scene_start(scenes[i + 1] if i < len(scenes) - 1 else None)

            self._log(f"Analyzing scene {i + 1}/{len(scenes)}: {scene.get('title', 'Unknown')}")

            analysis, patch = self._analyze_scene(
                scene=scene,
                prev_ending=prev_ending,
                next_start=next_start,
                gap_context=gap_context,
            )

            scene_analyses.append(analysis)
            total_issues += len(analysis.issues)

            if patch:
                storytelling_patches.append(patch)

        # Calculate overall score
        if scene_analyses:
            overall_score = sum(s.scores.overall for s in scene_analyses) / len(scene_analyses)
        else:
            overall_score = 0.0

        result = NarrationRefinementResult(
            project_id=self.project.id,
            scene_analyses=scene_analyses,
            overall_storytelling_score=overall_score,
            total_issues_found=total_issues,
        )

        # Combine all patches
        all_patches = phase1_patches + storytelling_patches

        self._log(f"Analysis complete. Overall score: {overall_score:.1f}/10")
        self._log(f"Total patches: {len(all_patches)} ({len(phase1_patches)} from Phase 1, {len(storytelling_patches)} from storytelling)")

        return all_patches, result

    def _format_gap_context(self, gap_analysis: GapAnalysisResult) -> str:
        """Format gap analysis for inclusion in scene analysis prompts.

        Args:
            gap_analysis: The gap analysis result

        Returns:
            Formatted string describing relevant gaps
        """
        lines = []

        if gap_analysis.missing_concepts:
            lines.append(f"Missing concepts: {', '.join(gap_analysis.missing_concepts)}")

        if gap_analysis.shallow_concepts:
            lines.append(f"Shallow coverage: {', '.join(gap_analysis.shallow_concepts)}")

        if gap_analysis.narrative_gaps:
            gap_descriptions = [
                f"Gap between {g.from_scene_title} -> {g.to_scene_title}: {g.gap_description}"
                for g in gap_analysis.narrative_gaps[:3]  # Limit to 3
            ]
            lines.append("Narrative gaps:\n  - " + "\n  - ".join(gap_descriptions))

        return "\n".join(lines) if lines else "No significant gaps identified."

    def _load_narrations(self) -> list[dict]:
        """Load narrations from the project.

        Returns:
            List of scene dictionaries with narrations
        """
        narrations_path = self.project.root_dir / "narration" / "narrations.json"
        if narrations_path.exists():
            with open(narrations_path) as f:
                data = json.load(f)
                return data.get("scenes", [])

        # Fall back to script.json
        script_path = self.project.root_dir / "script" / "script.json"
        if script_path.exists():
            with open(script_path) as f:
                data = json.load(f)
                return data.get("scenes", [])

        return []

    def _get_scene_ending(self, scene: Optional[dict]) -> str:
        """Get the last ~100 characters of a scene's narration."""
        if scene is None:
            return "(start of video)"

        narration = scene.get("narration", scene.get("voiceover", ""))
        if len(narration) <= 100:
            return narration
        return "..." + narration[-100:]

    def _get_scene_start(self, scene: Optional[dict]) -> str:
        """Get the first ~100 characters of a scene's narration."""
        if scene is None:
            return "(end of video)"

        narration = scene.get("narration", scene.get("voiceover", ""))
        if len(narration) <= 100:
            return narration
        return narration[:100] + "..."

    def _analyze_scene(
        self,
        scene: dict,
        prev_ending: str,
        next_start: str,
        gap_context: str,
    ) -> tuple[SceneNarrationAnalysis, Optional[ScriptPatch]]:
        """Analyze a single scene's narration and generate patch if needed.

        Args:
            scene: Scene dictionary
            prev_ending: End of previous scene's narration
            next_start: Start of next scene's narration
            gap_context: Context from gap analysis

        Returns:
            Tuple of (SceneNarrationAnalysis, optional ModifyScenePatch)
        """
        scene_id = scene.get("scene_id", "unknown")
        scene_title = scene.get("title", "Unknown")
        duration = scene.get("duration_seconds", 30)
        narration = scene.get("narration", scene.get("voiceover", ""))
        word_count = len(narration.split())
        expected_words = int(duration * 2.5)  # 150 wpm

        prompt = SINGLE_SCENE_ANALYSIS_PROMPT.format(
            scene_id=scene_id,
            scene_title=scene_title,
            duration_seconds=duration,
            expected_words=expected_words,
            actual_words=word_count,
            prev_ending=prev_ending,
            next_start=next_start,
            gap_context=gap_context,
            narration=narration,
            principles=format_principles_for_prompt(),
        )

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=NARRATION_ANALYSIS_SYSTEM_PROMPT,
            )

            # Parse scores
            scores_data = response.get("scores", {})
            scores = NarrationScores(
                hook=scores_data.get("hook", 5.0),
                flow=scores_data.get("flow", 5.0),
                tension=scores_data.get("tension", 5.0),
                insight=scores_data.get("insight", 5.0),
                engagement=scores_data.get("engagement", 5.0),
                accuracy=scores_data.get("accuracy", 5.0),
                length=scores_data.get("length", 5.0),
                specificity=scores_data.get("specificity", 5.0),
                mechanism=scores_data.get("mechanism", 5.0),
            )

            # Parse issues
            issues = []
            for issue_data in response.get("issues", []):
                try:
                    issue_type = NarrationIssueType(issue_data.get("issue_type", "other"))
                except ValueError:
                    issue_type = NarrationIssueType.OTHER

                issues.append(
                    NarrationIssue(
                        scene_id=scene_id,
                        issue_type=issue_type,
                        description=issue_data.get("description", ""),
                        current_text=issue_data.get("current_text", ""),
                        severity=issue_data.get("severity", "medium"),
                        suggested_fix=issue_data.get("suggested_fix"),
                    )
                )

            # Parse patch
            patch = None
            patch_data = response.get("patch")
            if patch_data and patch_data.get("patch_type") == "modify_scene":
                patch = ModifyScenePatch(
                    reason=patch_data.get("reason", "Storytelling improvement"),
                    priority=patch_data.get("priority", "medium"),
                    scene_id=patch_data.get("scene_id", scene_id),
                    field_name=patch_data.get("field_name", "narration"),
                    old_value=patch_data.get("old_value", narration),
                    new_value=patch_data.get("new_value", ""),
                )

            analysis = SceneNarrationAnalysis(
                scene_id=scene_id,
                scene_title=scene_title,
                current_narration=narration,
                duration_seconds=duration,
                word_count=word_count,
                scores=scores,
                issues=issues,
                suggested_revision=patch.new_value if patch else None,
            )

            return analysis, patch

        except Exception as e:
            self._log(f"Error analyzing scene {scene_id}: {e}")
            return SceneNarrationAnalysis(
                scene_id=scene_id,
                scene_title=scene_title,
                current_narration=narration,
                duration_seconds=duration,
                word_count=word_count,
                scores=NarrationScores(),
                issues=[
                    NarrationIssue(
                        scene_id=scene_id,
                        issue_type=NarrationIssueType.OTHER,
                        description=f"Analysis failed: {e}",
                        current_text="",
                        severity="low",
                    )
                ],
            ), None

    def apply_patch(self, patch: ScriptPatch) -> bool:
        """Apply a single patch to script.json and/or narrations.json.

        Args:
            patch: The patch to apply

        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(patch, AddScenePatch):
                return self._apply_add_scene_patch(patch)
            elif isinstance(patch, ModifyScenePatch):
                return self._apply_modify_scene_patch(patch)
            elif isinstance(patch, ExpandScenePatch):
                return self._apply_expand_scene_patch(patch)
            elif isinstance(patch, AddBridgePatch):
                return self._apply_add_bridge_patch(patch)
            else:
                self._log(f"Unknown patch type: {type(patch)}")
                return False
        except Exception as e:
            self._log(f"Error applying patch: {e}")
            return False

    def _apply_add_scene_patch(self, patch: AddScenePatch) -> bool:
        """Add a new scene to script and narrations."""
        # Update narrations.json
        narrations_path = self.project.root_dir / "narration" / "narrations.json"
        if narrations_path.exists():
            with open(narrations_path) as f:
                data = json.load(f)

            scenes = data.get("scenes", [])

            # Find insertion point using flexible matching
            insert_index = len(scenes)  # Default to end if not found
            if patch.insert_after_scene_id:
                for i, scene in enumerate(scenes):
                    if self._match_scene_id(scene, patch.insert_after_scene_id):
                        insert_index = i + 1
                        break

            # Create new scene for narrations.json
            new_scene = {
                "scene_id": patch.new_scene_id,
                "title": patch.title,
                "narration": patch.narration,
                "duration_seconds": patch.duration_seconds,
            }

            scenes.insert(insert_index, new_scene)
            data["scenes"] = scenes

            # Update total duration
            data["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in scenes
            )

            with open(narrations_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._log(f"Added scene '{patch.title}' at position {insert_index + 1} in narrations.json")

        # Also update script.json if it exists
        script_path = self.project.root_dir / "script" / "script.json"
        if script_path.exists():
            with open(script_path) as f:
                script_data = json.load(f)

            script_scenes = script_data.get("scenes", [])

            # Find insertion point using flexible matching
            insert_index = len(script_scenes)  # Default to end if not found
            if patch.insert_after_scene_id:
                for i, scene in enumerate(script_scenes):
                    if self._match_scene_id(scene, patch.insert_after_scene_id):
                        insert_index = i + 1
                        break

            # Create proper visual_cue structure (matching existing scene format)
            visual_cue = {
                "description": patch.visual_description,
                "visual_type": "animation",
                "elements": [patch.visual_description],  # Start with description as single element
                "duration_seconds": patch.duration_seconds,
            }

            new_script_scene = {
                "scene_id": patch.new_scene_id,
                "title": patch.title,
                "voiceover": patch.narration,
                "visual_cue": visual_cue,
                "duration_seconds": patch.duration_seconds,
            }

            script_scenes.insert(insert_index, new_script_scene)
            script_data["scenes"] = script_scenes

            # Update total duration
            script_data["total_duration_seconds"] = sum(
                s.get("duration_seconds", 0) for s in script_scenes
            )

            with open(script_path, "w") as f:
                json.dump(script_data, f, indent=2, ensure_ascii=False)

            self._log(f"Added scene '{patch.title}' at position {insert_index + 1} in script.json")

        return True

    def _apply_modify_scene_patch(self, patch: ModifyScenePatch) -> bool:
        """Modify an existing scene's field."""
        updated_narrations = False
        updated_script = False

        # Update narrations.json
        narrations_path = self.project.root_dir / "narration" / "narrations.json"
        if narrations_path.exists():
            with open(narrations_path) as f:
                data = json.load(f)

            for scene in data.get("scenes", []):
                if self._match_scene_id(scene, patch.scene_id):
                    scene[patch.field_name] = patch.new_value
                    updated_narrations = True
                    break

            if updated_narrations:
                with open(narrations_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        # Also update script.json if modifying voiceover/narration
        script_path = self.project.root_dir / "script" / "script.json"
        if script_path.exists() and patch.field_name in ("narration", "voiceover"):
            with open(script_path) as f:
                script_data = json.load(f)

            for scene in script_data.get("scenes", []):
                if self._match_scene_id(scene, patch.scene_id):
                    # script.json uses "voiceover" instead of "narration"
                    scene["voiceover"] = patch.new_value
                    updated_script = True
                    break

            if updated_script:
                with open(script_path, "w") as f:
                    json.dump(script_data, f, indent=2, ensure_ascii=False)
            elif updated_narrations:
                self._log(f"Warning: Updated narrations.json but scene '{patch.scene_id}' not found in script.json")

        if updated_narrations:
            self._log(f"Modified scene '{patch.scene_id}' {patch.field_name}")

        return updated_narrations or updated_script

    def _apply_expand_scene_patch(self, patch: ExpandScenePatch) -> bool:
        """Expand a scene with additional content."""
        updated_narrations = False
        updated_script = False

        # Update narrations.json
        narrations_path = self.project.root_dir / "narration" / "narrations.json"
        if narrations_path.exists():
            with open(narrations_path) as f:
                data = json.load(f)

            for scene in data.get("scenes", []):
                if self._match_scene_id(scene, patch.scene_id):
                    scene["narration"] = patch.expanded_narration
                    if patch.additional_duration_seconds > 0:
                        scene["duration_seconds"] = (
                            scene.get("duration_seconds", 30) + patch.additional_duration_seconds
                        )
                    updated_narrations = True
                    break

            if updated_narrations:
                with open(narrations_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        # Also update script.json
        script_path = self.project.root_dir / "script" / "script.json"
        if script_path.exists():
            with open(script_path) as f:
                script_data = json.load(f)

            for scene in script_data.get("scenes", []):
                if self._match_scene_id(scene, patch.scene_id):
                    scene["voiceover"] = patch.expanded_narration
                    if patch.additional_duration_seconds > 0:
                        scene["duration_seconds"] = (
                            scene.get("duration_seconds", 30) + patch.additional_duration_seconds
                        )
                    updated_script = True
                    break

            if updated_script:
                with open(script_path, "w") as f:
                    json.dump(script_data, f, indent=2, ensure_ascii=False)
            elif updated_narrations:
                self._log(f"Warning: Updated narrations.json but scene '{patch.scene_id}' not found in script.json")

        if updated_narrations:
            self._log(f"Expanded scene '{patch.scene_id}'")

        return updated_narrations or updated_script

    def _apply_add_bridge_patch(self, patch: AddBridgePatch) -> bool:
        """Add bridging content to a scene."""
        updated_narrations = False
        updated_script = False

        # Update narrations.json
        narrations_path = self.project.root_dir / "narration" / "narrations.json"
        if narrations_path.exists():
            with open(narrations_path) as f:
                data = json.load(f)

            for scene in data.get("scenes", []):
                if self._match_scene_id(scene, patch.modify_scene_id):
                    scene["narration"] = patch.new_text
                    updated_narrations = True
                    break

            if updated_narrations:
                with open(narrations_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        # Also update script.json
        script_path = self.project.root_dir / "script" / "script.json"
        if script_path.exists():
            with open(script_path) as f:
                script_data = json.load(f)

            for scene in script_data.get("scenes", []):
                if self._match_scene_id(scene, patch.modify_scene_id):
                    scene["voiceover"] = patch.new_text
                    updated_script = True
                    break

            if updated_script:
                with open(script_path, "w") as f:
                    json.dump(script_data, f, indent=2, ensure_ascii=False)
            elif updated_narrations:
                self._log(f"Warning: Updated narrations.json but scene '{patch.modify_scene_id}' not found in script.json")

        if updated_narrations:
            self._log(f"Added bridge content to scene '{patch.modify_scene_id}'")

        return updated_narrations or updated_script

    def save_result(
        self,
        result: NarrationRefinementResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save the narration refinement result to a JSON file.

        Args:
            result: The refinement result to save
            output_path: Optional custom output path

        Returns:
            Path to the saved file
        """
        if output_path is None:
            refinement_dir = self.project.root_dir / "refinement"
            refinement_dir.mkdir(parents=True, exist_ok=True)
            output_path = refinement_dir / "narration_analysis.json"

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path


# Keep the old name as an alias for backwards compatibility
NarrationRefiner = ScriptRefiner
