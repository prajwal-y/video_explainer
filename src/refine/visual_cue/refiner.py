"""
Visual Cue Refiner - Generates patches to improve visual_cue specifications.

Analyzes existing visual_cues in script.json and generates patches to:
- Add missing visual specifications
- Improve descriptions to follow established patterns (dark glass, 3D depth, etc.)
- Ensure consistency between visual_cue and scene implementation
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ...project import Project
from ...understanding.llm_provider import ClaudeCodeLLMProvider, LLMProvider, get_llm_provider
from ..models import ScriptPatch, UpdateVisualCuePatch


# =============================================================================
# Prompts for Visual Cue Analysis
# =============================================================================

VISUAL_CUE_ANALYSIS_SYSTEM_PROMPT = """You are an expert motion designer and video producer analyzing visual specifications for educational video scenes.

Your task is to improve visual_cue specifications to ensure they:
1. EXPLICITLY separate BACKGROUND from UI COMPONENTS (this is critical!)
2. Follow established visual patterns (dark glass panels, 3D depth, multi-layer shadows)
3. Provide specific, actionable descriptions for animators
4. List all required visual elements explicitly

CRITICAL DISTINCTION - Background vs UI Components:
- BACKGROUND: The scene canvas/backdrop (can be gradients, colors, patterns)
  - Example: "Light gradient (#f4f4f5 to #ffffff)" or "Soft gray with subtle grid pattern"
  - Use LIGHT backgrounds (#f0f0f5, #fafafa, #ffffff range) - NOT dark backgrounds!
  - The background is NOT the same as dark glass panels!

- UI COMPONENTS: The floating panels/cards/windows that sit ON TOP of the background
  - These use dark glass styling: rgba(18,20,25,0.98) backgrounds
  - Multi-layer drop shadows (5-7 layers) to create floating effect
  - Bezel borders: light top/left, dark bottom/right
  - Inner shadows for recessed depth

Visual Styling Principles for UI Components:
- Dark glass panels: Uniformly dark backgrounds (rgba 18-22 range)
- 3D depth through shadows: Multi-layer drop shadows, NOT perspective transforms
- Bezel borders: Light top/left edges, dark bottom/right edges
- Inner shadows: Recessed depth effect inside components
- Colored accent glows: Subtle glow underneath based on accent color
- Top edge highlights: Thin 1px line simulating light catching

TEXT STYLING REQUIREMENTS (CRITICAL):
- All text on dark glass panels MUST be white (#ffffff) or light gray (rgba(255,255,255,0.85))
- NEVER use black or dark text on dark panels - it will be INVISIBLE
- Minimum font sizes for readability:
  - Body text: 16-18px * scale
  - Titles: 22-28px * scale
  - Annotations/labels: 14-16px * scale
- Always specify text colors explicitly in visual_cue descriptions

ELEMENT CONTAINMENT REQUIREMENTS:
- All labels, badges, and annotations must stay INSIDE their parent containers
- If a label needs to appear "above" a bar chart, allocate space WITHIN the panel
- Never position elements with negative offsets that go outside panels
- Leave adequate padding (20-30px scaled) inside panels for content

SPACING REQUIREMENTS:
- Content panels should start at LAYOUT.title.y + 140 minimum (not crowding the title)
- Leave at least 80-100px between title area and first content panel
- Bottom of panels should not extend beyond height - 100px (leave room for Reference)
- Gap between panels: 25-50px scaled

The description MUST start with "BACKGROUND:" specifying the scene backdrop, then "UI COMPONENTS:" describing the floating panels.
"""

VISUAL_CUE_ANALYSIS_PROMPT = """Analyze and improve the visual_cue for this scene.

## Scene Information
- Scene ID: {scene_id}
- Scene Title: {scene_title}
- Scene Type: {scene_type}

## Narration
"{narration}"

## Current Visual Cue
{current_visual_cue_json}

## Scene Implementation (if available)
{scene_implementation}

## Instructions
Analyze the current visual_cue and generate an improved version that:

1. **EXPLICITLY separates BACKGROUND from UI COMPONENTS** (CRITICAL!)
   - The description MUST start with "BACKGROUND:" describing the scene backdrop (gradient, color, pattern)
   - Then "UI COMPONENTS:" describing the floating dark glass panels

2. **Lists ALL visual elements with clear layer identification**:
   - First element should always be the BACKGROUND specification
   - Subsequent elements describe UI components that float on top

3. **Specifies exact colors/values where possible**:
   - Background: LIGHT colors like #f4f4f5, #fafafa, #ffffff (soft grays/whites)
   - UI Components: rgba(18,20,25,0.98) for dark glass panels (contrast against light bg)

4. **Matches the narration**: The visual should support what's being said.

5. **Is specific and actionable**: An animator should implement this without guessing.

If the current visual_cue already clearly separates background from UI components, return it with minor improvements only.

Respond with JSON:
{{
    "needs_update": true,
    "reason": "Why this visual_cue needs improvement",
    "improved_visual_cue": {{
        "description": "BACKGROUND: [describe LIGHT scene backdrop - soft gradients, light colors]. UI COMPONENTS: [describe dark glass panels that float on top with their styling].",
        "visual_type": "animation",
        "elements": [
            "BACKGROUND: Light gradient (#f4f4f5 to #ffffff) or soft gray (#fafafa) with subtle grid pattern",
            "Main dark glass panel (rgba(18,20,25,0.98)) with multi-layer shadows - provides contrast against light background",
            "Additional UI component with specific styling",
            "..."
        ],
        "duration_seconds": {duration_seconds}
    }}
}}

If no update is needed:
{{
    "needs_update": false,
    "reason": "Visual cue already clearly separates background from UI components"
}}
"""


@dataclass
class VisualCueRefinerResult:
    """Result of visual cue refinement analysis."""

    project_id: str
    scenes_analyzed: int = 0
    scenes_needing_update: int = 0
    patches: list[UpdateVisualCuePatch] = field(default_factory=list)
    analysis_notes: str = ""
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "scenes_analyzed": self.scenes_analyzed,
            "scenes_needing_update": self.scenes_needing_update,
            "patches": [p.to_dict() for p in self.patches],
            "analysis_notes": self.analysis_notes,
            "error_message": self.error_message,
        }


class VisualCueRefiner:
    """Analyzes and improves visual_cue specifications in script.json."""

    def __init__(
        self,
        project: Project,
        llm_provider: Optional[LLMProvider] = None,
        verbose: bool = True,
    ):
        """Initialize the visual cue refiner.

        Args:
            project: The project to analyze
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

    def analyze(self, scene_indices: Optional[list[int]] = None) -> VisualCueRefinerResult:
        """Analyze visual_cues and generate improvement patches.

        Args:
            scene_indices: Optional list of scene indices to analyze (0-based).
                          If None, analyzes all scenes.

        Returns:
            VisualCueRefinerResult with patches to improve visual_cues
        """
        self._log("Starting visual cue analysis...")

        # Load script.json
        script_data = self._load_script()
        if not script_data:
            return VisualCueRefinerResult(
                project_id=self.project.id,
                error_message="ERROR: Could not load script.json",
            )

        scenes = script_data.get("scenes", [])
        if not scenes:
            return VisualCueRefinerResult(
                project_id=self.project.id,
                error_message="ERROR: No scenes found in script.json",
            )

        # Determine which scenes to analyze
        if scene_indices is None:
            scene_indices = list(range(len(scenes)))
        else:
            # Validate indices
            scene_indices = [i for i in scene_indices if 0 <= i < len(scenes)]

        self._log(f"Analyzing {len(scene_indices)} scenes...")

        patches = []
        scenes_needing_update = 0

        for scene_idx in scene_indices:
            scene = scenes[scene_idx]
            scene_title = scene.get("title", "Untitled")
            scene_id = scene.get("scene_id", f"scene_{scene_idx + 1}")

            self._log(f"\nScene {scene_idx + 1}: {scene_title}")

            # Analyze this scene's visual_cue
            patch = self._analyze_scene_visual_cue(scene, scene_idx)

            if patch:
                patches.append(patch)
                scenes_needing_update += 1
                self._log(f"  -> Needs update: {patch.reason[:60]}...")
            else:
                self._log(f"  -> OK (no changes needed)")

        result = VisualCueRefinerResult(
            project_id=self.project.id,
            scenes_analyzed=len(scene_indices),
            scenes_needing_update=scenes_needing_update,
            patches=patches,
            analysis_notes=f"Analyzed {len(scene_indices)} scenes, {scenes_needing_update} need visual_cue updates",
        )

        self._log(f"\nAnalysis complete: {scenes_needing_update}/{len(scene_indices)} scenes need updates")

        return result

    def _load_script(self) -> Optional[dict]:
        """Load script.json."""
        script_path = self.project.root_dir / "script" / "script.json"
        if not script_path.exists():
            return None

        try:
            with open(script_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._log(f"Error loading script.json: {e}")
            return None

    def _find_scene_file(self, scene: dict) -> Optional[Path]:
        """Find the scene implementation file (.tsx)."""
        scenes_dir = self.project.root_dir / "scenes"
        if not scenes_dir.exists():
            return None

        # Extract scene name from title
        title = scene.get("title", "")
        # Convert "The Impossible Leap" -> "TheImpossibleLeap" or "impossible_leap"
        scene_name_pascal = title.replace(" ", "").replace(":", "").replace("-", "")
        scene_name_snake = title.lower().replace(" ", "_").replace(":", "").replace("-", "_")

        # Try various patterns
        patterns = [
            f"*{scene_name_pascal}*.tsx",
            f"*{scene_name_snake}*.tsx",
            f"*{scene_name_pascal.lower()}*.tsx",
        ]

        for pattern in patterns:
            matches = list(scenes_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _analyze_scene_visual_cue(
        self, scene: dict, scene_idx: int
    ) -> Optional[UpdateVisualCuePatch]:
        """Analyze a single scene's visual_cue and generate a patch if needed.

        Args:
            scene: The scene dictionary from script.json
            scene_idx: The scene index (0-based)

        Returns:
            UpdateVisualCuePatch if the visual_cue needs improvement, None otherwise
        """
        scene_title = scene.get("title", "Untitled")
        scene_id = scene.get("scene_id", f"scene_{scene_idx + 1}")
        scene_type = scene.get("scene_type", "unknown")
        narration = scene.get("voiceover", "")
        current_visual_cue = scene.get("visual_cue")
        duration = scene.get("duration_seconds", 25.0)

        # Try to load scene implementation
        scene_file = self._find_scene_file(scene)
        scene_implementation = ""
        if scene_file and scene_file.exists():
            try:
                # Read first 3000 chars to get an idea of the implementation
                content = scene_file.read_text()
                scene_implementation = content[:3000]
                if len(content) > 3000:
                    scene_implementation += "\n... (truncated)"
            except IOError:
                pass

        # Format current visual_cue as JSON
        if current_visual_cue:
            current_visual_cue_json = json.dumps(current_visual_cue, indent=2)
        else:
            current_visual_cue_json = "(No visual_cue specified)"

        # Build the prompt
        prompt = VISUAL_CUE_ANALYSIS_PROMPT.format(
            scene_id=scene_id,
            scene_title=scene_title,
            scene_type=scene_type,
            narration=narration[:500] if narration else "(No narration)",
            current_visual_cue_json=current_visual_cue_json,
            scene_implementation=scene_implementation if scene_implementation else "(No implementation found)",
            duration_seconds=duration,
        )

        try:
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=VISUAL_CUE_ANALYSIS_SYSTEM_PROMPT,
            )

            if not response.get("needs_update", False):
                return None

            improved_visual_cue = response.get("improved_visual_cue", {})
            reason = response.get("reason", "Visual cue needs improvement")

            return UpdateVisualCuePatch(
                reason=reason,
                priority="medium",
                scene_id=scene_id,
                scene_title=scene_title,
                current_visual_cue=current_visual_cue,
                new_visual_cue=improved_visual_cue,
            )

        except Exception as e:
            self._log(f"  Error analyzing scene: {e}")
            return None

    def apply_patches(self, patches: list[UpdateVisualCuePatch]) -> int:
        """Apply visual_cue patches to script.json.

        Args:
            patches: List of patches to apply

        Returns:
            Number of patches successfully applied
        """
        if not patches:
            return 0

        script_path = self.project.root_dir / "script" / "script.json"
        if not script_path.exists():
            self._log("ERROR: script.json not found")
            return 0

        try:
            with open(script_path) as f:
                script_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._log(f"ERROR loading script.json: {e}")
            return 0

        scenes = script_data.get("scenes", [])
        applied = 0

        for patch in patches:
            # Find the scene by scene_id
            for scene in scenes:
                if scene.get("scene_id") == patch.scene_id:
                    scene["visual_cue"] = patch.new_visual_cue
                    applied += 1
                    self._log(f"Applied patch to scene: {patch.scene_title}")
                    break

        # Write back to script.json
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(script_data, f, indent=2, ensure_ascii=False)
            self._log(f"Saved {applied} updates to script.json")
        except IOError as e:
            self._log(f"ERROR saving script.json: {e}")
            return 0

        return applied

    def save_result(
        self, result: VisualCueRefinerResult, output_path: Optional[Path] = None
    ) -> Path:
        """Save the analysis result to a JSON file.

        Args:
            result: The analysis result to save
            output_path: Optional custom output path

        Returns:
            Path to the saved file
        """
        if output_path is None:
            refinement_dir = self.project.root_dir / "refinement"
            refinement_dir.mkdir(parents=True, exist_ok=True)
            output_path = refinement_dir / "visual_cue_analysis.json"

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path
