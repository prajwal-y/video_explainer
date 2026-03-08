"""
Visual Inspector - Main orchestrator for visual refinement.

Coordinates beat parsing, visual inspection, AI analysis, and fix application.
Uses Claude Code with --chrome flag for browser-based visual inspection.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from ...config import LLMConfig
from ...project import Project
from ...understanding.llm_provider import (
    ClaudeCodeLLMProvider,
    ClaudeCodeResult,
    LLMProvider,
    MockLLMProvider,
    get_llm_provider,
)
from ..models import (
    Beat,
    Fix,
    FixStatus,
    Issue,
    IssueType,
    SceneRefinementResult,
)
from ..principles import format_principles_for_prompt, GUIDING_PRINCIPLES
from ..validation import ProjectValidator
from .beat_parser import BeatParser, MockBeatParser
from .screenshot import (
    CapturedScreenshot,
    MockScreenshotCapture,
    ScreenshotCapture,
    check_remotion_running,
    PLAYWRIGHT_AVAILABLE,
)

# Default Remotion URL - uses SingleScenePlayer for isolated scene inspection
REMOTION_BASE_URL = "http://localhost:3000"
SINGLE_SCENE_PLAYER_URL = "http://localhost:3000/SingleScenePlayer"


VISUAL_ANALYSIS_SYSTEM_PROMPT = """You are an expert video editor and motion designer reviewing educational video content.

Your task is to analyze screenshots from a video scene and identify issues based on the guiding principles for high-quality educational videos.

When analyzing, be specific about:
1. What principle is violated
2. Where exactly in the visual the issue occurs
3. What the visual SHOULD look like instead

Be constructive and actionable in your feedback.
"""


VISUAL_ANALYSIS_PROMPT_TEMPLATE = """Analyze this scene from an educational video for quality issues.

## Scene Information
- Scene: {scene_title} (Scene {scene_index})
- Scene file: {scene_file}
- Total duration: {duration_seconds:.1f} seconds

## Narration
"{narration_text}"

## Visual Beats and Screenshots
I have captured screenshots at key moments in the narration. For each beat, analyze if the visual meets the quality principles.

{beats_info}

## The 13 Guiding Principles
{principles}

## Instructions
1. First, READ each screenshot file to see what the visual looks like
2. Compare what you see against what the narration describes at that moment
3. Evaluate against the 13 guiding principles
4. Identify specific issues with specific fixes

Respond with JSON in this format:
{{
    "issues": [
        {{
            "beat_index": 0,
            "principle_violated": "show_dont_tell",
            "description": "Specific description of the issue",
            "severity": "high",
            "suggested_fix": "Specific suggestion for how to fix"
        }}
    ],
    "overall_assessment": "Brief overall assessment of the scene quality",
    "passes_quality_bar": false
}}

Principle codes: show_dont_tell, animation_reveals, progressive_disclosure, text_complements, visual_hierarchy, breathing_room, purposeful_motion, emotional_resonance, professional_polish, sync_with_narration, screen_space_utilization, material_depth, visual_spec_match, other

Severity: low, medium, high

If the scene looks good and meets all principles, return an empty issues array and passes_quality_bar: true.
"""


FIX_GENERATION_PROMPT_TEMPLATE = """You need to fix visual issues in a Remotion scene component.

## Scene Information
- Scene file: {scene_file}
- Scene: {scene_title}

## Issues to Fix
{issues_description}

## The 13 Guiding Principles (for context)
{principles}

## Instructions
1. READ the scene file at {scene_file}
2. Understand the current implementation
3. Apply fixes for each issue using the Edit tool
4. Make sure your edits follow Remotion/React best practices
5. After making edits, explain what you changed

Key patterns for Remotion scenes:
- Use `interpolate` for opacity/position animations synced to frame number
- Use `spring` for bouncy/pop effects
- Use `Easing.out(Easing.exp)` for explosive growth animations
- Define a PHASE object to organize timing phases
- Keep colors in a single object for consistency
- Elements should appear WHEN narration mentions them (progressive disclosure)

Make the necessary edits now, then summarize what you changed.
"""


VERIFICATION_PROMPT_TEMPLATE = """Verify that the fixes improved the scene quality.

## Original Issues
{original_issues}

## Changes Made
{changes_made}

## New Screenshots
I have captured new screenshots after the fixes were applied. Please verify:

{new_beats_info}

## Instructions
1. READ each new screenshot file
2. Compare against the original issues
3. Determine if each issue was properly fixed

Respond with JSON:
{{
    "issues_resolved": [
        {{
            "original_issue": "Description of the original issue",
            "resolution_status": "resolved",
            "notes": "How it was fixed or why it wasn't"
        }}
    ],
    "remaining_issues": [
        {{
            "description": "Any new or unresolved issues",
            "severity": "medium"
        }}
    ],
    "verification_passed": true,
    "summary": "Brief summary of verification results"
}}

Resolution status: resolved, partially_resolved, not_resolved
"""


CLAUDE_CODE_VISUAL_INSPECTION_PROMPT = """## FIRST: Load the Remotion skill
Before doing ANYTHING else, invoke the `/remotion` skill by using the Skill tool with skill="remotion".
This loads essential Remotion best practices you'll need for fixes.
DO THIS NOW before proceeding.

---

Inspect and fix visuals for Scene {scene_number}: "{scene_title}"

## CRITICAL: DO NOT read storyboard.json or index.ts. All info is provided below.

## Quick Reference
- Scene file to edit: {scene_file}
- Scene duration: {duration_seconds:.1f}s ({total_frames} frames)
- Total beats to inspect: {num_beats}
- Frames to check: {beat_frames_list}

## ⚠️ VISUAL SPECIFICATION - THIS IS THE SOURCE OF TRUTH ⚠️
**THE SCENE MUST IMPLEMENT THIS SPECIFICATION EXACTLY. NOT APPROXIMATELY - EXACTLY.**

{visual_cue_section}

**CRITICAL: Every element listed above MUST be present in the scene. Every animation described MUST be implemented.
If the spec says "3D tree" → there must be a tree with 3D depth effects (shadows, layers, perspective).
If the spec says "branching animation" → branches must animate growing outward.
If the spec says "withering effect" → there must be visible withering/decay animation.
If the spec says "blooming" → there must be visible blooming/flourishing animation.
MISSING ANY ELEMENT = visual_spec_match FAIL. IMPLEMENT THE SPEC EXACTLY.**

## Remotion Studio Navigation - IMPORTANT
To go to a specific frame:
1. Click on the frame counter (shows "Frame XX" in the top toolbar)
2. Type the frame number (e.g., "397")
3. **PRESS ENTER** to confirm - this is required!

Without pressing Enter, the frame won't change. If navigation fails, try:
- Click the frame counter again, clear it, type the number, press Enter
- Or use Left/Right arrow keys to step frame-by-frame
- Or click "Skip to start" (|◀) then arrow-key forward

## CRITICAL: Complete Coverage Required
- You MUST take a screenshot at EVERY frame listed: {beat_frames_list}
- You MUST evaluate EVERY screenshot against ALL 13 principles (see checklist below)
- You MUST verify the visual matches the VISUAL SPECIFICATION above
- Stopping early or skipping frames is NOT acceptable
- The scene is {duration_seconds:.1f}s long - ensure you inspect content THROUGHOUT the entire duration
- Partial inspection = failure. Inspect ALL {num_beats} beats.

## Beats to Inspect
{beats_info}

## Narration
"{narration_text}"

## The 13 Guiding Principles
{principles}

Principle codes for JSON: show_dont_tell, animation_reveals, progressive_disclosure, text_complements, visual_hierarchy, breathing_room, purposeful_motion, emotional_resonance, professional_polish, sync_with_narration, screen_space_utilization, material_depth, visual_spec_match

---

## WORKFLOW (Follow in order - do NOT skip steps)

### Phase 1: COMPLETE INSPECTION (no fixes yet)
For EACH frame in [{beat_frames_list}]:
1. Navigate to that exact frame number (use tips above)
2. Take a screenshot
3. **Evaluate against ALL 13 principles using this checklist:**
   ```
   Beat X @ Frame Y - Principle Checklist:
   [ ] 1. Show don't tell: PASS/ISSUE - [reason]
   [ ] 2. Animation reveals: PASS/ISSUE - [reason]
   [ ] 3. Progressive disclosure: PASS/ISSUE - [reason]
   [ ] 4. Text complements: PASS/ISSUE - [reason]
   [ ] 5. Visual hierarchy: PASS/ISSUE - [reason]
   [ ] 6. Breathing room: PASS/ISSUE - [reason]
   [ ] 7. Purposeful motion: PASS/ISSUE - [reason]
   [ ] 8. Emotional resonance: PASS/ISSUE - [reason]
   [ ] 9. Professional polish: PASS/ISSUE - [reason]
   [ ] 10. Sync with narration: PASS/ISSUE - [reason]
   [ ] 11. Screen space utilization: PASS/ISSUE - [reason]
   [ ] 12. 3D depth & material design: PASS/ISSUE - [reason]
   [ ] 13. Follows visual specification: PASS/ISSUE - [list each element from spec and whether it's present]
   ```
4. Document the issues found (principles marked ISSUE)

⚠️ DO NOT proceed to Phase 2 until you have inspected ALL {num_beats} beats.
⚠️ DO NOT fix anything during Phase 1 - just document issues.
⚠️ You MUST explicitly evaluate ALL 13 principles for EACH beat - no shortcuts!

### Phase 2: FIX ISSUES
After completing Phase 1 for ALL beats:
1. Read the scene file: {scene_file}
2. **PRIORITY: Fix visual_spec_match issues FIRST.**
   - Go back to the VISUAL SPECIFICATION section above
   - For EACH element listed, ensure it exists in the code
   - If an element is missing, ADD IT
   - If an animation is described but not implemented, IMPLEMENT IT
   - The spec is the EXACT requirement - not a suggestion
3. Then fix other principle violations
4. Common patterns:
   - `interpolate(frame, [start, end], [0, 1])` for fade/move
   - `spring({{frame, fps, config: {{damping: 15}}}})` for bounce
   - Increase font sizes for screen_space_utilization issues (aim for 24px+ body, 48px+ headlines)
   - Scale up diagrams to use 60-80% of frame width
   - Add padding/margins for breathing_room issues

**For 3D depth & material design (Principle 12), use these patterns:**

Multi-layer shadows (creates floating depth):
```tsx
boxShadow: `
  0 ${{1 * scale}}px ${{2 * scale}}px rgba(0,0,0,0.08),
  0 ${{2 * scale}}px ${{4 * scale}}px rgba(0,0,0,0.08),
  0 ${{4 * scale}}px ${{8 * scale}}px rgba(0,0,0,0.08),
  0 ${{8 * scale}}px ${{16 * scale}}px rgba(0,0,0,0.1),
  0 ${{16 * scale}}px ${{32 * scale}}px rgba(0,0,0,0.12),
  0 ${{32 * scale}}px ${{64 * scale}}px rgba(0,0,0,0.14)
`
```

Bezel border (3D raised edge - light top/left, dark bottom/right):
```tsx
border: `${{2 * scale}}px solid transparent`,
borderTopColor: "rgba(255,255,255,0.12)",
borderLeftColor: "rgba(255,255,255,0.08)",
borderRightColor: "rgba(0,0,0,0.25)",
borderBottomColor: "rgba(0,0,0,0.35)",
```

Inner shadows (recessed depth):
```tsx
boxShadow: `
  inset 0 ${{1 * scale}}px ${{0}}px rgba(255,255,255,0.1),
  inset 0 ${{-4 * scale}}px ${{15 * scale}}px rgba(0,0,0,0.5),
  inset ${{4 * scale}}px 0 ${{15 * scale}}px rgba(0,0,0,0.3),
  inset ${{-4 * scale}}px 0 ${{15 * scale}}px rgba(0,0,0,0.3)
`
```

Dark glass background (NO grey overlays):
```tsx
background: `linear-gradient(180deg,
  rgba(18, 22, 35, 0.98) 0%,
  rgba(12, 15, 28, 0.99) 50%,
  rgba(8, 10, 22, 0.99) 100%)`
```

DON'T DO:
- ❌ `transform: "perspective(1200px) rotateX(2deg)"` - tilting for fake depth
- ❌ Grey gradient overlays at top of dark components
- ❌ Single-layer shadows
- ❌ Light background colors like `rgba(45, 52, 70, 0.95)` for dark glass

### Phase 3: VERIFY
After fixing, re-check at least 2-3 key frames where you found issues:
1. Navigate to a frame that had issues
2. Take a screenshot
3. **FIRST: Verify visual_spec_match by checking EACH element from the spec:**
   ```
   Visual Specification Checklist:
   - [Element 1 from spec]: ✅ PRESENT / ❌ MISSING
   - [Element 2 from spec]: ✅ PRESENT / ❌ MISSING
   - [Animation 1 from spec]: ✅ IMPLEMENTED / ❌ NOT VISIBLE
   ... (check EVERY element listed in the VISUAL SPECIFICATION section)
   ```
4. **Then confirm other principles:**
   ```
   Verification @ Frame Y:
   - screen_space_utilization: Was ISSUE → Now PASS (elements now use 70% of frame)
   - visual_hierarchy: Was ISSUE → Now PASS (74% is 3x larger than 4%)
   ```
5. If visual_spec_match or any principle still fails, go back to Phase 2

---

## Output Required (JSON)
After completing ALL phases, output:
```json
{{
  "visual_spec_checklist": {{
    "description_summary": "Brief summary of the visual_cue description",
    "elements_checked": [
      {{"element": "Element 1 from spec", "present": true, "notes": "How it's implemented"}},
      {{"element": "Element 2 from spec", "present": false, "notes": "MISSING - needs to be added"}},
      {{"element": "Animation X from spec", "present": true, "notes": "Visible at frame Y"}}
    ],
    "all_elements_present": false,
    "spec_compliance_score": "3/5 elements present"
  }},
  "beats_inspected": [
    {{
      "beat_index": 0,
      "frame": 0,
      "principle_checklist": {{
        "show_dont_tell": "PASS",
        "animation_reveals": "PASS",
        "progressive_disclosure": "PASS",
        "text_complements": "ISSUE - text repeats narration",
        "visual_hierarchy": "PASS",
        "breathing_room": "PASS",
        "purposeful_motion": "PASS",
        "emotional_resonance": "ISSUE - no wow moment",
        "professional_polish": "PASS",
        "sync_with_narration": "PASS",
        "screen_space_utilization": "ISSUE - elements only use 20% of frame",
        "material_depth": "ISSUE - flat single-layer shadows, no bezel",
        "visual_spec_match": "ISSUE - missing 2 elements from spec"
      }},
      "issues_summary": ["text repeats narration", "no wow moment", "elements too small", "missing spec elements"]
    }}
  ],
  "total_beats_expected": {num_beats},
  "total_beats_inspected": <number>,
  "issues_found": [
    {{"beat_index": 0, "frame": 0, "principle_violated": "visual_spec_match", "description": "Missing: Element X, Animation Y from specification", "severity": "high"}},
    {{"beat_index": 0, "frame": 0, "principle_violated": "screen_space_utilization", "description": "...", "severity": "high"}}
  ],
  "fixes_applied": [
    {{"description": "Added Element X as specified in visual_cue", "file": "{scene_file}", "lines_changed": "10-25"}},
    {{"description": "Implemented Animation Y per specification", "file": "{scene_file}", "lines_changed": "30-45"}}
  ],
  "verification_results": [
    {{
      "frame": 0,
      "visual_spec_elements_now_present": ["Element X", "Animation Y"],
      "visual_spec_elements_still_missing": [],
      "principles_fixed": [
        {{"principle": "visual_spec_match", "was": "ISSUE", "now": "PASS", "evidence": "all 5 spec elements now present"}},
        {{"principle": "screen_space_utilization", "was": "ISSUE", "now": "PASS", "evidence": "elements now use 70% of frame"}}
      ],
      "principles_still_failing": []
    }}
  ],
  "verification_passed": true
}}
```

---

## START NOW
1. Navigate to: {remotion_url}
2. The scene starts at frame 0 - no navigation math needed
3. You MUST inspect ALL {num_beats} beats at frames: {beat_frames_list}
4. Begin Phase 1: Navigate to frame {first_beat_frame}, take screenshot, evaluate ALL 13 principles using the checklist.
"""



class ClaudeCodeVisualInspector:
    """
    Visual inspector that uses Claude Code with --chrome flag for browser-based inspection.

    This approach lets Claude Code directly see and interact with Remotion Studio,
    eliminating the need for Playwright screenshot capture.
    """

    def __init__(
        self,
        project: Project,
        verbose: bool = True,
        timeout: int = 900,  # 15 minutes for full inspection
        live_output: bool = False,  # Stream Claude Code output in real-time
    ):
        """
        Initialize the Claude Code visual inspector.

        Args:
            project: The project to refine.
            verbose: Whether to print progress messages.
            timeout: Timeout in seconds for Claude Code subprocess.
            live_output: Whether to stream Claude Code output in real-time.
        """
        self.project = project
        self.verbose = verbose
        self.timeout = timeout
        self.live_output = live_output
        self.validator = ProjectValidator(project)

        # Set up beat parser with Claude Code provider for quality beat detection
        self.beat_parser = BeatParser(working_dir=project.root_dir)

        # Find repo root (where .claude/skills is located)
        # Projects are at: repo_root/projects/{project_id}/
        self.repo_root = self.project.root_dir.parent.parent
        if not (self.repo_root / ".claude").exists():
            # Fallback to project root if .claude not found at expected location
            self.repo_root = self.project.root_dir

    def _log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def refine_scene(self, scene_index: int) -> SceneRefinementResult:
        """
        Refine a single scene using Claude Code with browser access.

        Args:
            scene_index: Zero-based index of the scene to refine.

        Returns:
            SceneRefinementResult with details of the refinement.
        """
        # Get scene information
        try:
            scene_info = self.validator.get_scene_info(scene_index)
        except ValueError as e:
            return SceneRefinementResult(
                scene_id=f"scene{scene_index + 1}",
                scene_title="Unknown",
                scene_file=Path(""),
                error_message=str(e),
            )

        self._log(f"\n{'='*60}")
        self._log(f"Refining Scene {scene_index + 1}: {scene_info['title']}")
        self._log(f"{'='*60}")

        # Find the scene file
        scene_file = self._find_scene_file(scene_info)
        if not scene_file:
            return SceneRefinementResult(
                scene_id=scene_info["id"],
                scene_title=scene_info["title"],
                scene_file=Path(""),
                error_message=f"Could not find scene file for type: {scene_info.get('type', 'unknown')}",
            )

        result = SceneRefinementResult(
            scene_id=scene_info["id"],
            scene_title=scene_info["title"],
            scene_file=scene_file,
        )

        # Step 1: Parse narration into beats
        self._log("\n📝 Step 1: Parsing narration into beats...")
        beats = self._parse_beats(scene_info)
        result.beats = beats
        self._log(f"   Found {len(beats)} visual beats")

        for beat in beats:
            self._log(f"   Beat {beat.index + 1} [{beat.start_seconds:.1f}s-{beat.end_seconds:.1f}s]: {beat.text[:50]}...")

        # Step 2: Spawn Claude Code with --chrome for visual inspection
        self._log("\n🔍 Step 2: Starting visual inspection with Claude Code...")
        self._log("   (Claude Code will start Remotion, navigate frames, inspect visuals, and apply fixes)")

        inspection_result = self._run_claude_code_inspection(
            scene_info, scene_file, beats, scene_index
        )

        if inspection_result.get("error"):
            result.error_message = inspection_result["error"]
            return result

        # Parse results from Claude Code
        result.issues_found = self._parse_issues_from_result(inspection_result)
        result.fixes_applied = self._parse_fixes_from_result(inspection_result, scene_file)
        result.verification_passed = inspection_result.get("verification_passed", False)

        # Log summary
        if result.issues_found:
            self._log(f"\n   Found {len(result.issues_found)} issues:")
            for issue in result.issues_found:
                self._log(f"   - [{issue.severity.upper()}] Beat {issue.beat_index + 1}: {issue.description[:60]}...")

        if result.fixes_applied:
            applied_count = sum(1 for f in result.fixes_applied if f.status == FixStatus.APPLIED)
            self._log(f"\n   Applied {applied_count}/{len(result.fixes_applied)} fixes")

        if result.verification_passed:
            self._log("\n   ✅ Verification passed! Scene quality improved.")
        else:
            self._log("\n   ⚠️ Some issues may still remain. Consider another refinement pass.")

        return result

    def _find_scene_file(self, scene_info: dict) -> Optional[Path]:
        """Find the scene component file."""
        scenes_dir = self.project.root_dir / "scenes"
        if not scenes_dir.exists():
            return None

        scene_type = scene_info.get("type", "")
        if "/" in scene_type:
            scene_name = scene_type.split("/")[-1]
        else:
            scene_name = scene_type

        # Try various naming conventions
        patterns = [
            f"*{scene_name}*.tsx",
            f"*{scene_name.replace('_', '')}*.tsx",
            f"*{scene_name.replace('_', '-')}*.tsx",
        ]

        for pattern in patterns:
            matches = list(scenes_dir.glob(pattern))
            if matches:
                return matches[0]

        # Fallback: search all .tsx files for matching content
        for tsx_file in scenes_dir.glob("*.tsx"):
            if scene_name.lower().replace("_", "") in tsx_file.stem.lower():
                return tsx_file

        return None

    def _parse_beats(self, scene_info: dict) -> list[Beat]:
        """Parse narration into visual beats."""
        narration = scene_info.get("narration", "")
        duration = scene_info.get("duration_seconds", 30)

        if not narration:
            # Create a single beat for the whole scene
            return [
                Beat(
                    index=0,
                    start_seconds=0,
                    end_seconds=duration,
                    text="Full scene",
                    expected_visual="Scene content",
                )
            ]

        return self.beat_parser.parse(narration, duration)

    def _ensure_remotion_running(self) -> bool:
        """
        Ensure Remotion dev server is running.

        Returns:
            True if Remotion is running (or was started), False if failed to start.
        """
        # Check if already running
        if check_remotion_running(REMOTION_BASE_URL):
            self._log("   ✓ Remotion is already running")
            return True

        # Try to start it
        self._log("   Starting Remotion dev server...")

        # Find remotion directory - it's at the repo root, not inside the project
        # Projects are typically at: repo_root/projects/{project_id}/
        # Remotion is at: repo_root/remotion/
        remotion_dir = self.project.root_dir.parent.parent / "remotion"
        if not remotion_dir.exists() or not (remotion_dir / "package.json").exists():
            # Fallback: try inside project (some setups)
            remotion_dir = self.project.root_dir / "remotion"
        if not remotion_dir.exists() or not (remotion_dir / "package.json").exists():
            self._log("   ✗ Could not find remotion directory with package.json")
            return False

        # Build environment with PROJECT variable
        env = subprocess.os.environ.copy()
        env["PROJECT"] = self.project.id

        try:
            # Start Remotion in background
            process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(remotion_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for it to start (up to 15 seconds)
            for i in range(15):
                time.sleep(1)
                if check_remotion_running(REMOTION_BASE_URL):
                    self._log(f"   ✓ Remotion started (took {i+1}s)")
                    return True

            # Failed to start
            process.terminate()
            self._log("   ✗ Remotion failed to start within 15 seconds")
            return False

        except Exception as e:
            self._log(f"   ✗ Failed to start Remotion: {e}")
            return False

    def _run_claude_code_inspection(
        self,
        scene_info: dict,
        scene_file: Path,
        beats: list[Beat],
        scene_index: int,
    ) -> dict:
        """
        Run Claude Code subprocess with --chrome flag for visual inspection.

        Returns:
            Dictionary with inspection results or error.
        """
        import urllib.parse

        fps = self.project.video.fps
        duration_seconds = scene_info["duration_seconds"]
        total_frames = int(duration_seconds * fps)

        # Step 1: Ensure Remotion is running
        if not self._ensure_remotion_running():
            return {"error": "Failed to start Remotion. Please start it manually with: cd remotion && npm run dev"}

        # Step 2: Build SingleScenePlayer URL with scene props
        # Scene type is like "thinking-models/beyond_linear_thinking"
        scene_type = scene_info.get("type", f"{self.project.id}/{scene_info['id']}")
        # Use Remotion's native ?props={JSON} format so calculateMetadata receives correct duration
        props_json = json.dumps({
            "sceneType": scene_type,
            "durationInSeconds": duration_seconds,
        })
        remotion_url = (
            f"{SINGLE_SCENE_PLAYER_URL}"
            f"?props={urllib.parse.quote(props_json)}"
        )

        # Build beats info - frames are now RELATIVE to scene start (frame 0)
        beats_info_lines = []
        frame_numbers = []
        for beat in beats:
            # Frame relative to scene start (not absolute)
            beat_frame = int(beat.mid_seconds * fps)
            frame_numbers.append(str(beat_frame))
            beats_info_lines.append(
                f"Beat {beat.index + 1} @ frame {beat_frame}:\n"
                f"  \"{beat.text[:80]}{'...' if len(beat.text) > 80 else ''}\""
            )

        beats_info = "\n".join(beats_info_lines)
        beat_frames_list = ", ".join(frame_numbers) if frame_numbers else "0"

        # Format visual_cue section from script.json
        visual_cue = scene_info.get("visual_cue")
        if visual_cue:
            visual_cue_lines = [
                f"**Description:** {visual_cue.get('description', 'N/A')}",
                "",
                "**Required Elements (ALL MUST BE PRESENT):**",
            ]
            elements = visual_cue.get("elements", [])
            for i, element in enumerate(elements, 1):
                visual_cue_lines.append(f"  {i}. [ ] {element}")
            visual_cue_lines.append("")
            visual_cue_lines.append(f"**Total elements to implement: {len(elements)}**")
            visual_cue_lines.append("**You MUST verify ALL elements are present. Any missing = FAIL.**")
            visual_cue_section = "\n".join(visual_cue_lines)
        else:
            visual_cue_section = "(No visual specification found in script.json)"

        # Build the prompt
        first_beat_frame = frame_numbers[0] if frame_numbers else "0"
        prompt = CLAUDE_CODE_VISUAL_INSPECTION_PROMPT.format(
            remotion_url=remotion_url,
            scene_number=scene_index + 1,
            scene_title=scene_info["title"],
            scene_file=scene_file,
            duration_seconds=duration_seconds,
            total_frames=total_frames,
            num_beats=len(beats),
            narration_text=scene_info.get("narration", "")[:500],
            beats_info=beats_info,
            beat_frames_list=beat_frames_list,
            first_beat_frame=first_beat_frame,
            principles=format_principles_for_prompt(),
            visual_cue_section=visual_cue_section,
        )

        # Write prompt to temp file for claude code
        prompt_file = Path(tempfile.mktemp(suffix=".md", prefix="refine_prompt_"))
        prompt_file.write_text(prompt)

        try:
            # Build command - use streaming JSON for live output
            cmd = [
                "claude",
                "--chrome",  # Enable browser access
                "--dangerously-skip-permissions",  # Skip permission prompts for automation
                "-p", str(prompt),  # Pass prompt directly
            ]

            if self.live_output:
                # Use stream-json for real-time output (requires --print --verbose)
                cmd.extend(["--print", "--verbose", "--output-format", "stream-json"])
                return self._run_with_streaming(cmd)
            else:
                # Use --print for standard output
                cmd.append("--print")
                return self._run_standard(cmd)

        except FileNotFoundError:
            return {"error": "Claude Code CLI not found. Make sure 'claude' is installed and in PATH."}
        except Exception as e:
            return {"error": f"Failed to run Claude Code: {e}"}
        finally:
            # Clean up temp file
            if prompt_file.exists():
                prompt_file.unlink()

    def _run_standard(self, cmd: list) -> dict:
        """Run Claude Code with standard output capture."""
        self._log(f"   Running: claude --chrome --print ...")

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),  # Run from repo root so Claude Code can find .claude/skills
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            return {
                "error": f"Claude Code failed: {result.stderr}",
                "stdout": result.stdout,
            }

        return self._parse_claude_code_output(result.stdout)

    def _run_with_streaming(self, cmd: list) -> dict:
        """Run Claude Code with streaming JSON output for real-time display."""
        import threading

        self._log(f"   Running: claude --chrome --print --verbose --output-format stream-json ...")
        self._log("\n" + "=" * 60)
        self._log("CLAUDE CODE OUTPUT (live)")
        self._log("=" * 60 + "\n")

        process = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),  # Run from repo root so Claude Code can find .claude/skills
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Accumulators
        all_output = []
        final_result_text = ""

        def read_stderr(proc, lines):
            """Read stderr in background."""
            try:
                for line in iter(proc.stderr.readline, ''):
                    if line:
                        lines.append(line)
                        print(f"[stderr] {line}", end='', flush=True)
            except Exception:
                pass

        stderr_lines = []
        stderr_thread = threading.Thread(target=read_stderr, args=(process, stderr_lines))
        stderr_thread.daemon = True
        stderr_thread.start()

        try:
            # Process streaming JSON output line by line
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                all_output.append(line)

                # Parse the streaming JSON event
                try:
                    event = json.loads(line.strip())
                    event_type = event.get("type", "")

                    # Handle different event types
                    if event_type == "assistant":
                        # Assistant message with content
                        message = event.get("message", {})
                        content = message.get("content", [])
                        for block in content:
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                print(text, flush=True)
                                final_result_text += text
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                print(f"\n[Tool: {tool_name}]", flush=True)

                    elif event_type == "content_block_delta":
                        # Incremental text delta
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            print(text, end='', flush=True)
                            final_result_text += text

                    elif event_type == "result":
                        # Final result
                        result_text = event.get("result", "")
                        if result_text and result_text not in final_result_text:
                            final_result_text = result_text

                except json.JSONDecodeError:
                    # Not JSON, print as-is
                    print(line, end='', flush=True)
                    final_result_text += line

            # Wait for process to complete
            process.wait(timeout=30)
            stderr_thread.join(timeout=5)

        except subprocess.TimeoutExpired:
            process.kill()
            stderr_thread.join(timeout=1)

            self._log("\n" + "=" * 60)
            self._log("TIMEOUT - Claude Code did not complete in time")
            self._log("=" * 60)

            return {
                "error": f"Claude Code timed out after {self.timeout} seconds",
                "partial_output": final_result_text,
            }

        self._log("\n" + "=" * 60)
        self._log("CLAUDE CODE OUTPUT (end)")
        self._log("=" * 60 + "\n")

        if process.returncode != 0:
            stderr = ''.join(stderr_lines)
            return {
                "error": f"Claude Code failed with exit code {process.returncode}: {stderr}",
                "stdout": final_result_text,
            }

        return self._parse_claude_code_output(final_result_text)

    def _parse_claude_code_output(self, output: str) -> dict:
        """Parse JSON results from Claude Code output."""
        import re

        # Try to find JSON block in output
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, output)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_obj_pattern = r"\{[\s\S]*\"issues_found\"[\s\S]*\}"
        match = re.search(json_obj_pattern, output)

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # If no JSON found, return empty result with the raw output
        return {
            "issues_found": [],
            "fixes_applied": [],
            "verification_passed": False,
            "raw_output": output,
        }

    def _parse_issues_from_result(self, result: dict) -> list[Issue]:
        """Parse issues from Claude Code result."""
        issues = []
        raw_issues = result.get("issues_found", [])

        for raw_issue in raw_issues:
            # Ensure beat_index is an int (Claude may return string)
            raw_beat_index = raw_issue.get("beat_index", 0)
            try:
                beat_index = int(raw_beat_index)
            except (ValueError, TypeError):
                beat_index = 0
            principle_code = raw_issue.get("principle_violated", "other")

            try:
                issue_type = IssueType(principle_code)
            except ValueError:
                issue_type = IssueType.OTHER

            # Ensure description and severity are strings
            description = str(raw_issue.get("description", ""))
            severity = str(raw_issue.get("severity", "medium"))

            issues.append(
                Issue(
                    beat_index=beat_index,
                    principle_violated=issue_type,
                    description=description,
                    severity=severity,
                )
            )

        return issues

    def _parse_fixes_from_result(self, result: dict, scene_file: Path) -> list[Fix]:
        """Parse fixes from Claude Code result."""
        fixes = []
        raw_fixes = result.get("fixes_applied", [])

        for raw_fix in raw_fixes:
            # Ensure beat_index is an int (Claude may return string)
            raw_beat_index = raw_fix.get("beat_index", 0)
            try:
                beat_index = int(raw_beat_index)
            except (ValueError, TypeError):
                beat_index = 0

            # Create a placeholder issue for the fix
            issue = Issue(
                beat_index=beat_index,
                principle_violated=IssueType.OTHER,
                description="Issue addressed by fix",
                severity="medium",
            )

            fixes.append(
                Fix(
                    issue=issue,
                    file_path=scene_file,
                    description=raw_fix.get("description", "Fix applied"),
                    code_change=raw_fix.get("lines_changed", ""),
                    status=FixStatus.APPLIED,
                )
            )

        return fixes


class VisualInspector:
    """
    Orchestrates the visual refinement process.

    Coordinates:
    1. Beat parsing from narration
    2. Screenshot capture at key moments
    3. AI analysis against quality principles
    4. Fix generation and application
    5. Verification of improvements
    """

    def __init__(
        self,
        project: Project,
        llm_provider: Optional[LLMProvider] = None,
        screenshots_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Initialize the visual inspector.

        Args:
            project: The project to refine.
            llm_provider: LLM provider for AI analysis. If None, creates ClaudeCodeLLMProvider.
            screenshots_dir: Directory for screenshots. If None, uses temp directory.
            verbose: Whether to print progress messages.
        """
        self.project = project
        self.verbose = verbose
        self.validator = ProjectValidator(project)

        # Set up screenshots directory
        if screenshots_dir:
            self.screenshots_dir = Path(screenshots_dir)
        else:
            self.screenshots_dir = Path(tempfile.mkdtemp(prefix="refine_screenshots_"))
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Set up LLM provider
        if llm_provider is None:
            from ...config import load_config
            self.llm = get_llm_provider(load_config())
        else:
            self.llm = llm_provider

        # Set up beat parser
        self.beat_parser = BeatParser(llm_provider=llm_provider)

    def _log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def refine_scene(self, scene_index: int) -> SceneRefinementResult:
        """
        Refine a single scene through the full visual inspection process.

        Args:
            scene_index: Zero-based index of the scene to refine.

        Returns:
            SceneRefinementResult with details of the refinement.
        """
        # Get scene information
        try:
            scene_info = self.validator.get_scene_info(scene_index)
        except ValueError as e:
            return SceneRefinementResult(
                scene_id=f"scene{scene_index + 1}",
                scene_title="Unknown",
                scene_file=Path(""),
                error_message=str(e),
            )

        self._log(f"\n{'='*60}")
        self._log(f"Refining Scene {scene_index + 1}: {scene_info['title']}")
        self._log(f"{'='*60}")

        # Find the scene file
        scene_file = self._find_scene_file(scene_info)
        if not scene_file:
            return SceneRefinementResult(
                scene_id=scene_info["id"],
                scene_title=scene_info["title"],
                scene_file=Path(""),
                error_message=f"Could not find scene file for type: {scene_info['type']}",
            )

        result = SceneRefinementResult(
            scene_id=scene_info["id"],
            scene_title=scene_info["title"],
            scene_file=scene_file,
        )

        # Step 1: Parse narration into beats
        self._log("\n📝 Step 1: Parsing narration into beats...")
        beats = self._parse_beats(scene_info)
        result.beats = beats
        self._log(f"   Found {len(beats)} visual beats")

        for beat in beats:
            self._log(f"   Beat {beat.index + 1} [{beat.start_seconds:.1f}s-{beat.end_seconds:.1f}s]: {beat.text[:50]}...")

        # Step 2: Capture screenshots
        self._log("\n📸 Step 2: Capturing screenshots...")
        screenshots = self._capture_screenshots(beats, scene_info, scene_index)

        if not screenshots:
            result.error_message = "Failed to capture screenshots"
            return result

        self._log(f"   Captured {len(screenshots)} screenshots")

        # Step 3: Analyze screenshots
        self._log("\n🔍 Step 3: Analyzing visuals against quality principles...")
        issues = self._analyze_screenshots(screenshots, beats, scene_info, scene_file)
        result.issues_found = issues

        if not issues:
            self._log("   ✅ No issues found! Scene meets quality bar.")
            result.verification_passed = True
            return result

        self._log(f"   Found {len(issues)} issues:")
        for issue in issues:
            self._log(f"   - [{issue.severity.upper()}] Beat {issue.beat_index + 1}: {issue.description[:60]}...")

        # Step 4: Generate and apply fixes
        self._log("\n🔧 Step 4: Generating and applying fixes...")
        fixes = self._apply_fixes(issues, scene_info, scene_file)
        result.fixes_applied = fixes

        applied_count = sum(1 for f in fixes if f.status == FixStatus.APPLIED)
        self._log(f"   Applied {applied_count}/{len(fixes)} fixes")

        # Step 5: Verify improvements
        self._log("\n✅ Step 5: Verifying improvements...")
        verification_passed = self._verify_fixes(
            screenshots, beats, scene_info, scene_index, issues, fixes
        )
        result.verification_passed = verification_passed

        if verification_passed:
            self._log("   ✅ Verification passed! Scene quality improved.")
        else:
            self._log("   ⚠️ Some issues may still remain. Consider another refinement pass.")

        return result

    def _find_scene_file(self, scene_info: dict) -> Optional[Path]:
        """Find the scene component file."""
        scenes_dir = self.project.root_dir / "scenes"
        if not scenes_dir.exists():
            return None

        scene_type = scene_info.get("type", "")
        if "/" in scene_type:
            scene_name = scene_type.split("/")[-1]
        else:
            scene_name = scene_type

        # Try various naming conventions
        patterns = [
            f"*{scene_name}*.tsx",
            f"*{scene_name.replace('_', '')}*.tsx",
            f"*{scene_name.replace('_', '-')}*.tsx",
        ]

        for pattern in patterns:
            matches = list(scenes_dir.glob(pattern))
            if matches:
                return matches[0]

        # Fallback: search all .tsx files for matching content
        for tsx_file in scenes_dir.glob("*.tsx"):
            if scene_name.lower().replace("_", "") in tsx_file.stem.lower():
                return tsx_file

        return None

    def _parse_beats(self, scene_info: dict) -> list[Beat]:
        """Parse narration into visual beats."""
        narration = scene_info.get("narration", "")
        duration = scene_info.get("duration_seconds", 30)

        if not narration:
            # Create a single beat for the whole scene
            return [
                Beat(
                    index=0,
                    start_seconds=0,
                    end_seconds=duration,
                    text="Full scene",
                    expected_visual="Scene content",
                )
            ]

        return self.beat_parser.parse(narration, duration)

    def _capture_screenshots(
        self,
        beats: list[Beat],
        scene_info: dict,
        scene_index: int,
    ) -> list[CapturedScreenshot]:
        """Capture screenshots for all beats."""
        # Check if Remotion is running
        if not check_remotion_running():
            self._log("   ⚠️ Remotion Studio not running. Using mock screenshots.")
            capture = MockScreenshotCapture(self.screenshots_dir, self.project.video.fps)
        elif not PLAYWRIGHT_AVAILABLE:
            self._log("   ⚠️ Playwright not available. Using mock screenshots.")
            capture = MockScreenshotCapture(self.screenshots_dir, self.project.video.fps)
        else:
            capture = ScreenshotCapture(
                self.screenshots_dir,
                fps=self.project.video.fps,
                headless=True,
            )

        start_frame = scene_info["start_frame"]

        try:
            with capture:
                return capture.capture_beats(beats, start_frame, scene_index)
        except Exception as e:
            self._log(f"   ❌ Screenshot capture failed: {e}")
            return []

    def _analyze_screenshots(
        self,
        screenshots: list[CapturedScreenshot],
        beats: list[Beat],
        scene_info: dict,
        scene_file: Path,
    ) -> list[Issue]:
        """Analyze screenshots using AI."""
        # Build beats info string
        beats_info_lines = []
        for i, (beat, screenshot) in enumerate(zip(beats, screenshots)):
            beats_info_lines.append(
                f"Beat {i + 1} [{beat.start_seconds:.1f}s - {beat.end_seconds:.1f}s]:\n"
                f"  Narration: \"{beat.text}\"\n"
                f"  Expected visual: {beat.expected_visual}\n"
                f"  Screenshot file: {screenshot.path}"
            )

        beats_info = "\n\n".join(beats_info_lines)

        prompt = VISUAL_ANALYSIS_PROMPT_TEMPLATE.format(
            scene_title=scene_info["title"],
            scene_index=scene_info["id"],
            scene_file=scene_file,
            duration_seconds=scene_info["duration_seconds"],
            narration_text=scene_info["narration"],
            beats_info=beats_info,
            principles=format_principles_for_prompt(),
        )

        try:
            # Use generate_with_file_access so Claude can read the screenshots
            result = self.llm.generate_with_file_access(
                prompt,
                VISUAL_ANALYSIS_SYSTEM_PROMPT,
                allow_writes=False,  # Read only for analysis
            )

            if not result.success:
                self._log(f"   ❌ Analysis failed: {result.error_message}")
                return []

            # Parse the response
            response_data = self._parse_json_from_response(result.response)
            return self._parse_issues(response_data, screenshots)

        except Exception as e:
            self._log(f"   ❌ Analysis error: {e}")
            return []

    def _parse_issues(
        self, response_data: dict, screenshots: list[CapturedScreenshot]
    ) -> list[Issue]:
        """Parse issues from AI response."""
        issues = []
        raw_issues = response_data.get("issues", [])

        for raw_issue in raw_issues:
            beat_index = raw_issue.get("beat_index", 0)
            principle_code = raw_issue.get("principle_violated", "other")

            try:
                issue_type = IssueType(principle_code)
            except ValueError:
                issue_type = IssueType.OTHER

            # Find corresponding screenshot
            screenshot_path = None
            for s in screenshots:
                if s.beat_index == beat_index:
                    screenshot_path = s.path
                    break

            issues.append(
                Issue(
                    beat_index=beat_index,
                    principle_violated=issue_type,
                    description=raw_issue.get("description", ""),
                    severity=raw_issue.get("severity", "medium"),
                    screenshot_path=screenshot_path,
                )
            )

        return issues

    def _apply_fixes(
        self,
        issues: list[Issue],
        scene_info: dict,
        scene_file: Path,
    ) -> list[Fix]:
        """Generate and apply fixes for issues."""
        if not issues:
            return []

        # Build issues description
        issues_desc_lines = []
        for i, issue in enumerate(issues):
            issues_desc_lines.append(
                f"{i + 1}. [{issue.severity.upper()}] Beat {issue.beat_index + 1} - "
                f"{issue.principle_violated.value}:\n"
                f"   {issue.description}"
            )

        issues_description = "\n\n".join(issues_desc_lines)

        prompt = FIX_GENERATION_PROMPT_TEMPLATE.format(
            scene_file=scene_file,
            scene_title=scene_info["title"],
            issues_description=issues_description,
            principles=format_principles_for_prompt(),
        )

        try:
            # Use generate_with_file_access with writes enabled
            result = self.llm.generate_with_file_access(
                prompt,
                VISUAL_ANALYSIS_SYSTEM_PROMPT,
                allow_writes=True,
            )

            fixes = []
            for issue in issues:
                fix = Fix(
                    issue=issue,
                    file_path=scene_file,
                    description="AI-generated fix",
                    code_change=result.response[:500] if result.response else "",
                    status=FixStatus.APPLIED if result.success else FixStatus.FAILED,
                    error_message=result.error_message if not result.success else None,
                )
                fixes.append(fix)

            return fixes

        except Exception as e:
            self._log(f"   ❌ Fix generation error: {e}")
            return [
                Fix(
                    issue=issue,
                    file_path=scene_file,
                    description="Fix failed",
                    code_change="",
                    status=FixStatus.FAILED,
                    error_message=str(e),
                )
                for issue in issues
            ]

    def _verify_fixes(
        self,
        original_screenshots: list[CapturedScreenshot],
        beats: list[Beat],
        scene_info: dict,
        scene_index: int,
        original_issues: list[Issue],
        fixes: list[Fix],
    ) -> bool:
        """Verify that fixes improved the scene."""
        # Capture new screenshots
        new_screenshots = self._capture_screenshots(beats, scene_info, scene_index)

        if not new_screenshots:
            return False

        # Build verification prompt
        original_issues_desc = "\n".join(
            f"- Beat {i.beat_index + 1}: {i.description}"
            for i in original_issues
        )

        changes_desc = "\n".join(
            f"- {f.description}: {f.status.value}"
            for f in fixes
        )

        new_beats_info_lines = []
        for i, (beat, screenshot) in enumerate(zip(beats, new_screenshots)):
            new_beats_info_lines.append(
                f"Beat {i + 1}:\n  Screenshot file: {screenshot.path}"
            )

        new_beats_info = "\n".join(new_beats_info_lines)

        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            original_issues=original_issues_desc,
            changes_made=changes_desc,
            new_beats_info=new_beats_info,
        )

        try:
            result = self.llm.generate_with_file_access(
                prompt,
                VISUAL_ANALYSIS_SYSTEM_PROMPT,
                allow_writes=False,
            )

            if not result.success:
                return False

            response_data = self._parse_json_from_response(result.response)
            return response_data.get("verification_passed", False)

        except Exception as e:
            self._log(f"   ❌ Verification error: {e}")
            return False

    def _parse_json_from_response(self, response: str) -> dict:
        """Extract and parse JSON from LLM response."""
        import re

        # Try to find JSON in the response
        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, response)

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}


class MockVisualInspector(VisualInspector):
    """Mock visual inspector for testing."""

    def __init__(self, project: Project, verbose: bool = False):
        """Initialize with mock providers."""
        config = LLMConfig(provider="mock")
        mock_llm = MockLLMProvider(config)

        super().__init__(
            project=project,
            llm_provider=mock_llm,
            verbose=verbose,
        )

        self.beat_parser = MockBeatParser()

    def _capture_screenshots(
        self,
        beats: list[Beat],
        scene_info: dict,
        scene_index: int,
    ) -> list[CapturedScreenshot]:
        """Use mock screenshot capture."""
        capture = MockScreenshotCapture(self.screenshots_dir, self.project.video.fps)
        start_frame = scene_info["start_frame"]

        with capture:
            return capture.capture_beats(beats, start_frame, scene_index)

    def _analyze_screenshots(
        self,
        screenshots: list[CapturedScreenshot],
        beats: list[Beat],
        scene_info: dict,
        scene_file: Path,
    ) -> list[Issue]:
        """Return mock issues for testing."""
        return [
            Issue(
                beat_index=0,
                principle_violated=IssueType.VISUAL_HIERARCHY,
                description="Mock issue: visual hierarchy needs improvement",
                severity="medium",
                screenshot_path=screenshots[0].path if screenshots else None,
            )
        ]

    def _apply_fixes(
        self,
        issues: list[Issue],
        scene_info: dict,
        scene_file: Path,
    ) -> list[Fix]:
        """Return mock fixes for testing."""
        return [
            Fix(
                issue=issue,
                file_path=scene_file,
                description="Mock fix applied",
                code_change="// Mock code change",
                status=FixStatus.APPLIED,
            )
            for issue in issues
        ]

    def _verify_fixes(
        self,
        original_screenshots: list[CapturedScreenshot],
        beats: list[Beat],
        scene_info: dict,
        scene_index: int,
        original_issues: list[Issue],
        fixes: list[Fix],
    ) -> bool:
        """Mock verification always passes."""
        return True
