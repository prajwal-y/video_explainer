"""Custom scene generator for YouTube Shorts beats.

Generates custom React/Remotion components for each beat in a shorts storyboard.
Uses actual scene code from the full video as inspiration to create visually
stunning, concept-specific animations.

Key differences from full video:
- Canvas: 1080x1920 vertical (not 1920x1080)
- Visual area: Top 70% only (bottom 30% reserved for captions)
- Duration: 3-8s per beat (not 20-30s per scene)
- Animation: Fast, punchy (10-15 frame entries, not 30+)
- Layout: Vertical-first (stacked rows, not quadrants)
"""

import json
import re
from pathlib import Path
from typing import Any

from ..config import Config, LLMConfig, load_config
from ..models import Script
from ..understanding.llm_provider import ClaudeCodeLLMProvider, get_llm_provider
from .models import ShortsStoryboard, ShortsBeat


# =============================================================================
# SHORTS-SPECIFIC CONSTRAINTS
# =============================================================================
# These are the ONLY shorts-specific requirements that aren't in the Remotion skill.
# General Remotion knowledge (interpolate, spring, etc.) comes from the skill files.

SHORTS_CONSTRAINTS = """
## YouTube Shorts Constraints

These constraints are specific to vertical YouTube Shorts format.

### Canvas & Layout
- Canvas: 1080x1920 (vertical, mobile-first)
- Visual area: Top 70% only (0-1344px) - bottom 30% reserved for captions
- All content must fit within this visual area
- Use `const visualHeight = height * 0.7;` to calculate visual area height

### Dark Theme (MANDATORY for Shorts)
- Background: Always use `COLORS.backgroundGradient` from "./styles"
- Text: White (`COLORS.text`) for primary, gray (`COLORS.textDim`) for secondary
- Accents: Vibrant colors (cyan, orange, green, purple) that pop on dark backgrounds
- NEVER use light/white backgrounds

### Animation Speed (FAST & PUNCHY for Shorts)
- Entry animations: 10-15 frames (NOT 30+ like full videos)
- Spring config: `{ damping: 12, stiffness: 120 }`
- Stagger delays: 5-8 frames between elements
- Start animations 5-10 frames BEFORE the word is spoken

### Typography (LARGE for Mobile)
- Main text: 72-96px (scaled)
- Secondary text: 48-64px (scaled)
- Labels: 36-48px (scaled)
- Everything must be readable on a phone screen

### Required Imports
```typescript
import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig, spring } from "remotion";
import { COLORS, FONTS } from "./styles";
```

### Component Interface
```typescript
interface BeatNSceneProps {
  startFrame?: number;
}
```

### Phase-Based Animation (CRITICAL for Shorts)
Each shorts scene MUST have 2-3 phases synced with the voiceover:
1. Each phase shows a DIFFERENT visual matching what's being said
2. Phase boundaries are calculated from word timestamps (seconds * 30 = frames)
3. Phases crossfade smoothly using interpolate with adaptive fade times
4. Use `getFadeTime = (duration) => Math.min(10, Math.floor(duration / 3))` to prevent monotonic errors

### Color Palette (from ./styles)
- COLORS.background: "#0a0a0f"
- COLORS.backgroundGradient: "linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%)"
- COLORS.text: "#ffffff"
- COLORS.textDim: "#b0b0b0"
- COLORS.primary: "#00d4ff" (cyan)
- COLORS.secondary: "#ff6b35" (orange)
- COLORS.success: "#22c55e" (green)
- COLORS.purple: "#a855f7"

### What to Avoid
- Positioning elements in bottom 30% (reserved for captions)
- Light/white backgrounds (dark theme only)
- Static visuals without animation
- Same visual for entire beat (use phases!)
- Slow transitions (keep it punchy)
"""

# Default Remotion rules to load from the skill
DEFAULT_REMOTION_RULES = [
    "animations.md",
    "timing.md",
    "text-animations.md",
]


SHORTS_SCENE_GENERATION_PROMPT = """Generate a custom React/Remotion scene for this YouTube Shorts beat.

## Beat Information

**Beat ID**: {beat_id}
**Duration**: {duration:.2f}s = {total_frames} frames at 30fps

**Caption** (what's being spoken):
"{caption_text}"

**Visual Description** (from source video):
{visual_description}

## Word Timestamps
{word_timestamps_section}

## Example Scene(s) from Full Video

Study these actual scene files for inspiration. Adapt their visual patterns for vertical format (1080x1920):

{example_scenes}

## Your Task

Create a component named `{component_name}` that:
1. SHOWS the concept from the caption (don't just display text)
2. Uses visual patterns inspired by the example scenes above
3. Follows all constraints (vertical layout, fast animations, large typography)
4. Syncs animations to the word timestamps

## Output

Return ONLY the TypeScript/React code. No markdown code blocks, no explanation - just the code.
The component should be saved to: {output_path}
"""


# Index template for custom beat scenes
SHORTS_SCENES_INDEX_TEMPLATE = '''/**
 * Custom Scene Registry for YouTube Shorts
 *
 * Auto-generated index mapping beat IDs to custom scene components.
 */

import React from "react";

{imports}

export type BeatSceneComponent = React.FC<{{ startFrame?: number }}>;

/**
 * Registry of custom beat scene components.
 * Keys match beat IDs in the storyboard (e.g., "beat_1", "beat_2", etc.)
 */
const BEAT_SCENE_REGISTRY: Record<string, BeatSceneComponent> = {{
{registry_entries}
}};

// Default export for easy importing
export default BEAT_SCENE_REGISTRY;

{exports}

export function getBeatScene(beatId: string): BeatSceneComponent | undefined {{
  return BEAT_SCENE_REGISTRY[beatId];
}}

export function getAvailableBeatScenes(): string[] {{
  return Object.keys(BEAT_SCENE_REGISTRY);
}}
'''


class ShortsCustomSceneGenerator:
    """Generates custom React scene components for YouTube Shorts beats."""

    MAX_RETRIES = 2

    def __init__(
        self,
        config: Config | None = None,
        working_dir: Path | None = None,
        timeout: int = 180,
        remotion_skill_path: Path | None = None,
    ):
        """Initialize the generator.

        Args:
            config: Configuration object.
            working_dir: Working directory for Claude Code.
            timeout: Timeout for LLM calls in seconds.
            remotion_skill_path: Path to Remotion skill rules directory.
                Defaults to .claude/skills/remotion/rules/ in working directory.
        """
        self.config = config or load_config()
        self.working_dir = working_dir or Path.cwd()
        self.timeout = timeout

        # Find Remotion skill rules directory
        if remotion_skill_path:
            self.remotion_skill_path = remotion_skill_path
        else:
            # Look for skill in working directory or project root
            candidates = [
                self.working_dir / ".claude" / "skills" / "remotion" / "rules",
                Path(__file__).parent.parent.parent / ".claude" / "skills" / "remotion" / "rules",
            ]
            self.remotion_skill_path = next(
                (p for p in candidates if p.exists()),
                None
            )

        # Cache for loaded rules
        self._remotion_rules_cache: dict[str, str] = {}

    def _load_remotion_rules(
        self,
        rule_files: list[str] | None = None,
    ) -> str:
        """Load Remotion skill rules from the skill directory.

        Args:
            rule_files: List of rule file names to load (e.g., ["animations.md", "timing.md"]).
                If None, uses DEFAULT_REMOTION_RULES.

        Returns:
            Combined content of all loaded rules.
        """
        if rule_files is None:
            rule_files = DEFAULT_REMOTION_RULES

        if not self.remotion_skill_path or not self.remotion_skill_path.exists():
            return "<!-- Remotion skill rules not found. Using shorts constraints only. -->"

        loaded_rules = []
        for rule_file in rule_files:
            # Check cache first
            if rule_file in self._remotion_rules_cache:
                loaded_rules.append(self._remotion_rules_cache[rule_file])
                continue

            rule_path = self.remotion_skill_path / rule_file
            if rule_path.exists():
                content = rule_path.read_text()
                # Cache the content
                self._remotion_rules_cache[rule_file] = content
                loaded_rules.append(f"### {rule_file}\n\n{content}")

        return "\n\n---\n\n".join(loaded_rules) if loaded_rules else "<!-- No Remotion rules loaded. -->"

    def _build_system_prompt(
        self,
        rule_files: list[str] | None = None,
    ) -> str:
        """Build the system prompt by combining Remotion rules with shorts constraints.

        Args:
            rule_files: List of Remotion rule files to load.

        Returns:
            Complete system prompt for scene generation.
        """
        remotion_rules = self._load_remotion_rules(rule_files)

        return f"""You are an expert React/Remotion developer creating animated scene components for YouTube Shorts.

## Remotion Best Practices

The following are Remotion best practices for animations, timing, and visual styling:

{remotion_rules}

---

{SHORTS_CONSTRAINTS}

## Your Task

Create visually stunning scenes that:
1. SHOW the concept (don't just display text)
2. Use MOTION to tell the story
3. Make elements LARGE and BOLD (mobile-first)
4. Use the dark theme with vibrant accent colors
5. Sync phase transitions with the voiceover timing
"""

    def generate_all_scenes(
        self,
        storyboard: ShortsStoryboard,
        source_script: Script | None,
        scenes_dir: Path,
        project_scenes_dir: Path | None = None,
        word_timestamps: list | None = None,
    ) -> dict[str, Any]:
        """Generate custom scene components for all beats in the storyboard.

        Args:
            storyboard: The shorts storyboard with beats.
            source_script: Original script for visual context.
            scenes_dir: Directory to write generated scene files to.
            project_scenes_dir: Directory containing full video scene files for inspiration.
            word_timestamps: Word-level timestamps from voiceover.

        Returns:
            Dict with generation results.
        """
        scenes_dir.mkdir(parents=True, exist_ok=True)

        # Generate styles.ts first
        self._generate_vertical_styles(scenes_dir, storyboard.title)

        # Load example scenes from full video for inspiration
        example_scenes = self._load_example_scenes(project_scenes_dir)

        results = {
            "scenes_dir": str(scenes_dir),
            "scenes": [],
            "errors": [],
        }

        generated_components = []

        for beat in storyboard.beats:
            # Skip CTA beat - it uses pre-built component
            if beat.id == "cta":
                continue

            # Skip beats without visual_description (use fallback visuals)
            if not beat.visual_description:
                print(f"  ⚠ Beat {beat.id}: No visual description, using generic visual")
                continue

            try:
                # Get the specific source scene file if available
                beat_example_scenes = example_scenes
                if beat.source_scene_file and project_scenes_dir:
                    source_scene_path = project_scenes_dir / beat.source_scene_file
                    if source_scene_path.exists():
                        with open(source_scene_path) as f:
                            source_code = f.read()
                        # Prioritize the source scene, then add others
                        beat_example_scenes = f"### Source Scene ({beat.source_scene_file}) - PRIMARY INSPIRATION:\n```typescript\n{source_code[:6000]}\n```\n\n{example_scenes}"

                result = self._generate_beat_scene(
                    beat=beat,
                    storyboard=storyboard,
                    scenes_dir=scenes_dir,
                    example_scenes=beat_example_scenes,
                )
                results["scenes"].append(result)
                generated_components.append(result)
                print(f"  ✓ Generated {result['component_name']} for {beat.id}")

            except Exception as e:
                error = {"beat_id": beat.id, "error": str(e)}
                results["errors"].append(error)
                print(f"  ✗ Failed to generate scene for {beat.id}: {e}")

        # Generate index.ts for scene registry
        if generated_components:
            self._generate_index(scenes_dir, generated_components)
            print(f"  ✓ Generated scenes/index.ts with {len(generated_components)} scenes")

        return results

    def _load_example_scenes(self, project_scenes_dir: Path | None) -> str:
        """Load example scene files from the full video for inspiration.

        Args:
            project_scenes_dir: Directory containing full video scene files.

        Returns:
            Formatted string with example scene code.
        """
        if not project_scenes_dir or not project_scenes_dir.exists():
            return "No example scenes available. Create visuals based on the caption content."

        # Get scene files (exclude styles.ts, index.ts)
        scene_files = [
            f for f in project_scenes_dir.glob("*.tsx")
            if f.name not in ("index.tsx", "styles.tsx")
        ]

        if not scene_files:
            return "No example scenes available. Create visuals based on the caption content."

        # Load up to 2 scenes for inspiration (to avoid overwhelming the prompt)
        examples = []
        for scene_file in scene_files[:2]:
            with open(scene_file) as f:
                code = f.read()
            # Truncate if too long
            if len(code) > 5000:
                code = code[:5000] + "\n// ... (truncated)"
            examples.append(f"### {scene_file.name}:\n```typescript\n{code}\n```")

        return "\n\n".join(examples)

    def _generate_beat_scene(
        self,
        beat: ShortsBeat,
        storyboard: ShortsStoryboard,
        scenes_dir: Path,
        example_scenes: str,
    ) -> dict:
        """Generate a single beat scene component.

        Args:
            beat: The beat to generate a scene for.
            storyboard: The full storyboard for context.
            scenes_dir: Output directory.
            example_scenes: Example scene code for inspiration.

        Returns:
            Dict with scene generation result.
        """
        # Generate component name from beat ID
        beat_num = beat.id.replace("beat_", "")
        component_name = f"Beat{beat_num}Scene"
        filename = f"{component_name}.tsx"
        output_path = scenes_dir / filename

        # Calculate duration
        duration = beat.end_seconds - beat.start_seconds
        total_frames = int(duration * 30)

        # Format word timestamps
        word_timestamps_section = self._format_word_timestamps(
            beat.word_timestamps, beat.caption_text, duration
        )

        # Build prompt
        prompt = SHORTS_SCENE_GENERATION_PROMPT.format(
            beat_id=beat.id,
            duration=duration,
            total_frames=total_frames,
            caption_text=beat.caption_text,
            visual_description=beat.visual_description or "Create a visual that illustrates the caption concept.",
            word_timestamps_section=word_timestamps_section,
            example_scenes=example_scenes,
            component_name=component_name,
            output_path=output_path,
        )

        # Generate scene using LLM
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                self._generate_scene_file(prompt, output_path)

                # Update beat with component name
                beat.component_name = component_name

                return {
                    "beat_id": beat.id,
                    "component_name": component_name,
                    "filename": filename,
                    "path": str(output_path),
                }

            except Exception as e:
                last_error = str(e)
                print(f"    ⚠ Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")

        raise RuntimeError(f"Failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _generate_scene_file(self, prompt: str, output_path: Path) -> None:
        """Generate a scene file using Claude Code.

        Args:
            prompt: The generation prompt.
            output_path: Path to write the scene file.
        """
        llm = get_llm_provider(self.config)

        # Build system prompt dynamically from Remotion skill + shorts constraints
        system_prompt = self._build_system_prompt()

        if isinstance(llm, ClaudeCodeLLMProvider):
            full_prompt = f"""## FIRST: Load the Remotion skill
Before doing ANYTHING else, invoke the `/remotion` skill by using the Skill tool with skill="remotion".
This loads essential Remotion best practices for animations, timing, and visual patterns.
DO THIS NOW before proceeding.

---

{system_prompt}

{prompt}

Write the complete component code to the file: {output_path}
"""

            result = llm.generate_with_file_access(full_prompt, allow_writes=True)

            if not result.success:
                raise RuntimeError(f"LLM generation failed: {result.error_message}")

            # Verify file was created
            if not output_path.exists():
                code = self._extract_code(result.response)
                if code:
                    with open(output_path, "w") as f:
                        f.write(code)
                else:
                    raise RuntimeError(f"Scene file not created: {output_path}")
        else:
            full_prompt = f"""{system_prompt}

{prompt}

Write the complete component code as a single TypeScript/TSX code block.
"""
            response = llm.generate(full_prompt)
            code = self._extract_code(response)
            if not code:
                raise RuntimeError("LLM returned no extractable code")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(code)

    def _generate_vertical_styles(self, scenes_dir: Path, title: str) -> None:
        """Generate styles.ts for vertical shorts format.

        Args:
            scenes_dir: Directory to write styles.ts to.
            title: Title for the comment header.
        """
        styles_content = f'''/**
 * Shared Style Constants for YouTube Shorts: {title}
 *
 * Vertical format (1080x1920) with dark theme for shorts.
 */

import React from "react";

// ===== CANVAS DIMENSIONS =====
const CANVAS_WIDTH = 1080;
const CANVAS_HEIGHT = 1920;
const VISUAL_AREA_HEIGHT = 1344;  // Top 70%
const CAPTION_AREA_HEIGHT = 576;  // Bottom 30%

// ===== COLOR PALETTE (DARK THEME) =====
export const COLORS = {{
  // Background (dark)
  background: "#0a0a0f",
  backgroundGradient: "linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%)",
  surface: "#1a1a2e",
  surfaceAlt: "#252538",

  // Text (light on dark)
  text: "#ffffff",
  textDim: "#b0b0b0",
  textMuted: "#707070",

  // Accents (vibrant colors that pop on dark backgrounds)
  primary: "#00d4ff",
  primaryGlow: "#00d4ff60",
  secondary: "#ff6b35",
  secondaryGlow: "#ff6b3560",
  success: "#22c55e",
  successGlow: "#22c55e60",
  warning: "#f59e0b",
  error: "#ef4444",
  errorGlow: "#ef444460",
  purple: "#a855f7",
  purpleGlow: "#a855f760",
  cyan: "#00d4ff",
  cyanGlow: "#00d4ff60",

  // Borders (subtle on dark)
  border: "#3a3a4a",
  borderLight: "#2a2a3a",
}};

// ===== FONTS =====
export const FONTS = {{
  primary: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
  heading: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
  mono: '"SF Mono", Monaco, Consolas, monospace',
}};

// ===== LAYOUT SYSTEM =====
export const LAYOUT = {{
  canvas: {{
    width: CANVAS_WIDTH,
    height: CANVAS_HEIGHT,
  }},
  visualArea: {{
    top: 0,
    height: VISUAL_AREA_HEIGHT,
    centerY: VISUAL_AREA_HEIGHT / 2,
  }},
  captionArea: {{
    top: VISUAL_AREA_HEIGHT,
    height: CAPTION_AREA_HEIGHT,
  }},
  margin: {{
    horizontal: 60,
    top: 80,
    bottom: 60,
  }},
  content: {{
    width: CANVAS_WIDTH - 120,  // Minus horizontal margins
    height: VISUAL_AREA_HEIGHT - 140,  // Minus top/bottom margins
    centerX: CANVAS_WIDTH / 2,
    centerY: VISUAL_AREA_HEIGHT / 2,
  }},
}};

// ===== LAYOUT HELPERS =====

/**
 * Get centered position in the visual area
 */
export const getCenteredPosition = (): {{
  cx: number;
  cy: number;
  width: number;
  height: number;
}} => ({{
  cx: LAYOUT.content.centerX,
  cy: LAYOUT.content.centerY,
  width: LAYOUT.content.width,
  height: LAYOUT.content.height,
}});

/**
 * Get three-row layout (ideal for vertical format)
 */
export const getThreeRowLayout = (): {{
  top: {{ cx: number; cy: number; height: number }};
  middle: {{ cx: number; cy: number; height: number }};
  bottom: {{ cx: number; cy: number; height: number }};
}} => {{
  const rowHeight = LAYOUT.content.height / 3;
  const startY = LAYOUT.margin.top;
  return {{
    top: {{
      cx: LAYOUT.content.centerX,
      cy: startY + rowHeight / 2,
      height: rowHeight,
    }},
    middle: {{
      cx: LAYOUT.content.centerX,
      cy: startY + rowHeight + rowHeight / 2,
      height: rowHeight,
    }},
    bottom: {{
      cx: LAYOUT.content.centerX,
      cy: startY + rowHeight * 2 + rowHeight / 2,
      height: rowHeight,
    }},
  }};
}};

/**
 * Get two-row layout
 */
export const getTwoRowLayout = (): {{
  top: {{ cx: number; cy: number; height: number }};
  bottom: {{ cx: number; cy: number; height: number }};
}} => {{
  const rowHeight = LAYOUT.content.height / 2;
  const startY = LAYOUT.margin.top;
  return {{
    top: {{
      cx: LAYOUT.content.centerX,
      cy: startY + rowHeight / 2,
      height: rowHeight,
    }},
    bottom: {{
      cx: LAYOUT.content.centerX,
      cy: startY + rowHeight + rowHeight / 2,
      height: rowHeight,
    }},
  }};
}};

/**
 * Get style for centering an element at a position
 */
export const getCenteredStyle = (
  pos: {{ cx: number; cy: number }},
  scale: number
): React.CSSProperties => ({{
  position: "absolute",
  left: pos.cx * scale,
  top: pos.cy * scale,
  transform: "translate(-50%, -50%)",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
}});

// ===== ANIMATION HELPERS =====
export const ANIMATION = {{
  // Fast spring for punchy animations
  spring: {{ damping: 12, stiffness: 120 }},
  // Entry animation duration in frames
  entryDuration: 12,
  // Stagger delay between elements
  stagger: 6,
}};

export default {{ COLORS, FONTS, LAYOUT, ANIMATION }};
'''
        styles_path = scenes_dir / "styles.ts"
        with open(styles_path, "w") as f:
            f.write(styles_content)

    def _generate_index(self, scenes_dir: Path, components: list[dict]) -> None:
        """Generate the index.ts file for scene registry.

        Args:
            scenes_dir: Directory containing scene files.
            components: List of component info dicts.
        """
        imports = []
        exports = []
        registry_entries = []

        for comp in components:
            name = comp["component_name"]
            filename = comp["filename"].replace(".tsx", "")
            beat_id = comp["beat_id"]

            imports.append(f'import {{ {name} }} from "./{filename}";')
            exports.append(f'export {{ {name} }} from "./{filename}";')
            registry_entries.append(f'  "{beat_id}": {name},')

        content = SHORTS_SCENES_INDEX_TEMPLATE.format(
            imports="\n".join(imports),
            exports="\n".join(exports),
            registry_entries="\n".join(registry_entries),
        )

        index_path = scenes_dir / "index.ts"
        with open(index_path, "w") as f:
            f.write(content)

    def _format_word_timestamps(
        self,
        word_timestamps: list[dict],
        caption_text: str,
        duration: float,
    ) -> str:
        """Format word timestamps for the prompt with phase timing recommendations.

        Args:
            word_timestamps: List of {word, start_seconds, end_seconds} dicts.
            caption_text: The caption text.
            duration: Beat duration in seconds.

        Returns:
            Formatted string for inclusion in the prompt.
        """
        total_frames = int(duration * 30)

        if not word_timestamps:
            # Estimate 3 equal phases if no timestamps
            phase1_end = total_frames // 3
            phase2_end = (total_frames * 2) // 3
            return f"""
**Word Timestamps**: NOT AVAILABLE

**RECOMMENDED PHASE TIMING** (estimate based on duration):
- Phase 1: frames 0-{phase1_end} (first third of beat)
- Phase 2: frames {phase1_end}-{phase2_end} (middle third)
- Phase 3: frames {phase2_end}-{total_frames} (final third)

Use these phase boundaries in your code:
```typescript
const phase1End = {phase1_end};
const phase2End = {phase2_end};
const phase3End = {total_frames};
```
"""

        # Build detailed timeline with phrase groupings
        timeline_entries = []
        for wt in word_timestamps:
            word = wt.get("word", "")
            start = wt.get("start_seconds", 0)
            end = wt.get("end_seconds", start + 0.5)
            frame = int(start * 30)
            timeline_entries.append(f'  "{word}" @ {start:.2f}s (frame {frame})')

        # Identify natural phrase breaks and calculate phase timing
        # Look for pauses > 0.5s or punctuation-like breaks
        phrase_breaks = self._identify_phrase_breaks(word_timestamps, duration)

        phase_recommendations = self._calculate_phase_timing(
            phrase_breaks, word_timestamps, total_frames
        )

        return f"""
**Word Timestamps** (sync phase transitions to these):
{chr(10).join(timeline_entries)}

**RECOMMENDED PHASE TIMING** (based on natural phrase breaks):
{phase_recommendations}

IMPORTANT: Use these phase boundaries in your code. Each phase should show a DIFFERENT visual
that matches what's being said at that moment in the voiceover.
"""

    def _identify_phrase_breaks(
        self,
        word_timestamps: list[dict],
        duration: float,
    ) -> list[tuple[int, float, str]]:
        """Identify natural phrase breaks in the word timestamps.

        Args:
            word_timestamps: List of word timestamp dicts.
            duration: Total duration in seconds.

        Returns:
            List of (word_index, timestamp, reason) tuples for phrase breaks.
        """
        breaks = []

        for i in range(len(word_timestamps) - 1):
            current = word_timestamps[i]
            next_word = word_timestamps[i + 1]

            current_end = current.get("end_seconds", 0)
            next_start = next_word.get("start_seconds", 0)
            gap = next_start - current_end

            # Significant pause (> 0.7s) indicates phrase break
            if gap > 0.7:
                breaks.append((i, current_end, f"pause of {gap:.1f}s"))

        # If we found good breaks, return them
        if len(breaks) >= 1:
            return breaks

        # Otherwise, divide into roughly equal thirds based on word count
        n_words = len(word_timestamps)
        if n_words >= 6:
            third = n_words // 3
            breaks = [
                (third - 1, word_timestamps[third - 1].get("end_seconds", duration / 3), "equal division"),
                (2 * third - 1, word_timestamps[2 * third - 1].get("end_seconds", 2 * duration / 3), "equal division"),
            ]
        elif n_words >= 3:
            half = n_words // 2
            breaks = [
                (half - 1, word_timestamps[half - 1].get("end_seconds", duration / 2), "equal division"),
            ]

        return breaks

    def _calculate_phase_timing(
        self,
        phrase_breaks: list[tuple[int, float, str]],
        word_timestamps: list[dict],
        total_frames: int,
    ) -> str:
        """Calculate phase timing recommendations from phrase breaks.

        Args:
            phrase_breaks: List of (word_index, timestamp, reason) tuples.
            word_timestamps: Full word timestamps list.
            total_frames: Total frames in the beat.

        Returns:
            Formatted phase timing recommendation string.
        """
        if not phrase_breaks:
            # Single phase
            return f"""- Phase 1: frames 0-{total_frames} (entire beat)

```typescript
// Single phase - show one main visual
const phase1End = {total_frames};
```"""

        # Build phase descriptions
        phases = []
        prev_end_frame = 0
        prev_word_idx = -1

        for i, (word_idx, timestamp, reason) in enumerate(phrase_breaks):
            end_frame = int(timestamp * 30)
            # Get words in this phrase
            words_in_phase = [wt.get("word", "") for wt in word_timestamps[prev_word_idx + 1:word_idx + 1]]
            words_str = " ".join(words_in_phase[:5])  # First 5 words
            if len(words_in_phase) > 5:
                words_str += "..."

            phases.append({
                "num": i + 1,
                "start": prev_end_frame,
                "end": end_frame,
                "words": words_str,
                "reason": reason,
            })
            prev_end_frame = end_frame
            prev_word_idx = word_idx

        # Final phase
        remaining_words = [wt.get("word", "") for wt in word_timestamps[prev_word_idx + 1:]]
        words_str = " ".join(remaining_words[:5])
        if len(remaining_words) > 5:
            words_str += "..."
        phases.append({
            "num": len(phrase_breaks) + 1,
            "start": prev_end_frame,
            "end": total_frames,
            "words": words_str,
            "reason": "final phrase",
        })

        # Format output
        lines = []
        for p in phases:
            lines.append(f"- Phase {p['num']}: frames {p['start']}-{p['end']} \"{p['words']}\"")

        # Add code snippet
        if len(phases) == 2:
            lines.append(f"""
```typescript
// Phase timing synced with voiceover:
// Phase 1: "{phases[0]['words']}"
// Phase 2: "{phases[1]['words']}"
const phase1End = {phases[0]['end']};
const phase2End = {phases[1]['end']};
```""")
        elif len(phases) >= 3:
            lines.append(f"""
```typescript
// Phase timing synced with voiceover:
// Phase 1: "{phases[0]['words']}"
// Phase 2: "{phases[1]['words']}"
// Phase 3: "{phases[2]['words']}"
const phase1End = {phases[0]['end']};
const phase2End = {phases[1]['end']};
const phase3End = {phases[2]['end']};
```""")

        return "\n".join(lines)

    def _fix_imports(self, code: str) -> str:
        """Fix common import path issues in generated code.

        LLMs sometimes generate incorrect relative paths like '../styles'
        when all files are in the same directory.

        Args:
            code: Generated TypeScript code.

        Returns:
            Code with corrected import paths.
        """
        # Fix incorrect relative path to styles (should be ./styles, not ../styles)
        code = re.sub(r'from\s+["\']\.\.\/styles["\']', 'from "./styles"', code)
        code = re.sub(r'from\s+["\']\.\.\/\.\.\/styles["\']', 'from "./styles"', code)
        return code

    def _extract_code(self, response: str) -> str | None:
        """Extract TypeScript code from response.

        Args:
            response: LLM response text.

        Returns:
            Extracted code or None.
        """
        patterns = [
            r"```(?:typescript|tsx|ts)?\s*([\s\S]*?)```",
            r"```\s*([\s\S]*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                code = match.group(1).strip()
                return self._fix_imports(code)

        # If response looks like code, return as-is
        if "import" in response and "export" in response:
            return self._fix_imports(response.strip())

        return None
