"""
Script Analyzer - Phase 1: Gap Analysis.

Compares source material against current script to identify:
- Missing concepts that should be covered
- Shallow coverage of important topics
- Narrative gaps between scenes
- Suggested new scenes to address gaps
"""

import json
from pathlib import Path
from typing import Optional

from ...ingestion import parse_pdf
from ...project import Project
from ...understanding.llm_provider import ClaudeCodeLLMProvider, LLMProvider, get_llm_provider
from ..models import (
    AddBridgePatch,
    AddScenePatch,
    ConceptCoverage,
    ConceptDepth,
    ExpandScenePatch,
    GapAnalysisResult,
    NarrativeGap,
    ScriptPatch,
    SourceConcept,
    SuggestedScene,
)


# =============================================================================
# Prompts for Gap Analysis
# =============================================================================

CONCEPT_EXTRACTION_SYSTEM_PROMPT = """You are an expert technical writer analyzing source material for educational video production.

Your task is to extract ALL concepts from a document - be COMPREHENSIVE, not selective. Every topic, idea, and technical detail in the source should be captured. The filtering and prioritization happens later.

For each concept, assess its importance relative to the main topic:
- critical: Core concept that the video MUST cover - without this, the video is incomplete
- high: Important concept that significantly enhances understanding of the main topic
- medium: Supporting concept that adds depth or context
- low: Tangential detail, edge case, or advanced topic that could be deferred

DO NOT filter out concepts based on whether they can be "explained visually" - any concept can be explained through narration, diagrams, text, or animations. Your job is to capture everything.
"""

CONCEPT_EXTRACTION_PROMPT = """Analyze this source material and extract ALL concepts that appear in it.

## Source Material
{source_content}

## Instructions
Extract EVERY concept from this document - be comprehensive. Do not filter or skip concepts. For each concept, identify:
1. The concept name (short, descriptive)
2. A brief description of what it means
3. Its importance level (critical, high, medium, low) - relative to the main topic
4. Prerequisites (what concepts must be understood first)
5. The section or part of the document where this concept is primarily discussed

IMPORTANT: Include ALL concepts, even:
- Minor details and edge cases (mark as "low" importance)
- Tangential topics (mark as "low" importance)
- Advanced concepts (note prerequisites)
- Concepts that seem hard to visualize (they can be explained via narration)

The goal is COMPLETE coverage of the source material. Nothing should be silently dropped.

Respond with JSON in this exact format:
{{
    "concepts": [
        {{
            "name": "Concept Name",
            "description": "Brief explanation of the concept",
            "importance": "critical|high|medium|low",
            "prerequisites": ["prerequisite1", "prerequisite2"],
            "section_reference": "Section name or description"
        }}
    ],
    "main_topic": "The overarching topic of this document",
    "total_concepts_found": 10
}}
"""

GAP_ANALYSIS_SYSTEM_PROMPT = """You are an expert video script analyst comparing source material coverage against an existing script.

Your task is to account for EVERY concept from the source material:
1. Covered concepts: Note how deeply they're explained
2. Missing concepts that SHOULD be added: Generate patches to add them
3. Intentionally omitted concepts: Explicitly note WHY they're okay to skip

Nothing should silently disappear. Every concept must be either covered, flagged for addition, or explicitly marked as an acceptable omission with a clear reason.

Also identify narrative flow issues where the script jumps between topics without proper transitions."""

GAP_ANALYSIS_PROMPT = """Compare these source concepts against the current script. Account for EVERY concept.

## Source Concepts (extracted from source material)
{concepts_json}

## Current Script Scenes
{scenes_json}

## Instructions
For EACH source concept, determine ONE of the following:

1. **Covered adequately**: The script explains this concept well enough
   - depth: "explained" or "deep_dive"
   - Include which scene(s) cover it

2. **Needs improvement**: The concept is mentioned but not explained well enough
   - depth: "mentioned"
   - Generate a patch to expand coverage

3. **Missing - should add**: Important concept that's not covered at all
   - depth: "not_covered"
   - Generate a patch to add it (if critical/high importance)

4. **Intentionally omitted**: Concept that's okay to skip, with explicit reason
   - depth: "not_covered" or "mentioned"
   - omission_reason: One of:
     - "too_tangential": Not central to the main topic
     - "too_advanced": Requires too much prerequisite knowledge for target audience
     - "time_constraints": Would make video too long, and it's lower priority
     - "covered_elsewhere": Better suited for a follow-up video
     - "implementation_detail": Technical detail not needed for conceptual understanding

IMPORTANT: Every "low" importance concept that isn't covered MUST have an omission_reason explaining why it's okay to skip. Nothing should be silently dropped.

Also identify narrative gaps:
- Abrupt topic changes between scenes
- Missing transitions or context
- Unexplained jumps in complexity

Generate ACTIONABLE PATCHES to fix identified issues:
- add_scene: Insert a new scene to cover missing concepts
- expand_scene: Add content to cover concepts more deeply
- add_bridge: Add transition content between scenes

For each patch, include COMPLETE narration text (not just descriptions).

Respond with JSON in this exact format:
{{
    "concept_coverage": [
        {{
            "concept_name": "Name from source concepts",
            "depth": "not_covered|mentioned|explained|deep_dive",
            "scene_ids": ["scene1", "scene2"],
            "coverage_notes": "How well this is covered",
            "suggestion": "Specific improvement suggestion or null",
            "omission_reason": "null if covered, otherwise: too_tangential|too_advanced|time_constraints|covered_elsewhere|implementation_detail"
        }}
    ],
    "narrative_gaps": [
        {{
            "from_scene_id": "scene5",
            "from_scene_title": "Scene 5 Title",
            "to_scene_id": "scene6",
            "to_scene_title": "Scene 6 Title",
            "gap_description": "What's missing between these scenes",
            "severity": "low|medium|high",
            "suggested_bridge": "How to fix this transition"
        }}
    ],
    "suggested_scenes": [
        {{
            "title": "New Scene Title",
            "reason": "Why this scene is needed",
            "suggested_position": 5,
            "concepts_addressed": ["concept1", "concept2"],
            "suggested_narration": "Draft narration for this scene"
        }}
    ],
    "patches": [
        {{
            "patch_type": "add_scene",
            "priority": "critical|high|medium|low",
            "reason": "Why this change is needed",
            "insert_after_scene_id": "scene5",
            "new_scene_id": "scene5b_bridge",
            "title": "New Scene Title",
            "narration": "Complete narration text for the new scene...",
            "visual_description": "Description of visuals for this scene",
            "duration_seconds": 30,
            "concepts_addressed": ["concept1", "concept2"]
        }},
        {{
            "patch_type": "expand_scene",
            "priority": "high|medium|low",
            "reason": "Why this scene needs expansion",
            "scene_id": "scene3",
            "current_narration": "Current narration text...",
            "expanded_narration": "New expanded narration with additional content...",
            "concepts_to_add": ["concept_name"],
            "additional_duration_seconds": 15
        }},
        {{
            "patch_type": "add_bridge",
            "priority": "high|medium|low",
            "reason": "Why this transition is needed",
            "from_scene_id": "scene5",
            "to_scene_id": "scene6",
            "bridge_type": "transition|recap|foreshadow",
            "modify_scene_id": "scene5",
            "current_text": "Current ending text of scene5...",
            "new_text": "New ending with bridge to next topic..."
        }}
    ],
    "intentional_omissions_summary": "Brief explanation of what was intentionally left out and why",
    "overall_coverage_score": 75.5,
    "analysis_notes": "Overall assessment of the script's coverage"
}}
"""


class ScriptAnalyzer:
    """Analyzes script coverage against source material."""

    def __init__(
        self,
        project: Project,
        llm_provider: Optional[LLMProvider] = None,
        verbose: bool = True,
    ):
        """Initialize the script analyzer.

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

    def analyze(self) -> GapAnalysisResult:
        """Run full gap analysis on the project.

        Returns:
            GapAnalysisResult with identified gaps and suggestions
        """
        self._log("Starting gap analysis...")

        # Step 1: Load source material
        source_path, source_content = self._load_source_material()
        if not source_content:
            return GapAnalysisResult(
                project_id=self.project.id,
                source_file="",
                analysis_notes="ERROR: No source material found in input/ directory",
            )

        self._log(f"Loaded source material: {source_path.name} ({len(source_content)} chars)")

        # Step 2: Extract concepts from source
        self._log("Extracting concepts from source material...")
        concepts = self._extract_concepts(source_content)
        self._log(f"Found {len(concepts)} key concepts")

        # Step 3: Load current script
        scenes = self._load_script_scenes()
        if not scenes:
            return GapAnalysisResult(
                project_id=self.project.id,
                source_file=str(source_path),
                analysis_notes="ERROR: No script found. Run 'script' command first.",
            )

        self._log(f"Loaded script with {len(scenes)} scenes")

        # Step 4: Analyze gaps
        self._log("Analyzing coverage gaps...")
        result = self._analyze_gaps(source_path, concepts, scenes)

        self._log(f"Analysis complete. Coverage score: {result.overall_coverage_score:.1f}%")
        if result.missing_concepts:
            self._log(f"Missing concepts: {', '.join(result.missing_concepts)}")
        if result.narrative_gaps:
            self._log(f"Narrative gaps found: {len(result.narrative_gaps)}")

        return result

    def _load_source_material(self) -> tuple[Path, str]:
        """Load source material from the input directory.

        Reads ALL markdown, text, and PDF files and combines their content.

        Returns:
            Tuple of (input directory path, combined content) or (Path(), "") if not found
        """
        input_dir = self.project.root_dir / "input"
        if not input_dir.exists():
            return Path(), ""

        all_content: list[str] = []
        files_loaded: list[str] = []

        # Load all markdown and text files
        for pattern in ["*.md", "*.txt", "*.markdown"]:
            for file_path in sorted(input_dir.glob(pattern)):
                try:
                    content = file_path.read_text()
                    if content.strip():
                        all_content.append(f"# Source: {file_path.name}\n\n{content}")
                        files_loaded.append(file_path.name)
                except Exception as e:
                    self._log(f"Warning: Failed to read {file_path}: {e}")

        # Load all PDF files
        for file_path in sorted(input_dir.glob("*.pdf")):
            try:
                parsed = parse_pdf(file_path)
                if parsed.raw_content.strip():
                    all_content.append(f"# Source: {file_path.name}\n\n{parsed.raw_content}")
                    files_loaded.append(file_path.name)
            except Exception as e:
                self._log(f"Warning: Failed to parse PDF {file_path}: {e}")

        if not all_content:
            return Path(), ""

        if files_loaded:
            self._log(f"Loaded {len(files_loaded)} source files: {', '.join(files_loaded)}")

        combined_content = "\n\n---\n\n".join(all_content)
        return input_dir, combined_content

    def _load_script_scenes(self) -> list[dict]:
        """Load scenes from the script.

        Returns:
            List of scene dictionaries from script.json or narrations.json
        """
        # Try narrations.json first (has the actual narration text)
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

    def _extract_concepts(self, source_content: str) -> list[SourceConcept]:
        """Extract key concepts from source material using LLM.

        Args:
            source_content: The source document text

        Returns:
            List of extracted SourceConcept objects
        """
        # Truncate source if very long (to fit in context)
        max_chars = 50000
        if len(source_content) > max_chars:
            source_content = source_content[:max_chars] + "\n\n[... truncated ...]"

        prompt = CONCEPT_EXTRACTION_PROMPT.format(source_content=source_content)

        try:
            self._log(f"Sending {len(source_content)} chars to LLM for concept extraction...")
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=CONCEPT_EXTRACTION_SYSTEM_PROMPT,
            )
            self._log("Concept extraction complete.")

            concepts = []
            for c in response.get("concepts", []):
                concepts.append(
                    SourceConcept(
                        name=c.get("name", "Unknown"),
                        description=c.get("description", ""),
                        importance=c.get("importance", "medium"),
                        prerequisites=c.get("prerequisites", []),
                        section_reference=c.get("section_reference", ""),
                    )
                )
            return concepts

        except Exception as e:
            self._log(f"Error extracting concepts: {e}")
            return []

    def _analyze_gaps(
        self,
        source_path: Path,
        concepts: list[SourceConcept],
        scenes: list[dict],
    ) -> GapAnalysisResult:
        """Analyze gaps between source concepts and script scenes.

        Args:
            source_path: Path to source file
            concepts: Extracted source concepts
            scenes: Current script scenes

        Returns:
            GapAnalysisResult with full analysis
        """
        # Prepare concepts as JSON for the prompt
        concepts_json = json.dumps(
            [c.to_dict() for c in concepts],
            indent=2,
        )

        # Prepare scenes as JSON
        scenes_json = json.dumps(scenes, indent=2)

        prompt = GAP_ANALYSIS_PROMPT.format(
            concepts_json=concepts_json,
            scenes_json=scenes_json,
        )

        try:
            self._log(f"Sending {len(concepts)} concepts and {len(scenes)} scenes to LLM for gap analysis...")
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=GAP_ANALYSIS_SYSTEM_PROMPT,
            )
            self._log("Gap analysis complete.")

            # Parse concept coverage
            coverage_list = []
            for cov in response.get("concept_coverage", []):
                # Find the matching SourceConcept
                concept_name = cov.get("concept_name", "")
                matching_concept = next(
                    (c for c in concepts if c.name.lower() == concept_name.lower()),
                    SourceConcept(name=concept_name, description=""),
                )

                coverage_list.append(
                    ConceptCoverage(
                        concept=matching_concept,
                        depth=ConceptDepth(cov.get("depth", "not_covered")),
                        scene_ids=cov.get("scene_ids", []),
                        coverage_notes=cov.get("coverage_notes", ""),
                        suggestion=cov.get("suggestion"),
                        omission_reason=cov.get("omission_reason"),
                    )
                )

            # Parse narrative gaps
            narrative_gaps = []
            for gap in response.get("narrative_gaps", []):
                narrative_gaps.append(
                    NarrativeGap(
                        from_scene_id=gap.get("from_scene_id", ""),
                        from_scene_title=gap.get("from_scene_title", ""),
                        to_scene_id=gap.get("to_scene_id", ""),
                        to_scene_title=gap.get("to_scene_title", ""),
                        gap_description=gap.get("gap_description", ""),
                        severity=gap.get("severity", "medium"),
                        suggested_bridge=gap.get("suggested_bridge"),
                    )
                )

            # Parse suggested scenes
            suggested_scenes = []
            for scene in response.get("suggested_scenes", []):
                suggested_scenes.append(
                    SuggestedScene(
                        title=scene.get("title", ""),
                        reason=scene.get("reason", ""),
                        suggested_position=scene.get("suggested_position", 0),
                        concepts_addressed=scene.get("concepts_addressed", []),
                        suggested_narration=scene.get("suggested_narration", ""),
                    )
                )

            # Parse patches
            patches = self._parse_patches(response.get("patches", []))

            return GapAnalysisResult(
                project_id=self.project.id,
                source_file=str(source_path),
                concepts=coverage_list,
                narrative_gaps=narrative_gaps,
                suggested_scenes=suggested_scenes,
                patches=patches,
                overall_coverage_score=response.get("overall_coverage_score", 0.0),
                analysis_notes=response.get("analysis_notes", ""),
                intentional_omissions_summary=response.get("intentional_omissions_summary", ""),
            )

        except Exception as e:
            self._log(f"Error analyzing gaps: {e}")
            return GapAnalysisResult(
                project_id=self.project.id,
                source_file=str(source_path),
                analysis_notes=f"ERROR: Gap analysis failed: {e}",
            )

    def _parse_patches(self, patches_data: list[dict]) -> list[ScriptPatch]:
        """Parse patch data from LLM response into ScriptPatch objects.

        Args:
            patches_data: List of patch dictionaries from LLM response

        Returns:
            List of ScriptPatch objects
        """
        patches: list[ScriptPatch] = []

        for patch_data in patches_data:
            patch_type = patch_data.get("patch_type", "")
            reason = patch_data.get("reason", "")
            priority = patch_data.get("priority", "medium")

            try:
                if patch_type == "add_scene":
                    patches.append(
                        AddScenePatch(
                            reason=reason,
                            priority=priority,
                            insert_after_scene_id=patch_data.get("insert_after_scene_id"),
                            new_scene_id=patch_data.get("new_scene_id", ""),
                            title=patch_data.get("title", ""),
                            narration=patch_data.get("narration", ""),
                            visual_description=patch_data.get("visual_description", ""),
                            duration_seconds=patch_data.get("duration_seconds", 30.0),
                            concepts_addressed=patch_data.get("concepts_addressed", []),
                        )
                    )
                elif patch_type == "expand_scene":
                    patches.append(
                        ExpandScenePatch(
                            reason=reason,
                            priority=priority,
                            scene_id=patch_data.get("scene_id", ""),
                            current_narration=patch_data.get("current_narration", ""),
                            expanded_narration=patch_data.get("expanded_narration", ""),
                            concepts_to_add=patch_data.get("concepts_to_add", []),
                            additional_duration_seconds=patch_data.get(
                                "additional_duration_seconds", 0.0
                            ),
                        )
                    )
                elif patch_type == "add_bridge":
                    patches.append(
                        AddBridgePatch(
                            reason=reason,
                            priority=priority,
                            from_scene_id=patch_data.get("from_scene_id", ""),
                            to_scene_id=patch_data.get("to_scene_id", ""),
                            bridge_type=patch_data.get("bridge_type", "transition"),
                            modify_scene_id=patch_data.get("modify_scene_id", ""),
                            current_text=patch_data.get("current_text", ""),
                            new_text=patch_data.get("new_text", ""),
                        )
                    )
                else:
                    self._log(f"Unknown patch type: {patch_type}")
            except Exception as e:
                self._log(f"Error parsing patch: {e}")

        return patches

    def save_result(self, result: GapAnalysisResult, output_path: Optional[Path] = None) -> Path:
        """Save the gap analysis result to a JSON file.

        Args:
            result: The analysis result to save
            output_path: Optional custom output path

        Returns:
            Path to the saved file
        """
        if output_path is None:
            refinement_dir = self.project.root_dir / "refinement"
            refinement_dir.mkdir(parents=True, exist_ok=True)
            output_path = refinement_dir / "gap_analysis.json"

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path
