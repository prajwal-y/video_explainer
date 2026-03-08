"""Fact checker for video scripts and narrations."""

import json
from pathlib import Path
from typing import Any

from ..config import LLMConfig
from ..ingestion import parse_document
from ..project.loader import Project
from ..understanding.llm_provider import (
    ClaudeCodeLLMProvider,
    ClaudeCodeError,
    MockLLMProvider,
    get_llm_provider,
)
from .models import (
    FactCheckIssue,
    FactCheckReport,
    FactCheckSummary,
    IssueCategory,
    IssueSeverity,
)
from .prompts import (
    FACT_CHECK_MOCK_RESPONSE,
    FACT_CHECK_PROMPT,
    FACT_CHECK_SYSTEM_PROMPT,
)


class FactCheckError(Exception):
    """Error during fact checking."""

    pass


class FactChecker:
    """Fact checks video scripts and narrations against source material."""

    # Tools needed for thorough fact checking
    FACT_CHECK_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]

    def __init__(
        self,
        project: Project,
        use_mock: bool = False,
        verbose: bool = False,
        timeout: int = 600,
    ):
        """Initialize the fact checker.

        Args:
            project: The project to fact-check
            use_mock: If True, use mock LLM for testing
            verbose: If True, print detailed progress
            timeout: LLM timeout in seconds (default: 600 = 10 minutes)
        """
        self.project = project
        self.use_mock = use_mock
        self.verbose = verbose
        self.timeout = timeout

        # Set up working directory (repo root for file access)
        self.repo_root = project.root_dir.parent.parent

        # Initialize LLM provider
        if use_mock:
            llm_config = LLMConfig(provider="mock")
            self.llm = MockLLMProvider(llm_config)
        else:
            from ..config import load_config
            self.llm = get_llm_provider(load_config())

    def _log(self, message: str, indent: int = 0) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{message}")

    def _load_script(self) -> dict[str, Any]:
        """Load the script from the project.

        Returns:
            Script data as dictionary

        Raises:
            FactCheckError: If script cannot be loaded
        """
        script_path = self.project.root_dir / "script" / "script.json"
        if not script_path.exists():
            raise FactCheckError(f"Script not found: {script_path}")

        with open(script_path) as f:
            return json.load(f)

    def _load_narrations(self) -> dict[str, Any]:
        """Load narrations from the project.

        Returns:
            Narrations data as dictionary

        Raises:
            FactCheckError: If narrations cannot be loaded
        """
        narration_path = self.project.root_dir / "narration" / "narrations.json"
        if not narration_path.exists():
            raise FactCheckError(f"Narrations not found: {narration_path}")

        with open(narration_path) as f:
            return json.load(f)

    def _load_source_material(self) -> tuple[str, list[str]]:
        """Load source material from the project's input directory.

        Returns:
            Tuple of (combined content string, list of source document names)

        Raises:
            FactCheckError: If no source material found
        """
        input_dir = self.project.root_dir / "input"
        if not input_dir.exists():
            raise FactCheckError(f"Input directory not found: {input_dir}")

        # Find all supported input files
        input_files = []
        for pattern in ["*.md", "*.markdown", "*.pdf"]:
            input_files.extend(input_dir.glob(pattern))

        if not input_files:
            raise FactCheckError(f"No source documents found in {input_dir}")

        # Parse and combine all source documents
        source_contents = []
        source_names = []

        for file_path in input_files:
            try:
                doc = parse_document(file_path)
                source_names.append(file_path.name)

                # Format the document content
                content = f"### Source: {file_path.name}\n\n"
                content += f"Title: {doc.title}\n\n"
                for section in doc.sections:
                    content += f"#### {section.heading}\n{section.content}\n\n"

                source_contents.append(content)
                self._log(f"Loaded source: {file_path.name}", indent=1)

            except Exception as e:
                self._log(f"Warning: Failed to load {file_path.name}: {e}", indent=1)

        if not source_contents:
            raise FactCheckError("Failed to load any source documents")

        return "\n---\n".join(source_contents), source_names

    def _format_script_content(self, script: dict[str, Any]) -> str:
        """Format script for fact checking prompt.

        Args:
            script: Script data dictionary

        Returns:
            Formatted script string
        """
        lines = []
        lines.append(f"**Title:** {script.get('title', 'Untitled')}")
        lines.append(f"**Duration:** {script.get('total_duration_seconds', 0)} seconds")
        lines.append("")

        for scene in script.get("scenes", []):
            scene_id = scene.get("scene_id", "unknown")
            scene_type = scene.get("scene_type", "unknown")
            title = scene.get("title", "Untitled")
            voiceover = scene.get("voiceover", "")
            visual_cue = scene.get("visual_cue", {})

            lines.append(f"### Scene: {scene_id} ({scene_type})")
            lines.append(f"**Title:** {title}")
            lines.append(f"**Voiceover:** {voiceover}")
            if visual_cue:
                lines.append(f"**Visual:** {visual_cue.get('description', '')}")
            lines.append("")

        return "\n".join(lines)

    def _format_narration_content(self, narrations: dict[str, Any]) -> str:
        """Format narrations for fact checking prompt.

        Args:
            narrations: Narrations data dictionary

        Returns:
            Formatted narrations string
        """
        lines = []

        for scene in narrations.get("scenes", []):
            scene_id = scene.get("scene_id", "unknown")
            title = scene.get("title", "Untitled")
            narration = scene.get("narration", "")

            lines.append(f"### {scene_id}: {title}")
            lines.append(narration)
            lines.append("")

        return "\n".join(lines)

    def _parse_fact_check_response(
        self, response: dict[str, Any], source_names: list[str]
    ) -> FactCheckReport:
        """Parse the LLM fact check response into a report.

        Args:
            response: Parsed JSON response from LLM
            source_names: List of source document names

        Returns:
            FactCheckReport object
        """
        issues = []
        for issue_data in response.get("issues", []):
            try:
                issue = FactCheckIssue(
                    id=issue_data.get("id", f"issue_{len(issues)+1}"),
                    severity=IssueSeverity(issue_data.get("severity", "medium")),
                    category=IssueCategory(issue_data.get("category", "factual_error")),
                    location=issue_data.get("location", "unknown"),
                    original_text=issue_data.get("original_text", ""),
                    issue_description=issue_data.get("issue_description", ""),
                    correction=issue_data.get("correction", ""),
                    source_reference=issue_data.get("source_reference", ""),
                    confidence=issue_data.get("confidence", 0.8),
                    verified_via_web=issue_data.get("verified_via_web", False),
                )
                issues.append(issue)
            except (ValueError, KeyError) as e:
                self._log(f"Warning: Failed to parse issue: {e}", indent=1)

        # Parse summary
        summary_data = response.get("summary", {})
        summary = FactCheckSummary(
            total_issues=summary_data.get("total_issues", len(issues)),
            critical_count=summary_data.get("critical_count", 0),
            high_count=summary_data.get("high_count", 0),
            medium_count=summary_data.get("medium_count", 0),
            low_count=summary_data.get("low_count", 0),
            info_count=summary_data.get("info_count", 0),
            scenes_with_issues=summary_data.get("scenes_with_issues", []),
            overall_accuracy_score=summary_data.get("overall_accuracy_score", 1.0),
            web_verified_count=summary_data.get("web_verified_count", 0),
        )

        # Get script title
        try:
            script = self._load_script()
            script_title = script.get("title", "Untitled")
        except FactCheckError:
            script_title = "Untitled"

        return FactCheckReport(
            project_id=self.project.id,
            script_title=script_title,
            issues=issues,
            summary=summary,
            source_documents=source_names,
            recommendations=response.get("recommendations", []),
        )

    def run_fact_check(self) -> FactCheckReport:
        """Run the fact checking workflow.

        Returns:
            FactCheckReport with all issues and recommendations

        Raises:
            FactCheckError: If fact checking fails
        """
        self._log(f"Starting fact check for project: {self.project.id}")

        # Load all required data
        self._log("Loading script...")
        script = self._load_script()
        script_title = script.get("title", "Untitled")

        self._log("Loading narrations...")
        narrations = self._load_narrations()

        self._log("Loading source material...")
        source_content, source_names = self._load_source_material()
        self._log(f"Loaded {len(source_names)} source document(s)", indent=1)

        # Format content for the prompt
        script_content = self._format_script_content(script)
        narration_content = self._format_narration_content(narrations)

        # Build the fact check prompt
        prompt = FACT_CHECK_PROMPT.format(
            source_content=source_content,
            script_title=script_title,
            script_content=script_content,
            narration_content=narration_content,
        )

        self._log("Running fact check analysis...")
        self._log("(This may take several minutes for thorough analysis)")

        # Run the fact check
        if self.use_mock:
            # Return mock response for testing
            response = FACT_CHECK_MOCK_RESPONSE
            raw_analysis = json.dumps(response, indent=2)
        else:
            try:
                # Use generate with file access for web search capability
                result = self.llm.generate_with_file_access(
                    prompt,
                    system_prompt=FACT_CHECK_SYSTEM_PROMPT,
                    allow_writes=False,  # Read-only, we just need web search
                )

                if not result.success:
                    raise FactCheckError(f"Fact check failed: {result.error_message}")

                raw_analysis = result.response

                # Parse the JSON response
                response = self._parse_json_from_response(raw_analysis)

            except ClaudeCodeError as e:
                raise FactCheckError(f"LLM error during fact check: {e}") from e

        # Parse response into report
        report = self._parse_fact_check_response(response, source_names)
        report.raw_analysis = raw_analysis

        self._log(f"Fact check complete: {report.summary.total_issues} issues found")
        self._log(f"Accuracy score: {report.summary.overall_accuracy_score:.0%}")

        return report

    def _parse_json_from_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from the LLM response.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON dictionary

        Raises:
            FactCheckError: If JSON parsing fails
        """
        import re

        text = response.strip()

        # Try to extract JSON from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, text)
        if matches:
            text = matches[0].strip()

        # Try to find JSON object
        json_pattern = r"(\{[\s\S]*\})"
        json_match = re.search(json_pattern, text)
        if json_match:
            text = json_match.group(1)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise FactCheckError(
                f"Failed to parse fact check response as JSON: {e}\n"
                f"Response preview: {response[:500]}"
            ) from e

    def save_report(self, report: FactCheckReport, output_path: Path | None = None) -> Path:
        """Save the fact check report to a file.

        Args:
            report: The fact check report to save
            output_path: Optional custom output path

        Returns:
            Path where the report was saved
        """
        if output_path is None:
            output_dir = self.project.root_dir / "factcheck"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "report.json"

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        self._log(f"Report saved to: {output_path}")
        return output_path


def run_fact_check(
    project: Project,
    use_mock: bool = False,
    verbose: bool = False,
    timeout: int = 600,
) -> FactCheckReport:
    """Convenience function to run fact check.

    Args:
        project: The project to fact-check
        use_mock: If True, use mock LLM
        verbose: If True, print detailed progress
        timeout: LLM timeout in seconds

    Returns:
        FactCheckReport with results
    """
    checker = FactChecker(
        project,
        use_mock=use_mock,
        verbose=verbose,
        timeout=timeout,
    )
    return checker.run_fact_check()
