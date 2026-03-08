"""LLM Provider abstraction and implementations."""

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import Config, LLMConfig
from ..models import ContentAnalysis, Concept, Script, ScriptScene, VisualCue


class ClaudeCodeError(Exception):
    """Error from Claude Code CLI execution."""

    pass


@dataclass
class ClaudeCodeResult:
    """Result from Claude Code execution with file access."""

    response: str
    modified_files: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The generated text response
        """
        pass

    @abstractmethod
    def generate_json(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        """Generate a JSON response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON response as a dictionary
        """
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns generic responses for testing.

    This provider returns realistic but generic mock responses suitable
    for testing the pipeline without requiring an actual LLM API.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a mock response based on prompt patterns."""
        return "This is a mock LLM response for testing purposes."

    def generate_json(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        """Generate mock JSON responses for known prompt patterns.

        Pattern matching order is important:
        1. Plan refinement (most specific - contains "refine" + "plan")
        2. Plan generation (contains "video plan" or "create a video plan")
        3. Storyboard (most specific - contains "storyboard" keyword)
        4. Script (contains "script" + "create/generate", but NOT "storyboard")
        5. Content analysis (contains "analyze" + "content/document")
        """
        prompt_lower = prompt.lower()

        # Plan refinement request (check first - most specific)
        if "refine" in prompt_lower and "plan" in prompt_lower:
            return self._mock_plan_refinement(prompt)

        # Plan generation request
        if "video plan" in prompt_lower or "create a video plan" in prompt_lower:
            return self._mock_plan_generation(prompt)

        # Storyboard generation request (check first - most specific)
        if "storyboard" in prompt_lower or "scene id:" in prompt_lower:
            return self._mock_storyboard_generation(prompt)

        # Narration generation request (check before script - narration prompts contain "script")
        # Use specific pattern to avoid matching script generation prompts
        if "generate narrations for video script" in prompt_lower or "narrations for video" in prompt_lower:
            return self._mock_narration_generation(prompt)

        # Script generation request (check before analysis - scripts may contain "analyze")
        if "script" in prompt_lower and (
            "generate" in prompt_lower or "create" in prompt_lower
        ):
            return self._mock_script_generation(prompt)

        # Content analysis request
        if "analyze" in prompt_lower and (
            "content" in prompt_lower or "document" in prompt_lower
        ):
            return self._mock_content_analysis(prompt)

        # Default empty response
        return {}

    def _mock_content_analysis(self, prompt: str) -> dict[str, Any]:
        """Return mock content analysis based on document content."""
        return {
            "core_thesis": "This document explains a technical concept with practical applications.",
            "key_concepts": [
                {
                    "name": "Core Concept",
                    "explanation": "The fundamental idea that drives the topic.",
                    "complexity": 5,
                    "prerequisites": ["basic understanding"],
                    "analogies": ["Like a simple real-world example"],
                    "visual_potential": "high",
                },
                {
                    "name": "Supporting Concept",
                    "explanation": "A related idea that helps understand the core concept.",
                    "complexity": 4,
                    "prerequisites": ["core concept"],
                    "analogies": ["Similar to another familiar concept"],
                    "visual_potential": "medium",
                },
                {
                    "name": "Application",
                    "explanation": "How this concept is used in practice.",
                    "complexity": 6,
                    "prerequisites": ["core concept", "supporting concept"],
                    "analogies": ["Like using a tool for a job"],
                    "visual_potential": "high",
                },
            ],
            "target_audience": "Technical professionals and enthusiasts",
            "estimated_duration_minutes": 3,
            "complexity_score": 5,
        }

    def _mock_script_generation(self, prompt: str) -> dict[str, Any]:
        """Return mock script for testing."""
        return {
            "title": "Understanding the Core Concept",
            "total_duration_seconds": 180,
            "source_document": "document.md",
            "scenes": [
                {
                    "scene_id": 1,
                    "scene_type": "hook",
                    "title": "The Problem",
                    "voiceover": "Every day, we encounter this challenge. What if there was a better way?",
                    "visual_cue": {
                        "description": "Show the problem visually",
                        "visual_type": "animation",
                        "elements": ["problem_illustration"],
                        "duration_seconds": 15.0,
                    },
                    "duration_seconds": 15.0,
                    "notes": "Build intrigue",
                },
                {
                    "scene_id": 2,
                    "scene_type": "context",
                    "title": "Background",
                    "voiceover": "To understand the solution, we first need to understand the context.",
                    "visual_cue": {
                        "description": "Show background context",
                        "visual_type": "animation",
                        "elements": ["context_diagram"],
                        "duration_seconds": 20.0,
                    },
                    "duration_seconds": 20.0,
                    "notes": "Set the stage",
                },
                {
                    "scene_id": 3,
                    "scene_type": "explanation",
                    "title": "The Core Concept",
                    "voiceover": "Here's how it works. The key insight is understanding the relationship between components.",
                    "visual_cue": {
                        "description": "Explain the core concept with visuals",
                        "visual_type": "animation",
                        "elements": ["concept_visualization"],
                        "duration_seconds": 30.0,
                    },
                    "duration_seconds": 30.0,
                    "notes": "Main explanation",
                },
                {
                    "scene_id": 4,
                    "scene_type": "insight",
                    "title": "The Key Insight",
                    "voiceover": "This is the breakthrough. Once you understand this, everything else falls into place.",
                    "visual_cue": {
                        "description": "Highlight the key insight",
                        "visual_type": "animation",
                        "elements": ["insight_highlight"],
                        "duration_seconds": 25.0,
                    },
                    "duration_seconds": 25.0,
                    "notes": "Aha moment",
                },
                {
                    "scene_id": 5,
                    "scene_type": "conclusion",
                    "title": "Putting It Together",
                    "voiceover": "Now you understand the concept. Let's see how it applies in practice.",
                    "visual_cue": {
                        "description": "Summary and application",
                        "visual_type": "animation",
                        "elements": ["summary"],
                        "duration_seconds": 20.0,
                    },
                    "duration_seconds": 20.0,
                    "notes": "Wrap up",
                },
            ],
        }

    def _mock_narration_generation(self, prompt: str = "") -> dict[str, Any]:
        """Return mock narration for testing."""
        return {
            "scenes": [
                {
                    "scene_id": "scene1_hook",
                    "title": "The Hook",
                    "duration_seconds": 15,
                    "narration": "What if I told you there's a better way to approach this problem?",
                },
                {
                    "scene_id": "scene2_context",
                    "title": "Setting the Context",
                    "duration_seconds": 20,
                    "narration": "To understand this, we need to first look at the bigger picture.",
                },
                {
                    "scene_id": "scene3_explanation",
                    "title": "How It Works",
                    "duration_seconds": 30,
                    "narration": "At its core, this works by processing information in a fundamentally different way.",
                },
                {
                    "scene_id": "scene4_conclusion",
                    "title": "Conclusion",
                    "duration_seconds": 15,
                    "narration": "And that's how this concept reshapes our understanding of what's possible.",
                },
            ],
            "total_duration_seconds": 80,
        }

    def _mock_plan_generation(self, prompt: str = "") -> dict[str, Any]:
        """Return mock video plan for testing."""
        return {
            "title": "Understanding the Core Concept",
            "central_question": "How does this technology actually work under the hood?",
            "target_audience": "Technical professionals and enthusiasts",
            "estimated_total_duration_seconds": 300,
            "core_thesis": "This concept transforms how we approach the problem by introducing a fundamentally different approach.",
            "key_concepts": [
                "Basic foundations",
                "Core mechanism",
                "Optimization techniques",
                "Practical applications",
            ],
            "complexity_score": 6,
            "visual_style": "Clean diagrams with animated data flow visualizations",
            "scenes": [
                {
                    "scene_number": 1,
                    "scene_type": "hook",
                    "title": "The Challenge",
                    "concept_to_cover": "The problem that needs solving",
                    "visual_approach": "Show a dramatic visualization of the scale of the problem, with numbers animating to emphasize the challenge.",
                    "ascii_visual": "┌─────────────────────────────────────────────────────┐\n│                                                     │\n│    ┌───┬───┬───┬───┐         ╔═══════════════╗     │\n│    │ A │ B │ C │...│  ───►   ║   MASSIVE     ║     │\n│    ├───┼───┼───┼───┤         ║   SCALE!      ║     │\n│    │...│...│...│...│         ╚═══════════════╝     │\n│    └───┴───┴───┴───┘              ↓                │\n│       Input Data           [explosion effect]      │\n│                                                     │\n└─────────────────────────────────────────────────────┘",
                    "estimated_duration_seconds": 45,
                    "key_points": [
                        "Hook the viewer with a surprising fact",
                        "Establish the scale of the problem",
                        "Create curiosity about the solution",
                    ],
                },
                {
                    "scene_number": 2,
                    "scene_type": "context",
                    "title": "Background Context",
                    "concept_to_cover": "Historical context and why this matters",
                    "visual_approach": "Timeline animation showing evolution of approaches, highlighting limitations of previous solutions.",
                    "ascii_visual": "┌─────────────────────────────────────────────────────┐\n│                                                     │\n│  2010        2015        2020        NOW            │\n│    ●──────────●──────────●──────────●               │\n│    │          │          │          │               │\n│  [Old]     [Better]  [Improved]  [Current]          │\n│    ↓          ↓          ↓          ↓               │\n│  Slow      Faster    Fast      BREAKTHROUGH         │\n│                                                     │\n└─────────────────────────────────────────────────────┘",
                    "estimated_duration_seconds": 60,
                    "key_points": [
                        "Previous approaches and their limitations",
                        "Why a new solution was needed",
                        "Setup for the main explanation",
                    ],
                },
                {
                    "scene_number": 3,
                    "scene_type": "explanation",
                    "title": "How It Works",
                    "concept_to_cover": "The core mechanism explained step by step",
                    "visual_approach": "Step-by-step animation showing data flow through the system, with each step clearly labeled and highlighted.",
                    "ascii_visual": "┌─────────────────────────────────────────────────────┐\n│                                                     │\n│    INPUT          PROCESS          OUTPUT           │\n│    ┌───┐         ┌───────┐         ┌───┐           │\n│    │ X │  ─────► │ MAGIC │ ─────►  │ Y │           │\n│    └───┘         │  BOX  │         └───┘           │\n│                  └───────┘                         │\n│                      │                             │\n│                      ▼                             │\n│              [Detailed breakdown]                  │\n│                                                     │\n└─────────────────────────────────────────────────────┘",
                    "estimated_duration_seconds": 90,
                    "key_points": [
                        "Input processing",
                        "Core transformation",
                        "Output generation",
                    ],
                },
                {
                    "scene_number": 4,
                    "scene_type": "insight",
                    "title": "The Key Insight",
                    "concept_to_cover": "Why this approach is so effective",
                    "visual_approach": "Side-by-side comparison showing old vs new approach, with dramatic performance metrics.",
                    "ascii_visual": "┌─────────────────────────────────────────────────────┐\n│                                                     │\n│     OLD WAY           vs           NEW WAY          │\n│    ┌───────┐                      ┌───────┐         │\n│    │ Slow  │                      │ Fast  │         │\n│    │ O(n²) │                      │ O(n)  │         │\n│    └───────┘                      └───────┘         │\n│        │                              │             │\n│        ▼                              ▼             │\n│      40x/s          ───►         3,500x/s          │\n│                                                     │\n└─────────────────────────────────────────────────────┘",
                    "estimated_duration_seconds": 60,
                    "key_points": [
                        "Performance comparison",
                        "Key architectural difference",
                        "Why this matters in practice",
                    ],
                },
                {
                    "scene_number": 5,
                    "scene_type": "conclusion",
                    "title": "Putting It All Together",
                    "concept_to_cover": "Summary and practical implications",
                    "visual_approach": "Recap animation connecting all concepts, ending with forward-looking applications.",
                    "ascii_visual": "┌─────────────────────────────────────────────────────┐\n│                                                     │\n│           KEY TAKEAWAYS                             │\n│           ═════════════                             │\n│                                                     │\n│    ✓ Challenge understood                          │\n│    ✓ Mechanism explained                           │\n│    ✓ Performance demonstrated                      │\n│                                                     │\n│              WHAT'S NEXT?                          │\n│              ────────────                          │\n│         [Future possibilities]                     │\n│                                                     │\n└─────────────────────────────────────────────────────┘",
                    "estimated_duration_seconds": 45,
                    "key_points": [
                        "Recap main points",
                        "Practical applications",
                        "Future directions",
                    ],
                },
            ],
        }

    def _mock_plan_refinement(self, prompt: str = "") -> dict[str, Any]:
        """Return mock refined video plan for testing."""
        # Return same structure as plan generation but with slightly modified content
        plan = self._mock_plan_generation(prompt)
        plan["title"] = "Understanding the Core Concept (Refined)"
        plan["scenes"][0]["visual_approach"] = "Refined visual approach based on feedback"
        plan["scenes"][0]["key_points"].append("Added based on user feedback")
        return plan

    def _mock_storyboard_generation(self, prompt: str = "") -> dict[str, Any]:
        """Return mock storyboard beats for testing."""
        return {
            "id": "test_storyboard",
            "title": "Test Storyboard",
            "duration_seconds": 60,
            "beats": [
                {
                    "id": "setup",
                    "start_seconds": 0,
                    "end_seconds": 10,
                    "voiceover": "Introduction to the concept.",
                    "elements": [
                        {
                            "id": "title",
                            "component": "title_card",
                            "props": {"heading": "Test Video", "subheading": "A demonstration"},
                            "position": {"x": "center", "y": "center"},
                            "enter": {"type": "fade", "duration_seconds": 0.5},
                            "exit": {"type": "fade", "duration_seconds": 0.5},
                        }
                    ],
                },
                {
                    "id": "main",
                    "start_seconds": 10,
                    "end_seconds": 50,
                    "voiceover": "The main explanation of the concept.",
                    "elements": [
                        {
                            "id": "content",
                            "component": "text_reveal",
                            "props": {"text": "Main content here"},
                            "position": {"x": "center", "y": "center"},
                        }
                    ],
                },
                {
                    "id": "conclusion",
                    "start_seconds": 50,
                    "end_seconds": 60,
                    "voiceover": "Summary and takeaways.",
                    "elements": [
                        {
                            "id": "outro",
                            "component": "title_card",
                            "props": {"heading": "Thank You"},
                            "position": {"x": "center", "y": "center"},
                        }
                    ],
                },
            ],
            "style": {
                "background_color": "#f4f4f5",
                "primary_color": "#00d9ff",
                "secondary_color": "#ff6b35",
                "font_family": "Inter",
            },
        }


class ClaudeCodeLLMProvider(LLMProvider):
    """LLM provider using Claude Code CLI in headless mode.

    This provider executes the claude CLI tool to generate responses,
    with the ability to read and modify files in the working directory.
    """

    DEFAULT_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

    def __init__(
        self,
        config: LLMConfig,
        working_dir: Path | None = None,
        timeout: int = 300,
    ):
        """Initialize the Claude Code provider.

        Args:
            config: LLM configuration
            working_dir: Working directory for file operations (default: cwd)
            timeout: Command timeout in seconds (default: 300)
        """
        super().__init__(config)
        self.working_dir = working_dir or Path.cwd()
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a text response via Claude Code CLI.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The generated text response

        Raises:
            ClaudeCodeError: If the CLI command fails
        """
        cmd = self._build_command(prompt, system_prompt, tools=[])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.working_dir),
            timeout=self.timeout,
        )

        if result.returncode != 0:
            raise ClaudeCodeError(f"Claude Code failed: {result.stderr}")

        return result.stdout.strip()

    def generate_json(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        """Generate a JSON response via Claude Code CLI.

        The prompt is augmented to request JSON output, and the response
        is parsed to extract the JSON content.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            ClaudeCodeError: If the CLI command fails or JSON parsing fails
        """
        json_prompt = f"{prompt}\n\nRespond with valid JSON only. No markdown code blocks."
        cmd = self._build_command(json_prompt, system_prompt, tools=[])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.working_dir),
            timeout=self.timeout,
        )

        if result.returncode != 0:
            raise ClaudeCodeError(f"Claude Code failed: {result.stderr}")

        return self._parse_json_response(result.stdout)

    def generate_with_file_access(
        self,
        prompt: str,
        system_prompt: str | None = None,
        allow_writes: bool = False,
        live_output: bool = False,
    ) -> ClaudeCodeResult:
        """Generate a response with file read/write capabilities.

        This method allows Claude Code to read and optionally modify files
        in the working directory.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            allow_writes: If True, allows Write and Edit tools
            live_output: If True, stream Claude Code output to terminal

        Returns:
            ClaudeCodeResult with response and list of modified files
        """
        if allow_writes:
            tools = self.DEFAULT_TOOLS
        else:
            tools = ["Read", "Glob", "Grep"]

        cmd = self._build_command(prompt, system_prompt, tools=tools)

        # Add verbose flag for live output
        if live_output:
            cmd.append("--verbose")

        try:
            if live_output:
                # Stream output in real-time
                return self._run_with_live_output(cmd, allow_writes)
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.working_dir),
                    timeout=self.timeout,
                )

                if result.returncode != 0:
                    return ClaudeCodeResult(
                        response="",
                        success=False,
                        error_message=f"Claude Code failed: {result.stderr}",
                    )

                # Extract modified files from output if writes were allowed
                modified_files = []
                if allow_writes:
                    modified_files = self._extract_modified_files(result.stdout)

                return ClaudeCodeResult(
                    response=result.stdout.strip(),
                    modified_files=modified_files,
                    success=True,
                )

        except subprocess.TimeoutExpired:
            return ClaudeCodeResult(
                response="",
                success=False,
                error_message=f"Claude Code timed out after {self.timeout}s",
            )

    def _run_with_live_output(
        self, cmd: list[str], allow_writes: bool
    ) -> ClaudeCodeResult:
        """Run Claude Code with live output streaming.

        Args:
            cmd: The command to run
            allow_writes: Whether writes were allowed (for extracting modified files)

        Returns:
            ClaudeCodeResult with response and modified files
        """
        import sys

        output_lines = []
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.working_dir),
            )

            print("\n" + "=" * 60)
            print("Claude Code Output:")
            print("=" * 60)

            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                output_lines.append(line)

            process.wait(timeout=self.timeout)
            print("=" * 60 + "\n")

            full_output = "".join(output_lines)

            if process.returncode != 0:
                return ClaudeCodeResult(
                    response=full_output,
                    success=False,
                    error_message=f"Claude Code exited with code {process.returncode}",
                )

            modified_files = []
            if allow_writes:
                modified_files = self._extract_modified_files(full_output)

            return ClaudeCodeResult(
                response=full_output.strip(),
                modified_files=modified_files,
                success=True,
            )

        except subprocess.TimeoutExpired:
            process.kill()
            return ClaudeCodeResult(
                response="",
                success=False,
                error_message=f"Claude Code timed out after {self.timeout}s",
            )

    def _build_command(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> list[str]:
        """Build the claude CLI command.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            tools: List of allowed tools (empty list = no tools)

        Returns:
            Command as list of strings
        """
        cmd = ["claude", "--print", "-p", prompt]

        # Add model if specified in config
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # Only add tools if list is non-empty
        if tools:
            cmd.extend(["--allowedTools", ",".join(tools)])
            cmd.append("--dangerously-skip-permissions")

        return cmd

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from Claude Code response.

        Handles responses that may include markdown code blocks.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ClaudeCodeError: If JSON parsing fails
        """
        text = response.strip()

        # Try to extract JSON from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, text)
        if matches:
            text = matches[0].strip()

        # Try to find JSON object or array
        json_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
        json_match = re.search(json_pattern, text)
        if json_match:
            text = json_match.group(1)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ClaudeCodeError(f"Failed to parse JSON response: {e}\nResponse: {response[:500]}")

    def _extract_modified_files(self, output: str) -> list[str]:
        """Extract list of modified files from Claude Code output.

        Uses git to detect actual file modifications, which is more reliable
        than parsing CLI output.

        Args:
            output: Raw CLI output (used as fallback)

        Returns:
            List of modified file paths
        """
        modified = []

        # Primary method: Use git to detect modified files
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
                cwd=str(self.working_dir),
                timeout=10,
            )
            if result.returncode == 0:
                git_modified = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
                if git_modified:
                    return git_modified
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Fall back to output parsing

        # Fallback: Look for common patterns in output
        patterns = [
            r"(?:Wrote|Created|Updated|Modified|Edited)\s+['\"]?([^\s'\"]+)['\"]?",
            r"Writing to\s+['\"]?([^\s'\"]+)['\"]?",
            r"File saved:\s+['\"]?([^\s'\"]+)['\"]?",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            modified.extend(matches)

        return list(set(modified))  # Remove duplicates


class OpenAILLMProvider(LLMProvider):
    """LLM provider using the OpenAI-compatible Chat Completions API.

    Works with OpenAI, Azure OpenAI, and any provider exposing
    a compatible /v1/chat/completions endpoint (set OPENAI_BASE_URL).

    Requires OPENAI_API_KEY environment variable.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: LLMConfig, timeout: int = 300):
        super().__init__(config)
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for the OpenAI provider"
            )
        return key

    @property
    def base_url(self) -> str:
        return os.environ.get("OPENAI_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")

    def _chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_format: dict | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if response_format:
            body["response_format"] = response_format

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._chat(prompt, system_prompt)

    def generate_json(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        json_prompt = f"{prompt}\n\nRespond with valid JSON only. No markdown code blocks."
        text = self._chat(
            json_prompt,
            system_prompt,
            response_format={"type": "json_object"},
        )
        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        json_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_block:
            text = json_block.group(1).strip()
        json_obj = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_obj:
            text = json_obj.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON from OpenAI response: {e}\nResponse: {text[:500]}")


class AnthropicLLMProvider(LLMProvider):
    """LLM provider using the Anthropic Messages API directly.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com"

    def __init__(self, config: LLMConfig, timeout: int = 300):
        super().__init__(config)
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for the Anthropic provider"
            )
        return key

    @property
    def base_url(self) -> str:
        return os.environ.get("ANTHROPIC_BASE_URL", self.DEFAULT_BASE_URL).rstrip("/")

    def _chat(self, prompt: str, system_prompt: str | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        resp = httpx.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        return data["content"][0]["text"]

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._chat(prompt, system_prompt)

    def generate_json(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        json_prompt = f"{prompt}\n\nRespond with valid JSON only. No markdown code blocks."
        text = self._chat(json_prompt, system_prompt)
        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        json_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_block:
            text = json_block.group(1).strip()
        json_obj = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_obj:
            text = json_obj.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse JSON from Anthropic response: {e}\nResponse: {text[:500]}"
            )


def get_llm_provider(config: Config | None = None) -> LLMProvider:
    """Get the appropriate LLM provider based on configuration.

    Args:
        config: Configuration object. If None, loads default config.

    Returns:
        An LLM provider instance.

    Raises:
        ValueError: If provider name is not recognized.
    """
    if config is None:
        from ..config import load_config

        config = load_config()

    provider_name = config.llm.provider.lower()

    if provider_name == "mock":
        return MockLLMProvider(config.llm)
    elif provider_name == "claude-code":
        return ClaudeCodeLLMProvider(config.llm)
    elif provider_name == "anthropic":
        return AnthropicLLMProvider(config.llm)
    elif provider_name == "openai":
        return OpenAILLMProvider(config.llm)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
