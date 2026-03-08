"""Tests for scene generation module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scenes.generator import (
    SceneGenerator,
    SCENE_SYSTEM_PROMPT,
    SCENE_GENERATION_PROMPT,
    STYLES_TEMPLATE,
    INDEX_TEMPLATE,
)
from src.cli.main import _title_to_scene_key as cli_title_to_scene_key


class TestSceneGeneratorHelpers:
    """Tests for SceneGenerator helper methods."""

    @pytest.fixture
    def generator(self):
        return SceneGenerator()

    def test_title_to_scene_key_simple(self, generator):
        """Test simple title to scene key conversion."""
        assert generator._title_to_scene_key("The Hook") == "hook"

    def test_title_to_scene_key_multiple_words(self, generator):
        """Test multi-word title to scene key."""
        assert generator._title_to_scene_key("The Tokenization Challenge") == "tokenization_challenge"

    def test_title_to_scene_key_with_article_prefix(self, generator):
        """Test title starting with article."""
        assert generator._title_to_scene_key("A New Beginning") == "new_beginning"
        assert generator._title_to_scene_key("An Example Scene") == "example_scene"

    def test_title_to_scene_key_long_title(self, generator):
        """Test longer title conversion."""
        assert generator._title_to_scene_key("Cutting Images Into Visual Words") == "cutting_images_into_visual_words"

    def test_title_to_scene_key_special_chars(self, generator):
        """Test title with special characters."""
        assert generator._title_to_scene_key("What's Next?") == "whats_next"

    def test_title_to_scene_key_with_numbers(self, generator):
        """Test title with numbers."""
        assert generator._title_to_scene_key("Phase 1 Introduction") == "phase_1_introduction"

    def test_title_to_component_name_simple(self, generator):
        """Test simple title conversion."""
        assert generator._title_to_component_name("The Hook") == "TheHookScene"

    def test_title_to_component_name_with_special_chars(self, generator):
        """Test title with special characters."""
        assert generator._title_to_component_name("What's Next?") == "WhatsNextScene"

    def test_title_to_component_name_numbers(self, generator):
        """Test title with numbers."""
        assert generator._title_to_component_name("Phase 1 Introduction") == "Phase1IntroductionScene"

    def test_title_to_component_name_single_word(self, generator):
        """Test single word title."""
        assert generator._title_to_component_name("Conclusion") == "ConclusionScene"

    def test_component_to_registry_key_simple(self, generator):
        """Test simple component name to key conversion."""
        assert generator._component_to_registry_key("TheHookScene") == "the_hook"

    def test_component_to_registry_key_complex(self, generator):
        """Test complex component name to key conversion."""
        # Note: consecutive capitals are not split (KV stays together)
        assert generator._component_to_registry_key("KVCacheExplanationScene") == "kvcache_explanation"

    def test_title_to_registry_name(self, generator):
        """Test registry name generation."""
        assert generator._title_to_registry_name("LLM Image Understanding") == "LLM_IMAGE_UNDERSTANDING_SCENES"

    def test_title_to_registry_name_long(self, generator):
        """Test registry name generation with long title."""
        result = generator._title_to_registry_name("A Very Long Project Title Here")
        assert result == "A_VERY_LONG_SCENES"

    def test_extract_code_from_markdown(self, generator):
        """Test code extraction from markdown response."""
        response = '''Here's the code:

```typescript
import React from "react";
export const TestScene = () => <div>Test</div>;
```

That's the component.'''
        code = generator._extract_code(response)
        assert code is not None
        assert "import React" in code
        assert "TestScene" in code

    def test_extract_code_from_tsx_block(self, generator):
        """Test code extraction from tsx code block."""
        response = '''```tsx
import React from "react";
export const TestScene = () => <div>Test</div>;
```'''
        code = generator._extract_code(response)
        assert code is not None
        assert "TestScene" in code

    def test_extract_code_plain_code(self, generator):
        """Test code extraction when response is plain code."""
        response = '''import React from "react";
export const TestScene = () => <div>Test</div>;'''
        code = generator._extract_code(response)
        assert code is not None
        assert "import" in code
        assert "export" in code

    def test_extract_code_no_code(self, generator):
        """Test code extraction when no code present."""
        response = "This is just a text response without any code."
        code = generator._extract_code(response)
        assert code is None


class TestStylesTemplate:
    """Tests for styles template generation."""

    def test_styles_template_has_colors(self):
        """Test that styles template includes colors."""
        content = STYLES_TEMPLATE.format(project_title="Test Project", sidebar_width=0)
        assert "COLORS" in content
        assert "#0066FF" in content  # primary
        assert "#FF6600" in content  # secondary

    def test_styles_template_has_typography(self):
        """Test that styles template includes fonts."""
        content = STYLES_TEMPLATE.format(project_title="Test Project", sidebar_width=0)
        assert "FONTS" in content
        assert "fontSize" in content

    def test_styles_template_has_helpers(self):
        """Test that styles template includes helper functions."""
        content = STYLES_TEMPLATE.format(project_title="Test Project", sidebar_width=0)
        assert "getSceneIndicatorStyle" in content
        assert "getScale" in content

    def test_styles_template_project_title(self):
        """Test that project title is included."""
        content = STYLES_TEMPLATE.format(project_title="My Custom Project", sidebar_width=0)
        assert "My Custom Project" in content


class TestIndexTemplate:
    """Tests for index template generation."""

    def test_index_template_structure(self):
        """Test that index template has correct structure."""
        content = INDEX_TEMPLATE.format(
            project_title="Test Project",
            imports='import { TestScene } from "./TestScene";',
            exports='export { TestScene } from "./TestScene";',
            registry_entries='  test: TestScene,',
        )
        assert "TestScene" in content
        assert "PROJECT_SCENES" in content  # Standard export name
        assert "SCENE_REGISTRY" in content  # Internal registry
        assert "getScene" in content
        assert "getAvailableSceneTypes" in content


class TestSceneGeneratorWithMocks:
    """Tests for SceneGenerator with mocked LLM."""

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory with script."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        # Create script directory and file
        script_dir = project_dir / "script"
        script_dir.mkdir()

        script_data = {
            "title": "Test Video",
            "total_duration_seconds": 120,
            "scenes": [
                {
                    "scene_id": 1,
                    "scene_type": "hook",
                    "title": "The Opening Hook",
                    "voiceover": "This is the opening narration.",
                    "visual_description": "Show an animated title card.",
                    "key_elements": ["title", "animation"],
                    "duration_seconds": 20,
                },
                {
                    "scene_id": 2,
                    "scene_type": "explanation",
                    "title": "Core Concept",
                    "voiceover": "Here's how it works.",
                    "visual_description": "Diagram showing the process.",
                    "key_elements": ["diagram", "arrows"],
                    "duration_seconds": 30,
                },
            ],
        }

        with open(script_dir / "script.json", "w") as f:
            json.dump(script_data, f)

        return project_dir

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response with scene code."""
        return MagicMock(
            success=True,
            response='''import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";

export const TestScene: React.FC = () => {
  const frame = useCurrentFrame();
  return <AbsoluteFill>Test</AbsoluteFill>;
};''',
            modified_files=[],
        )

    def test_generate_styles(self, temp_project_dir):
        """Test styles.ts generation."""
        generator = SceneGenerator()
        scenes_dir = temp_project_dir / "scenes"
        scenes_dir.mkdir()

        generator._generate_styles(scenes_dir, "Test Project")

        styles_path = scenes_dir / "styles.ts"
        assert styles_path.exists()

        content = styles_path.read_text()
        assert "COLORS" in content
        assert "Test Project" in content

    def test_generate_index(self, temp_project_dir):
        """Test index.ts generation."""
        generator = SceneGenerator()
        scenes_dir = temp_project_dir / "scenes"
        scenes_dir.mkdir()

        scenes = [
            {"scene_number": 1, "title": "The Hook", "component_name": "HookScene", "filename": "HookScene.tsx", "scene_type": "hook", "scene_key": "hook"},
            {"scene_number": 2, "title": "Core Explanation", "component_name": "ExplanationScene", "filename": "ExplanationScene.tsx", "scene_type": "explanation", "scene_key": "core_explanation"},
        ]

        generator._generate_index(scenes_dir, scenes, "Test Video")

        index_path = scenes_dir / "index.ts"
        assert index_path.exists()

        content = index_path.read_text()
        assert "HookScene" in content
        assert "ExplanationScene" in content
        assert "PROJECT_SCENES" in content  # Standard export name
        # Registry keys should match scene_key (derived from title)
        assert "hook: HookScene" in content
        assert "core_explanation: ExplanationScene" in content

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_scene_creates_file(self, mock_get_provider, temp_project_dir, mock_llm_response):
        """Test that scene generation creates a file."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        generator = SceneGenerator()
        scenes_dir = temp_project_dir / "scenes"
        scenes_dir.mkdir()

        scene = {
            "title": "Test Scene",
            "scene_type": "hook",
            "duration_seconds": 20,
            "voiceover": "Test narration",
            "visual_description": "Test visual",
            "key_elements": ["element1"],
        }

        result = generator._generate_scene(
            scene=scene,
            scene_number=1,
            scenes_dir=scenes_dir,
            example_scene="// Example scene code",
        )

        assert result["component_name"] == "TestSceneScene"
        assert result["filename"] == "TestSceneScene.tsx"
        assert (scenes_dir / "TestSceneScene.tsx").exists()

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_all_scenes(self, mock_get_provider, temp_project_dir, mock_llm_response):
        """Test full scene generation pipeline."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        generator = SceneGenerator()

        results = generator.generate_all_scenes(
            project_dir=temp_project_dir,
            force=True,
        )

        assert len(results["scenes"]) == 2
        assert len(results["errors"]) == 0

        scenes_dir = temp_project_dir / "scenes"
        assert (scenes_dir / "styles.ts").exists()
        assert (scenes_dir / "index.ts").exists()

    def test_generate_scenes_no_script_error(self, tmp_path):
        """Test error when script doesn't exist."""
        project_dir = tmp_path / "empty-project"
        project_dir.mkdir()

        generator = SceneGenerator()

        with pytest.raises(FileNotFoundError) as exc_info:
            generator.generate_all_scenes(project_dir=project_dir)

        assert "Script not found" in str(exc_info.value)

    def test_generate_scenes_existing_no_force(self, temp_project_dir):
        """Test error when scenes exist without force flag."""
        scenes_dir = temp_project_dir / "scenes"
        scenes_dir.mkdir()
        (scenes_dir / "ExistingScene.tsx").write_text("// existing")

        generator = SceneGenerator()

        with pytest.raises(FileExistsError) as exc_info:
            generator.generate_all_scenes(project_dir=temp_project_dir)

        assert "Use --force" in str(exc_info.value)


class TestScenePrompts:
    """Tests for scene generation prompts."""

    def test_system_prompt_has_remotion_guidance(self):
        """Test system prompt includes Remotion guidance."""
        assert "Remotion" in SCENE_SYSTEM_PROMPT
        assert "useCurrentFrame" in SCENE_SYSTEM_PROMPT
        assert "interpolate" in SCENE_SYSTEM_PROMPT

    def test_system_prompt_has_animation_principles(self):
        """Test system prompt includes animation principles."""
        assert "Frame-based timing" in SCENE_SYSTEM_PROMPT
        assert "spring" in SCENE_SYSTEM_PROMPT

    def test_system_prompt_has_color_guidance(self):
        """Test system prompt includes color guidance."""
        assert "#00d9ff" in SCENE_SYSTEM_PROMPT
        assert "primary" in SCENE_SYSTEM_PROMPT

    def test_generation_prompt_template_has_placeholders(self):
        """Test generation prompt has required placeholders."""
        assert "{scene_number}" in SCENE_GENERATION_PROMPT
        assert "{title}" in SCENE_GENERATION_PROMPT
        assert "{voiceover}" in SCENE_GENERATION_PROMPT
        assert "{visual_description}" in SCENE_GENERATION_PROMPT
        assert "{component_name}" in SCENE_GENERATION_PROMPT

    def test_generation_prompt_formatting(self):
        """Test that generation prompt can be formatted."""
        formatted = SCENE_GENERATION_PROMPT.format(
            scene_number=1,
            title="Test Scene",
            scene_type="hook",
            duration=20,
            total_frames=600,
            voiceover="Test narration",
            visual_description="Test visual",
            elements="- element1\n- element2",
            component_name="TestScene",
            example_scene="// Example code",
            output_path="/path/to/scene.tsx",
            word_timestamps_section="## Word Timestamps (test section)",
        )
        assert "Test Scene" in formatted
        assert "TestScene" in formatted
        assert "600 frames" in formatted


class TestSceneGeneratorLoadExample:
    """Tests for loading example scenes."""

    def test_load_example_from_llm_inference(self):
        """Test loading example from llm-inference project."""
        # This test only works if the llm-inference project exists
        generator = SceneGenerator()
        example = generator._load_example_scene()

        # Should find an example or return placeholder
        assert example is not None
        assert len(example) > 0

    def test_load_example_from_custom_dir(self, tmp_path):
        """Test loading example from custom directory."""
        example_dir = tmp_path / "examples"
        example_dir.mkdir()
        (example_dir / "TestScene.tsx").write_text("// Test scene code")

        generator = SceneGenerator()
        example = generator._load_example_scene(example_dir)

        assert "Test scene code" in example

    def test_load_example_empty_dir(self, tmp_path):
        """Test loading example from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        generator = SceneGenerator()
        example = generator._load_example_scene(empty_dir)

        assert "No example scene available" in example


class TestSceneGeneratorOldFormat:
    """Tests for handling old script format with visual_cue."""

    @pytest.fixture
    def old_format_script(self, tmp_path):
        """Create script with old visual_cue format."""
        project_dir = tmp_path / "old-format"
        project_dir.mkdir()
        script_dir = project_dir / "script"
        script_dir.mkdir()

        script_data = {
            "title": "Old Format Video",
            "scenes": [
                {
                    "scene_id": 1,
                    "scene_type": "hook",
                    "title": "Hook Scene",
                    "voiceover": "Test narration",
                    "visual_cue": {
                        "description": "Old format visual description",
                        "visual_type": "animation",
                        "elements": ["element1", "element2"],
                        "duration_seconds": 20,
                    },
                    "duration_seconds": 20,
                },
            ],
        }

        with open(script_dir / "script.json", "w") as f:
            json.dump(script_data, f)

        return project_dir

    @patch("src.scenes.generator.get_llm_provider")
    def test_handles_old_visual_cue_format(self, mock_get_provider, old_format_script):
        """Test that old visual_cue format is handled correctly."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = MagicMock(
            success=True,
            response='import React from "react"; export const HookSceneScene = () => null;',
        )
        mock_get_provider.return_value = mock_llm

        generator = SceneGenerator()
        results = generator.generate_all_scenes(
            project_dir=old_format_script,
            force=True,
        )

        assert len(results["scenes"]) == 1
        assert len(results["errors"]) == 0

        # Verify the prompt included the visual description
        call_args = mock_llm.generate_with_file_access.call_args
        prompt = call_args[0][0]
        assert "Old format visual description" in prompt


class TestWordTimestampFormatting:
    """Tests for word timestamp formatting for animation-to-narration sync."""

    @pytest.fixture
    def generator(self):
        return SceneGenerator()

    @pytest.fixture
    def sample_word_timestamps(self):
        """Sample word timestamps like those from voiceover manifest."""
        return [
            {"word": "You", "start_seconds": 0.0, "end_seconds": 0.28},
            {"word": "type", "start_seconds": 0.28, "end_seconds": 0.54},
            {"word": "a", "start_seconds": 0.54, "end_seconds": 0.7},
            {"word": "question,", "start_seconds": 0.7, "end_seconds": 1.06},
            {"word": "a", "start_seconds": 1.72, "end_seconds": 1.88},
            {"word": "quarter", "start_seconds": 1.88, "end_seconds": 2.1},
            {"word": "second", "start_seconds": 2.1, "end_seconds": 2.46},
            {"word": "later,", "start_seconds": 2.46, "end_seconds": 2.84},
            {"word": "and", "start_seconds": 3.18, "end_seconds": 3.3},
            {"word": "answer.", "start_seconds": 3.3, "end_seconds": 3.64},
        ]

    def test_format_with_timestamps_includes_critical_header(self, generator, sample_word_timestamps):
        """Test that formatted output includes critical timing header."""
        result = generator._format_word_timestamps_for_prompt(
            sample_word_timestamps, "You type a question", 6.0
        )
        assert "USE THESE FOR ANIMATION TIMING" in result
        assert "CRITICAL" in result

    def test_format_with_timestamps_includes_frame_numbers(self, generator, sample_word_timestamps):
        """Test that frame numbers are calculated correctly (30fps)."""
        result = generator._format_word_timestamps_for_prompt(
            sample_word_timestamps, "You type a question", 6.0
        )
        # "You" at 0.0s should be frame 0
        assert "frame 0" in result
        # "type" at 0.28s should be frame 8 (0.28 * 30 = 8.4 -> 8)
        assert "frame 8" in result

    def test_format_with_timestamps_includes_word_timeline(self, generator, sample_word_timestamps):
        """Test that word timeline is included."""
        result = generator._format_word_timestamps_for_prompt(
            sample_word_timestamps, "You type a question", 6.0
        )
        assert '"You"' in result
        assert '"type"' in result
        assert '"question,"' in result

    def test_format_with_timestamps_includes_scene_duration(self, generator, sample_word_timestamps):
        """Test that scene duration is included."""
        result = generator._format_word_timestamps_for_prompt(
            sample_word_timestamps, "You type a question", 6.0
        )
        assert "6.00s" in result
        assert "180 frames" in result  # 6 * 30 = 180

    def test_format_with_timestamps_includes_do_not_guidance(self, generator, sample_word_timestamps):
        """Test that DO NOT guidance is included."""
        result = generator._format_word_timestamps_for_prompt(
            sample_word_timestamps, "You type a question", 6.0
        )
        assert "DO NOT" in result
        assert "percentage-based timing" in result.lower()

    def test_format_with_empty_timestamps_fallback(self, generator):
        """Test fallback when no word timestamps available."""
        result = generator._format_word_timestamps_for_prompt(
            [], "Some voiceover text", 10.0
        )
        assert "NOT AVAILABLE" in result
        assert "percentage-based timing as a fallback" in result
        assert "phase1:" in result
        assert "phase2:" in result

    def test_format_with_long_timestamps_truncates(self, generator):
        """Test that very long word lists are truncated."""
        # Create 50 word timestamps
        long_timestamps = [
            {"word": f"word{i}", "start_seconds": i * 0.5, "end_seconds": i * 0.5 + 0.4}
            for i in range(50)
        ]
        result = generator._format_word_timestamps_for_prompt(
            long_timestamps, "Long narration", 25.0
        )
        # Should show first 20 and last 10, with ellipsis
        assert "..." in result
        assert "more words" in result

    def test_format_preserves_word_order(self, generator, sample_word_timestamps):
        """Test that words appear in chronological order."""
        result = generator._format_word_timestamps_for_prompt(
            sample_word_timestamps, "You type a question", 6.0
        )
        # Find positions of key words
        you_pos = result.find('"You"')
        type_pos = result.find('"type"')
        question_pos = result.find('"question,"')

        assert you_pos < type_pos < question_pos

    def test_format_with_special_characters_in_words(self, generator):
        """Test handling of punctuation in words."""
        timestamps = [
            {"word": "What's", "start_seconds": 0.0, "end_seconds": 0.5},
            {"word": "next?", "start_seconds": 0.5, "end_seconds": 1.0},
        ]
        result = generator._format_word_timestamps_for_prompt(
            timestamps, "What's next?", 2.0
        )
        assert "What's" in result
        assert "next?" in result


class TestGenerateAllScenesWithManifest:
    """Tests for generate_all_scenes with voiceover manifest integration."""

    @pytest.fixture
    def temp_project_with_manifest(self, tmp_path):
        """Create a temporary project with script and voiceover manifest."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        # Create script directory and file
        script_dir = project_dir / "script"
        script_dir.mkdir()

        script_data = {
            "title": "Test Video",
            "total_duration_seconds": 60,
            "scenes": [
                {
                    "scene_id": "scene1_hook",
                    "scene_type": "hook",
                    "title": "The Opening Hook",
                    "voiceover": "This is the opening narration for the hook.",
                    "visual_description": "Show an animated title card.",
                    "key_elements": ["title", "animation"],
                    "duration_seconds": 20,
                },
                {
                    "scene_id": "scene2_explanation",
                    "scene_type": "explanation",
                    "title": "Core Concept",
                    "voiceover": "Here is how it works step by step.",
                    "visual_description": "Diagram showing the process.",
                    "key_elements": ["diagram", "arrows"],
                    "duration_seconds": 40,
                },
            ],
        }

        with open(script_dir / "script.json", "w") as f:
            json.dump(script_data, f)

        # Create voiceover directory and manifest
        voiceover_dir = project_dir / "voiceover"
        voiceover_dir.mkdir()

        manifest_data = {
            "scenes": [
                {
                    "scene_id": "scene1_hook",
                    "audio_path": str(voiceover_dir / "scene1_hook.mp3"),
                    "duration_seconds": 20.0,
                    "word_timestamps": [
                        {"word": "This", "start_seconds": 0.0, "end_seconds": 0.3},
                        {"word": "is", "start_seconds": 0.3, "end_seconds": 0.5},
                        {"word": "the", "start_seconds": 0.5, "end_seconds": 0.7},
                        {"word": "opening", "start_seconds": 0.7, "end_seconds": 1.1},
                        {"word": "narration", "start_seconds": 1.1, "end_seconds": 1.6},
                    ],
                },
                {
                    "scene_id": "scene2_explanation",
                    "audio_path": str(voiceover_dir / "scene2_explanation.mp3"),
                    "duration_seconds": 40.0,
                    "word_timestamps": [
                        {"word": "Here", "start_seconds": 0.0, "end_seconds": 0.3},
                        {"word": "is", "start_seconds": 0.3, "end_seconds": 0.5},
                        {"word": "how", "start_seconds": 0.5, "end_seconds": 0.8},
                        {"word": "it", "start_seconds": 0.8, "end_seconds": 1.0},
                        {"word": "works", "start_seconds": 1.0, "end_seconds": 1.4},
                    ],
                },
            ]
        }

        with open(voiceover_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        return project_dir

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response with scene code."""
        return MagicMock(
            success=True,
            response='''import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";

export const TestScene: React.FC = () => {
  const frame = useCurrentFrame();
  return <AbsoluteFill>Test</AbsoluteFill>;
};''',
            modified_files=[],
        )

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_with_manifest_passes_timestamps(
        self, mock_get_provider, temp_project_with_manifest, mock_llm_response
    ):
        """Test that word timestamps from manifest are passed to scene generation."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        generator = SceneGenerator()
        manifest_path = temp_project_with_manifest / "voiceover" / "manifest.json"

        results = generator.generate_all_scenes(
            project_dir=temp_project_with_manifest,
            voiceover_manifest_path=manifest_path,
            force=True,
        )

        assert len(results["scenes"]) == 2
        assert len(results["errors"]) == 0

        # Verify the prompts included word timestamp information
        calls = mock_llm.generate_with_file_access.call_args_list
        assert len(calls) == 2

        # First scene prompt should include timestamps
        first_prompt = calls[0][0][0]
        assert "USE THESE FOR ANIMATION TIMING" in first_prompt
        assert '"This"' in first_prompt or '"opening"' in first_prompt

        # Second scene prompt should include timestamps
        second_prompt = calls[1][0][0]
        assert "USE THESE FOR ANIMATION TIMING" in second_prompt
        assert '"Here"' in second_prompt or '"works"' in second_prompt

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_without_manifest_uses_fallback(
        self, mock_get_provider, temp_project_with_manifest, mock_llm_response
    ):
        """Test that generation works without manifest using fallback timing."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        generator = SceneGenerator()

        # Don't pass manifest path
        results = generator.generate_all_scenes(
            project_dir=temp_project_with_manifest,
            voiceover_manifest_path=None,
            force=True,
        )

        assert len(results["scenes"]) == 2

        # Prompts should use fallback timing
        calls = mock_llm.generate_with_file_access.call_args_list
        first_prompt = calls[0][0][0]
        assert "NOT AVAILABLE" in first_prompt or "percentage-based timing as a fallback" in first_prompt

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_with_nonexistent_manifest(
        self, mock_get_provider, temp_project_with_manifest, mock_llm_response
    ):
        """Test that generation works when manifest path doesn't exist."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        generator = SceneGenerator()
        nonexistent_path = temp_project_with_manifest / "nonexistent" / "manifest.json"

        results = generator.generate_all_scenes(
            project_dir=temp_project_with_manifest,
            voiceover_manifest_path=nonexistent_path,
            force=True,
        )

        # Should still work, just without timestamps
        assert len(results["scenes"]) == 2

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_with_missing_scene_in_manifest(
        self, mock_get_provider, temp_project_with_manifest, mock_llm_response
    ):
        """Test handling when manifest doesn't have timestamps for all scenes."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        # Modify manifest to only have one scene
        manifest_path = temp_project_with_manifest / "voiceover" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["scenes"] = [manifest["scenes"][0]]  # Only keep first scene
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        generator = SceneGenerator()

        results = generator.generate_all_scenes(
            project_dir=temp_project_with_manifest,
            voiceover_manifest_path=manifest_path,
            force=True,
        )

        assert len(results["scenes"]) == 2

        # First scene should have timestamps, second should use fallback
        calls = mock_llm.generate_with_file_access.call_args_list
        first_prompt = calls[0][0][0]
        second_prompt = calls[1][0][0]

        assert "USE THESE FOR ANIMATION TIMING" in first_prompt
        assert "NOT AVAILABLE" in second_prompt


class TestGenerateSceneWithWordTimestamps:
    """Tests for _generate_scene method with word timestamps."""

    @pytest.fixture
    def generator(self):
        return SceneGenerator()

    @pytest.fixture
    def sample_scene(self):
        return {
            "title": "Test Scene",
            "scene_type": "explanation",
            "duration_seconds": 30,
            "voiceover": "This explains the concept clearly.",
            "visual_description": "Animated diagram",
            "key_elements": ["diagram", "labels"],
        }

    @pytest.fixture
    def sample_timestamps(self):
        return [
            {"word": "This", "start_seconds": 0.0, "end_seconds": 0.3},
            {"word": "explains", "start_seconds": 0.3, "end_seconds": 0.7},
            {"word": "the", "start_seconds": 0.7, "end_seconds": 0.9},
            {"word": "concept", "start_seconds": 0.9, "end_seconds": 1.4},
            {"word": "clearly.", "start_seconds": 1.4, "end_seconds": 1.9},
        ]

    @pytest.fixture
    def mock_llm_response(self):
        return MagicMock(
            success=True,
            response='''import React from "react";
import { AbsoluteFill } from "remotion";
export const TestSceneScene: React.FC = () => <AbsoluteFill>Test</AbsoluteFill>;''',
            modified_files=[],
        )

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_scene_includes_timestamps_in_prompt(
        self, mock_get_provider, generator, sample_scene, sample_timestamps, mock_llm_response, tmp_path
    ):
        """Test that _generate_scene includes word timestamps in the prompt."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        generator._generate_scene(
            scene=sample_scene,
            scene_number=1,
            scenes_dir=scenes_dir,
            example_scene="// Example",
            word_timestamps=sample_timestamps,
        )

        # Verify prompt included timestamps
        call_args = mock_llm.generate_with_file_access.call_args
        prompt = call_args[0][0]
        assert "USE THESE FOR ANIMATION TIMING" in prompt
        assert '"This"' in prompt
        assert '"concept"' in prompt

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_scene_without_timestamps_uses_fallback(
        self, mock_get_provider, generator, sample_scene, mock_llm_response, tmp_path
    ):
        """Test that _generate_scene uses fallback when no timestamps provided."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        generator._generate_scene(
            scene=sample_scene,
            scene_number=1,
            scenes_dir=scenes_dir,
            example_scene="// Example",
            word_timestamps=None,
        )

        # Verify prompt used fallback
        call_args = mock_llm.generate_with_file_access.call_args
        prompt = call_args[0][0]
        assert "NOT AVAILABLE" in prompt

    @patch("src.scenes.generator.get_llm_provider")
    def test_generate_scene_with_empty_timestamps_uses_fallback(
        self, mock_get_provider, generator, sample_scene, mock_llm_response, tmp_path
    ):
        """Test that _generate_scene uses fallback when timestamps list is empty."""
        from src.understanding.llm_provider import ClaudeCodeLLMProvider
        mock_llm = MagicMock(spec=ClaudeCodeLLMProvider)
        mock_llm.generate_with_file_access.return_value = mock_llm_response
        mock_get_provider.return_value = mock_llm

        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        generator._generate_scene(
            scene=sample_scene,
            scene_number=1,
            scenes_dir=scenes_dir,
            example_scene="// Example",
            word_timestamps=[],
        )

        # Verify prompt used fallback
        call_args = mock_llm.generate_with_file_access.call_args
        prompt = call_args[0][0]
        assert "NOT AVAILABLE" in prompt


class TestScenePromptWithTimestamps:
    """Tests for SCENE_GENERATION_PROMPT with word_timestamps_section placeholder."""

    def test_prompt_has_word_timestamps_placeholder(self):
        """Test that prompt template includes word_timestamps_section placeholder."""
        assert "{word_timestamps_section}" in SCENE_GENERATION_PROMPT

    def test_prompt_can_be_formatted_with_timestamps_section(self):
        """Test that prompt can be formatted with timestamps section."""
        timestamps_section = """
## Word Timestamps (USE THESE FOR ANIMATION TIMING)
- "test" at 0.0s (frame 0)
"""
        formatted = SCENE_GENERATION_PROMPT.format(
            scene_number=1,
            title="Test Scene",
            scene_type="hook",
            duration=20,
            total_frames=600,
            voiceover="Test narration",
            visual_description="Test visual",
            elements="- element1",
            component_name="TestScene",
            example_scene="// Example",
            output_path="/path/to/scene.tsx",
            word_timestamps_section=timestamps_section,
        )
        assert "USE THESE FOR ANIMATION TIMING" in formatted

    def test_prompt_step1_references_word_timestamps(self):
        """Test that STEP 1 guidance references word timestamps."""
        assert "Sync Animations to Narration" in SCENE_GENERATION_PROMPT
        assert "word timestamps" in SCENE_GENERATION_PROMPT.lower()

    def test_prompt_warns_against_percentage_timing(self):
        """Test that prompt warns against percentage-based timing."""
        assert "DON'T DO THIS" in SCENE_GENERATION_PROMPT or "NEVER DO THIS" in SCENE_GENERATION_PROMPT
        assert "percentage" in SCENE_GENERATION_PROMPT.lower()

    def test_prompt_shows_correct_timing_example(self):
        """Test that prompt shows correct timing approach."""
        assert "// GOOD:" in SCENE_GENERATION_PROMPT
        assert "// BAD:" in SCENE_GENERATION_PROMPT

    def test_prompt_requirement_9_updated(self):
        """Test that requirement 9 mentions word timestamps, not percentages."""
        # Requirement 9 should say timestamps, not "proportional to durationInFrames"
        assert "word timestamps" in SCENE_GENERATION_PROMPT.lower()
        # The old guidance should be gone
        assert "proportional to durationInFrames" not in SCENE_GENERATION_PROMPT


class TestSceneKeyConsistency:
    """Tests to ensure scene key generation is consistent across CLI and scene generator.

    This is critical: if these functions produce different keys, scenes won't be found
    at render time because the storyboard type won't match the registry key.
    """

    @pytest.fixture
    def generator(self):
        return SceneGenerator()

    def test_cli_and_generator_produce_same_keys(self, generator):
        """CRITICAL: CLI and scene generator must produce identical keys from titles."""
        titles = [
            "The Hook",
            "The Tokenization Challenge",
            "Cutting Images Into Visual Words",
            "The Special Summary Token",
            "A New Approach",
            "What's Next?",
            "Phase 1 Introduction",
            "Vision Meets Language",
            "The Vision Revolution",
        ]

        for title in titles:
            generator_key = generator._title_to_scene_key(title)
            cli_key = cli_title_to_scene_key(title)
            assert generator_key == cli_key, (
                f"Key mismatch for title '{title}': "
                f"generator='{generator_key}', cli='{cli_key}'"
            )

    def test_actual_project_titles_produce_consistent_keys(self, generator):
        """Test with actual titles from llm-image-understanding project."""
        # These are the actual titles from the project
        titles = [
            "The Pixel Problem",
            "The Tokenization Challenge",
            "Cutting Images Into Visual Words",
            "The Special Summary Token",
            "Teaching Patches Their Location",
            "Every Patch Talks to Every Patch",
            "Learning by Hiding",
            "Vision Meets Language",
            "Adding the Time Dimension",
            "Divided Space-Time Attention",
            "How LLMs See Images",
            "The Resolution Trade-off",
            "The Key Parallel",
            "Training Without Next-Token Prediction",
            "The Vision Revolution",
        ]

        for title in titles:
            generator_key = generator._title_to_scene_key(title)
            cli_key = cli_title_to_scene_key(title)
            assert generator_key == cli_key, (
                f"Key mismatch for actual project title '{title}': "
                f"generator='{generator_key}', cli='{cli_key}'"
            )


class TestSyncSceneTiming:
    """Tests for the sync_all_scenes and _sync_scene_timing methods."""

    @pytest.fixture
    def project_with_scenes(self, tmp_path):
        """Create a project with scenes and voiceover manifest for sync testing."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        # Create script with scene info
        script_dir = project_dir / "script"
        script_dir.mkdir()
        script = {
            "title": "Test Video",
            "scenes": [
                {
                    "scene_id": "scene1_hook",
                    "title": "The Hook",
                    "scene_type": "hook",
                    "voiceover": "This is a test hook for our video.",
                    "duration_seconds": 15.0,
                },
                {
                    "scene_id": "scene2_context",
                    "title": "The Context",
                    "scene_type": "context",
                    "voiceover": "Here is some context for the topic.",
                    "duration_seconds": 20.0,
                }
            ]
        }
        with open(script_dir / "script.json", "w") as f:
            json.dump(script, f)

        # Create scenes
        scenes_dir = project_dir / "scenes"
        scenes_dir.mkdir()

        hook_scene = '''import { AbsoluteFill, useCurrentFrame } from "remotion";
import { COLORS } from "./styles";

export const HookScene = () => {
  const frame = useCurrentFrame();
  // Old timing: "test" at frame 60 (2.0s)
  const phase1End = 60;
  const phase2End = 150;

  return (
    <AbsoluteFill style={{ background: COLORS.background }}>
      <h1>Hook Scene</h1>
    </AbsoluteFill>
  );
};
'''
        with open(scenes_dir / "HookScene.tsx", "w") as f:
            f.write(hook_scene)

        context_scene = '''import { AbsoluteFill } from "remotion";
export const ContextScene = () => {
  return <AbsoluteFill>Context</AbsoluteFill>;
};
'''
        with open(scenes_dir / "ContextScene.tsx", "w") as f:
            f.write(context_scene)

        # Create voiceover manifest with timestamps
        voiceover_dir = project_dir / "voiceover"
        voiceover_dir.mkdir()
        manifest = {
            "scenes": [
                {
                    "scene_id": "scene1_hook",
                    "audio_path": str(voiceover_dir / "scene1_hook.mp3"),
                    "duration_seconds": 15.0,
                    "word_timestamps": [
                        {"word": "This", "start_seconds": 0.0, "end_seconds": 0.3},
                        {"word": "is", "start_seconds": 0.3, "end_seconds": 0.5},
                        {"word": "a", "start_seconds": 0.5, "end_seconds": 0.6},
                        {"word": "test", "start_seconds": 3.0, "end_seconds": 3.5},  # Changed from 2.0s to 3.0s
                        {"word": "hook", "start_seconds": 3.5, "end_seconds": 4.0},
                    ]
                },
                {
                    "scene_id": "scene2_context",
                    "audio_path": str(voiceover_dir / "scene2_context.mp3"),
                    "duration_seconds": 20.0,
                    "word_timestamps": [
                        {"word": "Here", "start_seconds": 0.0, "end_seconds": 0.5},
                        {"word": "is", "start_seconds": 0.5, "end_seconds": 0.8},
                    ]
                }
            ]
        }
        with open(voiceover_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        return project_dir

    def test_sync_all_scenes_missing_scenes_dir(self, tmp_path):
        """Test sync fails if scenes directory doesn't exist."""
        from src.scenes.generator import SceneGenerator

        project_dir = tmp_path / "empty-project"
        project_dir.mkdir()

        generator = SceneGenerator(working_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc:
            generator.sync_all_scenes(project_dir)

        assert "Scenes directory not found" in str(exc.value)

    def test_sync_all_scenes_missing_voiceover_manifest(self, tmp_path):
        """Test sync fails if voiceover manifest doesn't exist."""
        from src.scenes.generator import SceneGenerator

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        scenes_dir = project_dir / "scenes"
        scenes_dir.mkdir()
        with open(scenes_dir / "TestScene.tsx", "w") as f:
            f.write("// Test")

        generator = SceneGenerator(working_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc:
            generator.sync_all_scenes(project_dir)

        assert "Voiceover manifest not found" in str(exc.value)

    def test_sync_all_scenes_filters_by_scene(self, project_with_scenes):
        """Test sync can filter to a specific scene."""
        from src.scenes.generator import SceneGenerator
        from unittest.mock import patch, MagicMock

        generator = SceneGenerator(working_dir=project_with_scenes.parent)

        # Mock _sync_scene_timing to avoid LLM calls
        with patch.object(generator, '_sync_scene_timing') as mock_sync:
            results = generator.sync_all_scenes(
                project_dir=project_with_scenes,
                scene_filter="HookScene.tsx"
            )

            # Should only try to sync HookScene
            assert mock_sync.call_count == 1
            call_args = mock_sync.call_args
            assert "HookScene.tsx" in str(call_args[1]["scene_file"])

    def test_sync_all_scenes_skips_scenes_without_timestamps(self, project_with_scenes):
        """Test sync skips scenes that don't have matching timestamps."""
        from src.scenes.generator import SceneGenerator
        from unittest.mock import patch

        # Add a scene that doesn't have timestamps
        extra_scene = project_with_scenes / "scenes" / "ExtraScene.tsx"
        with open(extra_scene, "w") as f:
            f.write("// Extra scene without timestamps")

        generator = SceneGenerator(working_dir=project_with_scenes.parent)

        with patch.object(generator, '_sync_scene_timing'):
            results = generator.sync_all_scenes(project_dir=project_with_scenes)

            # ExtraScene should be in skipped
            skipped_files = [s["filename"] for s in results["skipped"]]
            assert "ExtraScene.tsx" in skipped_files

    def test_sync_all_scenes_scene_not_found(self, project_with_scenes):
        """Test sync fails if filtered scene doesn't exist."""
        from src.scenes.generator import SceneGenerator

        generator = SceneGenerator(working_dir=project_with_scenes.parent)

        with pytest.raises(FileNotFoundError) as exc:
            generator.sync_all_scenes(
                project_dir=project_with_scenes,
                scene_filter="NonexistentScene.tsx"
            )

        assert "Scene not found" in str(exc.value)

    def test_sync_prompt_contains_existing_code(self):
        """Test the sync prompt template includes existing code placeholder."""
        from src.scenes.generator import SYNC_SCENE_PROMPT

        assert "{existing_code}" in SYNC_SCENE_PROMPT
        assert "{word_timestamps_section}" in SYNC_SCENE_PROMPT
        assert "{duration}" in SYNC_SCENE_PROMPT
        assert "{total_frames}" in SYNC_SCENE_PROMPT

    def test_sync_prompt_emphasizes_timing_only(self):
        """Test the sync prompt emphasizes timing-only updates."""
        from src.scenes.generator import SYNC_SCENE_PROMPT

        # Should emphasize what NOT to change
        assert "DO NOT CHANGE" in SYNC_SCENE_PROMPT
        assert "Visual structure" in SYNC_SCENE_PROMPT or "layout" in SYNC_SCENE_PROMPT

        # Should emphasize what TO change
        assert "ONLY UPDATE" in SYNC_SCENE_PROMPT
        assert "Frame numbers" in SYNC_SCENE_PROMPT or "timing" in SYNC_SCENE_PROMPT.lower()
