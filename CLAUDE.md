# CLAUDE.md

## Project Overview

Video Explainer is a system that generates high-quality explainer videos from technical documents (research papers, articles, PDFs, URLs). It combines a Python backend pipeline with a React/Remotion frontend for programmatic video rendering.

## Architecture

```
Document → Ingest → Script → Narration → TTS → Storyboard → Scenes → Render → Video
```

TTS runs **before** storyboard creation because word-level timestamps from audio are required for visual synchronization.

- **Python backend** (`src/`): Pipeline orchestration, LLM integration, audio processing
- **Remotion frontend** (`remotion/`): React-based video rendering with Three.js for 3D

## Common Commands

### Python

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests excluding slow/network-dependent ones
pytest -m "not slow"

# Run a specific test file
pytest tests/test_scenes.py

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Remotion (from `remotion/` directory)

```bash
# Install dependencies
npm install

# Run tests
npm test

# Start Remotion Studio
npm run dev

# Render a composition
npm run render
```

### CLI

```bash
python -m src.cli <command>
```

## Code Style

### Python
- **Formatter/Linter**: ruff (line-length 100, target Python 3.10+)
- **Lint rules**: E, F, I, N, W (E501 ignored)
- **Type checking**: mypy with `disallow_untyped_defs = true`
- All functions require type annotations

### TypeScript
- **Target**: ES2022, strict mode
- **JSX**: react-jsx
- **Tests**: Vitest

## Project Structure

```
src/                    # Python pipeline modules
├── cli/                # CLI entry point
├── ingestion/          # Document parsing (PDF, MD, URL, HTML)
├── script/             # Video script generation
├── narration/          # Scene narration
├── scenes/             # Remotion component generation
├── voiceover/          # TTS audio generation
├── storyboard/         # JSON timeline syncing visuals to audio
├── animation/          # Remotion renderer integration
├── sound/              # SFX planning and generation
├── music/              # AI music generation (MusicGen)
├── composition/        # Audio mixing
├── short/              # Shorts/Reels generation (1080x1920)
├── refine/             # 4-phase quality refinement
├── planning/           # Interactive video planning
├── factcheck/          # Accuracy verification
├── sync/               # Audio-visual timing sync
├── models.py           # Core Pydantic data models
└── config.py           # YAML config loading

remotion/               # Node.js Remotion project
├── src/
│   ├── Root.tsx         # Root composition
│   ├── scenes/          # Scene players & renderers
│   ├── shorts/          # Short-form video components
│   ├── components/      # Reusable animation components
│   ├── types/           # TypeScript type definitions
│   └── utils/           # Utilities (voiceover sync, etc.)
├── package.json
└── tsconfig.json

tests/                  # Python test suite (~1192 tests)
config.yaml             # Application configuration
```

## Configuration

- Global config: `config.yaml` (LLM provider, TTS, video settings, budget limits)
- Per-project config: `projects/<id>/config.json`
- LLM provider set to `claude-code` for production, `mock` for testing
- TTS provider set to `mock` by default; use `elevenlabs` with `ELEVENLABS_API_KEY`

## Testing Notes

- Python tests use pytest with `asyncio_mode = "auto"`
- LLM integration tests are gated behind `--run-llm-tests` flag
- Tests tagged `@pytest.mark.slow` require network access
- JavaScript tests use Vitest (`npm test` in `remotion/`)
- Mock providers exist for both LLM and TTS to enable offline testing

## Dependencies

- **Runtime**: Python 3.10+, Node.js 20+, FFmpeg
- **Key Python packages**: pydantic, click, httpx, edge-tts, pymupdf, rich
- **Key JS packages**: remotion 4.0.242, react 18, three.js, zod, zustand
