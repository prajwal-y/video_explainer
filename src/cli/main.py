"""Main CLI entry point for video explainer pipeline.

Usage:
    python -m src.cli list                                    # List all projects
    python -m src.cli info <project>                          # Show project info
    python -m src.cli create <project_id>                     # Create new project
    python -m src.cli generate <project>                      # Run full pipeline end-to-end
    python -m src.cli generate <project> --from scenes        # Start from a specific step
    python -m src.cli generate <project> --to voiceover       # Stop at a specific step
    python -m src.cli generate <project> --force              # Force regenerate all steps
    python -m src.cli script <project>                        # Generate script from docs
    python -m src.cli script <project> --url <url>            # Generate script from URL
    python -m src.cli script <project> -i doc.pdf             # Generate script from PDF
    python -m src.cli narration <project>                     # Generate narrations
    python -m src.cli scenes <project>                        # Generate Remotion scenes
    python -m src.cli voiceover <project>                     # Generate voiceovers
    python -m src.cli storyboard <project> --view             # View storyboard
    python -m src.cli sound <project> plan                    # Plan sound effects
    python -m src.cli sound <project> library --list          # List sound library
    python -m src.cli sound <project> mix                     # Mix audio
    python -m src.cli music <project> generate                # Generate background music
    python -m src.cli render <project>                        # Render video
    python -m src.cli render <project> -r 4k                  # Render in 4K
    python -m src.cli feedback <project> add "<text>"         # Process feedback
    python -m src.cli feedback <project> list                 # List feedback
    python -m src.cli factcheck <project>                     # Fact-check script
    python -m src.cli refine <project>                        # Refine video quality
    python -m src.cli refine <project> --scene 1              # Refine specific scene

Input formats supported:
    - Markdown files (.md, .markdown)
    - PDF files (.pdf)
    - Web URLs (https://...)

Pipeline workflow:
    1. create    - Create new project with config
    2. script    - Generate script from input documents (MD, PDF, or URL)
    3. narration - Generate narrations for each scene
    4. scenes    - Generate Remotion scene components (React/TypeScript)
    5. voiceover - Generate audio files from narrations
    6. storyboard - Create storyboard linking scenes with audio
    7. sound     - Plan and mix sound effects
    8. music     - Generate AI background music (optional)
    9. render    - Render final video
    10. feedback - Iterate on video with natural language feedback
    11. refine   - Refine video quality to professional standards
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _title_to_scene_key(title: str) -> str:
    """Convert a scene title to a scene key for registry/storyboard.

    This must match the logic in src/scenes/generator.py to ensure
    scene registry keys match storyboard scene types.

    Examples:
        "The Pixel Problem" -> "pixel_problem"
        "The Tokenization Challenge" -> "tokenization_challenge"
        "Cutting Images Into Visual Words" -> "cutting_images_into_visual_words"
    """
    # Remove common prefixes like "The", "A", "An"
    words = title.split()
    if words and words[0].lower() in ("the", "a", "an"):
        words = words[1:]

    # Convert to snake_case: lowercase with underscores
    key = "_".join(word.lower() for word in words)

    # Remove any non-alphanumeric characters except underscores
    key = re.sub(r"[^a-z0-9_]", "", key)

    # Collapse multiple underscores
    key = re.sub(r"_+", "_", key).strip("_")

    return key


def cmd_list(args: argparse.Namespace) -> int:
    """List all available projects."""
    from ..project import list_projects

    projects = list_projects(args.projects_dir)

    if not projects:
        print(f"No projects found in {args.projects_dir}/")
        return 0

    print(f"Found {len(projects)} project(s):\n")
    for project in projects:
        print(f"  {project.id}")
        print(f"    Title: {project.title}")
        print(f"    Path: {project.root_dir}")
        print()

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed project information."""
    from ..project import load_project

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Project: {project.id}")
    print(f"Title: {project.title}")
    print(f"Description: {project.description}")
    print(f"Version: {project.version}")
    print(f"Path: {project.root_dir}")
    print()

    print("Video Settings:")
    print(f"  Resolution: {project.video.width}x{project.video.height}")
    print(f"  FPS: {project.video.fps}")
    print(f"  Target Duration: {project.video.target_duration_seconds}s")
    print()

    print("TTS Settings:")
    print(f"  Provider: {project.tts.provider}")
    print(f"  Voice ID: {project.tts.voice_id}")
    print()

    # Check what files exist
    print("Files:")
    narration_path = project.get_path("narration")
    print(f"  Narrations: {'[exists]' if narration_path.exists() else '[missing]'} {narration_path}")

    voiceover_files = project.get_voiceover_files()
    print(f"  Voiceovers: {len(voiceover_files)} audio files")

    storyboard_path = project.get_path("storyboard")
    print(f"  Storyboard: {'[exists]' if storyboard_path.exists() else '[missing]'} {storyboard_path}")

    output_files = list(project.output_dir.glob("*.mp4"))
    print(f"  Output: {len(output_files)} video files")

    return 0


def cmd_voiceover(args: argparse.Namespace) -> int:
    """Generate voiceovers for a project."""
    from ..project import load_project
    from ..audio import get_tts_provider, ManualVoiceoverProvider
    from ..config import load_config, TTSConfig

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load narrations
    try:
        narrations = project.load_narrations()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Handle --export-script: generate recording script and exit
    if args.export_script:
        return _export_recording_script(project, narrations, args)

    print(f"Generating voiceovers for {project.id}")
    print(f"Found {len(narrations)} scenes")
    print()

    # Determine TTS provider
    provider_name = args.provider or project.tts.provider
    if args.mock:
        provider_name = "mock"

    # Handle manual provider
    if provider_name == "manual":
        if not args.audio_dir:
            print("Error: --audio-dir is required when using manual provider", file=sys.stderr)
            return 1

        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}", file=sys.stderr)
            return 1

        print(f"Using manual voiceover provider")
        print(f"Audio directory: {audio_dir}")

        # Check for missing recordings
        config = load_config()
        tts = ManualVoiceoverProvider(
            config.tts,
            audio_dir=audio_dir,
            whisper_model=args.whisper_model or "base",
        )

        scene_ids = [n.scene_id for n in narrations]
        missing = tts.list_missing_scenes(scene_ids)
        if missing:
            print(f"\nWarning: Missing audio files for {len(missing)} scene(s):")
            for scene_id in missing:
                print(f"  - {scene_id}.mp3 (or .wav, .m4a)")
            if not args.continue_on_error:
                print("\nUse --continue-on-error to skip missing scenes")
                return 1
            print()
    else:
        print(f"Using TTS provider: {provider_name}")

        # Create TTS config
        config = load_config()
        config.tts.provider = provider_name
        if project.tts.voice_id:
            config.tts.voice_id = project.tts.voice_id

        tts = get_tts_provider(config)

    # Generate voiceovers
    output_dir = project.voiceover_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_duration = 0.0

    for narration in narrations:
        print(f"  Processing: {narration.title}...")
        output_path = output_dir / f"{narration.scene_id}.mp3"

        try:
            # ManualVoiceoverProvider needs scene_id passed explicitly
            if provider_name == "manual":
                result = tts.generate_with_timestamps(
                    narration.narration,
                    output_path,
                    scene_id=narration.scene_id,
                )
            else:
                result = tts.generate_with_timestamps(narration.narration, output_path)

            results.append({
                "scene_id": narration.scene_id,
                "audio_path": str(output_path),
                "duration_seconds": result.duration_seconds,
                "word_timestamps": [
                    {
                        "word": ts.word,
                        "start_seconds": ts.start_seconds,
                        "end_seconds": ts.end_seconds,
                    }
                    for ts in result.word_timestamps
                ],
            })
            total_duration += result.duration_seconds
            print(f"    Duration: {result.duration_seconds:.2f}s")
            if provider_name == "manual":
                print(f"    Words transcribed: {len(result.word_timestamps)}")
        except FileNotFoundError as e:
            print(f"    Skipped: {e}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
        except Exception as e:
            print(f"    Error: {e}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    # Save manifest
    manifest = {
        "scenes": results,
        "total_duration_seconds": total_duration,
        "output_dir": str(output_dir),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Processed {len(results)} voiceovers")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.1f} min)")
    print(f"Manifest saved to: {manifest_path}")

    # Auto-sync storyboard durations with voiceover durations
    if not args.no_sync:
        sync_result = _sync_storyboard_durations(project, results)
        if sync_result > 0:
            print(f"\nSynced {sync_result} scene duration(s) in storyboard")

    return 0


def _sync_storyboard_durations(project, voiceover_results: list[dict]) -> int:
    """Sync storyboard scene durations with actual voiceover durations.

    Args:
        project: The project
        voiceover_results: List of voiceover results with scene_id and duration_seconds

    Returns:
        Number of scenes updated
    """
    storyboard_path = project.get_path("storyboard")
    if not storyboard_path.exists():
        return 0

    try:
        with open(storyboard_path) as f:
            storyboard = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0

    # Create lookup for voiceover durations
    vo_durations = {vo["scene_id"]: vo["duration_seconds"] for vo in voiceover_results}

    # Update storyboard scene durations
    updated_count = 0
    for scene in storyboard.get("scenes", []):
        scene_id = scene.get("id") or scene.get("scene_id")
        if scene_id and scene_id in vo_durations:
            old_dur = scene.get("audio_duration_seconds", 0)
            # Add a small buffer (0.5s) to ensure voiceover fits comfortably
            new_dur = round(vo_durations[scene_id] + 0.5, 2)

            # Only update if difference is significant (> 0.5s)
            if abs(new_dur - old_dur) > 0.5:
                scene["audio_duration_seconds"] = new_dur
                updated_count += 1

    if updated_count > 0:
        # Update total duration
        total_duration = sum(
            scene.get("audio_duration_seconds", 0)
            for scene in storyboard.get("scenes", [])
        )
        storyboard["total_duration_seconds"] = round(total_duration, 2)

        # Save updated storyboard
        with open(storyboard_path, "w") as f:
            json.dump(storyboard, f, indent=2)

    return updated_count


def _export_recording_script(project, narrations, args) -> int:
    """Export a recording script for manual voiceover recording."""
    from ..voiceover.delivery_tags import add_delivery_tags

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project.root_dir / "recording_script.txt"

    # Check if we should add delivery tags
    include_tags = getattr(args, 'with_tags', False)

    # Calculate estimated durations (~150 words per minute)
    lines = []
    lines.append("=" * 70)
    lines.append(f"RECORDING SCRIPT: {project.title}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Instructions:")
    lines.append("1. Record each scene as a separate audio file")
    lines.append("2. Name files by scene_id (e.g., scene1_hook.mp3)")
    lines.append("3. Speak naturally - aim for conversational tone")
    lines.append("4. Leave ~0.5s silence at start and end of each recording")
    if include_tags:
        lines.append("")
        lines.append("Delivery Tags Guide:")
        lines.append("  [thoughtful] - Reflective, contemplative")
        lines.append("  [puzzled] - Questions, uncertainty")
        lines.append("  [excited] - Impressive facts, revelations")
        lines.append("  [serious] - Important technical info")
        lines.append("  [wonder] - Awe-inspiring moments")
        lines.append("  [dramatic] - Building tension, key reveals")
        lines.append("  [warm] - Human, personal moments")
        lines.append("  [curious] - Inquisitive, exploring")
        lines.append("  [confident] - Assured, authoritative")
        lines.append("  [playful] - Light, fun tone")
        lines.append("  [reverent] - Respectful awe, profound")
        lines.append("  [urgent] - Time pressure, high stakes")
        lines.append("  [satisfied] - Resolution, conclusion")
        lines.append("  [intrigued] - Mystery, hook")
    lines.append("")
    lines.append("-" * 70)
    lines.append("")

    total_words = 0
    for i, narration in enumerate(narrations, 1):
        words = len(narration.narration.split())
        total_words += words
        duration_estimate = (words / 150) * 60  # seconds

        lines.append(f"=== Scene {i}: {narration.scene_id} ===")
        lines.append(f"Title: {narration.title}")
        lines.append(f"Words: {words} (~{duration_estimate:.0f} seconds)")
        lines.append(f"Output file: {narration.scene_id}.mp3")
        lines.append("")

        # Add delivery tags if requested
        if include_tags:
            print(f"  Adding delivery tags for scene {i}...")
            narration_text = add_delivery_tags(
                narration.narration,
                working_dir=project.root_dir.parent.parent,
            )
        else:
            narration_text = narration.narration

        lines.append(f'"{narration_text}"')
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    total_duration = (total_words / 150) * 60
    lines.append(f"TOTAL: {len(narrations)} scenes, {total_words} words")
    lines.append(f"Estimated recording time: {total_duration:.0f} seconds ({total_duration/60:.1f} minutes)")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Recording script exported to: {output_path}")
    print(f"Scenes: {len(narrations)}")
    print(f"Estimated duration: {total_duration/60:.1f} minutes")
    if include_tags:
        print(f"Delivery tags: included")
    print()
    print("Next steps:")
    print(f"  1. Record audio files following the script")
    print(f"  2. Place files in a directory (e.g., ./recordings/)")
    print(f"  3. Import with: python -m src.cli voiceover {project.id} --provider manual --audio-dir ./recordings/")

    return 0


def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


def cmd_storyboard(args: argparse.Namespace) -> int:
    """Generate or view storyboard for a project."""
    from ..project import load_project

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    storyboard_path = project.get_path("storyboard")

    if args.view:
        # View existing storyboard
        if not storyboard_path.exists():
            print(f"Error: Storyboard not found: {storyboard_path}", file=sys.stderr)
            return 1

        with open(storyboard_path) as f:
            storyboard = json.load(f)

        print(f"Storyboard: {storyboard.get('title', 'Untitled')}")
        print(f"Scenes: {len(storyboard.get('scenes', []))}")
        total_duration = storyboard.get('total_duration_seconds', 0)
        print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print()

        for i, scene in enumerate(storyboard.get("scenes", []), 1):
            duration = scene.get('audio_duration_seconds', 0)
            print(f"  {i}. {scene.get('id', 'unnamed')}: {scene.get('title', '')}")
            print(f"     Type: {scene.get('type', 'unknown')} | Duration: {duration:.1f}s")
            if scene.get('sfx_cues'):
                print(f"     SFX: {len(scene['sfx_cues'])} cues")

        return 0

    # Check if storyboard exists and --force not specified
    if storyboard_path.exists() and not args.force:
        print(f"Storyboard already exists: {storyboard_path}")
        print("Use --force to regenerate, or --view to view.")
        return 0

    # Generate storyboard
    print(f"Generating storyboard for {project.id}")

    # Check for required files
    narration_path = project.get_path("narration")
    voiceover_manifest = project.root_dir / "voiceover" / "manifest.json"
    scenes_dir = project.root_dir / "scenes"

    if not narration_path.exists():
        print(f"Error: Narrations not found: {narration_path}", file=sys.stderr)
        print("Run 'narration' command first.")
        return 1

    # Load narrations
    with open(narration_path) as f:
        narrations = json.load(f)

    # Load voiceover manifest if it exists (for audio durations)
    voiceover_data = {}
    if voiceover_manifest.exists():
        with open(voiceover_manifest) as f:
            manifest = json.load(f)
            for scene in manifest.get("scenes", []):
                voiceover_data[scene["scene_id"]] = {
                    "audio_file": Path(scene["audio_path"]).name,
                    "duration": scene.get("duration_seconds", 0),
                }
        print(f"Found voiceover manifest with {len(voiceover_data)} scenes")
    else:
        print("Warning: No voiceover manifest found. Using narration durations.")

    # Check for scenes directory to determine scene types
    scene_types = {}
    if scenes_dir.exists():
        index_path = scenes_dir / "index.ts"
        if index_path.exists():
            # Parse scene types from index.ts
            index_content = index_path.read_text()
            import re
            # Match patterns like: hook: HookScene,
            matches = re.findall(r'(\w+):\s*(\w+Scene)', index_content)
            for key, component in matches:
                scene_types[key] = component
            print(f"Found {len(scene_types)} scene types in scenes/index.ts")

    # Build storyboard
    storyboard = {
        "title": project.title,
        "description": f"Storyboard for {project.title}",
        "version": "2.0.0",
        "project": project.id,
        "video": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
        },
        "style": {
            "background_color": "#f4f4f5",
            "primary_color": "#00d9ff",
            "secondary_color": "#ff6b35",
            "font_family": "Inter",
        },
        "scenes": [],
        "audio": {
            "background_music": None,
            "music_volume": 0.15,
        },
        "total_duration_seconds": 0,
    }

    total_duration = 0
    for scene in narrations.get("scenes", []):
        scene_id = scene.get("scene_id", "")
        title = scene.get("title", "")

        # Derive scene type from title - this must match the scene generator's logic
        # to ensure storyboard scene types match scene registry keys
        scene_type_key = _title_to_scene_key(title)
        scene_type = f"{project.id}/{scene_type_key}"

        # Get audio info - prefer actual file duration over manifest (manifest may be stale)
        if scene_id in voiceover_data:
            audio_file = voiceover_data[scene_id]["audio_file"]
            audio_path = project.voiceover_dir / audio_file
            if audio_path.exists():
                audio_duration = _get_audio_duration(audio_path)
            else:
                audio_duration = voiceover_data[scene_id]["duration"]
        else:
            audio_file = f"{scene_id}.mp3"
            audio_path = project.voiceover_dir / audio_file
            if audio_path.exists():
                audio_duration = _get_audio_duration(audio_path)
            else:
                audio_duration = scene.get("duration_seconds", 20)

        storyboard_scene = {
            "id": scene_id,
            "type": scene_type,
            "title": title,
            "audio_file": audio_file,
            "audio_duration_seconds": audio_duration,
            "sfx_cues": [],  # Empty by default, can be added later
        }

        storyboard["scenes"].append(storyboard_scene)
        total_duration += audio_duration

    storyboard["total_duration_seconds"] = total_duration

    # Write storyboard
    storyboard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(storyboard_path, "w") as f:
        json.dump(storyboard, f, indent=2)

    print(f"\nGenerated storyboard with {len(storyboard['scenes'])} scenes")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Saved to: {storyboard_path}")

    if args.verbose:
        print("\nScenes:")
        for scene in storyboard["scenes"]:
            print(f"  {scene['id']}: {scene['title']} ({scene['audio_duration_seconds']:.1f}s)")

    return 0


def cmd_script(args: argparse.Namespace) -> int:
    """Generate a script from input documents."""
    from ..project import load_project
    from ..ingestion import parse_document
    from ..understanding import ContentAnalyzer
    from ..script import ScriptGenerator
    from ..planning import PlanGenerator
    from ..config import load_config

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Generating script for {project.id}")

    # Check for approved plan
    plan_path = project.plan_dir / "plan.json"
    plan = None
    skip_plan = getattr(args, "skip_plan", False)

    if plan_path.exists() and not skip_plan:
        plan = PlanGenerator.load_plan(plan_path)
        if plan.status != "approved":
            print(f"Warning: Plan exists but is not approved (status: {plan.status})")
            print("Use 'plan review' to approve, or --skip-plan to generate without plan.")
            return 1
        print(f"Using approved plan: {plan.title}")

    documents = []

    # Handle URL input
    if args.url:
        print(f"Fetching content from URL: {args.url}")
        try:
            doc = parse_document(args.url)
            documents.append(doc)
            print(f"  Parsed: {doc.title}")
        except Exception as e:
            print(f"Error fetching URL: {e}", file=sys.stderr)
            return 1

    # Handle file input (--input option)
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            return 1

        print(f"Parsing input file: {input_path}")
        try:
            doc = parse_document(input_path)
            documents.append(doc)
            print(f"  Parsed: {doc.title}")
        except Exception as e:
            print(f"Error parsing file: {e}", file=sys.stderr)
            return 1

    # If no explicit input, look for files in input directory
    if not documents:
        input_dir = project.input_dir
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
            print("Add source documents to the input/ directory, or use --url or --input.")
            return 1

        # Find all supported input files (markdown and PDF)
        input_files = []
        for pattern in ["*.md", "*.markdown", "*.pdf"]:
            input_files.extend(input_dir.glob(pattern))

        if not input_files:
            print(f"Error: No supported files found in {input_dir}", file=sys.stderr)
            print("Supported formats: .md, .markdown, .pdf")
            print("You can also use --url to fetch from a web page.")
            return 1

        print(f"Found {len(input_files)} input file(s)")

        # Parse documents
        for f in input_files:
            print(f"  Parsing: {f.name}")
            try:
                doc = parse_document(f)
                documents.append(doc)
            except Exception as e:
                print(f"    Error: {e}", file=sys.stderr)
                if not args.continue_on_error:
                    return 1

    if not documents:
        print("Error: No documents were successfully parsed.", file=sys.stderr)
        return 1

    print(f"\nSuccessfully parsed {len(documents)} document(s)")

    # Analyze content
    print("\nAnalyzing content...")
    config = load_config()
    if args.mock:
        config.llm.provider = "mock"

    analyzer = ContentAnalyzer(config)
    analysis = analyzer.analyze(documents[0])  # Use first document

    print(f"  Thesis: {analysis.core_thesis[:60]}...")
    print(f"  Concepts: {len(analysis.key_concepts)}")

    # Generate script
    print("\nGenerating script...")
    generator = ScriptGenerator(config)

    if plan:
        # Generate script constrained by approved plan
        print(f"  Using plan with {len(plan.scenes)} scenes")
        script = generator.generate_from_plan(
            plan=plan,
            document=documents[0],
            analysis=analysis,
        )
    else:
        # Original behavior - generate script directly
        script = generator.generate(
            documents[0],
            analysis,
            target_duration=args.duration or project.video.target_duration_seconds,
        )

    print(f"  Generated {len(script.scenes)} scenes")
    print(f"  Total duration: {script.total_duration_seconds}s")

    # Save script
    script_path = project.root_dir / "script" / "script.json"
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, "w") as f:
        json.dump(script.model_dump(), f, indent=2)

    print(f"\nScript saved to: {script_path}")

    # Show scenes
    if args.verbose:
        print("\nScenes:")
        for scene in script.scenes:
            print(f"  {scene.scene_id}. {scene.title} ({scene.duration_seconds}s)")
            print(f"      Type: {scene.scene_type}")
            print(f"      Voiceover: {scene.voiceover[:50]}...")
            print()

    return 0


def cmd_narration(args: argparse.Namespace) -> int:
    """Generate narrations for a project."""
    from ..project import load_project
    from ..narration import NarrationGenerator
    from ..script.generator import ScriptGenerator
    from ..ingestion import parse_document
    from ..config import load_config

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Generating narrations for {project.id}")

    narration_path = project.get_path("narration")

    # Check if narrations already exist
    if narration_path.exists() and not args.force:
        print(f"Narrations already exist: {narration_path}")
        print("Use --force to regenerate.")
        return 0

    # Load script
    script_path = project.root_dir / "script" / "script.json"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        print("Run 'video-explainer script' first to generate a script.")
        return 1

    script = ScriptGenerator.load_script(str(script_path))
    print(f"  Loaded script with {len(script.scenes)} scenes")

    # Load source documents
    source_documents = []
    input_dir = project.input_dir
    if input_dir.exists():
        input_files = []
        for pattern in ["*.md", "*.markdown", "*.pdf"]:
            input_files.extend(input_dir.glob(pattern))

        for input_file in input_files:
            try:
                doc = parse_document(input_file)
                source_documents.append(doc)
                print(f"  Loaded source: {input_file.name}")
            except Exception as e:
                print(f"  Warning: Could not parse {input_file.name}: {e}")

    # Determine topic
    topic = args.topic or project.title

    # Create generator with appropriate config
    config = load_config()
    if args.mock:
        config.llm.provider = "mock"

    generator = NarrationGenerator(config=config)

    # Generate narrations
    print(f"Generating narrations for topic: {topic}")

    if args.mock:
        narrations = generator.generate_mock(topic)
    else:
        try:
            narrations = generator.generate(
                script=script,
                source_documents=source_documents if source_documents else None,
                topic=topic,
            )
        except Exception as e:
            print(f"Error generating narrations: {e}", file=sys.stderr)
            return 1

    # Save narrations
    generator.save_narrations(narrations, narration_path)

    scene_count = len(narrations.get("scenes", []))
    print(f"\nGenerated {scene_count} narrations")
    print(f"Saved to: {narration_path}")

    if args.verbose:
        print("\nScenes:")
        for scene in narrations.get("scenes", []):
            print(f"  {scene.get('scene_id')}: {scene.get('title')}")

    return 0


def _generate_mock_narrations(topic: str) -> dict:
    """Generate mock narrations for testing.

    DEPRECATED: Use NarrationGenerator.generate_mock() instead.
    """
    from ..narration import NarrationGenerator
    return NarrationGenerator().generate_mock(topic)


# Keep old function for backward compatibility
def _old_generate_mock_narrations(topic: str) -> dict:
    """Generate mock narrations for testing (old implementation)."""
    return {
        "scenes": [
            {
                "scene_id": "scene1_hook",
                "title": "The Hook",
                "narration": f"What if I told you that {topic} could change everything you know about technology?",
            },
            {
                "scene_id": "scene2_context",
                "title": "Setting the Context",
                "narration": f"To understand {topic}, we need to first look at the bigger picture.",
            },
            {
                "scene_id": "scene3_explanation",
                "title": "How It Works",
                "narration": f"At its core, {topic} works by processing information in a fundamentally different way.",
            },
            {
                "scene_id": "scene4_conclusion",
                "title": "The Takeaway",
                "narration": f"So remember, {topic} isn't just a technology - it's a paradigm shift.",
            },
        ]
    }


def cmd_scenes(args: argparse.Namespace) -> int:
    """Generate Remotion scene components from script."""
    from ..project import load_project
    from ..scenes import SceneGenerator, SyntaxVerifier

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Handle verify-only mode
    if getattr(args, "verify", False):
        return _cmd_scenes_verify(args, project)

    # Handle sync mode
    if args.sync:
        return _cmd_scenes_sync(args, project)

    # Handle single scene regeneration
    if args.scene:
        return _cmd_scenes_regenerate_single(args, project)

    print(f"Generating scenes for {project.id}")

    # Check for script
    script_path = project.root_dir / "script" / "script.json"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        print("Run 'script' command first to generate a script.")
        return 1

    # Check if scenes already exist
    scenes_dir = project.root_dir / "scenes"
    if scenes_dir.exists() and list(scenes_dir.glob("*.tsx")) and not args.force:
        print(f"Scenes already exist in {scenes_dir}")
        print("Use --force to regenerate.")
        return 0

    # Load script to show info
    with open(script_path) as f:
        script = json.load(f)

    scene_count = len(script.get("scenes", []))
    print(f"Script: {script.get('title', 'Untitled')}")
    print(f"Scenes to generate: {scene_count}")

    # Check for voiceover manifest (for animation-to-narration timing sync)
    voiceover_manifest_path = project.root_dir / "voiceover" / "manifest.json"
    if voiceover_manifest_path.exists():
        print(f"Voiceover manifest found - will use word timestamps for animation timing")
    else:
        print(f"Note: No voiceover manifest found at {voiceover_manifest_path}")
        print("  Animation timing will use percentage-based estimation.")
        print("  Run 'voiceover' command first for precise animation-to-narration sync.")
        voiceover_manifest_path = None
    print()

    # Generate scenes (validation is internal - generator retries on validation failure)
    generator = SceneGenerator(
        working_dir=project.root_dir.parent.parent,  # Repo root
        timeout=args.timeout,
        skip_validation=getattr(args, 'no_validate', False),
    )

    try:
        print("Generating scene components...")
        results = generator.generate_all_scenes(
            project_dir=project.root_dir,
            script_path=script_path,
            voiceover_manifest_path=voiceover_manifest_path,
            force=args.force,
        )

        # Report results
        print()
        print(f"Generated {len(results['scenes'])} scenes")
        if results["errors"]:
            print(f"Failed scenes: {len(results['errors'])}")
            for err in results["errors"]:
                scene_num = err['scene_number']
                title = err.get('title', f'Scene {scene_num}')
                print(f"  Scene {scene_num} ({title}): {err['error']}")
            print()
            print("To regenerate failed scenes individually:")
            for err in results["errors"]:
                print(f"  python -m src.cli scenes {args.project} --scene {err['scene_number']}")
            return 1

        print(f"\nOutput directory: {results['scenes_dir']}")

        if args.verbose:
            print("\nGenerated files:")
            for scene in results["scenes"]:
                print(f"  {scene['filename']}: {scene['title']}")

    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error generating scenes: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_scenes_verify(args: argparse.Namespace, project) -> int:
    """Verify syntax of existing scene files."""
    from ..scenes import SyntaxVerifier

    scenes_dir = project.root_dir / "scenes"
    if not scenes_dir.exists() or not list(scenes_dir.glob("*.tsx")):
        print(f"Error: No scenes found in {scenes_dir}", file=sys.stderr)
        print("Run 'scenes' command first to generate scenes.")
        return 1

    print(f"Verifying scene syntax for {project.id}")
    print(f"Scenes directory: {scenes_dir}")
    print()

    auto_fix = not getattr(args, "no_auto_fix", False)
    verifier = SyntaxVerifier(remotion_dir=project.root_dir.parent.parent / "remotion")

    result = verifier.verify_scenes(scenes_dir, auto_fix=auto_fix)

    if result.success:
        print("✓ All scenes pass syntax verification")
        if result.fixed_files:
            print(f"\nAuto-fixed files: {', '.join(result.fixed_files)}")
        return 0

    # Report errors
    print(f"✗ Found {result.error_count} syntax error(s)")
    print()

    if result.fixed_files:
        print(f"Auto-fixed: {', '.join(result.fixed_files)}")

    if result.unfixed_files:
        print(f"Files with remaining errors: {', '.join(result.unfixed_files)}")
        print()
        print("Errors:")
        for error in result.errors:
            print(f"  {error}")

    return 1 if result.unfixed_files else 0


def _cmd_scenes_sync(args: argparse.Namespace, project) -> int:
    """Sync scene timing to updated voiceover timestamps."""
    from ..scenes import SceneGenerator

    if args.scene:
        print(f"Syncing scene {args.scene} for {project.id}")
    else:
        print(f"Syncing all scenes for {project.id}")

    # Check scenes exist
    scenes_dir = project.root_dir / "scenes"
    if not scenes_dir.exists() or not list(scenes_dir.glob("*.tsx")):
        print(f"Error: No scenes found in {scenes_dir}", file=sys.stderr)
        print("Run 'scenes' command first to generate scenes.")
        return 1

    # Check voiceover manifest exists
    voiceover_manifest_path = project.root_dir / "voiceover" / "manifest.json"
    if not voiceover_manifest_path.exists():
        print(f"Error: Voiceover manifest not found at {voiceover_manifest_path}", file=sys.stderr)
        print("Run 'voiceover' command first to generate voiceover with timestamps.")
        return 1

    print(f"Using word timestamps from: {voiceover_manifest_path}")
    print()

    generator = SceneGenerator(
        working_dir=project.root_dir.parent.parent,  # Repo root
        timeout=args.timeout,
    )

    try:
        print("Syncing scene timing to voiceover...")
        results = generator.sync_all_scenes(
            project_dir=project.root_dir,
            voiceover_manifest_path=voiceover_manifest_path,
            scene_filter=args.scene,
        )

        # Report results
        print()
        print(f"Synced: {len(results['synced'])} scenes")
        if results["skipped"]:
            print(f"Skipped: {len(results['skipped'])} scenes")
            if args.verbose:
                for skip in results["skipped"]:
                    print(f"  {skip['filename']}: {skip['reason']}")
        if results["errors"]:
            print(f"Failed: {len(results['errors'])} scenes")
            for err in results["errors"]:
                print(f"  {err['filename']}: {err['error']}")
            return 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error syncing scenes: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_scenes_regenerate_single(args: argparse.Namespace, project) -> int:
    """Regenerate a single scene by number or filename."""
    from ..scenes import SceneGenerator

    scene_spec = args.scene

    # Check for script
    script_path = project.root_dir / "script" / "script.json"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        return 1

    # Load script
    with open(script_path) as f:
        script = json.load(f)

    scenes = script.get("scenes", [])

    # Find the scene to regenerate
    scene_index = None
    scene_data = None

    # Try to match by scene number (e.g., "6", "scene6", "scene_6")
    scene_num_match = re.match(r"^(?:scene[_]?)?(\d+)$", scene_spec.lower())
    if scene_num_match:
        scene_num = int(scene_num_match.group(1))
        if 1 <= scene_num <= len(scenes):
            scene_index = scene_num - 1
            scene_data = scenes[scene_index]

    # Try to match by filename (e.g., "HookScene.tsx" or "HookScene")
    if scene_data is None:
        scene_filename = scene_spec.replace(".tsx", "").replace("Scene", "").lower()
        for idx, scene in enumerate(scenes):
            title = scene.get("title", f"Scene {idx + 1}")
            # Convert title to component name (same logic as generator)
            words = re.sub(r"[^a-zA-Z0-9\s]", "", title).split()
            component_name = "".join(word.capitalize() for word in words) + "Scene"
            # Also try without common prefixes like "The", "A", "An"
            title_words = title.split()
            if title_words and title_words[0].lower() in ("the", "a", "an"):
                title_words = title_words[1:]
            component_name_no_prefix = "".join(re.sub(r"[^a-zA-Z0-9]", "", w).capitalize() for w in title_words) + "Scene"

            if (component_name.lower() == (scene_filename + "scene").lower() or
                component_name.lower() == scene_filename.lower() or
                component_name_no_prefix.lower() == (scene_filename + "scene").lower() or
                component_name_no_prefix.lower() == scene_filename.lower()):
                scene_index = idx
                scene_data = scene
                break

    if scene_data is None:
        print(f"Error: Scene '{scene_spec}' not found.", file=sys.stderr)
        print(f"Valid options: 1-{len(scenes)}, or scene filename (e.g., HookScene.tsx)")
        return 1

    scene_number = scene_index + 1
    title = scene_data.get("title", f"Scene {scene_number}")
    print(f"Regenerating scene {scene_number}: {title}")

    # Check for voiceover manifest
    voiceover_manifest_path = project.root_dir / "voiceover" / "manifest.json"
    word_timestamps = []
    if voiceover_manifest_path.exists():
        with open(voiceover_manifest_path) as f:
            manifest = json.load(f)
        scene_id = scene_data.get("scene_id", f"scene{scene_number}")
        for scene_manifest in manifest.get("scenes", []):
            if scene_manifest.get("scene_id") == scene_id:
                word_timestamps = scene_manifest.get("word_timestamps", [])
                break
        if word_timestamps:
            print(f"Using word timestamps from voiceover manifest")

    # Generate the scene
    generator = SceneGenerator(
        working_dir=project.root_dir.parent.parent,
        timeout=args.timeout,
        skip_validation=getattr(args, 'no_validate', False),
    )

    scenes_dir = project.root_dir / "scenes"
    example_scene = generator._load_example_scene()

    try:
        print()
        result = generator._generate_scene(
            scene=scene_data,
            scene_number=scene_number,
            scenes_dir=scenes_dir,
            example_scene=example_scene,
            word_timestamps=word_timestamps,
        )
        print(f"\n✓ Regenerated: {result['filename']}")

        # Update index.ts to ensure scene is registered
        generator._generate_index(
            scenes_dir,
            _collect_scene_info(scenes_dir, script),
            script.get("title", "Untitled"),
        )
        print(f"✓ Updated index.ts")

    except Exception as e:
        print(f"\n✗ Failed to regenerate scene: {e}", file=sys.stderr)
        return 1

    return 0


def _collect_scene_info(scenes_dir: Path, script: dict) -> list[dict]:
    """Collect scene info from existing scene files for index generation."""
    import re
    scenes_info = []

    for idx, scene in enumerate(script.get("scenes", [])):
        scene_num = idx + 1
        title = scene.get("title", f"Scene {scene_num}")

        # Generate component name from title
        words = re.sub(r"[^a-zA-Z0-9\s]", "", title).split()
        component_name = "".join(word.capitalize() for word in words) + "Scene"
        filename = f"{component_name}.tsx"

        # Check if file exists
        if (scenes_dir / filename).exists():
            # Derive scene_key from title
            key_words = title.split()
            if key_words and key_words[0].lower() in ("the", "a", "an"):
                key_words = key_words[1:]
            scene_key = "_".join(word.lower() for word in key_words)
            scene_key = re.sub(r"[^a-z0-9_]", "", scene_key)
            scene_key = re.sub(r"_+", "_", scene_key).strip("_")

            scenes_info.append({
                "scene_number": scene_num,
                "title": title,
                "component_name": component_name,
                "filename": filename,
                "scene_type": scene.get("scene_type", "explanation"),
                "scene_key": scene_key,
            })

    return scenes_info


# Resolution presets
RESOLUTION_PRESETS = {
    "4k": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
}

# Shorts resolution presets (vertical 9:16 aspect ratio)
SHORTS_RESOLUTION_PRESETS = {
    "4k": (2160, 3840),
    "1440p": (1440, 2560),
    "1080p": (1080, 1920),
    "720p": (720, 1280),
    "480p": (480, 854),
}


def _merge_render_chunks(project, resolution_name: str, is_short: bool, variant: str) -> int:
    """Merge chunked render files into a single video.

    Finds all chunk files matching the pattern and concatenates them using ffmpeg.
    """
    import subprocess
    import re

    # Determine output directory and base filename
    if is_short:
        output_dir = project.root_dir / "short" / variant / "output"
        if resolution_name != "1080p":
            base_name = f"short-{resolution_name}"
        else:
            base_name = "short"
    else:
        output_dir = project.output_dir
        if resolution_name != "1080p":
            base_name = f"final-{resolution_name}"
        else:
            base_name = "final"

    # Find all chunk files matching pattern: {base_name}-frames-*.mp4
    chunk_pattern = f"{base_name}-frames-*.mp4"
    chunk_files = list(output_dir.glob(chunk_pattern))

    if not chunk_files:
        print(f"Error: No chunk files found matching '{chunk_pattern}' in {output_dir}", file=sys.stderr)
        return 1

    # Sort chunks by start frame number
    def get_start_frame(path: Path) -> int:
        # Extract start frame from filename like "final-4k-frames-0-2500.mp4" or "final-4k-frames-7501-end.mp4"
        match = re.search(r'-frames-(\d+)-', path.name)
        if match:
            return int(match.group(1))
        return 0

    chunk_files.sort(key=get_start_frame)

    print(f"Found {len(chunk_files)} chunk files to merge:")
    for f in chunk_files:
        print(f"  - {f.name}")

    # Create concat file
    concat_file = output_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for chunk in chunk_files:
            f.write(f"file '{chunk.name}'\n")

    # Determine output filename
    output_path = output_dir / f"{base_name}.mp4"

    # Check if output already exists
    if output_path.exists():
        print(f"Warning: {output_path} already exists, will be overwritten")

    # Run ffmpeg to concatenate
    print(f"\nMerging chunks into {output_path}...")
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, cwd=str(output_dir), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: ffmpeg failed with exit code {result.returncode}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return result.returncode
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.", file=sys.stderr)
        return 1

    # Clean up concat file
    concat_file.unlink()

    print(f"\nSuccessfully merged {len(chunk_files)} chunks into: {output_path}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render video for a project."""
    import subprocess

    from ..project import load_project

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    is_short = getattr(args, "short", False)
    variant = getattr(args, "variant", "default")

    # Handle --merge-chunks option
    if hasattr(args, 'merge_chunks') and args.merge_chunks:
        resolution_name = args.resolution or "1080p"
        return _merge_render_chunks(project, resolution_name, is_short, variant)

    if is_short:
        print(f"Rendering short for {project.id} (variant: {variant})")
    else:
        print(f"Rendering video for {project.id}")

    # Determine composition and setup
    remotion_dir = Path(__file__).parent.parent.parent / "remotion"
    render_script = remotion_dir / "scripts" / "render.mjs"

    if not render_script.exists():
        print(f"Error: Render script not found: {render_script}", file=sys.stderr)
        return 1

    # Check for storyboard (different paths for shorts vs full video)
    if is_short:
        storyboard_path = project.root_dir / "short" / variant / "storyboard" / "shorts_storyboard.json"
        voiceover_dir = project.root_dir / "short" / variant / "voiceover"
        audio_dir = f"short/{variant}/voiceover"
        composition_id = "ShortsPlayer"
    else:
        storyboard_path = project.get_path("storyboard")
        voiceover_dir = project.voiceover_dir
        audio_dir = "voiceover"
        composition_id = "ScenePlayer"

    if not storyboard_path.exists():
        print(f"Error: Storyboard not found: {storyboard_path}", file=sys.stderr)
        if is_short:
            print(f"Run 'python -m src.cli short {project.id}' first")
        else:
            print("Run storyboard generation first or create storyboard/storyboard.json")
        return 1

    # Check for voiceover files
    voiceover_files = list(voiceover_dir.glob("*.mp3"))
    print(f"Found {len(voiceover_files)} voiceover files")

    # Determine resolution (use shorts presets for shorts)
    resolution_name = args.resolution or "1080p"
    if is_short:
        if resolution_name not in SHORTS_RESOLUTION_PRESETS:
            print(f"Error: Unknown resolution '{resolution_name}'", file=sys.stderr)
            print(f"Available: {', '.join(SHORTS_RESOLUTION_PRESETS.keys())}", file=sys.stderr)
            return 1
        width, height = SHORTS_RESOLUTION_PRESETS[resolution_name]
    else:
        if resolution_name not in RESOLUTION_PRESETS:
            print(f"Error: Unknown resolution '{resolution_name}'", file=sys.stderr)
            print(f"Available: {', '.join(RESOLUTION_PRESETS.keys())}", file=sys.stderr)
            return 1
        width, height = RESOLUTION_PRESETS[resolution_name]

    # Determine output path
    if is_short:
        short_output_dir = project.root_dir / "short" / variant / "output"
        short_output_dir.mkdir(parents=True, exist_ok=True)
        if args.preview:
            output_path = short_output_dir / "preview.mp4"
        elif resolution_name != "1080p":
            output_path = short_output_dir / f"short-{resolution_name}.mp4"
        else:
            output_path = short_output_dir / "short.mp4"
    else:
        if args.preview:
            output_path = project.output_dir / "preview" / "preview.mp4"
        else:
            # Include resolution in filename for non-1080p renders
            if resolution_name != "1080p":
                output_path = project.output_dir / f"final-{resolution_name}.mp4"
            else:
                output_path = project.get_path("final_video")

    # Modify output filename if --frames is specified for chunked rendering
    if hasattr(args, 'frames') and args.frames:
        # Parse frame range to create descriptive filename
        # e.g., "0-2500" -> "final-4k-frames-0-2500.mp4"
        # e.g., "7501-" -> "final-4k-frames-7501-end.mp4"
        frame_suffix = args.frames
        if frame_suffix.endswith("-"):
            frame_suffix = f"{frame_suffix}end"
        stem = output_path.stem  # e.g., "final-4k"
        output_path = output_path.parent / f"{stem}-frames-{frame_suffix}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build render command using new data-driven architecture
    # The render script uses the project directory to serve static files (voiceovers/mixed audio)
    cmd = [
        "node",
        str(render_script),
        "--project", str(project.root_dir),
        "--output", str(output_path),
        "--width", str(width),
        "--height", str(height),
        "--voiceover-path", audio_dir,
        "--composition", composition_id,
    ]

    # For shorts, pass the storyboard path explicitly
    if is_short:
        cmd.extend(["--storyboard", str(storyboard_path)])

    # Performance options
    if args.fast:
        cmd.append("--fast")
    if args.concurrency:
        cmd.extend(["--concurrency", str(args.concurrency)])
    if hasattr(args, 'gl') and args.gl:
        cmd.extend(["--gl", args.gl])
    if hasattr(args, 'frames') and args.frames:
        cmd.extend(["--frames", args.frames])

    print(f"Project: {project.root_dir}")
    print(f"Composition: {composition_id}")
    print(f"Audio: {audio_dir}")
    print(f"Resolution: {resolution_name} ({width}x{height})")
    print(f"Output: {output_path}")
    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, cwd=str(remotion_dir))
        if result.returncode != 0:
            print(f"Render failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
    except FileNotFoundError:
        print("Error: Node.js not found. Please install Node.js.", file=sys.stderr)
        return 1

    print()
    print(f"Video rendered to: {output_path}")
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    """Process or view feedback for a project."""
    from ..project import load_project
    from ..refine.feedback import FeedbackProcessor, FeedbackStore, FeedbackStatus

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.feedback_command:
        print("Usage: python -m src.cli feedback <project> <command>")
        print("\nCommands:")
        print("  add <text>     Add and process new feedback")
        print("  list           List all feedback for the project")
        print("  show <id>      Show details of a feedback item")
        print("  retry <id>     Retry a failed feedback item")
        return 1

    if args.feedback_command == "add":
        # Process new feedback
        live_output = getattr(args, 'live', False)
        processor = FeedbackProcessor(
            project,
            verbose=True,  # Always show detailed progress
            live_output=live_output,
        )

        print(f"Processing feedback for {project.id}...")
        print(f"Feedback: {args.feedback_text}")
        print()

        if args.dry_run:
            print("[DRY RUN] Analyzing feedback only, no changes will be made")
            print()
        if live_output:
            print("[LIVE] Streaming Claude Code output")
            print()

        item = processor.process(args.feedback_text, dry_run=args.dry_run)

        print(f"\nFeedback ID: {item.id}")
        print(f"Status: {item.status.value}")

        if item.intent:
            print(f"Intent: {item.intent.value}")

        if item.interpretation:
            print(f"\nInterpretation:")
            print(f"  {item.interpretation}")

        if item.target:
            print(f"\nScope: {item.target.scope.value}")
            if item.target.scene_ids:
                print(f"Affected scenes: {', '.join(item.target.scene_ids)}")

        if item.patches:
            print(f"\nPatches generated: {len(item.patches)}")
            for i, patch in enumerate(item.patches, 1):
                patch_type = patch.get("patch_type", "unknown") if isinstance(patch, dict) else "unknown"
                print(f"  {i}. {patch_type}")

        if item.files_modified:
            print(f"\nFiles modified:")
            for f in item.files_modified:
                print(f"  - {f}")

        if item.verification_passed is not None:
            print(f"\nVerification: {'passed' if item.verification_passed else 'failed'}")

        if item.error_message:
            print(f"\nError: {item.error_message}", file=sys.stderr)

        return 0 if item.status != FeedbackStatus.FAILED else 1

    elif args.feedback_command == "list":
        # List all feedback
        store = FeedbackStore(project)
        items = store.list_all()

        if not items:
            print(f"No feedback found for {project.id}")
            return 0

        print(f"Feedback for {project.id} ({len(items)} items):\n")

        for item in items:
            status_icon = {
                "pending": "⏳",
                "analyzing": "🔍",
                "generating": "⚙️",
                "applying": "📝",
                "verifying": "✓",
                "applied": "✅",
                "failed": "💥",
            }.get(item.status.value, "?")

            print(f"  {status_icon} {item.id}")
            print(f"    Status: {item.status.value}")
            if item.intent:
                print(f"    Intent: {item.intent.value}")
            print(f"    Feedback: {item.feedback_text[:60]}{'...' if len(item.feedback_text) > 60 else ''}")
            if item.target and item.target.scene_ids:
                print(f"    Scenes: {', '.join(item.target.scene_ids)}")
            print()

        return 0

    elif args.feedback_command == "show":
        # Show detailed feedback
        store = FeedbackStore(project)
        item = store.get_item(args.feedback_id)

        if not item:
            print(f"Error: Feedback not found: {args.feedback_id}", file=sys.stderr)
            return 1

        print(f"Feedback: {item.id}")
        print(f"Status: {item.status.value}")
        print(f"Timestamp: {item.timestamp}")
        print()
        print("Original feedback:")
        print(f"  {item.feedback_text}")
        print()

        if item.intent:
            print(f"Intent: {item.intent.value}")
            if item.sub_intents:
                print(f"Sub-intents: {', '.join(i.value for i in item.sub_intents)}")
            print()

        if item.interpretation:
            print("Interpretation:")
            print(f"  {item.interpretation}")
            print()

        if item.target:
            print(f"Scope: {item.target.scope.value}")
            if item.target.scene_ids:
                print(f"Affected scenes: {', '.join(item.target.scene_ids)}")
            print()

        if item.patches:
            print(f"Patches ({len(item.patches)}):")
            print(json.dumps(item.patches, indent=2))
            print()

        if item.files_modified:
            print("Files modified:")
            for f in item.files_modified:
                print(f"  - {f}")

        if item.verification_passed is not None:
            print(f"\nVerification: {'passed' if item.verification_passed else 'failed'}")

        if item.error_message:
            print(f"\nError: {item.error_message}")

        return 0

    elif args.feedback_command == "retry":
        # Retry a failed feedback item
        processor = FeedbackProcessor(
            project,
            verbose=True,
        )

        print(f"Retrying feedback {args.feedback_id}...")

        item = processor.process_item(args.feedback_id, dry_run=args.dry_run)

        if not item:
            print(f"Error: Feedback not found: {args.feedback_id}", file=sys.stderr)
            return 1

        print(f"\nStatus: {item.status.value}")
        if item.files_modified:
            print(f"Files modified: {', '.join(item.files_modified)}")
        if item.error_message:
            print(f"Error: {item.error_message}", file=sys.stderr)

        return 0 if item.status != FeedbackStatus.FAILED else 1

    return 0


def cmd_create(args: argparse.Namespace) -> int:
    """Create a new project."""
    from ..project.loader import create_project

    try:
        project = create_project(
            project_id=args.project_id,
            title=args.title or args.project_id.replace("-", " ").title(),
            projects_dir=args.projects_dir,
            description=args.description or "",
        )
        print(f"Created project: {project.id}")
        print(f"Path: {project.root_dir}")
        print()
        print("Next steps:")
        print(f"  1. Add source document to {project.input_dir}/")
        print(f"  2. Add narrations to {project.narration_dir}/narrations.json")
        print(f"  3. Run: python -m src.cli voiceover {project.id}")
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_music(args: argparse.Namespace) -> int:
    """Generate AI background music for a project."""
    from ..project import load_project
    from ..music import MusicGenerator, MusicConfig
    from ..music.generator import generate_for_project, get_music_prompt

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.music_command:
        print("Usage: python -m src.cli music <project> <command>")
        print("\nCommands:")
        print("  generate    Generate background music for full video")
        print("  short       Generate punchy background music for YouTube Short")
        print("  info        Show music generation info")
        return 1

    if args.music_command == "generate":
        print(f"Generating background music for {project.id}")
        print()

        # Determine target duration from storyboard
        storyboard_path = project.get_path("storyboard")
        target_duration = args.duration

        if not target_duration and storyboard_path.exists():
            try:
                with open(storyboard_path) as f:
                    storyboard = json.load(f)
                target_duration = int(storyboard.get("total_duration_seconds", 60))
                print(f"Duration from storyboard: {target_duration}s")
            except (json.JSONDecodeError, KeyError):
                target_duration = 60

        if not target_duration:
            target_duration = 60

        # Determine topic
        topic = args.topic or project.title
        print(f"Topic: {topic}")

        # Show music style
        style = args.style or get_music_prompt(topic)
        print(f"Style: {style}")
        print()

        # Generate music
        result = generate_for_project(
            project_dir=project.root_dir,
            topic=topic,
            target_duration=target_duration,
            update_storyboard=not args.no_update,
        )

        if result.success:
            print()
            print(f"Generated: {result.output_path}")
            print(f"Duration: {result.duration_seconds:.1f}s")
            print(f"Segments: {result.segments_generated}")

            if not args.no_update:
                print()
                print("Storyboard updated with background music config.")
                print("Run render to include music in video.")
            return 0
        else:
            print(f"Error: {result.error_message}", file=sys.stderr)
            return 1

    elif args.music_command == "short":
        # Generate punchy music for YouTube Short
        from ..music.generator import generate_for_short, get_shorts_music_prompt, analyze_shorts_mood, SHORTS_STYLE_PRESETS

        variant = getattr(args, 'variant', 'default') or 'default'
        print(f"Generating punchy background music for {project.id} short (variant: {variant})")
        print()

        # Check if shorts storyboard exists
        storyboard_path = project.root_dir / "short" / variant / "storyboard" / "shorts_storyboard.json"
        if not storyboard_path.exists():
            print(f"Error: Shorts storyboard not found at {storyboard_path}", file=sys.stderr)
            print("Run 'python -m src.cli short storyboard <project>' first.", file=sys.stderr)
            return 1

        # Determine topic
        topic = args.topic or project.title
        print(f"Topic: {topic}")

        # Show what style will be used
        with open(storyboard_path) as f:
            storyboard = json.load(f)
        beats = storyboard.get("beats", [])

        mood = analyze_shorts_mood(beats)
        print(f"Detected mood: {mood['primary_mood']}")

        style = args.style or get_shorts_music_prompt(topic, beats)
        print(f"Style: {style}")
        print()

        # Generate music
        result = generate_for_short(
            project_dir=project.root_dir,
            topic=topic,
            variant=variant,
            target_duration=args.duration,
            custom_style=args.style,
            update_storyboard=not args.no_update,
        )

        if result.success:
            print()
            print(f"Generated: {result.output_path}")
            print(f"Duration: {result.duration_seconds:.1f}s")
            print(f"Segments: {result.segments_generated}")

            if not args.no_update:
                print()
                print("Shorts storyboard updated with background music config.")
                print("Run render to include music in short.")
            return 0
        else:
            print(f"Error: {result.error_message}", file=sys.stderr)
            return 1

    elif args.music_command == "info":
        # Show music generation info
        print("Music Generation Info")
        print("=" * 40)
        print()

        # Check for existing music
        music_path = project.root_dir / "music" / "background.mp3"
        if music_path.exists():
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(music_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info.get("format", {}).get("duration", 0))
                print(f"Existing music: {music_path}")
                print(f"Duration: {duration:.1f}s")
            else:
                print(f"Existing music: {music_path}")
        else:
            print("No background music generated yet.")

        print()
        print("Available style presets (full video):")
        from ..music.generator import MUSIC_STYLE_PRESETS, SHORTS_STYLE_PRESETS
        for name, desc in MUSIC_STYLE_PRESETS.items():
            print(f"  {name}: {desc[:60]}...")

        print()
        print("Available style presets (shorts - more punchy):")
        for name, desc in SHORTS_STYLE_PRESETS.items():
            print(f"  {name}: {desc[:60]}...")

        print()
        print("Device support:")
        import torch
        print(f"  MPS (Apple Silicon): {'Available' if torch.backends.mps.is_available() else 'Not available'}")
        print(f"  CUDA (NVIDIA GPU): {'Available' if torch.cuda.is_available() else 'Not available'}")
        print(f"  CPU: Always available (slower)")

        return 0

    return 0


def cmd_sound(args: argparse.Namespace) -> int:
    """Sound design commands: generate SFX library for Remotion."""
    from ..project import load_project
    from ..sound.library import SoundLibrary, SOUND_MANIFEST

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # SFX files go in project's sfx/ directory (used by Remotion)
    sfx_dir = project.root_dir / "sfx"

    if not args.sound_command:
        print("Usage: python -m src.cli sound <project> <command>")
        print("\nCommands:")
        print("  library    Generate/manage SFX library")
        print("  analyze    Analyze scenes for sound moments (dry run)")
        print("  generate   Generate SFX cues and write to storyboard")
        print("  clear      Remove SFX cues from storyboard")
        print("\nNote: SFX cues are defined in storyboard.json and rendered by Remotion.")
        return 1

    if args.sound_command == "library":
        library = SoundLibrary(sfx_dir)

        if args.list:
            print(f"Sound Library ({len(SOUND_MANIFEST)} sounds)")
            print("=" * 50)

            for name, info in SOUND_MANIFEST.items():
                status = "[ready]" if library.sound_exists(name) else "[missing]"
                print(f"  {status} {name}")
                print(f"         {info['description']}")

            missing = library.get_missing_sounds()
            if missing:
                print(f"\n{len(missing)} sounds need to be generated.")
                print("Run: python -m src.cli sound <project> library --generate")

            return 0

        elif args.download or args.generate:
            print(f"Generating SFX library for {project.id}")

            generated = library.generate_all()
            print(f"Generated {len(generated)} sounds to: {sfx_dir}")

            for name in generated:
                print(f"  - {name}.wav")

            return 0

        else:
            print("Usage: python -m src.cli sound <project> library [options]")
            print("\nOptions:")
            print("  --list       List all available sounds")
            print("  --generate   Generate all sound effect files")
            return 1

    elif args.sound_command == "analyze":
        return _cmd_sound_analyze(project, args)

    elif args.sound_command == "generate":
        return _cmd_sound_generate(project, args)

    elif args.sound_command == "clear":
        return _cmd_sound_clear(project, args)

    else:
        print(f"Unknown sound command: {args.sound_command}")
        return 1

    return 0


def _cmd_sound_analyze(project, args: argparse.Namespace) -> int:
    """Analyze scenes for sound moments (dry run)."""
    from ..sound.sfx_orchestrator import SFXOrchestrator

    print(f"Analyzing scenes for {project.id}...")
    print()

    orchestrator = SFXOrchestrator(project_dir=project.root_dir)

    try:
        preview = orchestrator.preview_analysis()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not preview:
        print("No scenes found to analyze.")
        return 0

    # Filter to specific scene if requested
    if args.scene:
        if args.scene not in preview:
            print(f"Error: Scene '{args.scene}' not found", file=sys.stderr)
            print(f"Available scenes: {', '.join(preview.keys())}")
            return 1
        preview = {args.scene: preview[args.scene]}

    total_moments = 0
    for scene_id, data in preview.items():
        print(f"Scene: {scene_id}")
        print(f"  Type: {data['scene_type']}")
        print(f"  Duration: {data['duration_frames']} frames ({data['duration_frames']/30:.1f}s)")
        print(f"  Detected moments: {data['total_moments']}")

        if data['moments_by_type']:
            print("  By type:")
            for moment_type, count in sorted(data['moments_by_type'].items()):
                print(f"    - {moment_type}: {count}")

        if data['notes']:
            print("  Notes:")
            for note in data['notes']:
                print(f"    - {note}")

        if args.verbose:
            moments = orchestrator.get_scene_moments(scene_id)
            if moments:
                print("  Detailed moments:")
                for m in moments:
                    print(f"    Frame {m.frame:4d}: {m.type:<18} (conf={m.confidence:.2f}, src={m.source})")
                    if m.context:
                        print(f"                    {m.context}")

        total_moments += data['total_moments']
        print()

    print(f"Total: {len(preview)} scenes, {total_moments} moments detected")
    print()
    print("Run 'sound generate' to create SFX cues in storyboard.json")

    return 0


def _cmd_sound_generate(project, args: argparse.Namespace) -> int:
    """Generate SFX cues and write to storyboard."""
    from ..sound.sfx_orchestrator import SFXOrchestrator
    from ..sound.generator import SoundTheme

    print(f"Generating SFX cues for {project.id}...")

    # Parse theme
    try:
        theme = SoundTheme(args.theme)
    except ValueError:
        theme = SoundTheme.TECH_AI

    orchestrator = SFXOrchestrator(
        project_dir=project.root_dir,
        theme=theme,
        use_library=True,  # Use pre-generated library sounds
    )

    use_llm = not args.no_llm

    if args.dry_run:
        print("[DRY RUN] Analyzing only, no changes will be written")
    print(f"Theme: {theme.value}")
    print(f"Max density: {args.max_density} sounds/second")
    print(f"Min gap: {args.min_gap} frames")
    print(f"LLM analysis: {'enabled' if use_llm else 'disabled'}")
    print()

    result = orchestrator.generate_sfx_cues(
        use_llm=use_llm,
        dry_run=args.dry_run,
        max_per_second=args.max_density,
        min_gap_frames=args.min_gap,
    )

    print(f"Scenes analyzed: {result.scenes_analyzed}")
    print(f"Moments detected: {result.moments_detected}")
    print(f"Cues generated: {result.cues_generated}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    if not args.dry_run:
        print("\nScenes updated:")
        for scene_id, success in result.scenes_updated.items():
            status = "[ok]" if success else "[failed]"
            print(f"  {status} {scene_id}")
        print(f"\nStoryboard updated: {project.root_dir}/storyboard/storyboard.json")
    else:
        print("\n[DRY RUN] No changes written. Run without --dry-run to update storyboard.")

    return 0


def _cmd_sound_clear(project, args: argparse.Namespace) -> int:
    """Clear SFX cues from storyboard."""
    from ..sound.storyboard_updater import load_storyboard

    storyboard_path = project.root_dir / "storyboard" / "storyboard.json"

    if not storyboard_path.exists():
        print(f"Error: Storyboard not found: {storyboard_path}", file=sys.stderr)
        return 1

    updater = load_storyboard(storyboard_path)

    if args.scene:
        # Clear specific scene
        if updater.clear_scene_cues(args.scene):
            print(f"Cleared SFX cues from scene: {args.scene}")
        else:
            print(f"Error: Scene '{args.scene}' not found", file=sys.stderr)
            return 1
    else:
        # Clear all scenes
        updater.clear_all_cues()
        print("Cleared SFX cues from all scenes")

    updater.save(backup=True)
    print(f"Storyboard updated: {storyboard_path}")

    return 0


def cmd_factcheck(args: argparse.Namespace) -> int:
    """Run fact checking on a project's script and narration."""
    from ..project import load_project
    from ..factcheck import FactChecker, FactCheckError

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Fact checking project: {project.id}")
    print()

    try:
        checker = FactChecker(
            project,
            use_mock=args.mock,
            verbose=args.verbose,
            timeout=args.timeout,
        )

        report = checker.run_fact_check()

        # Display results
        print()
        print("=" * 60)
        print("FACT CHECK REPORT")
        print("=" * 60)
        print(f"Project: {report.project_id}")
        print(f"Script: {report.script_title}")
        print(f"Source Documents: {', '.join(report.source_documents)}")
        print()

        # Summary
        print("SUMMARY")
        print("-" * 40)
        print(f"Total Issues: {report.summary.total_issues}")
        print(f"  Critical: {report.summary.critical_count}")
        print(f"  High: {report.summary.high_count}")
        print(f"  Medium: {report.summary.medium_count}")
        print(f"  Low: {report.summary.low_count}")
        print(f"  Info: {report.summary.info_count}")
        print(f"Accuracy Score: {report.summary.overall_accuracy_score:.0%}")
        print(f"Web Verified: {report.summary.web_verified_count} issues")
        print()

        # Issues
        if report.issues:
            print("ISSUES")
            print("-" * 40)
            for issue in report.issues:
                severity_icon = {
                    "critical": "[CRITICAL]",
                    "high": "[HIGH]",
                    "medium": "[MEDIUM]",
                    "low": "[LOW]",
                    "info": "[INFO]",
                }.get(issue.severity.value, "[?]")

                print(f"\n{severity_icon} {issue.id}")
                print(f"  Location: {issue.location}")
                print(f"  Category: {issue.category.value}")
                print(f"  Original: \"{issue.original_text[:100]}{'...' if len(issue.original_text) > 100 else ''}\"")
                print(f"  Issue: {issue.issue_description}")
                print(f"  Correction: {issue.correction}")
                print(f"  Source: {issue.source_reference}")
                print(f"  Confidence: {issue.confidence:.0%}")
                if issue.verified_via_web:
                    print("  (Verified via web search)")

        # Recommendations
        if report.recommendations:
            print()
            print("RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

        # Save report
        if not args.no_save:
            output_path = checker.save_report(report)
            print()
            print(f"Report saved to: {output_path}")

        # Return code based on critical issues
        if report.has_critical_issues():
            print()
            print("WARNING: Critical issues found! Review and fix before publishing.")
            return 1

        return 0

    except FactCheckError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_short(args: argparse.Namespace) -> int:
    """Generate YouTube Short from existing project.

    This command creates a vertical short (1080x1920) optimized for
    YouTube Shorts, Instagram Reels, and TikTok from an existing
    full-length video project.

    Prerequisites:
    - Script must be generated (run 'script' command first)
    - Narrations must be generated (run 'narration' command first)
    """
    import json
    from ..project import load_project
    from ..short import ShortGenerator, ShortSceneGenerator
    from ..short.generator import normalize_script_format
    from ..models import Script

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Validate prerequisites
    narration_path = project.get_path("narration")
    if not narration_path.exists():
        print("Error: Narrations not found. Run 'narration' command first.", file=sys.stderr)
        return 1

    script_path = project.get_path("script")
    if not script_path.exists():
        print("Error: Script not found. Run 'script' command first.", file=sys.stderr)
        return 1

    # Load source script for visual descriptions
    # Normalize script format (handles visual_description -> visual_cue conversion)
    source_script = None
    try:
        with open(script_path) as f:
            script_data = json.load(f)
        script_data = normalize_script_format(script_data)
        source_script = Script(**script_data)
    except Exception as e:
        print(f"  Warning: Could not load script ({type(e).__name__}: {e})")
        print("  Will use generic visual components instead.")

    # Get mode and duration
    mode = getattr(args, "mode", "hook")
    duration = args.duration

    # Set default duration based on mode if not specified
    if duration is None:
        duration = 60 if mode == "summary" else 45

    print(f"Generating YouTube Short for: {project.id}")
    print(f"  Variant: {args.variant}")
    print(f"  Mode: {mode}")
    print(f"  Duration: {duration}s")
    print(f"  Custom scenes: {not args.skip_custom_scenes}")
    print()

    # Parse scene override if provided (only for hook mode)
    scene_ids = None
    if args.scenes:
        if mode == "summary":
            print("  Warning: --scenes is ignored in summary mode (uses all scenes)")
        else:
            scene_ids = [s.strip() for s in args.scenes.split(",")]
            print(f"  Using specified scenes: {scene_ids}")

    # Initialize generators
    generator = ShortGenerator()
    scene_generator = ShortSceneGenerator()

    # Generate short script
    if mode == "summary":
        print("Analyzing script for summary sweep...")
    else:
        print("Analyzing script for best hook...")
    result = generator.generate_short(
        project,
        variant=args.variant,
        duration=duration,
        scene_ids=scene_ids,
        mode=mode,
        force=args.force,
        mock=args.mock,
    )

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    print(f"  Generated short script: {result.short_script_path}")

    # Setup variant directory and scenes
    variant_dir = project.short_dir / args.variant
    scenes_dir = variant_dir / "scenes"
    storyboard_dir = variant_dir / "storyboard"
    storyboard_dir.mkdir(parents=True, exist_ok=True)
    storyboard_path = storyboard_dir / "shorts_storyboard.json"

    # Setup vertical scenes (styles.ts, CTA scene)
    print("Setting up vertical scenes...")
    short_script = ShortGenerator.load_short_script(result.short_script_path)
    scene_paths = scene_generator.setup_short_scenes(
        project, short_script, variant=args.variant
    )
    print(f"  Generated styles: {scene_paths['styles_path']}")
    print(f"  Generated CTA scene: {scene_paths['cta_path']}")

    # Generate voiceover FIRST, then create storyboard from actual word timings
    # This ensures captions perfectly match the spoken audio
    shorts_storyboard = None
    if not args.skip_voiceover:
        from ..voiceover import VoiceoverGenerator

        print("Generating voiceover with word timestamps...")
        voiceover_generator = VoiceoverGenerator()
        voiceover_dir = variant_dir / "voiceover"

        try:
            short_voiceover = voiceover_generator.generate_short_voiceover(
                short_script,
                voiceover_dir,
            )

            # Get relative path for Remotion
            relative_voiceover_path = short_voiceover.audio_path.relative_to(project.root_dir)

            print(f"  Voiceover: {relative_voiceover_path}")
            print(f"  Duration: {short_voiceover.duration_seconds:.2f}s")

            # Generate storyboard with custom scenes (new enhanced pipeline)
            if not args.skip_custom_scenes:
                print("Generating shorts storyboard with custom scenes...")
                # Get the full video scenes directory for inspiration
                project_scenes_dir = project.root_dir / "scenes"
                shorts_storyboard = generator.generate_shorts_with_custom_scenes(
                    short_script,
                    short_voiceover.word_timestamps,
                    short_voiceover.duration_seconds,
                    source_script,
                    scenes_dir,
                    project_scenes_dir=project_scenes_dir if project_scenes_dir.exists() else None,
                    selected_scene_ids=scene_ids,
                    mock=args.mock,
                )
            else:
                # Fallback to old pipeline without custom scenes
                print("Generating shorts storyboard from voiceover...")
                shorts_storyboard = generator.generate_shorts_storyboard_from_voiceover(
                    short_script,
                    short_voiceover.word_timestamps,
                    short_voiceover.duration_seconds,
                    mock=args.mock,
                )

            shorts_storyboard.voiceover_path = str(relative_voiceover_path)

        except Exception as e:
            print(f"  Warning: Voiceover generation failed: {e}")
            print("  Falling back to timestamp-independent storyboard...")
            shorts_storyboard = None
    else:
        print("Skipping voiceover generation (--skip-voiceover)")

    # Fallback: generate storyboard without voiceover timing
    if shorts_storyboard is None:
        print("Generating shorts storyboard (without voiceover sync)...")
        shorts_storyboard = generator.generate_shorts_storyboard(
            short_script,
            mock=args.mock,
        )

    # Save final storyboard
    generator.save_shorts_storyboard(shorts_storyboard, storyboard_path)
    print(f"  Generated shorts storyboard: {storyboard_path}")
    print(f"  Total beats: {len(shorts_storyboard.beats)}")

    # Report on custom scenes generated
    custom_scene_count = sum(1 for b in shorts_storyboard.beats if b.component_name)
    if custom_scene_count > 0:
        print(f"  Custom scenes generated: {custom_scene_count}")

    print()
    print("=" * 60)
    print("SHORT GENERATION COMPLETE")
    print("=" * 60)
    print()
    print(f"Short script: {result.short_script_path}")
    print(f"Shorts storyboard: {storyboard_path}")
    print(f"Scenes directory: {scene_paths['scenes_dir']}")
    if shorts_storyboard.voiceover_path:
        print(f"Voiceover: {shorts_storyboard.voiceover_path}")
    print()
    print("To preview in Remotion:")
    print(f"  cd remotion && npm run dev")
    print("  Select 'ShortsPlayer' composition and provide the storyboard")

    return 0


# ============================================================================
# Short Subcommands (YouTube Shorts Pipeline)
# ============================================================================


def cmd_short_script(args: argparse.Namespace) -> int:
    """Generate short script from full video project.

    Analyzes the full video script to find the best hook and generates
    a condensed narration optimized for YouTube Shorts.
    """
    import json
    from ..project import load_project
    from ..short import ShortGenerator
    from ..short.generator import normalize_script_format
    from ..models import Script

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Validate prerequisites
    narration_path = project.get_path("narration")
    if not narration_path.exists():
        print("Error: Narrations not found. Run 'narration' command first.", file=sys.stderr)
        return 1

    script_path = project.get_path("script")
    if not script_path.exists():
        print("Error: Script not found. Run 'script' command first.", file=sys.stderr)
        return 1

    # Get mode and duration
    mode = getattr(args, "mode", "hook")
    duration = args.duration

    # Set default duration based on mode if not specified
    if duration is None:
        duration = 60 if mode == "summary" else 45

    print(f"Generating short script for: {project.id}")
    print(f"  Variant: {args.variant}")
    print(f"  Mode: {mode}")
    print(f"  Duration: {duration}s")
    print()

    # Parse scene override if provided (only for hook mode)
    scene_ids = None
    if args.scenes:
        if mode == "summary":
            print("  Warning: --scenes is ignored in summary mode (uses all scenes)")
        else:
            scene_ids = [s.strip() for s in args.scenes.split(",")]
            print(f"  Using specified scenes: {scene_ids}")

    generator = ShortGenerator()

    if mode == "summary":
        print("Analyzing script for summary sweep...")
    else:
        print("Analyzing script for best hook...")
    result = generator.generate_short(
        project,
        variant=args.variant,
        duration=duration,
        scene_ids=scene_ids,
        mode=mode,
        force=args.force,
        mock=args.mock,
    )

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    print(f"  Generated short script: {result.short_script_path}")
    print()
    print("Next step: python -m src.cli short scenes " + project.id)

    return 0


def cmd_short_scenes(args: argparse.Namespace) -> int:
    """Generate vertical scene components for shorts.

    Creates the styles.ts and CTA scene components needed for
    vertical 1080x1920 rendering.
    """
    from ..project import load_project
    from ..short import ShortGenerator, ShortSceneGenerator

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Check if short script exists
    variant_dir = project.short_dir / args.variant
    short_script_path = variant_dir / "short_script.json"

    if not short_script_path.exists():
        print(f"Error: Short script not found at {short_script_path}", file=sys.stderr)
        print("Run 'short script' command first.", file=sys.stderr)
        return 1

    short_script = ShortGenerator.load_short_script(short_script_path)

    print(f"Setting up vertical scenes for: {project.id}")
    print(f"  Variant: {args.variant}")
    print()

    scene_generator = ShortSceneGenerator()
    scene_paths = scene_generator.setup_short_scenes(
        project, short_script, variant=args.variant
    )

    print(f"  Generated styles: {scene_paths['styles_path']}")
    print(f"  Generated CTA scene: {scene_paths['cta_path']}")
    print(f"  Generated index: {scene_paths['index_path']}")
    print()
    print("Next step: python -m src.cli short voiceover " + project.id)

    return 0


def cmd_short_voiceover(args: argparse.Namespace) -> int:
    """Generate or process voiceover for shorts.

    Can either:
    - Generate TTS voiceover from the short script
    - Export a recording script for manual recording
    - Process a manually recorded audio file with Whisper
    """
    from ..project import load_project
    from ..short import ShortGenerator
    from ..voiceover import VoiceoverGenerator

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Check if short script exists
    variant_dir = project.short_dir / args.variant
    short_script_path = variant_dir / "short_script.json"

    if not short_script_path.exists():
        print(f"Error: Short script not found at {short_script_path}", file=sys.stderr)
        print("Run 'short script' command first.", file=sys.stderr)
        return 1

    short_script = ShortGenerator.load_short_script(short_script_path)
    voiceover_dir = variant_dir / "voiceover"

    # Handle --export-script
    if args.export_script:
        output_path = args.output if args.output else variant_dir / "recording_script.txt"
        script_path = VoiceoverGenerator.export_short_recording_script(
            short_script, output_path
        )
        print(f"Recording script exported to: {script_path}")
        print()
        print("Next steps:")
        print("  1. Record your voiceover following the script")
        print("  2. Save as short_voiceover.mp3 (or .wav, .m4a)")
        print(f"  3. Run: python -m src.cli short voiceover {project.id} --audio <path>")
        return 0

    # Determine provider
    provider = args.provider
    if args.mock:
        provider = "mock"

    voiceover_generator = VoiceoverGenerator(provider=provider)

    # Handle --audio (manual recording)
    if args.audio:
        print(f"Processing manual voiceover for: {project.id}")
        print(f"  Variant: {args.variant}")
        print(f"  Audio: {args.audio}")
        print()

        try:
            short_voiceover = voiceover_generator.process_manual_short_voiceover(
                Path(args.audio),
                voiceover_dir,
                whisper_model=args.whisper_model,
            )
            print()
            print("Next step: python -m src.cli short storyboard " + project.id)
            return 0
        except Exception as e:
            print(f"Error processing audio: {e}", file=sys.stderr)
            return 1

    # Default: generate TTS voiceover
    print(f"Generating voiceover for: {project.id}")
    print(f"  Variant: {args.variant}")
    print(f"  Provider: {provider}")
    print()

    try:
        short_voiceover = voiceover_generator.generate_short_voiceover(
            short_script,
            voiceover_dir,
        )
        print()
        print("Next step: python -m src.cli short storyboard " + project.id)
        return 0
    except Exception as e:
        print(f"Error generating voiceover: {e}", file=sys.stderr)
        return 1


def cmd_short_storyboard(args: argparse.Namespace) -> int:
    """Generate storyboard for shorts from voiceover.

    Creates the storyboard with beat timing synced to the voiceover,
    and optionally generates custom scene components.
    """
    import json
    from ..project import load_project
    from ..short import ShortGenerator
    from ..short.generator import normalize_script_format
    from ..short.custom_scene_generator import ShortsCustomSceneGenerator
    from ..voiceover.generator import ShortVoiceover
    from ..models import Script

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    variant_dir = project.short_dir / args.variant
    short_script_path = variant_dir / "short_script.json"
    voiceover_manifest_path = variant_dir / "voiceover" / "short_voiceover_manifest.json"
    storyboard_dir = variant_dir / "storyboard"
    storyboard_path = storyboard_dir / "shorts_storyboard.json"
    scenes_dir = variant_dir / "scenes"

    # Check prerequisites
    if not short_script_path.exists():
        print(f"Error: Short script not found. Run 'short script' first.", file=sys.stderr)
        return 1

    short_script = ShortGenerator.load_short_script(short_script_path)

    # Load voiceover manifest if it exists
    short_voiceover = None
    if voiceover_manifest_path.exists():
        short_voiceover = ShortVoiceover.load_manifest(voiceover_manifest_path)
    else:
        print("Warning: Voiceover not found. Run 'short voiceover' first for synced captions.")
        print("Generating storyboard without voiceover sync...")

    # Load source script for visual descriptions
    source_script = None
    script_path = project.get_path("script")
    if script_path.exists():
        try:
            with open(script_path) as f:
                script_data = json.load(f)
            script_data = normalize_script_format(script_data)
            source_script = Script(**script_data)
        except Exception:
            pass

    print(f"Generating storyboard for: {project.id}")
    print(f"  Variant: {args.variant}")
    print(f"  Custom scenes: {not args.skip_custom_scenes}")
    print()

    storyboard_dir.mkdir(parents=True, exist_ok=True)
    generator = ShortGenerator()

    if short_voiceover:
        # Convert WordTimestamp objects to dicts
        word_timestamps = [
            {
                "word": ts.word,
                "start_seconds": ts.start_seconds,
                "end_seconds": ts.end_seconds,
            }
            for ts in short_voiceover.word_timestamps
        ]

        if not args.skip_custom_scenes and source_script:
            print("Generating storyboard with custom scenes...")
            project_scenes_dir = project.root_dir / "scenes"
            shorts_storyboard = generator.generate_shorts_with_custom_scenes(
                short_script,
                word_timestamps,
                short_voiceover.duration_seconds,
                source_script,
                scenes_dir,
                project_scenes_dir=project_scenes_dir if project_scenes_dir.exists() else None,
                mock=args.mock,
            )
        else:
            print("Generating storyboard from voiceover...")
            shorts_storyboard = generator.generate_shorts_storyboard_from_voiceover(
                short_script,
                word_timestamps,
                short_voiceover.duration_seconds,
                mock=args.mock,
            )

        # Set voiceover path relative to project root
        relative_voiceover_path = short_voiceover.audio_path.relative_to(project.root_dir)
        shorts_storyboard.voiceover_path = str(relative_voiceover_path)
    else:
        # Fallback without voiceover
        shorts_storyboard = generator.generate_shorts_storyboard(
            short_script,
            mock=args.mock,
        )

    # Save storyboard
    generator.save_shorts_storyboard(shorts_storyboard, storyboard_path)
    print(f"  Generated storyboard: {storyboard_path}")
    print(f"  Total beats: {len(shorts_storyboard.beats)}")

    custom_scene_count = sum(1 for b in shorts_storyboard.beats if b.component_name)
    if custom_scene_count > 0:
        print(f"  Custom scenes: {custom_scene_count}")

    print()
    print("Next step: python -m src.cli render " + project.id + " --short")

    return 0


def cmd_short_timing(args: argparse.Namespace) -> int:
    """Generate timing.ts file from storyboard phase markers.

    This regenerates the timing constants that scene components use
    for animation synchronization. Run this after changing voiceover
    timing to update all scene files automatically.
    """
    from ..project import load_project
    from ..short.generator import ShortGenerator
    from ..short.timing_generator import generate_timing_file

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    variant_dir = project.short_dir / args.variant
    storyboard_path = variant_dir / "storyboard" / "shorts_storyboard.json"
    scenes_dir = variant_dir / "scenes"

    if not storyboard_path.exists():
        print(f"Error: Storyboard not found at {storyboard_path}", file=sys.stderr)
        print("Run 'short storyboard' first to generate the storyboard.", file=sys.stderr)
        return 1

    print(f"Generating timing.ts for: {project.id}")
    print(f"  Variant: {args.variant}")
    print()

    # Load storyboard
    storyboard = ShortGenerator.load_shorts_storyboard(storyboard_path)

    # Check if any beats have phase markers
    beats_with_markers = [b for b in storyboard.beats if b.phase_markers]
    if not beats_with_markers:
        print("Warning: No beats have phase markers defined.")
        print("Add phase_markers to beats in the storyboard to enable automatic timing sync.")
        print()
        print("Example phase_markers in storyboard JSON:")
        print('  "phase_markers": [')
        print('    {"id": "phase1End", "end_word": "insight.", "description": "End of intro"}')
        print('  ]')
        return 1

    # Generate timing file
    timing_path = scenes_dir / "timing.ts"
    timing_data = generate_timing_file(storyboard, timing_path, fps=args.fps)

    print(f"  Generated timing for {len(timing_data)} beats")
    print()
    print("Timing file can be imported in scene components:")
    print('  import { TIMING } from "./timing";')
    print()
    print("Usage in scenes:")
    print("  const phase1End = TIMING.beat_1.phase1End;")

    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Create, review, or manage video plans."""
    from ..project import load_project
    from ..ingestion import parse_document
    from ..understanding import ContentAnalyzer
    from ..planning import PlanGenerator, PlanEditor
    from ..config import load_config

    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    plan_dir = project.plan_dir
    plan_path = plan_dir / "plan.json"

    # Handle subcommands
    subcommand = getattr(args, "plan_command", None)

    if subcommand == "show":
        # Display existing plan
        if not plan_path.exists():
            print(f"Error: No plan found at {plan_path}", file=sys.stderr)
            print(f"Run 'python -m src.cli plan create {args.project}' to create one.")
            return 1

        plan = PlanGenerator.load_plan(plan_path)
        config = load_config()
        if args.mock:
            config.llm.provider = "mock"
        generator = PlanGenerator(config=config)
        print(generator.format_for_display(plan))
        return 0

    elif subcommand == "approve":
        # Approve existing plan without interactive session
        if not plan_path.exists():
            print(f"Error: No plan found at {plan_path}", file=sys.stderr)
            return 1

        plan = PlanGenerator.load_plan(plan_path)
        if plan.status == "approved":
            print(f"Plan is already approved (at {plan.approved_at})")
            return 0

        config = load_config()
        generator = PlanGenerator(config=config)
        editor = PlanEditor(generator=generator, plan_dir=plan_dir)
        plan = editor.approve_plan(plan)
        print(f"✓ Plan approved!")
        print(f"Run 'python -m src.cli script {args.project}' to generate the script.")
        return 0

    elif subcommand == "review":
        # Review/refine existing plan interactively
        if not plan_path.exists():
            print(f"Error: No plan found at {plan_path}", file=sys.stderr)
            print(f"Run 'python -m src.cli plan create {args.project}' to create one.")
            return 1

        plan = PlanGenerator.load_plan(plan_path)
        config = load_config()
        if args.mock:
            config.llm.provider = "mock"
        generator = PlanGenerator(config=config)
        editor = PlanEditor(generator=generator, plan_dir=plan_dir)

        plan, was_approved = editor.run_interactive_session(plan)
        return 0 if was_approved else 1

    elif subcommand == "create" or subcommand is None:
        # Create new plan (default behavior)
        print(f"Creating video plan for {project.id}")

        # Check if plan exists and --force not specified
        if plan_path.exists() and not args.force:
            print(f"Plan already exists: {plan_path}")
            print("Use --force to regenerate, or 'plan review' to refine.")
            return 0

        # Load input documents
        documents = []
        input_dir = project.input_dir
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
            print("Add source documents to the input/ directory first.")
            return 1

        # Find all supported input files
        input_files = []
        for pattern in ["*.md", "*.markdown", "*.pdf"]:
            input_files.extend(input_dir.glob(pattern))

        if not input_files:
            print(f"Error: No supported files found in {input_dir}", file=sys.stderr)
            return 1

        print(f"Found {len(input_files)} input file(s)")

        # Parse documents
        for f in input_files:
            print(f"  Parsing: {f.name}")
            try:
                doc = parse_document(f)
                documents.append(doc)
            except Exception as e:
                print(f"    Error: {e}", file=sys.stderr)
                return 1

        if not documents:
            print("Error: No documents were successfully parsed.", file=sys.stderr)
            return 1

        # Analyze content
        print("\nAnalyzing content...")
        config = load_config()
        if args.mock:
            config.llm.provider = "mock"

        analyzer = ContentAnalyzer(config)
        analysis = analyzer.analyze(documents[0])

        print(f"  Thesis: {analysis.core_thesis[:60]}...")
        print(f"  Concepts: {len(analysis.key_concepts)}")

        # Generate plan
        print("\nGenerating video plan...")
        generator = PlanGenerator(config=config)
        plan = generator.generate(
            documents[0],
            analysis,
            target_duration=args.duration or project.video.target_duration_seconds,
        )

        # Save plan
        json_path, md_path = generator.save_plan(plan, plan_dir)
        print(f"\nPlan saved to: {json_path}")

        # Interactive mode or just display
        if args.no_interactive:
            print(generator.format_for_display(plan))
            print(f"\nRun 'python -m src.cli plan review {args.project}' to refine.")
        else:
            editor = PlanEditor(generator=generator, plan_dir=plan_dir)
            plan, was_approved = editor.run_interactive_session(plan)
            if was_approved:
                return 0
            else:
                print(f"\nDraft saved. Run 'python -m src.cli plan review {args.project}' to continue.")
                return 0

        return 0

    else:
        print(f"Unknown plan command: {subcommand}")
        return 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Run the full video generation pipeline end-to-end.

    This command orchestrates all the steps needed to create a video:
    1. plan - Generate video plan (auto-approved unless --interactive)
    2. script - Generate script from plan/docs
    3. narration - Generate narrations for scenes
    4. scenes - Generate Remotion scene components
    5. voiceover - Generate audio from narrations
    6. storyboard - Create storyboard linking scenes + audio
    7. render - Render final video
    """
    from ..project import load_project

    # Define pipeline steps in order
    PIPELINE_STEPS = ["plan", "script", "narration", "scenes", "voiceover", "storyboard", "render"]

    # Parse --from and --to options
    start_step = args.from_step.lower() if args.from_step else "plan"
    end_step = args.to_step.lower() if args.to_step else "render"

    if start_step not in PIPELINE_STEPS:
        print(f"Error: Unknown step '{start_step}'")
        print(f"Available steps: {', '.join(PIPELINE_STEPS)}")
        return 1

    if end_step not in PIPELINE_STEPS:
        print(f"Error: Unknown step '{end_step}'")
        print(f"Available steps: {', '.join(PIPELINE_STEPS)}")
        return 1

    start_idx = PIPELINE_STEPS.index(start_step)
    end_idx = PIPELINE_STEPS.index(end_step)

    if start_idx > end_idx:
        print(f"Error: --from step '{start_step}' comes after --to step '{end_step}'")
        return 1

    steps_to_run = PIPELINE_STEPS[start_idx : end_idx + 1]

    # Load project
    try:
        project = load_project(Path(args.projects_dir) / args.project)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"{'=' * 60}")
    print(f"VIDEO GENERATION PIPELINE")
    print(f"{'=' * 60}")
    print(f"Project: {project.id}")
    print(f"Title: {project.title}")
    print(f"Steps: {' → '.join(steps_to_run)}")
    if args.force:
        print(f"Mode: Force regenerate all steps")
    else:
        print(f"Mode: Skip completed steps")
    print(f"{'=' * 60}\n")

    # Helper to check if step output exists
    def step_output_exists(step: str) -> bool:
        """Check if a step's output already exists."""
        if step == "plan":
            plan_path = project.plan_dir / "plan.json"
            if not plan_path.exists():
                return False
            # Check if plan is approved
            try:
                with open(plan_path) as f:
                    plan_data = json.load(f)
                return plan_data.get("status") == "approved"
            except (json.JSONDecodeError, KeyError):
                return False
        elif step == "script":
            return (project.root_dir / "script" / "script.json").exists()
        elif step == "narration":
            return (project.root_dir / "narration" / "narrations.json").exists()
        elif step == "scenes":
            scenes_dir = project.root_dir / "scenes"
            if not scenes_dir.exists():
                return False
            tsx_files = list(scenes_dir.glob("*Scene.tsx"))
            return len(tsx_files) > 0
        elif step == "voiceover":
            return (project.root_dir / "voiceover" / "manifest.json").exists()
        elif step == "storyboard":
            return (project.root_dir / "storyboard" / "storyboard.json").exists()
        elif step == "render":
            output_dir = project.root_dir / "output"
            if not output_dir.exists():
                return False
            mp4_files = list(output_dir.glob("*.mp4"))
            return len(mp4_files) > 0
        return False

    # Run each step
    for step_num, step in enumerate(steps_to_run, 1):
        step_header = f"[{step_num}/{len(steps_to_run)}] {step.upper()}"
        print(f"\n{'─' * 60}")
        print(f"{step_header}")
        print(f"{'─' * 60}")

        # Check if we can skip this step
        if not args.force and step_output_exists(step):
            print(f"✓ Output already exists, skipping (use --force to regenerate)")
            continue

        # Create args namespace for the step command
        step_args = argparse.Namespace(
            project=args.project,
            projects_dir=args.projects_dir,
        )

        # Add step-specific arguments
        if step == "plan":
            # Check if input files exist
            input_dir = project.root_dir / "input"
            if not input_dir.exists() or not list(input_dir.glob("*")):
                print(f"Error: No input files found in {input_dir}")
                print("Please add input documents (PDF, MD, or run with --url)")
                return 1
            step_args.plan_command = "create"
            step_args.duration = None
            step_args.mock = args.mock
            step_args.force = args.force or not step_output_exists(step)
            # Auto-approve plan unless --interactive is specified
            step_args.no_interactive = not getattr(args, "interactive", False)
            result = cmd_plan(step_args)

            # If interactive mode and plan wasn't approved, stop pipeline
            if getattr(args, "interactive", False) and result != 0:
                print("\nPipeline paused. Approve the plan to continue.")
                return result

            # Auto-approve plan if not interactive
            if not getattr(args, "interactive", False):
                plan_path = project.plan_dir / "plan.json"
                if plan_path.exists():
                    from ..planning import PlanGenerator
                    plan = PlanGenerator.load_plan(plan_path)
                    if plan.status != "approved":
                        from datetime import datetime
                        plan.status = "approved"
                        plan.approved_at = datetime.now().isoformat()
                        plan_dir = project.plan_dir
                        plan_dir.mkdir(parents=True, exist_ok=True)
                        with open(plan_path, "w") as f:
                            json.dump(plan.model_dump(), f, indent=2)
                        print("✓ Plan auto-approved")

        elif step == "script":
            # Check if input files exist
            input_dir = project.root_dir / "input"
            if not input_dir.exists() or not list(input_dir.glob("*")):
                print(f"Error: No input files found in {input_dir}")
                print("Please add input documents (PDF, MD, or run with --url)")
                return 1
            step_args.input = None
            step_args.url = None
            step_args.duration = None
            step_args.mock = args.mock
            step_args.timeout = args.timeout
            step_args.force = args.force or not step_output_exists(step)
            step_args.verbose = False
            step_args.skip_plan = False  # Use plan if available
            step_args.continue_on_error = False
            result = cmd_script(step_args)

        elif step == "narration":
            step_args.mock = args.mock
            step_args.timeout = args.timeout
            step_args.force = args.force or not step_output_exists(step)
            step_args.topic = None  # Use project title
            step_args.verbose = False
            result = cmd_narration(step_args)

        elif step == "scenes":
            step_args.mock = args.mock
            step_args.timeout = args.timeout
            step_args.force = args.force or not step_output_exists(step)
            step_args.sync = False  # Generate mode, not sync
            step_args.scene = None
            step_args.no_validate = False
            step_args.verbose = False
            result = cmd_scenes(step_args)

        elif step == "voiceover":
            step_args.provider = args.voice_provider or project.tts.provider or "edge"
            step_args.mock = args.mock
            step_args.continue_on_error = False
            step_args.export_script = False
            step_args.output = None
            step_args.audio_dir = None
            step_args.whisper_model = "base"
            step_args.no_sync = False
            step_args.with_tags = False
            result = cmd_voiceover(step_args)

        elif step == "storyboard":
            step_args.view = False
            step_args.mock = args.mock
            step_args.timeout = args.timeout
            step_args.force = args.force or not step_output_exists(step)
            step_args.verbose = False
            result = cmd_storyboard(step_args)

        elif step == "render":
            step_args.resolution = args.resolution
            step_args.quality = "high"
            step_args.output = None
            step_args.scenes = None
            step_args.preview = False
            step_args.open_output = False
            step_args.dev = False
            step_args.fast = False
            step_args.concurrency = None
            result = cmd_render(step_args)

        else:
            print(f"Error: Unknown step '{step}'")
            return 1

        # Check result
        if result != 0:
            print(f"\n{'=' * 60}")
            print(f"PIPELINE FAILED at step: {step}")
            print(f"{'=' * 60}")
            return result

        print(f"✓ {step.upper()} completed successfully")

    # Success
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 60}")

    # Show output location
    output_dir = project.root_dir / "output"
    mp4_files = list(output_dir.glob("*.mp4")) if output_dir.exists() else []
    if mp4_files:
        latest_video = max(mp4_files, key=lambda f: f.stat().st_mtime)
        print(f"\nOutput video: {latest_video}")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Video Explainer Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--projects-dir",
        default="projects",
        help="Path to projects directory (default: projects)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List all projects")
    list_parser.set_defaults(func=cmd_list)

    # info command
    info_parser = subparsers.add_parser("info", help="Show project information")
    info_parser.add_argument("project", help="Project ID")
    info_parser.set_defaults(func=cmd_info)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("project_id", help="Project ID (used as directory name)")
    create_parser.add_argument("--title", help="Project title")
    create_parser.add_argument("--description", help="Project description")
    create_parser.set_defaults(func=cmd_create)

    # generate command (full pipeline)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Run the full video generation pipeline end-to-end",
        description="""
Run all steps to generate a video from input documents:
  1. plan       - Generate and auto-approve video plan
  2. script     - Generate script from approved plan
  3. narration  - Generate narrations for scenes
  4. scenes     - Generate Remotion scene components
  5. voiceover  - Generate audio from narrations
  6. storyboard - Create storyboard linking scenes + audio
  7. render     - Render final video

By default, skips steps that have already been completed.
Use --force to regenerate all steps.
Use --interactive to pause for plan review before continuing.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generate_parser.add_argument("project", help="Project ID")
    generate_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force regenerate all steps even if output exists",
    )
    generate_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause for interactive plan review before continuing",
    )
    generate_parser.add_argument(
        "--from",
        dest="from_step",
        choices=["plan", "script", "narration", "scenes", "voiceover", "storyboard", "render"],
        help="Start from this step (skip earlier steps)",
    )
    generate_parser.add_argument(
        "--to",
        dest="to_step",
        choices=["plan", "script", "narration", "scenes", "voiceover", "storyboard", "render"],
        help="Stop after this step (skip later steps)",
    )
    generate_parser.add_argument(
        "--resolution", "-r",
        choices=["720p", "1080p", "4k"],
        default="1080p",
        help="Video resolution for render step (default: 1080p)",
    )
    generate_parser.add_argument(
        "--voice-provider",
        choices=["elevenlabs", "edge"],
        default=None,
        help="TTS provider for voiceover step (default: from project/config)",
    )
    generate_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM/TTS for testing (faster, no API calls)",
    )
    generate_parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="LLM timeout in seconds (default: 600)",
    )
    generate_parser.set_defaults(func=cmd_generate)

    # refine command - imported from refine module
    from ..refine.command import add_refine_parser
    add_refine_parser(subparsers)

    # voiceover command
    voiceover_parser = subparsers.add_parser("voiceover", help="Generate voiceovers")
    voiceover_parser.add_argument("project", help="Project ID")
    voiceover_parser.add_argument(
        "--provider",
        choices=["elevenlabs", "edge", "mock", "manual"],
        help="TTS provider to use (manual = import your own recordings)",
    )
    voiceover_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock TTS (for testing)",
    )
    voiceover_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue even if some scenes fail",
    )
    voiceover_parser.add_argument(
        "--export-script",
        action="store_true",
        help="Export a recording script for manual voiceover recording",
    )
    voiceover_parser.add_argument(
        "--output", "-o",
        help="Output path for recording script (with --export-script)",
    )
    voiceover_parser.add_argument(
        "--audio-dir",
        help="Directory containing recorded audio files (required with --provider manual)",
    )
    voiceover_parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size for transcription (default: base)",
    )
    voiceover_parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Don't auto-sync storyboard durations with voiceover durations",
    )
    voiceover_parser.add_argument(
        "--with-tags",
        action="store_true",
        help="Add delivery tags to guide voice actor (with --export-script)",
    )
    voiceover_parser.set_defaults(func=cmd_voiceover)

    # storyboard command
    storyboard_parser = subparsers.add_parser("storyboard", help="Generate or view storyboard")
    storyboard_parser.add_argument("project", help="Project ID")
    storyboard_parser.add_argument(
        "--view",
        action="store_true",
        help="View existing storyboard instead of generating",
    )
    storyboard_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing storyboard",
    )
    storyboard_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    storyboard_parser.set_defaults(func=cmd_storyboard)

    # script command
    script_parser = subparsers.add_parser("script", help="Generate script from input documents")
    script_parser.add_argument("project", help="Project ID")
    script_parser.add_argument(
        "--url",
        help="URL to fetch content from (web page)",
    )
    script_parser.add_argument(
        "--input", "-i",
        help="Path to input file (PDF or Markdown)",
    )
    script_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (for testing)",
    )
    script_parser.add_argument(
        "--duration",
        type=int,
        help="Target duration in seconds",
    )
    script_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue even if some files fail to parse",
    )
    script_parser.add_argument(
        "--skip-plan",
        action="store_true",
        help="Generate script without using approved plan (backward compatible mode)",
    )
    script_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    script_parser.set_defaults(func=cmd_script)

    # plan command (with subcommands)
    plan_parser = subparsers.add_parser(
        "plan",
        help="Create and manage video plans before script generation",
        description="""
Interactive video planning - create and refine a structured plan before generating scripts.

Commands:
  plan create <project>     Generate new plan (interactive by default)
  plan review <project>     Review and refine existing plan
  plan show <project>       Display current plan
  plan approve <project>    Approve plan without interactive session
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan_subparsers = plan_parser.add_subparsers(
        dest="plan_command",
        help="Plan commands",
    )

    # plan create
    plan_create_parser = plan_subparsers.add_parser(
        "create",
        help="Generate a new video plan",
    )
    plan_create_parser.add_argument("project", help="Project ID")
    plan_create_parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Generate plan without interactive review",
    )
    plan_create_parser.add_argument(
        "--duration",
        type=int,
        help="Target duration in seconds",
    )
    plan_create_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing plan",
    )
    plan_create_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (for testing)",
    )

    # plan review
    plan_review_parser = plan_subparsers.add_parser(
        "review",
        help="Review and refine existing plan interactively",
    )
    plan_review_parser.add_argument("project", help="Project ID")
    plan_review_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for refinement (for testing)",
    )

    # plan show
    plan_show_parser = plan_subparsers.add_parser(
        "show",
        help="Display current plan",
    )
    plan_show_parser.add_argument("project", help="Project ID")
    plan_show_parser.add_argument(
        "--mock",
        action="store_true",
        help="Unused, for consistency",
    )

    # plan approve
    plan_approve_parser = plan_subparsers.add_parser(
        "approve",
        help="Approve plan without interactive session",
    )
    plan_approve_parser.add_argument("project", help="Project ID")

    plan_parser.set_defaults(func=cmd_plan)

    # narration command
    narration_parser = subparsers.add_parser("narration", help="Generate narrations for a project")
    narration_parser.add_argument("project", help="Project ID")
    narration_parser.add_argument(
        "--topic",
        help="Topic to generate narrations for (default: project title)",
    )
    narration_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock narrations (for testing)",
    )
    narration_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing narrations",
    )
    narration_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    narration_parser.set_defaults(func=cmd_narration)

    # scenes command
    scenes_parser = subparsers.add_parser(
        "scenes",
        help="Generate Remotion scene components from script",
    )
    scenes_parser.add_argument("project", help="Project ID")
    scenes_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scenes",
    )
    scenes_parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync existing scenes to updated voiceover timing (timing-only update)",
    )
    scenes_parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Regenerate a specific scene by number (e.g., 6) or filename (e.g., HookScene.tsx). With --sync, syncs only that scene.",
    )
    scenes_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per scene generation in seconds (default: 300)",
    )
    scenes_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation (generate scene without checking for errors). Use when LLM keeps failing validation.",
    )
    scenes_parser.add_argument(
        "--verify",
        action="store_true",
        help="Run syntax verification only (no generation). Checks existing scenes for syntax errors and attempts to fix them.",
    )
    scenes_parser.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="With --verify, report errors but don't attempt automatic fixes.",
    )
    scenes_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    scenes_parser.set_defaults(func=cmd_scenes)

    # render command
    render_parser = subparsers.add_parser("render", help="Render video")
    render_parser.add_argument("project", help="Project ID")
    render_parser.add_argument(
        "--preview",
        action="store_true",
        help="Quick preview render",
    )
    render_parser.add_argument(
        "--resolution", "-r",
        choices=["4k", "1440p", "1080p", "720p", "480p"],
        default="1080p",
        help="Output resolution (default: 1080p)",
    )
    render_parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster encoding (trades some quality for speed)",
    )
    render_parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of parallel threads for rendering",
    )
    render_parser.add_argument(
        "--short",
        action="store_true",
        help="Render a short video instead of the full video",
    )
    render_parser.add_argument(
        "--variant",
        default="default",
        help="Short variant to render (default: 'default')",
    )
    render_parser.add_argument(
        "--gl",
        choices=["angle", "egl", "swiftshader", "swangle", "vulkan"],
        help="OpenGL renderer for 3D content (use 'angle' or 'swangle' if WebGL fails)",
    )
    render_parser.add_argument(
        "--frames",
        help="Frame range to render (e.g., '0-3500', '3501-7000', '7001-' for chunked rendering)",
    )
    render_parser.add_argument(
        "--merge-chunks",
        action="store_true",
        help="Merge previously rendered chunks into final video (uses ffmpeg)",
    )
    render_parser.set_defaults(func=cmd_render)

    # feedback command
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Process or view feedback for a project",
    )
    feedback_parser.add_argument("project", help="Project ID")

    feedback_subparsers = feedback_parser.add_subparsers(
        dest="feedback_command",
        help="Feedback commands",
    )

    # feedback add
    feedback_add_parser = feedback_subparsers.add_parser(
        "add",
        help="Add and process new feedback",
    )
    feedback_add_parser.add_argument(
        "feedback_text",
        help="The feedback text (natural language)",
    )
    feedback_add_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze feedback without applying changes",
    )
    feedback_add_parser.add_argument(
        "--live",
        action="store_true",
        help="Stream Claude Code output in real-time",
    )

    # feedback list
    feedback_subparsers.add_parser(
        "list",
        help="List all feedback for the project",
    )

    # feedback show
    feedback_show_parser = feedback_subparsers.add_parser(
        "show",
        help="Show details of a feedback item",
    )
    feedback_show_parser.add_argument(
        "feedback_id",
        help="Feedback ID (e.g., fb_0001_1234567890)",
    )

    # feedback retry
    feedback_retry_parser = feedback_subparsers.add_parser(
        "retry",
        help="Retry a failed feedback item",
    )
    feedback_retry_parser.add_argument(
        "feedback_id",
        help="Feedback ID to retry",
    )
    feedback_retry_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze feedback without applying changes",
    )

    feedback_parser.set_defaults(func=cmd_feedback)

    # factcheck command
    factcheck_parser = subparsers.add_parser(
        "factcheck",
        help="Fact-check script and narration against source material",
    )
    factcheck_parser.add_argument("project", help="Project ID")
    factcheck_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for testing",
    )
    factcheck_parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="LLM timeout in seconds (default: 600)",
    )
    factcheck_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the report to file",
    )
    factcheck_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    factcheck_parser.set_defaults(func=cmd_factcheck)

    # short command (with subcommands)
    short_parser = subparsers.add_parser(
        "short",
        help="YouTube Shorts generation pipeline",
        description="""
YouTube Shorts generation pipeline.

Full pipeline (recommended):
  python -m src.cli short generate <project>    # Runs everything end-to-end

Or run individual steps:
  python -m src.cli short script <project>      # 1. Generate short script
  python -m src.cli short scenes <project>      # 2. Generate scene components
  python -m src.cli short voiceover <project>   # 3. Generate voiceover
  python -m src.cli short storyboard <project>  # 4. Create storyboard
  python -m src.cli short timing <project>      # 5. Generate timing.ts (optional)

Then render with:
  python -m src.cli render <project> --short

For manual voiceover recording:
  python -m src.cli short voiceover <project> --export-script
  # Record your voiceover
  python -m src.cli short voiceover <project> --audio <recording.mp3>
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    short_subparsers = short_parser.add_subparsers(
        dest="short_command",
        help="Short generation commands",
    )

    # short generate (full pipeline)
    short_generate_parser = short_subparsers.add_parser(
        "generate",
        help="Run full shorts pipeline (script → scenes → voiceover → storyboard)",
    )
    short_generate_parser.add_argument("project", help="Project ID")
    short_generate_parser.add_argument(
        "--mode", "-m",
        choices=["hook", "summary"],
        default="hook",
        help="Generation mode: 'hook' (deep dive into selected scenes) or 'summary' (rapid sweep of all scenes)",
    )
    short_generate_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        help="Target duration in seconds (default: 45 for hook, 60 for summary, max: 60)",
    )
    short_generate_parser.add_argument(
        "--variant",
        default="default",
        help="Variant name for multiple shorts from same project",
    )
    short_generate_parser.add_argument(
        "--scenes",
        help="Override scene selection (comma-separated scene IDs, hook mode only)",
    )
    short_generate_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force regenerate even if files exist",
    )
    short_generate_parser.add_argument(
        "--skip-voiceover",
        action="store_true",
        help="Skip voiceover generation",
    )
    short_generate_parser.add_argument(
        "--skip-custom-scenes",
        action="store_true",
        help="Skip custom scene generation (use generic components)",
    )
    short_generate_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for testing (no API calls)",
    )
    short_generate_parser.set_defaults(func=cmd_short)

    # short script
    short_script_parser = short_subparsers.add_parser(
        "script",
        help="Generate short script (hook analysis + condensed narration)",
    )
    short_script_parser.add_argument("project", help="Project ID")
    short_script_parser.add_argument(
        "--mode", "-m",
        choices=["hook", "summary"],
        default="hook",
        help="Generation mode: 'hook' (deep dive into selected scenes) or 'summary' (rapid sweep of all scenes)",
    )
    short_script_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        help="Target duration in seconds (default: 45 for hook, 60 for summary, max: 60)",
    )
    short_script_parser.add_argument(
        "--variant",
        default="default",
        help="Variant name for multiple shorts from same project",
    )
    short_script_parser.add_argument(
        "--scenes",
        help="Override scene selection (comma-separated scene IDs, hook mode only)",
    )
    short_script_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force regenerate even if files exist",
    )
    short_script_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for testing (no API calls)",
    )
    short_script_parser.set_defaults(func=cmd_short_script)

    # short scenes
    short_scenes_parser = short_subparsers.add_parser(
        "scenes",
        help="Generate vertical scene components (styles, CTA)",
    )
    short_scenes_parser.add_argument("project", help="Project ID")
    short_scenes_parser.add_argument(
        "--variant",
        default="default",
        help="Variant name",
    )
    short_scenes_parser.set_defaults(func=cmd_short_scenes)

    # short voiceover
    short_voiceover_parser = short_subparsers.add_parser(
        "voiceover",
        help="Generate or process voiceover for shorts",
    )
    short_voiceover_parser.add_argument("project", help="Project ID")
    short_voiceover_parser.add_argument(
        "--variant",
        default="default",
        help="Variant name",
    )
    short_voiceover_parser.add_argument(
        "--provider",
        choices=["elevenlabs", "edge", "mock"],
        default="edge",
        help="TTS provider to use (default: edge)",
    )
    short_voiceover_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock TTS (for testing)",
    )
    short_voiceover_parser.add_argument(
        "--export-script",
        action="store_true",
        help="Export a recording script for manual voiceover",
    )
    short_voiceover_parser.add_argument(
        "--audio",
        help="Path to manually recorded audio file (uses Whisper for timestamps)",
    )
    short_voiceover_parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size for transcription (default: base)",
    )
    short_voiceover_parser.add_argument(
        "--output", "-o",
        help="Output path for recording script (with --export-script)",
    )
    short_voiceover_parser.set_defaults(func=cmd_short_voiceover)

    # short storyboard
    short_storyboard_parser = short_subparsers.add_parser(
        "storyboard",
        help="Generate storyboard from voiceover timing",
    )
    short_storyboard_parser.add_argument("project", help="Project ID")
    short_storyboard_parser.add_argument(
        "--variant",
        default="default",
        help="Variant name",
    )
    short_storyboard_parser.add_argument(
        "--skip-custom-scenes",
        action="store_true",
        help="Skip custom scene generation (use generic components)",
    )
    short_storyboard_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for testing",
    )
    short_storyboard_parser.set_defaults(func=cmd_short_storyboard)

    # short timing
    short_timing_parser = short_subparsers.add_parser(
        "timing",
        help="Generate timing.ts from storyboard phase markers",
    )
    short_timing_parser.add_argument("project", help="Project ID")
    short_timing_parser.add_argument(
        "--variant",
        default="default",
        help="Variant name",
    )
    short_timing_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    short_timing_parser.set_defaults(func=cmd_short_timing)

    # Keep legacy cmd_short for backward compatibility (runs full pipeline)
    short_parser.set_defaults(func=cmd_short)

    # music command
    music_parser = subparsers.add_parser(
        "music",
        help="Generate AI background music using MusicGen",
    )
    music_parser.add_argument("project", help="Project ID")

    music_subparsers = music_parser.add_subparsers(
        dest="music_command",
        help="Music commands",
    )

    # music generate
    music_generate_parser = music_subparsers.add_parser(
        "generate",
        help="Generate background music using AI",
    )
    music_generate_parser.add_argument(
        "--duration",
        type=int,
        help="Target duration in seconds (default: from storyboard)",
    )
    music_generate_parser.add_argument(
        "--topic",
        help="Topic for music style (default: project title)",
    )
    music_generate_parser.add_argument(
        "--style",
        help="Custom music style prompt",
    )
    music_generate_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't update storyboard.json with music config",
    )

    # music short
    music_short_parser = music_subparsers.add_parser(
        "short",
        help="Generate punchy background music for YouTube Short",
    )
    music_short_parser.add_argument(
        "--variant",
        default="default",
        help="Short variant name (default: 'default')",
    )
    music_short_parser.add_argument(
        "--duration",
        type=int,
        help="Target duration in seconds (default: from shorts storyboard)",
    )
    music_short_parser.add_argument(
        "--topic",
        help="Topic for music style (default: project title)",
    )
    music_short_parser.add_argument(
        "--style",
        help="Custom music style prompt (overrides auto-detection)",
    )
    music_short_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't update shorts_storyboard.json with music config",
    )

    # music info
    music_subparsers.add_parser(
        "info",
        help="Show music generation info and device support",
    )

    music_parser.set_defaults(func=cmd_music)

    # sound command
    sound_parser = subparsers.add_parser(
        "sound",
        help="Sound design: generate SFX library for Remotion",
    )
    sound_parser.add_argument("project", help="Project ID")

    sound_subparsers = sound_parser.add_subparsers(
        dest="sound_command",
        help="Sound commands",
    )

    # sound library
    sound_library_parser = sound_subparsers.add_parser(
        "library",
        help="Generate/manage SFX library",
    )
    sound_library_parser.add_argument(
        "--list",
        action="store_true",
        help="List all available sounds",
    )
    sound_library_parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate all SFX files to project's sfx/ directory",
    )
    sound_library_parser.add_argument(
        "--download",
        action="store_true",
        help="Alias for --generate (for backwards compatibility)",
    )

    # sound analyze - analyze scenes for sound moments
    sound_analyze_parser = sound_subparsers.add_parser(
        "analyze",
        help="Analyze scenes and show detected sound moments (dry run)",
    )
    sound_analyze_parser.add_argument(
        "--scene",
        type=str,
        help="Analyze only a specific scene ID",
    )
    sound_analyze_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed moment information",
    )

    # sound generate - generate SFX cues
    sound_generate_parser = sound_subparsers.add_parser(
        "generate",
        help="Generate SFX cues and write to storyboard.json",
    )
    sound_generate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing to storyboard",
    )
    sound_generate_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM semantic analysis (faster, pattern-only)",
    )
    sound_generate_parser.add_argument(
        "--theme",
        type=str,
        default="tech_ai",
        choices=["tech_ai", "science", "finance", "space", "nature", "abstract"],
        help="Sound theme (default: tech_ai)",
    )
    sound_generate_parser.add_argument(
        "--max-density",
        type=float,
        default=3.0,
        help="Maximum sounds per second (default: 3.0)",
    )
    sound_generate_parser.add_argument(
        "--min-gap",
        type=int,
        default=10,
        help="Minimum frames between sounds (default: 10)",
    )

    # sound clear - remove SFX cues
    sound_clear_parser = sound_subparsers.add_parser(
        "clear",
        help="Remove all SFX cues from storyboard",
    )
    sound_clear_parser.add_argument(
        "--scene",
        type=str,
        help="Clear only a specific scene ID",
    )

    sound_parser.set_defaults(func=cmd_sound)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
