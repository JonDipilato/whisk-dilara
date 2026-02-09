"""
=================================================================
  STORY VIDEO MAKER

  Creates an animated story video from a simple config file.

  HOW TO USE:
  1. Edit 'story_config.json' with your story details
  2. Run: python run_story.py
  3. Find your videos in the 'output/videos/' folder

  REQUIREMENTS:
  - Google Chrome installed
  - Python packages: selenium, edge-tts, rich
  - ffmpeg installed
  - Be logged into Google Whisk in Chrome
=================================================================
"""

import os
import sys
import json
import time
import shutil
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if HAS_RICH:
    console = Console()
    def status(msg, style="cyan"):
        console.print(f"[{style}]{msg}[/{style}]")
    def error(msg):
        console.print(f"[bold red]ERROR: {msg}[/bold red]")
    def success(msg):
        console.print(f"[bold green]{msg}[/bold green]")
    def header(msg):
        console.print(Panel(f"[bold magenta]{msg}[/bold magenta]"))
else:
    def status(msg, style=None): print(f"  {msg}")
    def error(msg): print(f"  ERROR: {msg}")
    def success(msg): print(f"  OK: {msg}")
    def header(msg): print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def parse_scene_spec(spec: str, max_scene: int) -> list[int]:
    """Parse scene specification into list of scene numbers.

    Supports:
        - Single: "51" -> [51]
        - Range: "51-55" -> [51, 52, 53, 54, 55]
        - List: "51,53,60" -> [51, 53, 60]
        - Mixed: "51,53-55,60" -> [51, 53, 54, 55, 60]
    """
    scenes = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start.strip()), int(end.strip())
            scenes.update(range(start, end + 1))
        else:
            scenes.add(int(part))
    # Filter to valid range and sort
    return sorted(s for s in scenes if 1 <= s <= max_scene)


# Theme-specific colors for click-optimized thumbnails
THUMBNAIL_STYLES = {
    "mushroom": {"accent": "#FF9F43", "glow": "#FFA502", "bg_tint": (255, 159, 67)},
    "starfall": {"accent": "#9B59B6", "glow": "#8E44AD", "bg_tint": (155, 89, 182)},
    "ocean": {"accent": "#00CEC9", "glow": "#0984E3", "bg_tint": (0, 206, 201)},
    "winter": {"accent": "#74B9FF", "glow": "#0984E3", "bg_tint": (116, 185, 255)},
    "garden": {"accent": "#00B894", "glow": "#55EFC4", "bg_tint": (0, 184, 148)},
    "aurora": {"accent": "#A29BFE", "glow": "#6C5CE7", "bg_tint": (162, 155, 254)},
    "desert": {"accent": "#FDCB6E", "glow": "#F39C12", "bg_tint": (253, 203, 110)},
    "sky_islands": {"accent": "#FD79A8", "glow": "#E84393", "bg_tint": (253, 121, 168)},
}


def _has_nvenc():
    """Check if NVIDIA NVENC hardware encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


class StoryVideoMaker:
    def __init__(self, config_path="story_config.json", output_dir=None):
        self.root = Path(__file__).parent
        self.config_path = self.root / config_path
        self.config = self._load_config()
        if output_dir:
            output_path = Path(output_dir)
            self.output_dir = output_path if output_path.is_absolute() else (self.root / output_path)
        else:
            # Auto-detect episode folder from config's environment paths
            detected_dir = self._detect_episode_folder()
            if detected_dir:
                self.output_dir = detected_dir
                status(f"Auto-detected episode folder: {detected_dir.name}", "cyan")
            else:
                self.output_dir = self.root / "output"
                status("[yellow]WARNING: Could not detect episode folder from config, using ./output[/yellow]")
                status("[yellow]Consider using --output-dir to specify the correct location[/yellow]")
        self.chars_dir = self.root / "data" / "characters"
        self.envs_dir = self.root / "data" / "environments"
        self.audio_dir = self.output_dir / "audio"
        self.videos_dir = self.output_dir / "videos"

        # Create directories
        for d in [self.output_dir, self.chars_dir, self.envs_dir, self.audio_dir, self.videos_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # GPU acceleration settings
        self.use_gpu = _has_nvenc()
        self.encoder = "h264_nvenc" if self.use_gpu else "libx264"
        self.preset = "p4" if self.use_gpu else "medium"  # nvenc uses p1-p7
        self.quality_flag = "-cq" if self.use_gpu else "-crf"
        if self.use_gpu:
            status("GPU acceleration enabled (NVIDIA NVENC)", "green")
        else:
            status("Using CPU encoding (libx264)", "yellow")

    def _load_config(self):
        if not self.config_path.exists():
            error(f"Config file not found: {self.config_path}")
            error("Create a 'story_config.json' file. See the example template.")
            sys.exit(1)

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Validate required fields
        required = ["title", "characters", "scene", "scenes", "narration"]
        missing = [r for r in required if r not in config]
        if missing:
            error(f"Missing required fields in config: {', '.join(missing)}")
            sys.exit(1)

        if len(config["characters"]) == 0:
            error("At least one character is required")
            sys.exit(1)

        if len(config["scenes"]) == 0:
            error("At least one scene is required")
            sys.exit(1)

        return config

    def _detect_episode_folder(self) -> Optional[Path]:
        """Auto-detect episode folder from config's environment/scene paths.

        Looks for paths like 'output/episodes/chars_ep11_20260130_080911/refs/...'
        and extracts the episode folder.
        """
        # Check environments for episode folder path
        environments = self.config.get("environments", {})
        for env_data in environments.values():
            img_path = env_data.get("image_path", "")
            if "episodes" in img_path and "/refs/" in img_path:
                # Extract episode folder: output/episodes/EPISODE_FOLDER/refs/...
                parts = img_path.replace("\\", "/").split("/")
                try:
                    eps_idx = parts.index("episodes")
                    if eps_idx + 1 < len(parts):
                        episode_folder = "/".join(parts[:eps_idx + 2])
                        full_path = Path(episode_folder)
                        if not full_path.is_absolute():
                            full_path = self.root / full_path
                        if full_path.exists():
                            return full_path
                except (ValueError, IndexError):
                    pass

        # Check scene image_path as backup
        scene = self.config.get("scene", {})
        img_path = scene.get("image_path", "")
        if "episodes" in img_path:
            parts = img_path.replace("\\", "/").split("/")
            try:
                eps_idx = parts.index("episodes")
                if eps_idx + 1 < len(parts):
                    episode_folder = "/".join(parts[:eps_idx + 2])
                    full_path = Path(episode_folder)
                    if not full_path.is_absolute():
                        full_path = self.root / full_path
                    if full_path.exists():
                        return full_path
            except (ValueError, IndexError):
                pass

        return None

    def _get_settings(self, key, default=None):
        return self.config.get("settings", {}).get(key, default)

    def _resolve_ref_path(self, path_value: str, fallback_path: Path) -> Path:
        if path_value:
            path = Path(path_value)
            return path if path.is_absolute() else (self.root / path)
        return fallback_path

    def _get_character_ref_path(self, char: dict) -> Path:
        name = char["name"].lower().replace(" ", "_")
        default_path = self.chars_dir / f"{name}_01.png"
        return self._resolve_ref_path(char.get("image_path", ""), default_path)

    def _get_scene_ref_path(self) -> Path:
        scene = self.config["scene"]
        scene_name = scene["name"].lower().replace(" ", "_")
        default_path = self.envs_dir / f"{scene_name}.png"
        return self._resolve_ref_path(scene.get("image_path", ""), default_path)

    def _resolve_env_path(self, env_data: dict) -> Path:
        """Resolve path for a multi-environment entry."""
        env_name = env_data.get("name", "env").lower().replace(" ", "_")
        default_path = self.envs_dir / f"grandma_{env_name}.png"
        return self._resolve_ref_path(env_data.get("image_path", ""), default_path)

    # =========================================================================
    # STEP 1: Generate character & scene reference images via Whisk
    # =========================================================================
    def generate_references(self):
        header("STEP 1: Generating Reference Images via Whisk")

        from src.config import load_config
        from src.whisk_controller import WhiskController

        app_config = load_config()
        refs_needed = []

        # Check characters
        for char in self.config["characters"]:
            name = char["name"].lower().replace(" ", "_")
            filepath = self._get_character_ref_path(char)
            if filepath.exists() and filepath.stat().st_size > 10000:
                status(f"Character '{char['name']}' already exists ({filepath.stat().st_size // 1024}KB)")
            else:
                refs_needed.append(("character", name, char["description"], filepath))

        # Check environments (multi-environment for grandma Excel configs)
        environments = self.config.get("environments", {})
        if environments:
            for env_code, env_data in environments.items():
                if env_code == "E1":
                    continue  # Plain bg, no ref needed
                env_path = self._resolve_env_path(env_data)
                if env_path.exists() and env_path.stat().st_size > 10000:
                    status(f"Environment '{env_data['name']}' already exists ({env_path.stat().st_size // 1024}KB)")
                else:
                    env_name = env_data["name"].lower().replace(" ", "_")
                    refs_needed.append(("environment", env_name, env_data["description"], env_path))
        else:
            # Existing single-scene check
            scene = self.config["scene"]
            scene_name = scene["name"].lower().replace(" ", "_")
            scene_path = self._get_scene_ref_path()
            if scene_path.exists() and scene_path.stat().st_size > 10000:
                status(f"Scene '{scene['name']}' already exists ({scene_path.stat().st_size // 1024}KB)")
            else:
                refs_needed.append(("scene", scene_name, scene["description"], scene_path))

        if not refs_needed:
            success("All reference images already exist!")
            return True

        status(f"Need to generate {len(refs_needed)} reference(s)...")

        controller = WhiskController(app_config)
        for ref_type, ref_id, prompt, save_path in refs_needed:
            status(f"Generating {ref_type}: {ref_id}...")

            try:
                controller.start()
                time.sleep(3)

                controller.clear_inputs()
                time.sleep(2)
                controller.set_prompt(prompt)
                time.sleep(1)
                controller.generate()
                controller.wait_for_generation(timeout=app_config.generation.download_timeout)

                temp_dir = self.output_dir / f"_ref_{ref_id}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                downloaded = controller.download_images(temp_dir, prefix=f"ref_{ref_id}", crop=False)

                if downloaded:
                    best = max(downloaded, key=lambda p: p.stat().st_size)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(best, save_path)
                    success(f"Saved: {save_path.name} ({save_path.stat().st_size // 1024}KB)")
                else:
                    error(f"Failed to generate {ref_id}. Try running again.")

            except Exception as e:
                error(f"Error generating {ref_id}: {e}")
            finally:
                controller.stop()
                time.sleep(2)

        return True

    # =========================================================================
    # STEP 2: Generate scene images using references
    # =========================================================================
    def generate_scenes(self, force_scenes: list[int] = None):
        """Generate scene images using Whisk.

        Args:
            force_scenes: Optional list of specific scene numbers to regenerate.
                         When provided, ONLY these scenes are processed (existing
                         images are cleared first). When None, auto-detects which
                         scenes need generation.
        """
        from src.config import load_config
        from src.whisk_controller import WhiskController
        from src.models import ImageFormat

        app_config = load_config()
        style = self.config.get("style", "")
        scenes = self.config["scenes"]
        scene_refs = self.config.get("scene_refs")

        # Determine which scenes to generate
        if force_scenes:
            # Force-regenerate specific scenes
            header(f"STEP 2: Regenerating {len(force_scenes)} Specific Scene(s)")
            scenes_to_generate = []
            for i in force_scenes:
                if 1 <= i <= len(scenes):
                    scene_dir = self.output_dir / f"scene_{i:03d}_batch_1"
                    # Clear ALL existing images to force fresh generation (no leftovers)
                    if scene_dir.exists():
                        for img in list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.webp")) + list(scene_dir.glob("*.jpg")):
                            img.unlink()
                        status(f"Cleared existing images in scene_{i:03d}_batch_1")
                    scenes_to_generate.append((i, scenes[i - 1]))
                else:
                    error(f"Scene {i} out of range (1-{len(scenes)})")
            status(f"Will regenerate scenes: {', '.join(str(s) for s in force_scenes)}")
        else:
            # Auto-detect which scenes need generation
            header(f"STEP 2: Generating {len(self.config['scenes'])} Scene Images")
            scenes_to_generate = []
            for i, scene_prompt in enumerate(scenes, 1):
                scene_dir = self.output_dir / f"scene_{i:03d}_batch_1"
                if scene_dir.exists():
                    images = list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.webp"))
                    if images and max(img.stat().st_size for img in images) > 30000:
                        continue  # Already has good images
                scenes_to_generate.append((i, scene_prompt))

        if not scenes_to_generate:
            success("All scene images already generated!")
            return True

        if not force_scenes:
            status(f"{len(scenes_to_generate)} scenes need generation (skipping {len(scenes) - len(scenes_to_generate)} already done)")

        controller = WhiskController(app_config)
        max_scene_retries = 3
        failed_scenes = []

        if scene_refs:
            # Per-scene character ref switching
            environments = self.config.get("environments", {})
            char_map = {c.get("code"): c for c in self.config["characters"] if c.get("code")}

            # Single-environment fallback (scene_refs but no environments dict)
            single_env_path = None
            if not environments:
                sp = self._get_scene_ref_path()
                if sp.exists():
                    single_env_path = sp

            for scene_idx, scene_prompt in scenes_to_generate:
                full_prompt = f"{scene_prompt}, {style}" if style else scene_prompt

                # Look up refs for this scene
                ref_info = scene_refs[scene_idx - 1]

                # Build character files for this scene
                char_files_for_scene = []
                for code in ref_info.get("character_codes", []):
                    char_data = char_map.get(code)
                    if char_data:
                        fp = self._get_character_ref_path(char_data)
                        if fp.exists():
                            char_files_for_scene.append(fp)

                # Build environment path for this scene
                env_code = ref_info.get("environment_code", "")
                if environments and env_code and env_code != "E1":
                    env_data = environments.get(env_code, {})
                    env_path = self._resolve_env_path(env_data) if env_data else None
                else:
                    # Single environment (default style)
                    env_path = single_env_path

                scene_ok = False
                for attempt in range(1, max_scene_retries + 1):
                    scene_dir = self.output_dir / f"scene_{scene_idx:03d}_batch_1"
                    scene_dir.mkdir(parents=True, exist_ok=True)

                    if attempt > 1:
                        status(f"Scene {scene_idx} retry {attempt}/{max_scene_retries}...")
                        # Clear failed images before retry (all formats)
                        for img in list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.webp")) + list(scene_dir.glob("*.jpg")):
                            img.unlink()
                        time.sleep(5)
                    else:
                        status(f"Scene {scene_idx}/{len(scenes)}: {scene_prompt[:60]}...")

                    try:
                        controller.start()
                        time.sleep(3)

                        # Set 16:9 landscape aspect ratio for all scenes
                        controller.set_format(ImageFormat.LANDSCAPE)
                        time.sleep(1)

                        # Upload scene-specific references
                        if char_files_for_scene:
                            controller.upload_all_images(
                                char_paths=char_files_for_scene,
                                env_path=env_path if env_path and env_path.exists() else None,
                            )
                            time.sleep(2)

                        # Set prompt and generate
                        controller.set_prompt(full_prompt)
                        time.sleep(1)
                        controller.generate()
                        controller.wait_for_generation(timeout=app_config.generation.download_timeout)

                        # Download results
                        downloaded = controller.download_images(scene_dir, prefix=f"scene_{scene_idx:03d}")
                        if downloaded:
                            success(f"Scene {scene_idx}: {len(downloaded)} images saved")
                            scene_ok = True
                        else:
                            error(f"Scene {scene_idx}: No images downloaded (attempt {attempt}/{max_scene_retries})")

                    except Exception as e:
                        error(f"Scene {scene_idx} error (attempt {attempt}/{max_scene_retries}): {e}")
                    finally:
                        controller.stop()
                        time.sleep(2)

                    if scene_ok:
                        break

                if not scene_ok:
                    failed_scenes.append(scene_idx)
        else:
            # Existing path: same refs for all scenes
            char_files = []
            for char in self.config["characters"]:
                filepath = self._get_character_ref_path(char)
                if filepath.exists():
                    char_files.append(filepath)
                else:
                    error(f"Character reference missing: {filepath}")
                    error("Run step 1 first (reference generation)")
                    return False

            scene_path = self._get_scene_ref_path()
            if not scene_path.exists():
                error(f"Scene reference missing: {scene_path}")
                return False

            for scene_idx, scene_prompt in scenes_to_generate:
                full_prompt = f"{scene_prompt}, {style}" if style else scene_prompt

                scene_ok = False
                for attempt in range(1, max_scene_retries + 1):
                    scene_dir = self.output_dir / f"scene_{scene_idx:03d}_batch_1"
                    scene_dir.mkdir(parents=True, exist_ok=True)

                    if attempt > 1:
                        status(f"Scene {scene_idx} retry {attempt}/{max_scene_retries}...")
                        for img in list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.webp")):
                            img.unlink()
                        time.sleep(5)
                    else:
                        status(f"Scene {scene_idx}/{len(scenes)}: {scene_prompt[:60]}...")

                    try:
                        controller.start()
                        time.sleep(3)

                        # Set 16:9 landscape aspect ratio for all scenes
                        controller.set_format(ImageFormat.LANDSCAPE)
                        time.sleep(1)

                        # Upload character and environment references
                        controller.upload_all_images(
                            char_paths=char_files,
                            env_path=scene_path,
                        )
                        time.sleep(2)

                        # Set prompt and generate
                        controller.set_prompt(full_prompt)
                        time.sleep(1)
                        controller.generate()
                        controller.wait_for_generation(timeout=app_config.generation.download_timeout)

                        # Download results
                        downloaded = controller.download_images(scene_dir, prefix=f"scene_{scene_idx:03d}")
                        if downloaded:
                            success(f"Scene {scene_idx}: {len(downloaded)} images saved")
                            scene_ok = True
                        else:
                            error(f"Scene {scene_idx}: No images downloaded (attempt {attempt}/{max_scene_retries})")

                    except Exception as e:
                        error(f"Scene {scene_idx} error (attempt {attempt}/{max_scene_retries}): {e}")
                    finally:
                        controller.stop()
                        time.sleep(2)

                    if scene_ok:
                        break

                if not scene_ok:
                    failed_scenes.append(scene_idx)

        # Warn about failed scenes (don't abort — hours of generation shouldn't be thrown away)
        if failed_scenes:
            scene_spec = ",".join(str(s) for s in failed_scenes)
            error(f"{len(failed_scenes)} scene(s) failed after {max_scene_retries} attempts each: {failed_scenes}")
            error(f"Fix with: python run_story.py --scene {scene_spec}")
            return False

        return True

    # =========================================================================
    # STEP 3: Generate narration audio (per-scene or combined)
    # =========================================================================
    def generate_narration(self):
        header("STEP 3: Generating Narration Audio")

        # Check for per-scene narrations (Excel-style)
        scene_narrations = self.config.get("scene_narrations", [])
        if scene_narrations and any(n.strip() for n in scene_narrations):
            return self._generate_per_scene_narration(scene_narrations)

        # Fall back to combined narration
        text = self.config.get("narration", "")
        if not text or not text.strip():
            status("No narration text in config (music-only video)")
            return None

        narration_path = self.audio_dir / "narration.mp3"
        if narration_path.exists() and narration_path.stat().st_size > 50000:
            status(f"Narration already exists ({narration_path.stat().st_size // 1024}KB)")
            return narration_path

        try:
            import edge_tts
        except ImportError:
            error("edge-tts not installed. Run: pip install edge-tts")
            return None

        voice = self._get_settings("voice", "en-US-AriaNeural")
        rate = self._get_settings("narration_speed", "-15%")

        status(f"Generating narration with voice: {voice}, rate: {rate}")

        async def _generate():
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            await communicate.save(str(narration_path))

        asyncio.run(_generate())

        if narration_path.exists() and narration_path.stat().st_size > 10000:
            success(f"Narration saved: {narration_path.stat().st_size // 1024}KB")
            return narration_path
        else:
            error("Narration generation failed")
            return None

    def _generate_per_scene_narration(self, scene_narrations: list):
        """Generate TTS for each scene individually (Excel-style sync)."""
        import edge_tts
        import subprocess

        voice = self._get_settings("voice", "en-US-AriaNeural")
        rate = self._get_settings("narration_speed", "-15%")

        scene_audio_dir = self.audio_dir / "scenes"
        scene_audio_dir.mkdir(parents=True, exist_ok=True)

        # Check if already generated
        durations_file = self.audio_dir / "scene_durations.json"
        if durations_file.exists():
            import json
            durations = json.loads(durations_file.read_text(encoding="utf-8"))
            if len(durations) == len(scene_narrations):
                status(f"Per-scene narration exists ({len(durations)} scenes)")
                # Also check combined file exists
                combined = self.audio_dir / "narration.mp3"
                if combined.exists():
                    return combined

        status(f"Generating per-scene narration ({len(scene_narrations)} scenes)...")

        scene_durations = []
        audio_files = []

        # Build list of scenes to generate
        scenes_to_generate = []
        for idx, narr_text in enumerate(scene_narrations):
            scene_audio = scene_audio_dir / f"scene_{idx+1:03d}.mp3"
            audio_files.append(scene_audio)
            if not scene_audio.exists() or scene_audio.stat().st_size < 1000:
                scenes_to_generate.append((idx, narr_text, scene_audio))

        # Generate all scenes in one async batch (fixes Windows asyncio issues)
        tts_errors = []

        async def _generate_all_scenes():
            for idx, text, out_path in scenes_to_generate:
                if text and text.strip():
                    try:
                        communicate = edge_tts.Communicate(text.strip(), voice=voice, rate=rate)
                        await communicate.save(str(out_path))
                    except Exception as e:
                        tts_errors.append((idx + 1, str(e)))
                        error(f"  Scene {idx+1} TTS FAILED: {e}")
                else:
                    # Create silent audio for scenes without narration (0.5s) - this is intentional
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                        "-t", "0.5", "-q:a", "9", str(out_path)
                    ], capture_output=True)

                if (idx + 1) % 20 == 0:
                    status(f"  Generated {idx + 1}/{len(scene_narrations)} scene audios...")

                # Small delay for rate limiting
                await asyncio.sleep(0.05)

        if scenes_to_generate:
            status(f"  Generating {len(scenes_to_generate)} new scene audio files...")
            asyncio.run(_generate_all_scenes())

            # Fail-fast: error if TTS failed for any scenes
            if tts_errors:
                error(f"TTS generation failed for {len(tts_errors)} scene(s):")
                for scene_num, err_msg in tts_errors[:5]:
                    error(f"  Scene {scene_num}: {err_msg}")
                if len(tts_errors) > 5:
                    error(f"  ...and {len(tts_errors) - 5} more")
                error("Check your internet connection and edge-tts installation")
                sys.exit(1)

        # Get durations for all scenes
        duration_errors = []
        for idx, scene_audio in enumerate(audio_files):
            if scene_audio.exists():
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(scene_audio)
                ], capture_output=True, text=True)
                try:
                    dur = float(result.stdout.strip())
                except (ValueError, AttributeError):
                    duration_errors.append(idx + 1)
                    dur = 0.0
            else:
                duration_errors.append(idx + 1)
                dur = 0.0

            scene_durations.append(dur)

        # Fail-fast: error if duration detection failed
        if duration_errors:
            error(f"Could not get duration for {len(duration_errors)} audio file(s): scenes {duration_errors[:10]}")
            sys.exit(1)

            if (idx + 1) % 20 == 0:
                status(f"  Generated {idx + 1}/{len(scene_narrations)} scene audios...")

        # Save durations for video assembly
        import json
        durations_file.write_text(json.dumps(scene_durations), encoding="utf-8")

        # Concatenate all scene audios into one file
        concat_file = self.audio_dir / "scene_concat.txt"
        with open(concat_file, "w") as f:
            for audio in audio_files:
                f.write(f"file '{audio.absolute()}'\n")

        combined = self.audio_dir / "narration.mp3"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(combined)
        ], capture_output=True)

        total_dur = sum(scene_durations)
        success(f"Per-scene narration complete: {len(scene_durations)} scenes, {total_dur:.1f}s total")
        return combined

    # =========================================================================
    # STEP 4: Find or set music (auto-rotates per episode)
    # =========================================================================
    def get_music_path(self):
        from src.music_library import MusicLibrary, MusicCategory

        episode_num = self.config.get("episode", 0)
        seed = self.config.get("seed", 0)

        library = MusicLibrary()
        track = library.get_track_for_episode(
            episode_num=episode_num,
            seed=seed,
            category=MusicCategory.CALM,
        )

        if track and track.exists:
            status(f"Using music: {track.name} (rotated for episode {episode_num})")
            return track.path

        # Fallback: scan assets/music/calm/ directly
        music_dir = self.root / "assets" / "music" / "calm"
        if music_dir.exists():
            audio_files = sorted(
                [f for f in music_dir.iterdir()
                 if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".ogg")
                 and f.stat().st_size > 100000],
                key=lambda f: f.name
            )
            if audio_files:
                pick = audio_files[episode_num % len(audio_files)]
                status(f"Using music: {pick.name} (rotated from {len(audio_files)} tracks)")
                return pick

        status("No music files found. Run: python download_music.py", "yellow")
        return None

    # =========================================================================
    # STEP 5: Assemble final videos (with transitions)
    # =========================================================================

    # Transition types for variety between scenes
    SCENE_TRANSITIONS = [
        "fade", "dissolve", "smoothleft", "smoothright",
        "wipeleft", "wiperight", "circleopen", "circlecrop",
        "radial", "smoothup", "smoothdown",
    ]

    def _get_scene_images(self):
        """Collect best image for each scene, with scene indices for duration matching."""
        scenes = self.config["scenes"]
        images = []
        scene_indices = []  # Track which scene index each image corresponds to
        missing_folders = []
        empty_folders = []

        for i in range(1, len(scenes) + 1):
            scene_dir = self.output_dir / f"scene_{i:03d}_batch_1"
            if not scene_dir.exists():
                missing_folders.append(f"scene_{i:03d}_batch_1")
                continue
            candidates = (
                list(scene_dir.glob("*.png")) +
                list(scene_dir.glob("*.jpg")) +
                list(scene_dir.glob("*.webp"))
            )
            if candidates:
                best = max(candidates, key=lambda p: p.stat().st_size)
                images.append(best)
                scene_indices.append(i - 1)  # 0-indexed for duration array
            else:
                empty_folders.append(f"scene_{i:03d}_batch_1")

        # Fail-fast: error if scenes are missing
        if missing_folders or empty_folders:
            error_msg = []
            if missing_folders:
                error_msg.append(f"Missing scene folders: {', '.join(missing_folders[:10])}")
                if len(missing_folders) > 10:
                    error_msg.append(f"  ...and {len(missing_folders) - 10} more")
            if empty_folders:
                error_msg.append(f"Empty scene folders: {', '.join(empty_folders[:10])}")
                if len(empty_folders) > 10:
                    error_msg.append(f"  ...and {len(empty_folders) - 10} more")
            error("\n".join(error_msg))
            error(f"Expected {len(scenes)} scenes, found {len(images)} valid images")
            error("Run scene generation first or check your output directory")
            sys.exit(1)

        self._scene_indices = scene_indices  # Store for duration matching
        return images

    def _build_act_video(self, images, secs_per_scene, act_num, width, height, fps, crf):
        """Build a single act video with zoompan + xfade transitions.

        Args:
            images: List of image paths for this act.
            secs_per_scene: Duration per scene in seconds.
            act_num: Act number (0-indexed, used for transition variety).
            width, height: Output resolution.
            fps: Frame rate.
            crf: Quality setting.

        Returns:
            Path to the generated act video, or None on failure.
        """
        if not images:
            return None

        act_video = self.output_dir / f"_act_{act_num + 1}.mp4"
        n = len(images)
        frames_per_scene = int(float(fps) * secs_per_scene)
        xfade_duration = min(0.8, secs_per_scene * 0.15)  # 15% of scene or 0.8s max

        # Build FFmpeg inputs - use raw image inputs, zoompan handles frame generation
        # (Using -loop -t caused incorrect xfade timing)
        inputs = []
        for img in images:
            inputs.extend(["-i", str(img)])

        # Build filter: zoompan each input, then chain xfade
        filters = []

        # Zoompan each image (subtle zoom-in with slight pan variation)
        # Use 2500px scale - enough headroom for zoom without excessive memory/time
        # (8000px caused timeouts on small source images like 773x422)
        zoompan_scale = 2500
        for i in range(n):
            zoom_rate = "0.0003"
            zoom_max = "1.025"
            filters.append(
                f"[{i}:v]scale={zoompan_scale}:-1,"
                f"zoompan=z='min(zoom+{zoom_rate},{zoom_max})':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames_per_scene}:s={width}x{height}:fps={fps},"
                f"setpts=PTS-STARTPTS[v{i}]"
            )

        # Chain xfade transitions between scenes
        if n == 1:
            # Single scene, just use it directly
            filters.append(f"[v0]fade=t=in:st=0:d=0.5,fade=t=out:st={secs_per_scene - 0.5}:d=0.5[vout]")
        else:
            # First xfade
            t_idx = (act_num * 15) % len(self.SCENE_TRANSITIONS)
            transition = self.SCENE_TRANSITIONS[t_idx]
            offset = secs_per_scene - xfade_duration
            filters.append(
                f"[v0][v1]xfade=transition={transition}:duration={xfade_duration:.3f}:"
                f"offset={offset:.3f}[xf0]"
            )

            # Subsequent xfades
            for i in range(2, n):
                prev_label = f"xf{i - 2}"
                out_label = f"xf{i - 1}"
                t_idx = (act_num * 15 + i) % len(self.SCENE_TRANSITIONS)
                transition = self.SCENE_TRANSITIONS[t_idx]
                # Offset accumulates: each scene adds (secs_per_scene - xfade_duration)
                offset = i * (secs_per_scene - xfade_duration)
                filters.append(
                    f"[{prev_label}][v{i}]xfade=transition={transition}:"
                    f"duration={xfade_duration:.3f}:offset={offset:.3f}[{out_label}]"
                )

            # Add fade-in at start and fade-out at end for act boundaries
            final_label = f"xf{n - 2}"
            total_act_duration = n * secs_per_scene - (n - 1) * xfade_duration
            filters.append(
                f"[{final_label}]fade=t=in:st=0:d=0.8,"
                f"fade=t=out:st={total_act_duration - 0.8:.3f}:d=0.8[vout]"
            )

        filter_complex = ";\n".join(filters)

        cmd = (
            ["ffmpeg", "-y"] + inputs +
            ["-filter_complex", filter_complex,
             "-map", "[vout]",
             "-c:v", self.encoder, "-preset", self.preset, self.quality_flag, str(crf),
             "-pix_fmt", "yuv420p",
             str(act_video)]
        )

        status(f"  Building Act {act_num + 1} ({n} scenes, transitions: {xfade_duration:.1f}s)...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if act_video.exists() and act_video.stat().st_size > 10000:
            return act_video
        else:
            # Fallback: simple concat without transitions if xfade fails
            error(f"  Act {act_num + 1} transition build failed, using simple concat fallback")
            if result.stderr:
                status(f"  FFmpeg: {result.stderr[-200:]}", "dim")
            return self._build_act_simple(images, secs_per_scene, act_num, width, height, fps, crf)

    def _build_act_simple(self, images, secs_per_scene, act_num, width, height, fps, crf, scene_durations=None):
        """Fallback: simple concat with fade in/out per act.

        Args:
            secs_per_scene: Default duration (used if scene_durations not provided)
            scene_durations: Optional list of per-scene durations (for Excel-style sync)
        """
        act_video = self.output_dir / f"_act_{act_num + 1}.mp4"
        concat_file = self.output_dir / f"_act_{act_num + 1}_concat.txt"

        lines = []
        total = 0
        for idx, img in enumerate(images):
            # Use per-scene duration if available, else default
            dur = scene_durations[idx] if scene_durations and idx < len(scene_durations) else secs_per_scene
            dur = max(dur, 0.5)  # Minimum 0.5s per scene
            lines.append(f"file '{img}'")
            lines.append(f"duration {dur:.3f}")
            total += dur
            # Debug: Print first few durations for act 1
            if act_num == 0 and idx < 3:
                status(f"[DEBUG] Act1 scene {idx+1}: duration={dur:.3f}s")
        # Hold last frame
        lines.append(f"file '{images[-1]}'")
        lines.append("duration 0.1")

        with open(concat_file, "w") as f:
            f.write("\n".join(lines))

        # Keep concat file for debugging
        debug_concat = self.output_dir / f"_debug_act_{act_num + 1}_concat.txt"
        shutil.copy(concat_file, debug_concat)

        # Build each scene with zoompan effect, then concat
        scene_videos = []
        for idx, img in enumerate(images):
            dur = scene_durations[idx] if scene_durations and idx < len(scene_durations) else secs_per_scene
            dur = max(dur, 0.5)
            frames = int(dur * int(fps))

            scene_video = self.output_dir / f"_act_{act_num + 1}_scene_{idx + 1}.mp4"

            # Alternate zoom direction for variety
            if (act_num + idx) % 2 == 0:
                # Slow zoom in (1.0 -> 1.08)
                zoom_expr = "min(zoom+0.0005,1.08)"
            else:
                # Slow zoom out (1.08 -> 1.0)
                zoom_expr = "if(eq(on,1),1.08,max(zoom-0.0005,1.0))"

            # Zoompan: slow zoom with center focus
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(img),
                "-vf",
                f"zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={width}x{height}:fps={fps}",
                "-t", str(dur),
                "-c:v", self.encoder, "-preset", self.preset, self.quality_flag, str(crf),
                "-pix_fmt", "yuv420p",
                str(scene_video)
            ]
            subprocess.run(cmd, capture_output=True)
            if scene_video.exists():
                scene_videos.append(scene_video)

        # Concat all scene videos for this act
        if not scene_videos:
            return None

        scene_concat = self.output_dir / f"_act_{act_num + 1}_scenes.txt"
        with open(scene_concat, "w") as f:
            for sv in scene_videos:
                f.write(f"file '{sv}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(scene_concat),
            "-c", "copy",
            str(act_video)
        ]
        subprocess.run(cmd, capture_output=True)

        # Cleanup scene videos
        for sv in scene_videos:
            sv.unlink(missing_ok=True)
        scene_concat.unlink(missing_ok=True)
        return act_video if act_video.exists() else None

    def assemble_videos(self, narration_path, music_path):
        header("STEP 5: Assembling Final Videos (with transitions)")

        # Check ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            error("ffmpeg not found. Install ffmpeg first.")
            return False

        scenes = self.config["scenes"]
        resolution = self._get_settings("resolution", "1920x1080")
        width, height = resolution.split("x")
        fps = self._get_settings("fps", 24)
        crf = self._get_settings("video_quality", 18)
        music_vol = self._get_settings("music_volume", 0.18)
        narr_vol = self._get_settings("narration_volume", 1.2)

        # Get narration duration
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(narration_path)],
            capture_output=True, text=True
        )
        narr_duration = float(result.stdout.strip()) if result.stdout.strip() else 300

        # Check for per-scene durations (Excel-style sync)
        scene_durations_file = self.audio_dir / "scene_durations.json"
        scene_durations = None
        if scene_durations_file.exists():
            import json
            scene_durations = json.loads(scene_durations_file.read_text(encoding="utf-8"))
            status(f"Using per-scene narration durations ({len(scene_durations)} scenes)")

        # Account for xfade overlap: total_duration = N*scene_dur - (N-1)*xfade
        # We want total_duration >= narr_duration, so:
        # scene_dur = (narr_duration + buffer + (N-1)*xfade) / N
        # Xfade is min(0.8, scene_dur*0.15) — use 0.8 (max) for conservative estimate
        num_scenes = len(scenes)
        xfade_estimate = 0.8  # Max xfade duration, ensures video is never too short
        buffer_seconds = 3    # Extra padding so narration never gets cut off
        secs_per_scene = (narr_duration + buffer_seconds + (num_scenes - 1) * xfade_estimate) / num_scenes
        status(f"Narration: {narr_duration:.0f}s, {secs_per_scene:.2f}s per scene (with {buffer_seconds}s buffer)")

        # Collect all scene images
        all_images = self._get_scene_images()
        if not all_images:
            error("No scene images found!")
            return False

        # CRITICAL: Map durations to actual images using scene indices
        if scene_durations and hasattr(self, '_scene_indices'):
            # Build matched durations list - one duration per actual image
            matched_durations = []
            missing_duration_indices = []
            for scene_idx in self._scene_indices:
                if scene_idx < len(scene_durations):
                    matched_durations.append(scene_durations[scene_idx])
                else:
                    missing_duration_indices.append(scene_idx + 1)  # 1-indexed for display

            # Fail-fast: error if durations don't match
            if missing_duration_indices:
                error(f"Duration data missing for scenes: {missing_duration_indices[:10]}")
                error(f"scene_durations.json has {len(scene_durations)} entries, but scene indices go up to {max(self._scene_indices) + 1}")
                sys.exit(1)

            if len(matched_durations) != len(all_images):
                error(f"Duration count mismatch: {len(matched_durations)} durations != {len(all_images)} images")
                error("This indicates a bug in duration matching logic")
                sys.exit(1)

            scene_durations = matched_durations
            status(f"Matched {len(scene_durations)} durations to {len(all_images)} images")

        encoder_msg = f"GPU ({self.encoder})" if self.use_gpu else f"CPU ({self.encoder})"
        status(f"Found {len(all_images)} scene images, building with transitions...")
        status(f"Using encoder: {encoder_msg}")

        # Split into acts (15 scenes each) and build per-act videos
        act_size = 15
        act_videos = []
        for act_idx in range(0, len(all_images), act_size):
            act_images = all_images[act_idx:act_idx + act_size]

            # Get per-scene durations for this act if available
            act_durations = None
            if scene_durations:
                act_durations = scene_durations[act_idx:act_idx + act_size]
                # Use simple builder for per-scene sync (no xfade complexity)
                act_video = self._build_act_simple(
                    act_images, secs_per_scene, act_idx // act_size,
                    width, height, fps, crf, scene_durations=act_durations
                )
            else:
                # Use full xfade transitions for uniform timing
                act_video = self._build_act_video(
                    act_images, secs_per_scene, act_idx // act_size,
                    width, height, fps, crf
                )

            if act_video:
                act_videos.append(act_video)

        if not act_videos:
            error("Failed to build any act videos!")
            return False

        success(f"Built {len(act_videos)} act segments with transitions")

        # Concatenate all acts (the fade in/out creates natural black gaps between acts)
        concat_file = self.output_dir / "ffmpeg_acts_concat.txt"
        with open(concat_file, "w") as f:
            for av in act_videos:
                f.write(f"file '{av}'\n")

        # Build silent video from acts
        silent_video = self.output_dir / "_full_video_silent.mp4"
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c:v", self.encoder, "-preset", self.preset, self.quality_flag, str(crf),
            "-pix_fmt", "yuv420p",
            str(silent_video)
        ]
        subprocess.run(concat_cmd, capture_output=True)

        if not silent_video.exists():
            error("Failed to concatenate act videos")
            return False

        total_duration = narr_duration + 1
        title_safe = self.config["title"].lower().replace(" ", "_").replace("-", "")[:30]
        versions = self._get_settings("output_versions", ["narrated", "music_only"])

        # Build narrated version (video + narration + music)
        if "narrated" in versions and narration_path:
            status("Mixing narrated version...")
            output_path = self.videos_dir / f"{title_safe}_narrated.mp4"

            if music_path:
                mixed_audio = self.audio_dir / "mixed_audio.wav"
                mix_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(narration_path),
                    "-i", str(music_path),
                    "-filter_complex",
                    f"[0:a]volume={narr_vol}[narr];"
                    f"[1:a]volume={music_vol},afade=t=out:st={narr_duration - 8}:d=8[mus];"
                    f"[narr][mus]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                    "-map", "[aout]",
                    "-t", str(total_duration),
                    "-ac", "2",
                    "-c:a", "pcm_s16le",
                    str(mixed_audio)
                ]
                subprocess.run(mix_cmd, capture_output=True)
                audio_input = mixed_audio
            else:
                audio_input = narration_path

            mux_cmd = [
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(audio_input),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path)
            ]
            subprocess.run(mux_cmd, capture_output=True)

            if output_path.exists() and output_path.stat().st_size > 100000:
                success(f"Narrated video: {output_path.name} ({output_path.stat().st_size // 1024 // 1024}MB)")
            else:
                error("Failed to create narrated video")

        # Build music-only version
        if "music_only" in versions and music_path:
            status("Mixing music-only version...")
            output_path = self.videos_dir / f"{title_safe}_music_only.mp4"

            mux_cmd = [
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(music_path),
                "-filter_complex",
                f"[1:a]volume=0.7,afade=t=in:st=0:d=3,"
                f"afade=t=out:st={narr_duration - 8}:d=8[mus]",
                "-map", "0:v", "-map", "[mus]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path)
            ]
            subprocess.run(mux_cmd, capture_output=True)

            if output_path.exists() and output_path.stat().st_size > 100000:
                success(f"Music-only video: {output_path.name} ({output_path.stat().st_size // 1024 // 1024}MB)")
            else:
                error("Failed to create music-only video")

        # Clean up temp act videos
        for av in act_videos:
            if av.exists():
                av.unlink()
        if silent_video.exists():
            silent_video.unlink()
        for f in self.output_dir.glob("_act_*_concat.txt"):
            f.unlink()

        return True

    def assemble_music_only_video(self, music_path):
        """Assemble a music-only video when no narration text is provided.

        Uses a fixed seconds-per-scene based on the music track duration,
        or a default of 5 seconds per scene.
        """
        header("STEP 5: Assembling Music-Only Video (no narration)")

        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            error("ffmpeg not found. Install ffmpeg first.")
            return False

        scenes = self.config["scenes"]
        resolution = self._get_settings("resolution", "1920x1080")
        width, height = resolution.split("x")
        fps = self._get_settings("fps", 24)
        crf = self._get_settings("video_quality", 18)

        # Get music duration for pacing
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(music_path)],
            capture_output=True, text=True
        )
        music_duration = float(result.stdout.strip()) if result.stdout.strip() else 0

        num_scenes = len(scenes)
        if music_duration > 0:
            secs_per_scene = music_duration / num_scenes
        else:
            secs_per_scene = 5.0

        status(f"Music: {music_duration:.0f}s, {secs_per_scene:.2f}s per scene")

        all_images = self._get_scene_images()
        if not all_images:
            error("No scene images found!")
            return False

        encoder_msg = f"GPU ({self.encoder})" if self.use_gpu else f"CPU ({self.encoder})"
        status(f"Found {len(all_images)} scene images, building with transitions...")
        status(f"Using encoder: {encoder_msg}")

        # Build act videos (same as assemble_videos)
        act_size = 15
        act_videos = []
        for act_idx in range(0, len(all_images), act_size):
            act_images = all_images[act_idx:act_idx + act_size]
            act_video = self._build_act_video(
                act_images, secs_per_scene, act_idx // act_size,
                width, height, fps, crf
            )
            if act_video:
                act_videos.append(act_video)

        if not act_videos:
            error("Failed to build any act videos!")
            return False

        success(f"Built {len(act_videos)} act segments with transitions")

        # Concatenate acts
        concat_file = self.output_dir / "ffmpeg_acts_concat.txt"
        with open(concat_file, "w") as f:
            for av in act_videos:
                f.write(f"file '{av}'\n")

        silent_video = self.output_dir / "_full_video_silent.mp4"
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c:v", self.encoder, "-preset", self.preset, self.quality_flag, str(crf),
            "-pix_fmt", "yuv420p",
            str(silent_video)
        ]
        subprocess.run(concat_cmd, capture_output=True)

        if not silent_video.exists():
            error("Failed to concatenate act videos")
            return False

        total_duration = music_duration if music_duration > 0 else secs_per_scene * num_scenes
        title_safe = self.config["title"].lower().replace(" ", "_").replace("-", "")[:30]
        output_path = self.videos_dir / f"{title_safe}_music_only.mp4"

        status("Mixing music-only version...")
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-i", str(music_path),
            "-filter_complex",
            f"[1:a]volume=0.7,afade=t=in:st=0:d=3,"
            f"afade=t=out:st={total_duration - 8:.3f}:d=8[mus]",
            "-map", "0:v", "-map", "[mus]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path)
        ]
        subprocess.run(mux_cmd, capture_output=True)

        if output_path.exists() and output_path.stat().st_size > 100000:
            success(f"Music-only video: {output_path.name} ({output_path.stat().st_size // 1024 // 1024}MB)")
        else:
            error("Failed to create music-only video")

        # Clean up
        for av in act_videos:
            if av.exists():
                av.unlink()
        if silent_video.exists():
            silent_video.unlink()
        for f in self.output_dir.glob("_act_*_concat.txt"):
            f.unlink()

        return True

    # =========================================================================
    # STEP 6: Generate YouTube metadata + thumbnail
    # =========================================================================
    def generate_metadata_and_thumbnail(self, episode_override: int = None):
        """Generate YouTube metadata and thumbnail.

        Args:
            episode_override: Optional episode number to use instead of config value.
                            Used by --fix-episode to correct wrong episode numbers.
        """
        header("STEP 6: Generating YouTube Metadata & Thumbnail")

        from src.youtube_metadata import YouTubeMetadataGenerator, generate_chapters
        from src.config import load_config

        app_config = load_config()
        meta_gen = YouTubeMetadataGenerator(app_config)

        # Extract info from story config
        char_name = self.config["characters"][0]["name"]
        # Use the actual story title for GLM-generated stories
        story_title = self.config.get("title", "")
        theme = self.config["scene"]["name"].replace("_", " ").title()
        style = self.config.get("style", "Studio Ghibli anime style")
        # Use story title as custom_title so it doesn't get overwritten by templates
        custom_title = story_title if story_title else self.config.get("custom_title")
        episode = episode_override if episode_override else self.config.get("episode")
        description_text = self.config.get("description")

        # Generate chapter timestamps based on scene count
        scenes = self.config["scenes"]
        narration_path = self.audio_dir / "narration.mp3"
        secs_per_scene = 4.0
        if narration_path.exists():
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0",
                 str(narration_path)],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                secs_per_scene = float(result.stdout.strip()) / len(scenes)

        # Create chapters from environment transitions (for GLM configs)
        # or fall back to act-based chapters
        environments = self.config.get("environments", {})
        scene_refs = self.config.get("scene_refs", [])

        if environments and scene_refs:
            # Build chapters from environment changes
            segments = []
            current_env = None
            scene_count_in_chapter = 0

            for idx, ref in enumerate(scene_refs):
                env_code = ref.get("environment_code", "")
                if env_code != current_env and env_code in environments:
                    if current_env is not None and scene_count_in_chapter > 0:
                        # Close previous chapter
                        segments[-1]["image_count"] = scene_count_in_chapter

                    # Start new chapter
                    env_data = environments[env_code]
                    env_name = env_data.get("name", env_code).replace("_", " ").title()
                    segments.append({
                        "scene_id": idx + 1,
                        "title": env_name,
                        "image_count": 1,
                    })
                    current_env = env_code
                    scene_count_in_chapter = 1
                else:
                    scene_count_in_chapter += 1

            # Close last chapter
            if segments and scene_count_in_chapter > 0:
                segments[-1]["image_count"] = scene_count_in_chapter
        else:
            # Extract location/setting names from scene prompts for meaningful chapters
            # Look for keywords that indicate scene locations
            location_keywords = ["village", "forest", "meadow", "stream", "cave", "mountain",
                               "valley", "garden", "path", "bridge", "lake", "shore", "hill",
                               "cottage", "tower", "clearing", "grove", "waterfall", "peak"]

            segments = []
            scenes_per_chapter = max(1, len(scenes) // 8)  # ~8 chapters

            for chapter_idx in range(0, len(scenes), scenes_per_chapter):
                chunk = scenes[chapter_idx:chapter_idx + scenes_per_chapter]
                if not chunk:
                    break

                # Try to extract a location from the first scene in this chunk
                first_prompt = chunk[0].lower() if isinstance(chunk[0], str) else str(chunk[0]).lower()
                chapter_title = None

                for keyword in location_keywords:
                    if keyword in first_prompt:
                        # Find the full phrase around the keyword
                        words = first_prompt.split()
                        for i, word in enumerate(words):
                            if keyword in word:
                                # Get 1-2 words before for context (e.g., "ancient forest")
                                start = max(0, i - 1)
                                phrase = " ".join(words[start:i + 1])
                                chapter_title = phrase.title().replace(",", "").strip()[:25]
                                break
                        if chapter_title:
                            break

                # Fallback titles based on story arc position
                if not chapter_title:
                    position = chapter_idx / len(scenes)
                    if position < 0.15:
                        chapter_title = "The Beginning"
                    elif position < 0.3:
                        chapter_title = "Setting Out"
                    elif position < 0.5:
                        chapter_title = "Into the Unknown"
                    elif position < 0.7:
                        chapter_title = "The Heart of It"
                    elif position < 0.85:
                        chapter_title = "Finding the Way"
                    else:
                        chapter_title = "Journey Home"

                segments.append({
                    "scene_id": chapter_idx + 1,
                    "title": chapter_title,
                    "image_count": len(chunk),
                })

        chapters = generate_chapters(segments, duration_per_image=secs_per_scene, transition_time=0)

        # Generate metadata
        metadata = meta_gen.generate_all_metadata(
            character_name=char_name,
            theme=theme,
            style="ghibli",
            summary=description_text,
            chapters=chapters,
            number=episode,
            custom_title=custom_title,
        )

        # Save metadata JSON
        metadata_path = self.videos_dir / "youtube_metadata.json"
        meta_gen.save_metadata(metadata, metadata_path)
        success(f"Metadata saved: {metadata_path.name}")

        # Generate thumbnail (pass episode override for correct badge)
        self._generate_thumbnail(metadata.title, episode_override=episode)

        return metadata

    def _draw_text_with_outline(self, draw, text, position, font, fill, outline_color, outline_width):
        """Draw text with a thick outline for readability at small sizes."""
        x, y = position
        # Draw outline by offsetting in all directions
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        # Draw main text
        draw.text((x, y), text, font=font, fill=fill)

    def _crop_character_face(self, char_image_path, target_height=400):
        """Crop character image to focus on face region (upper 60%)."""
        from PIL import Image
        img = Image.open(char_image_path).convert("RGBA")

        # Take upper 60% of the image (face region)
        face_height = int(img.height * 0.6)
        img = img.crop((0, 0, img.width, face_height))

        # Scale to target height while maintaining aspect ratio
        scale = target_height / img.height
        new_width = int(img.width * scale)
        img = img.resize((new_width, target_height), Image.LANCZOS)

        return img

    def _draw_episode_badge(self, draw, episode_num, position, font):
        """Draw episode badge with rounded background."""
        badge_text = f"Ep {episode_num}"
        x, y = position

        # Get text dimensions
        bbox = draw.textbbox((0, 0), badge_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Badge padding
        pad_x, pad_y = 16, 8
        badge_width = text_width + pad_x * 2
        badge_height = text_height + pad_y * 2

        # Draw badge background (dark with slight transparency effect via border)
        draw.rounded_rectangle(
            [(x, y), (x + badge_width, y + badge_height)],
            radius=8,
            fill=(20, 20, 30, 230),
            outline=(255, 255, 255, 180),
            width=2
        )

        # Draw badge text
        draw.text((x + pad_x, y + pad_y), badge_text, fill=(255, 255, 255), font=font)

        return badge_width

    def _enhance_colors(self, img, saturation_boost=1.3, brightness_boost=1.05):
        """Boost saturation and brightness for more eye-catching thumbnail."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_boost)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_boost)
        return img

    def _generate_thumbnail(self, title, episode_override: int = None):
        """Create a click-optimized thumbnail for YouTube.

        Features:
        - Character face prominently displayed
        - Large bold text with thick outline
        - Episode badge
        - Theme-specific accent colors
        - Enhanced saturation for visibility

        Args:
            title: Video title for the thumbnail text.
            episode_override: Optional episode number override for the badge.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
        except ImportError:
            error("Pillow not installed. Run: pip install Pillow")
            return None

        # Get theme-specific colors
        theme = self.config.get("theme", "starfall")
        style = THUMBNAIL_STYLES.get(theme, THUMBNAIL_STYLES["starfall"])
        accent_hex = style["accent"]
        accent_rgb = tuple(int(accent_hex[i:i+2], 16) for i in (1, 3, 5))

        # Find best scene image for background
        scene_dir = self.output_dir / "scene_001_batch_1"
        if not scene_dir.exists():
            error("No scene images found for thumbnail")
            return None

        images = list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.webp")) + list(scene_dir.glob("*.jpg"))
        if not images:
            error("No scene images found for thumbnail")
            return None

        best_image = max(images, key=lambda p: p.stat().st_size)

        # Load and prepare background
        bg = Image.open(best_image).convert("RGBA")
        bg = bg.resize((1280, 720), Image.LANCZOS)

        # Enhance colors for more pop
        bg_rgb = bg.convert("RGB")
        bg_rgb = self._enhance_colors(bg_rgb, saturation_boost=1.35, brightness_boost=1.08)
        bg = bg_rgb.convert("RGBA")

        # Create gradient overlay at bottom for text readability
        overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Bottom gradient for text
        for y in range(400, 720):
            alpha = int((y - 400) / 320 * 180)
            overlay_draw.rectangle([(0, y), (1280, y + 1)], fill=(0, 0, 0, alpha))

        # Composite background with gradient
        bg = Image.alpha_composite(bg, overlay)

        # Convert to RGB for drawing
        final = bg.convert("RGBA")
        draw = ImageDraw.Draw(final)

        # Load fonts (sized for readability at thumbnail scale)
        title_font_size = 96
        subtitle_font_size = 48
        badge_font_size = 36

        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/impact.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]

        title_font = None
        for fp in font_paths:
            try:
                title_font = ImageFont.truetype(fp, title_font_size)
                subtitle_font = ImageFont.truetype(fp, subtitle_font_size)
                badge_font = ImageFont.truetype(fp, badge_font_size)
                break
            except (OSError, IOError):
                continue

        if title_font is None:
            title_font = ImageFont.load_default()
            subtitle_font = title_font
            badge_font = title_font

        # Draw episode badge (top-left) - use override if provided
        episode_num = episode_override if episode_override else self.config.get("episode")
        if episode_num:
            self._draw_episode_badge(draw, episode_num, (30, 30), badge_font)

        # Extract title parts for thumbnail text
        # Use the story title as the big text
        main_title = self.config.get("title", self.config["scene"]["name"])
        # Clean up any underscores in the title
        main_title = main_title.replace("_", " ").title()
        subtitle = "A Bedtime Story"
        if episode_num:
            subtitle = f"Episode {episode_num}"

        # Word-wrap main title
        max_width = 1180  # Full width minus margins
        words = main_title.split()
        lines = []
        current_line = ""
        for word in words:
            test = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=title_font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Calculate text position (bottom-left area)
        line_height = title_font_size + 12
        total_text_height = len(lines) * line_height + subtitle_font_size + 20
        text_y = 720 - total_text_height - 50

        # Draw main title with thick outline (centered)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            line_width = bbox[2] - bbox[0]
            text_x = (1280 - line_width) // 2
            self._draw_text_with_outline(
                draw, line, (text_x, text_y),
                title_font, fill=(255, 255, 255), outline_color=(0, 0, 0), outline_width=4
            )
            text_y += line_height

        # Draw subtitle with accent color (centered)
        text_y += 10
        bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = bbox[2] - bbox[0]
        subtitle_x = (1280 - subtitle_width) // 2
        self._draw_text_with_outline(
            draw, subtitle, (subtitle_x, text_y),
            subtitle_font, fill=accent_rgb, outline_color=(0, 0, 0), outline_width=3
        )

        # Add subtle bedtime star decoration (top-right)
        star_text = "★ BEDTIME ★"
        bbox = draw.textbbox((0, 0), star_text, font=badge_font)
        star_width = bbox[2] - bbox[0]
        self._draw_text_with_outline(
            draw, star_text, (1280 - star_width - 30, 35),
            badge_font, fill=accent_rgb, outline_color=(0, 0, 0), outline_width=2
        )

        # Save thumbnail
        thumbnails_dir = self.output_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumbnails_dir / "youtube_thumbnail.png"
        final.convert("RGB").save(thumb_path, "PNG", quality=95)
        success(f"Thumbnail saved: {thumb_path.name} ({thumb_path.stat().st_size // 1024}KB)")
        return thumb_path

    # =========================================================================
    # STEP 7: Upload to YouTube (optional)
    # =========================================================================
    def upload_to_youtube(self, schedule_hours=None, immediate=False):
        """Upload video to YouTube with auto-scheduling.

        Args:
            schedule_hours: Deprecated - use auto-scheduler instead.
            immediate: If True, publish immediately (bypass auto-scheduler).
        """
        header("STEP 7: Uploading to YouTube")

        from src.youtube_uploader import YouTubeUploader

        uploader = YouTubeUploader()
        if not uploader.authenticate():
            error("YouTube authentication failed. See setup instructions above.")
            return None

        # Find the best video to upload (prefer narrated)
        title_safe = self.config["title"].lower().replace(" ", "_").replace("-", "")[:30]
        narrated = self.videos_dir / f"{title_safe}_narrated.mp4"
        music_only = self.videos_dir / f"{title_safe}_music_only.mp4"
        video_path = narrated if narrated.exists() else music_only

        if not video_path or not video_path.exists():
            error("No video file found to upload")
            return None

        # Use metadata file if available
        metadata_path = self.videos_dir / "youtube_metadata.json"
        thumbnail_path = self.output_dir / "thumbnails" / "youtube_thumbnail.png"
        thumb = str(thumbnail_path) if thumbnail_path.exists() else None

        # Determine scheduling
        schedule_datetime = None
        if not immediate and schedule_hours:
            schedule_datetime = datetime.now(timezone.utc) + timedelta(hours=schedule_hours)
            status(f"Scheduling {schedule_hours}h from now: {schedule_datetime.strftime('%b %d, %Y at %H:%M UTC')}")

        video_title = self.config["title"]

        if metadata_path.exists():
            video_id = uploader.upload_with_metadata_file(
                video_path=str(video_path),
                metadata_path=str(metadata_path),
                thumbnail_path=thumb,
                schedule_datetime=schedule_datetime,
            )
        else:
            video_id = uploader.upload(
                video_path=str(video_path),
                title=video_title,
                description=self.config.get("description", ""),
                tags=["bedtime stories", "kids", "ghibli", "animated stories"],
                thumbnail_path=thumb,
                schedule_datetime=schedule_datetime,
            )

        if video_id:
            success(f"Uploaded: https://www.youtube.com/watch?v={video_id}")

        return video_id

    # =========================================================================
    # MAIN RUN
    # =========================================================================
    def run(self, skip_whisk=False, upload=False, schedule_hours=None, upload_immediate=False):
        header(f"STORY VIDEO MAKER: {self.config['title']}")
        status(f"Characters: {', '.join(c['name'] for c in self.config['characters'])}")
        status(f"Scene: {self.config['scene']['name']}")
        status(f"Total scenes: {len(self.config['scenes'])}")
        print()

        start = datetime.now()

        if not skip_whisk:
            # Step 1: Generate references
            self.generate_references()
            print()

            # Step 2: Generate scene images
            self.generate_scenes()
            print()
        else:
            status("Skipping Whisk generation (using existing images)")
            print()

        # Step 3: Generate narration
        narration_path = self.generate_narration()
        print()

        # Step 4: Get music
        music_path = self.get_music_path()
        print()

        # Step 5: Assemble videos
        if narration_path:
            self.assemble_videos(narration_path, music_path)
        elif music_path:
            self.assemble_music_only_video(music_path)

        # Step 6: Generate metadata + thumbnail
        print()
        self.generate_metadata_and_thumbnail()

        # Step 7: Upload (optional)
        if upload:
            print()
            self.upload_to_youtube(schedule_hours=schedule_hours, immediate=upload_immediate)

        elapsed = (datetime.now() - start).total_seconds()
        print()
        header("COMPLETE!")
        status(f"Time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
        status(f"Videos saved to: {self.videos_dir}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Story Video Maker - Create animated story videos from a config file"
    )
    parser.add_argument("--config", default="story_config.json",
                        help="Path to story config JSON (default: story_config.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for this episode (default: ./output)")
    parser.add_argument("--skip-whisk", action="store_true",
                        help="Skip image generation (use existing images)")
    parser.add_argument("--video-only", action="store_true",
                        help="Only rebuild videos from existing images + audio")
    parser.add_argument("--upload", action="store_true",
                        help="Upload finished video to YouTube")
    parser.add_argument("--schedule", type=int, default=None,
                        help="Schedule upload N hours from now (implies --upload)")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload existing video (skip all generation)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only regenerate YouTube metadata + thumbnail")
    parser.add_argument("--update-video", type=str, default=None,
                        help="Update metadata/thumbnail for existing YouTube video (provide video ID)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Regenerate specific scenes only. Accepts: single (51), range (51-55), or list (51,53,60)")
    parser.add_argument("--fix-episode", type=int, default=None,
                        help="Override episode number when using --update-video (fixes wrong episode in description)")
    parser.add_argument("--list-videos", action="store_true",
                        help="List recent YouTube uploads with video IDs")
    parser.add_argument("--fix-title", type=str, default=None,
                        help="Override video title when using --update-video (use with --fix-episode)")
    parser.add_argument("--upload-now", action="store_true",
                        help="Upload and publish immediately")

    args = parser.parse_args()
    maker = StoryVideoMaker(config_path=args.config, output_dir=args.output_dir)

    upload = args.upload or args.schedule is not None

    if args.list_videos:
        # List recent YouTube uploads with video IDs
        from src.youtube_uploader import YouTubeUploader
        uploader = YouTubeUploader()
        if uploader.authenticate():
            videos = uploader.list_my_uploads(max_results=20)
            if videos:
                print("\n  Recent YouTube Uploads:")
                print("  " + "-" * 80)
                for v in videos:
                    print(f"  ID: {v['video_id']}")
                    print(f"      Title: {v['title']}")
                    print(f"      Date:  {v['published_at'][:10]}")
                    print()
            else:
                print("  No videos found")
        sys.exit(0)
    elif args.update_video and args.fix_title:
        # Quick fix: directly update YouTube title and episode WITHOUT regenerating
        import re
        from src.youtube_uploader import YouTubeUploader

        uploader = YouTubeUploader()
        if uploader.authenticate():
            # Get current video details
            response = uploader.service.videos().list(
                part="snippet",
                id=args.update_video
            ).execute()

            if not response.get("items"):
                error(f"Video {args.update_video} not found")
                sys.exit(1)

            snippet = response["items"][0]["snippet"]
            old_title = snippet["title"]
            old_desc = snippet["description"]

            # Update title
            snippet["title"] = args.fix_title

            # Fix episode in description if --fix-episode provided
            if args.fix_episode:
                snippet["description"] = re.sub(
                    r'Episode \d+',
                    f'Episode {args.fix_episode}',
                    old_desc
                )

            # Push update
            uploader.service.videos().update(
                part="snippet",
                body={"id": args.update_video, "snippet": snippet}
            ).execute()

            success(f"Updated: '{old_title}' → '{args.fix_title}'")
            if args.fix_episode:
                success(f"Episode number fixed to: {args.fix_episode}")
        sys.exit(0)
    elif args.update_video:
        # Regenerate metadata/thumbnail and update existing YouTube video
        # Pass episode override so thumbnail and metadata use correct episode number
        maker.generate_metadata_and_thumbnail(episode_override=args.fix_episode)
        from src.youtube_uploader import YouTubeUploader
        import json
        import re

        uploader = YouTubeUploader()
        if uploader.authenticate():
            # Load metadata
            metadata_path = maker.videos_dir / "youtube_metadata.json"
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))

                # Fix episode number in description if --fix-episode provided
                # (metadata generator already used the override, but double-check description)
                if args.fix_episode:
                    description = meta.get("description", "")
                    # Replace "Episode N" with correct episode number
                    description = re.sub(
                        r'Episode \d+',
                        f'Episode {args.fix_episode}',
                        description
                    )
                    meta["description"] = description
                    status(f"Fixed episode number to: Episode {args.fix_episode}")

                thumb_path = maker.output_dir / "thumbnails" / "youtube_thumbnail.png"
                uploader.update_video_metadata(
                    video_id=args.update_video,
                    title=meta.get("title"),
                    description=meta.get("description"),
                    tags=meta.get("tags"),
                    thumbnail_path=str(thumb_path) if thumb_path.exists() else None,
                )
            else:
                error("No metadata file found - run --metadata-only first")
    elif args.metadata_only:
        maker.generate_metadata_and_thumbnail()
    elif args.upload_only:
        maker.upload_to_youtube(schedule_hours=args.schedule, immediate=args.upload_now)
    elif args.video_only:
        narration = maker.generate_narration()
        music = maker.get_music_path()
        if narration:
            maker.assemble_videos(narration, music)
        maker.generate_metadata_and_thumbnail()
        if upload:
            maker.upload_to_youtube(schedule_hours=args.schedule, immediate=args.upload_now)
    elif args.scene:
        # Regenerate specific scenes only
        max_scenes = len(maker.config["scenes"])
        force_scenes = parse_scene_spec(args.scene, max_scenes)
        if not force_scenes:
            error(f"No valid scenes in range 1-{max_scenes} from spec: {args.scene}")
            sys.exit(1)
        status(f"Regenerating {len(force_scenes)} scene(s): {', '.join(str(s) for s in force_scenes)}")
        maker.generate_scenes(force_scenes=force_scenes)
        success(f"Scene regeneration complete. Run with --video-only to rebuild videos.")
    else:
        maker.run(skip_whisk=args.skip_whisk, upload=upload, schedule_hours=args.schedule, upload_immediate=args.upload_now)
