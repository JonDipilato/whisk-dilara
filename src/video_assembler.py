"""Video assembly module for creating YouTube-ready videos from generated images."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
import json

try:
    from moviepy import (
        ImageClip,
        AudioFileClip,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_videoclips,
        concatenate_audioclips,
        VideoClip,
    )
    # For MoviePy 2.x, effects are applied differently
    MOVIEPY_AVAILABLE = True
    MOVIEPY_VERSION = 2
except ImportError:
    try:
        # Fallback to moviepy.editor for older versions
        from moviepy.editor import (
            ImageClip,
            AudioFileClip,
            CompositeVideoClip,
            CompositeAudioClip,
            concatenate_videoclips,
            concatenate_audioclips,
        )
        from moviepy.audio.fx import volumex
        from moviepy.video.fx import fadein, fadeout
        MOVIEPY_AVAILABLE = True
        MOVIEPY_VERSION = 1
    except ImportError:
        MOVIEPY_AVAILABLE = False
        MOVIEPY_VERSION = 0

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from src.config import AppConfig, load_config


console = Console()


@dataclass
class VideoSegment:
    """A single video segment composed of one or more images."""
    image_paths: List[Path]
    duration: float
    scene_id: Optional[int] = None


class VideoAssembler:
    """Assemble images into videos with transitions and audio."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the video assembler.

        Args:
            config: Application configuration. Uses default if not provided.
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError(
                "MoviePy is not installed. Install it with: pip install moviepy>=2.0.0"
            )

        self.config = config or load_config()
        self.video_settings = self.config.video

    def create_video_from_images(
        self,
        image_paths: List[Path],
        output_path: Path,
        duration_per_image: Optional[float] = None,
        add_transitions: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Create a video from a list of images.

        Args:
            image_paths: List of image file paths.
            output_path: Output video file path.
            duration_per_image: Duration for each image in seconds.
            add_transitions: Whether to add fade transitions between images.
            progress_callback: Optional callback for progress updates (0-1).

        Returns:
            Path to the created video file.
        """
        if not image_paths:
            raise ValueError("No images provided for video creation")

        duration = duration_per_image or self.video_settings.duration_per_image
        resolution = tuple(self.video_settings.resolution)
        fps = self.video_settings.fps

        console.print(f"[cyan]Creating video from {len(image_paths)} images...[/cyan]")

        # Create image clips
        clips = []
        for i, img_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i / len(image_paths))

            clip = ImageClip(str(img_path), duration=duration)
            # MoviePy 2.x uses resized(), 1.x uses resize()
            if MOVIEPY_VERSION >= 2:
                clip = clip.resized(height=resolution[1])
                if clip.size[0] != resolution[0]:
                    clip = clip.resized(width=resolution[0])
            else:
                clip = clip.resize(resolution)

            if add_transitions and self.video_settings.transition_duration > 0:
                fade_duration = self.video_settings.transition_duration / 2
                # MoviePy 2.x uses with_effects(), 1.x uses fx()
                if MOVIEPY_VERSION >= 2:
                    from moviepy.video.fx.FadeIn import FadeIn
                    from moviepy.video.fx.FadeOut import FadeOut
                    clip = clip.with_effects([FadeIn(fade_duration), FadeOut(fade_duration)])
                else:
                    clip = clip.fx(fadein, fade_duration)
                    clip = clip.fx(fadeout, fade_duration)

            clips.append(clip)

        # Concatenate clips
        final_clip = concatenate_videoclips(clips, method="compose")

        # Write video file (without audio for now)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec=self.video_settings.codec,
            audio_codec=self.video_settings.audio_codec,
            preset="medium",
            ffmpeg_params=["-crf", str(self.video_settings.crf)],
            logger=None,
        )

        final_clip.close()

        if progress_callback:
            progress_callback(1.0)

        console.print(f"[green]Video created: {output_path}[/green]")
        return output_path

    def create_video_from_scenes(
        self,
        scene_folders: List[Path],
        output_path: Path,
        images_per_scene: int = 1,
        duration_per_image: Optional[float] = None,
        add_transitions: bool = True,
    ) -> Tuple[Path, List[VideoSegment]]:
        """Create a video from multiple scene folders.

        Args:
            scene_folders: List of scene folder paths containing images.
            output_path: Output video file path.
            images_per_scene: Number of images to use per scene.
            duration_per_image: Duration for each image in seconds.
            add_transitions: Whether to add fade transitions.

        Returns:
            Tuple of (video_path, segments_used)
        """
        all_images = []
        segments = []

        for folder in sorted(scene_folders):
            folder = Path(folder)
            if not folder.is_dir():
                continue

            # Get images from folder
            image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
            images = sorted([
                f for f in folder.iterdir()
                if f.suffix.lower() in image_extensions
            ])

            # Limit to images_per_scene
            images = images[:images_per_scene]

            if images:
                all_images.extend(images)
                scene_id = folder.name.split("_")[1] if "_" in folder.name else None
                segments.append(VideoSegment(
                    image_paths=images,
                    duration=duration_per_image or self.video_settings.duration_per_image,
                    scene_id=scene_id,
                ))

        if not all_images:
            raise ValueError("No images found in scene folders")

        video_path = self.create_video_from_images(
            all_images,
            output_path,
            duration_per_image,
            add_transitions,
        )

        return video_path, segments

    def create_video_from_output_directory(
        self,
        output_dir: Path,
        output_path: Path,
        scene_filter: Optional[str] = None,
        images_per_scene: int = 1,
        duration_per_image: Optional[float] = None,
    ) -> Tuple[Path, List[VideoSegment]]:
        """Create a video from the output directory containing scene folders.

        Args:
            output_dir: Output directory containing scene_X_batch_Y folders.
            output_path: Output video file path.
            scene_filter: Optional filter pattern (e.g., "scene_1") for specific scenes.
            images_per_scene: Number of images to use per scene.
            duration_per_image: Duration for each image in seconds.

        Returns:
            Tuple of (video_path, segments_used)
        """
        output_dir = Path(output_dir)

        if not output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")

        # Get all scene folders
        scene_folders = sorted([
            f for f in output_dir.iterdir()
            if f.is_dir() and f.name.startswith("scene_")
        ])

        # Apply filter if specified
        if scene_filter:
            if scene_filter.lower() == "all":
                pass  # Use all scenes
            else:
                scene_folders = [
                    f for f in scene_folders
                    if scene_filter in f.name
                ]

        if not scene_folders:
            raise ValueError(f"No scene folders found in {output_dir}")

        console.print(f"[cyan]Found {len(scene_folders)} scene folders[/cyan]")

        return self.create_video_from_scenes(
            scene_folders,
            output_path,
            images_per_scene,
            duration_per_image,
        )

    def add_audio_to_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None,
        audio_volume: float = 1.0,
        fade_in: float = 0.5,
        fade_out: float = 1.0,
    ) -> Path:
        """Add audio track to a video.

        Args:
            video_path: Path to the video file.
            audio_path: Path to the audio file.
            output_path: Output path (defaults to overwriting video).
            audio_volume: Volume multiplier for the audio.
            fade_in: Fade in duration in seconds.
            fade_out: Fade out duration in seconds.

        Returns:
            Path to the video with audio.
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        if output_path is None:
            output_path = video_path

        console.print(f"[cyan]Adding audio to video...[/cyan]")

        # Load clips
        if MOVIEPY_VERSION >= 2:
            from moviepy import VideoFileClip
            video_clip = VideoFileClip(str(video_path))
        else:
            from moviepy.editor import VideoFileClip
            video_clip = VideoFileClip(str(video_path))
        audio_clip = AudioFileClip(str(audio_path))

        # Apply volume
        if audio_volume != 1.0:
            if MOVIEPY_VERSION >= 2:
                from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
                audio_clip = audio_clip.with_effects([MultiplyVolume(audio_volume)])
            else:
                audio_clip = audio_clip.fx(volumex, audio_volume)

        # Fade in/out
        if fade_in > 0 or fade_out > 0:
            effects = []
            if fade_in > 0:
                if MOVIEPY_VERSION >= 2:
                    from moviepy.audio.fx.AudioFadeIn import AudioFadeIn
                    effects.append(AudioFadeIn(fade_in))
                else:
                    audio_clip = audio_clip.audio_fadein(fade_in)
            if fade_out > 0:
                if MOVIEPY_VERSION >= 2:
                    from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
                    effects.append(AudioFadeOut(fade_out))
                else:
                    audio_clip = audio_clip.audio_fadeout(fade_out)
            if MOVIEPY_VERSION >= 2 and effects:
                audio_clip = audio_clip.with_effects(effects)

        # Set audio to video
        video_clip = video_clip.with_audio(audio_clip)

        # Write output
        video_clip.write_videofile(
            str(output_path),
            fps=self.video_settings.fps,
            codec=self.video_settings.codec,
            audio_codec=self.video_settings.audio_codec,
            preset="medium",
            logger=None,
        )

        video_clip.close()
        audio_clip.close()

        console.print(f"[green]Audio added: {output_path}[/green]")
        return output_path

    def add_music_to_video(
        self,
        video_path: Path,
        music_path: Path,
        output_path: Optional[Path] = None,
        music_volume: float = 0.25,
        fade_in: float = 0.5,
        fade_out: float = 1.0,
    ) -> Path:
        """Add background music to a video (preserving existing audio).

        Args:
            video_path: Path to the video file.
            music_path: Path to the music file.
            output_path: Output path (defaults to overwriting video).
            music_volume: Volume for background music (0-1).
            fade_in: Fade in duration in seconds.
            fade_out: Fade out duration in seconds.

        Returns:
            Path to the video with music mixed in.
        """
        video_path = Path(video_path)
        music_path = Path(music_path)

        if output_path is None:
            output_path = video_path

        console.print(f"[cyan]Adding background music to video...[/cyan]")

        # Load video
        if MOVIEPY_VERSION >= 2:
            from moviepy import VideoFileClip
            video_clip = VideoFileClip(str(video_path))
        else:
            from moviepy.editor import VideoFileClip
            video_clip = VideoFileClip(str(video_path))

        # Get video duration for looping music
        video_duration = video_clip.duration

        # Load and loop music
        music_clip = AudioFileClip(str(music_path))
        music_duration = music_clip.duration

        # Loop music to match video duration
        if music_duration < video_duration:
            loops_needed = int(video_duration / music_duration) + 1
            music_clips = [music_clip] * loops_needed
            music_clip = concatenate_audioclips(music_clips)

        # Trim to video duration
        if MOVIEPY_VERSION >= 2:
            music_clip = music_clip.subclipped(0, video_duration)
        else:
            music_clip = music_clip.subclip(0, video_duration)

        # Apply volume
        if music_volume != 1.0:
            if MOVIEPY_VERSION >= 2:
                from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
                music_clip = music_clip.with_effects([MultiplyVolume(music_volume)])
            else:
                music_clip = music_clip.fx(volumex, music_volume)

        # Fade in/out
        effects = []
        # Combine with existing audio if present
        original_audio = video_clip.audio
        if original_audio:
            if MOVIEPY_VERSION >= 2:
                final_audio = CompositeAudioClip([
                    original_audio,
                    music_clip.with_start(0)
                ])
            else:
                final_audio = CompositeAudioClip([
                    original_audio,
                    music_clip.set_start(0)
                ])
        else:
            final_audio = music_clip

        if MOVIEPY_VERSION >= 2:
            video_clip = video_clip.with_audio(final_audio)
        else:
            video_clip = video_clip.set_audio(final_audio)

        # Write output
        video_clip.write_videofile(
            str(output_path),
            fps=self.video_settings.fps,
            codec=self.video_settings.codec,
            audio_codec=self.video_settings.audio_codec,
            preset="medium",
            logger=None,
        )

        video_clip.close()
        music_clip.close()

        console.print(f"[green]Music added: {output_path}[/green]")
        return output_path

    def export_for_youtube(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Export video with YouTube-optimized settings.

        Args:
            video_path: Path to the source video.
            output_path: Output path (defaults to adding _youtube suffix).

        Returns:
            Path to the YouTube-ready video.
        """
        video_path = Path(video_path)

        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_youtube{video_path.suffix}"

        console.print(f"[cyan]Exporting for YouTube...[/cyan]")

        # Re-export with optimal YouTube settings
        if MOVIEPY_VERSION >= 2:
            from moviepy import VideoFileClip
            clip = VideoFileClip(str(video_path))
        else:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(video_path))

        clip.write_videofile(
            str(output_path),
            fps=self.video_settings.fps,
            codec=self.video_settings.codec,
            audio_codec=self.video_settings.audio_codec,
            preset="slow",
            ffmpeg_params=[
                "-crf", str(self.video_settings.crf),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
            logger=None,
        )

        clip.close()

        console.print(f"[green]YouTube-ready video: {output_path}[/green]")
        return output_path

    def get_video_info(self, video_path: Path) -> dict:
        """Get information about a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with video information.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Try using ffmpeg to get video info
        import subprocess
        import json

        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", str(video_path)],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(result.stdout)

            # Get video stream info
            video_stream = None
            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio":
                    audio_stream = stream

            format_info = data.get("format", {})

            # Parse frame rate safely (format: "numerator/denominator")
            fps = 24
            if video_stream and "r_frame_rate" in video_stream:
                fps_str = video_stream["r_frame_rate"]
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = int(num) / int(den)
                else:
                    fps = int(fps_str)

            info = {
                "path": str(video_path),
                "duration": float(format_info.get("duration", 0)),
                "size": [
                    int(video_stream.get("width", 1920)) if video_stream else 1920,
                    int(video_stream.get("height", 1080)) if video_stream else 1080
                ],
                "fps": fps,
                "has_audio": audio_stream is not None,
            }
            return info
        except Exception:
            # Fallback: use moviepy VideoFileClip
            try:
                from moviepy import VideoFileClip
                clip = VideoFileClip(str(video_path))
                info = {
                    "path": str(video_path),
                    "duration": clip.duration,
                    "size": clip.size,
                    "fps": clip.fps,
                    "has_audio": clip.audio is not None,
                }
                clip.close()
                return info
            except:
                # Last resort: return default values
                return {
                    "path": str(video_path),
                    "duration": 12.0,
                    "size": [1920, 1080],
                    "fps": 24,
                    "has_audio": False,
                }


def create_video_from_scenes(
    scene_folders: List[str],
    output_path: str,
    config: Optional[AppConfig] = None,
    **kwargs
) -> Tuple[str, List[VideoSegment]]:
    """Convenience function to create a video from scene folders.

    Args:
        scene_folders: List of scene folder paths.
        output_path: Output video file path.
        config: Application configuration.
        **kwargs: Additional arguments for VideoAssembler.

    Returns:
        Tuple of (video_path, segments_used)
    """
    assembler = VideoAssembler(config)
    folder_paths = [Path(f) for f in scene_folders]
    video_path, segments = assembler.create_video_from_scenes(
        folder_paths,
        Path(output_path),
        **kwargs
    )
    return str(video_path), segments


def create_video_from_output(
    output_dir: str,
    output_path: str,
    config: Optional[AppConfig] = None,
    **kwargs
) -> Tuple[str, List[VideoSegment]]:
    """Convenience function to create a video from the output directory.

    Args:
        output_dir: Output directory containing scene folders.
        output_path: Output video file path.
        config: Application configuration.
        **kwargs: Additional arguments for VideoAssembler.

    Returns:
        Tuple of (video_path, segments_used)
    """
    assembler = VideoAssembler(config)
    video_path, segments = assembler.create_video_from_output_directory(
        Path(output_dir),
        Path(output_path),
        **kwargs
    )
    return str(video_path), segments
