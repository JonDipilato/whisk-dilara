"""Full pipeline module for end-to-end video generation from Excel to YouTube."""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeRemainingColumn, TaskID
)
from rich.panel import Panel
from rich.table import Table

from src.config import AppConfig, load_config
from src.models import Scene, VideoProject, VideoMetadata, AudioVersionType, Chapter
from src.video_assembler import VideoAssembler, VideoSegment
from src.audio_generator import AudioGenerator, NarrationSegment, TTSVoice
try:
    from src.youtube_metadata import (
        YouTubeMetadataGenerator, generate_chapters_from_scenes,
        generate_title, format_timestamp
    )
    HAS_YOUTUBE_METADATA = True
except ImportError:
    HAS_YOUTUBE_METADATA = False
from src.music_library import MusicLibrary, MusicCategory, setup_music_library


console = Console()


@dataclass
class PipelineResult:
    """Result of the video generation pipeline."""
    success: bool
    project_id: str
    video_paths: Dict[str, Path] = field(default_factory=dict)
    audio_paths: Dict[str, Path] = field(default_factory=dict)
    metadata_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    duration: float = 0.0
    error_message: Optional[str] = None
    generation_time_seconds: float = 0.0
    scenes_processed: int = 0
    total_images_used: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "project_id": self.project_id,
            "video_paths": {k: str(v) for k, v in self.video_paths.items()},
            "audio_paths": {k: str(v) for k, v in self.audio_paths.items()},
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "duration": self.duration,
            "error_message": self.error_message,
            "generation_time_seconds": self.generation_time_seconds,
            "scenes_processed": self.scenes_processed,
            "total_images_used": self.total_images_used,
        }


@dataclass
class PipelineConfig:
    """Configuration for the video generation pipeline."""
    character_name: str = "Grandmother"
    theme: str = "Garden Adventure"
    style: str = "ghibli"
    narration_text: Optional[str] = None
    narration_scenes: Optional[List[Dict]] = None
    music_category: str = "calm"
    music_path: Optional[Path] = None
    voice: str = "aria"
    images_per_scene: int = 1
    duration_per_image: float = 4.0
    generate_both_versions: bool = True
    export_youtube_ready: bool = True
    custom_title: Optional[str] = None
    custom_description: Optional[str] = None
    summary: Optional[str] = None
    lesson: Optional[str] = None


class VideoPipeline:
    """End-to-end video generation pipeline."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the video pipeline.

        Args:
            config: Application configuration. Uses default if not provided.
        """
        self.config = config or load_config()
        self.video_assembler = VideoAssembler(self.config)
        self.audio_generator = AudioGenerator(self.config)
        self.metadata_generator = YouTubeMetadataGenerator(self.config) if HAS_YOUTUBE_METADATA else None
        self.music_library = setup_music_library(config=self.config)

        # Setup output directories
        self.video_output_dir = Path(self.config.paths.videos)
        self.audio_output_dir = Path(self.config.paths.audio)
        self.thumbnail_output_dir = Path(self.config.paths.thumbnails)

        for dir_path in [self.video_output_dir, self.audio_output_dir, self.thumbnail_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_project(
        self,
        project_id: str,
        scenes: List[Scene],
        pipeline_config: PipelineConfig,
    ) -> VideoProject:
        """Create a video project from scenes and config.

        Args:
            project_id: Unique project identifier.
            scenes: List of scenes for the video.
            pipeline_config: Pipeline configuration.

        Returns:
            VideoProject object.
        """
        project = VideoProject(
            project_id=project_id,
            title=pipeline_config.custom_title or (generate_title(
                character_name=pipeline_config.character_name,
                theme=pipeline_config.theme,
                style=pipeline_config.style,
            ) if HAS_YOUTUBE_METADATA else f"{pipeline_config.character_name} - {pipeline_config.theme}"),
            scenes=scenes,
            style_preset=pipeline_config.style,
            character_name=pipeline_config.character_name,
            theme=pipeline_config.theme,
            narration_script=pipeline_config.narration_text,
        )

        return project

    def generate_video_from_images(
        self,
        image_folders: List[Path],
        output_path: Path,
        pipeline_config: PipelineConfig,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[Path, List[VideoSegment]]:
        """Generate a video from image folders.

        Args:
            image_folders: List of scene folders containing images.
            output_path: Output video path.
            pipeline_config: Pipeline configuration.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (video_path, segments).
        """
        video_path, segments = self.video_assembler.create_video_from_scenes(
            scene_folders=image_folders,
            output_path=output_path,
            images_per_scene=pipeline_config.images_per_scene,
            duration_per_image=pipeline_config.duration_per_image,
            add_transitions=True,
        )

        return video_path, segments

    def generate_audio_for_video(
        self,
        video_path: Path,
        pipeline_config: PipelineConfig,
        project_id: str,
        target_duration: Optional[float] = None,
    ) -> Dict[AudioVersionType, Path]:
        """Generate audio tracks for the video.

        Args:
            video_path: Path to the silent video.
            pipeline_config: Pipeline configuration.
            project_id: Project ID for naming.
            target_duration: Target duration for audio.

        Returns:
            Dictionary mapping audio version types to file paths.
        """
        # Get music track
        if pipeline_config.music_path:
            music_path = pipeline_config.music_path
        else:
            music_category = MusicCategory(pipeline_config.music_category)
            music_track = self.music_library.get_best_track(
                category=music_category,
                target_duration=target_duration,
            )

            if not music_track:
                console.print(f"[yellow]Warning: No music found in category '{pipeline_config.music_category}'[/yellow]")
                console.print("[yellow]Video will be created without music[/yellow]")
                return {}

            music_path = music_track.path

        narration_text = pipeline_config.narration_text

        if not narration_text and pipeline_config.narration_scenes:
            # Combine scene narrations
            narration_text = " ".join([
                s.get("narration", "") for s in pipeline_config.narration_scenes
            ])

        # Generate audio versions
        audio_paths = {}

        if pipeline_config.generate_both_versions:
            # Generate both narrated and ASMR versions
            narrated, asmr = self.audio_generator.generate_dual_audio_versions(
                narration_text=narration_text,
                music_path=music_path,
                output_dir=self.audio_output_dir,
                project_name=project_id,
                voice=pipeline_config.voice,
                target_duration=target_duration,
            )

            audio_paths[AudioVersionType.NARRATED] = narrated.path
            audio_paths[AudioVersionType.ASMR] = asmr.path
        else:
            # Generate only narrated version
            if narration_text:
                narration_path = self.audio_output_dir / f"{project_id}_narration.mp3"
                self.audio_generator.text_to_speech(
                    text=narration_text,
                    output_path=narration_path,
                    voice=TTSVoice.get_voice(pipeline_config.voice),
                )

            mix_path = self.audio_output_dir / f"{project_id}_audio.mp3"
            self.audio_generator.create_audio_mix(
                narration_path=narration_path if narration_text else None,
                music_path=music_path,
                output_path=mix_path,
                music_volume=self.config.audio.music_volume,
                target_duration=target_duration,
            )

            audio_paths[AudioVersionType.NARRATED] = mix_path

        return audio_paths

    def combine_video_and_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> Path:
        """Combine video and audio into final video file.

        Args:
            video_path: Path to silent video.
            audio_path: Path to audio file.
            output_path: Output video path.

        Returns:
            Path to combined video.
        """
        try:
            result = self.video_assembler.add_audio_to_video(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                audio_volume=1.0,
            )
            return result
        except Exception as e:
            console.print(f"[yellow]Warning: Could not combine audio: {e}[/yellow]")
            # Return original video path
            return video_path

    def generate_metadata(
        self,
        pipeline_config: PipelineConfig,
        project_id: str,
        segments: Optional[List[VideoSegment]] = None,
        video_duration: Optional[float] = None,
    ) -> VideoMetadata:
        """Generate YouTube metadata for the video.

        Args:
            pipeline_config: Pipeline configuration.
            project_id: Project ID.
            segments: Video segments for chapter generation.
            video_duration: Video duration in seconds.

        Returns:
            VideoMetadata object.
        """
        if not HAS_YOUTUBE_METADATA or not self.metadata_generator:
            console.print("[yellow]Skipping metadata generation (youtube_metadata not available)[/yellow]")
            return None

        # Generate chapters from segments
        chapters = None
        if segments:
            segment_data = [
                {
                    "scene_id": s.scene_id,
                    "image_count": len(s.image_paths),
                    "duration": s.duration,
                }
                for s in segments
            ]
            chapters = generate_chapters_from_scenes(
                scenes=segment_data,
                scenes_config={"duration_per_image": pipeline_config.duration_per_image},
            )

        # Generate metadata
        metadata = self.metadata_generator.generate_all_metadata(
            character_name=pipeline_config.character_name,
            theme=pipeline_config.theme,
            style=pipeline_config.style,
            summary=pipeline_config.summary,
            lesson=pipeline_config.lesson,
            chapters=chapters,
            custom_title=pipeline_config.custom_title,
        )

        # Save metadata
        metadata_path = self.video_output_dir / f"{project_id}_metadata.json"
        self.metadata_generator.save_metadata(metadata, metadata_path)

        return metadata

    def generate_thumbnail(
        self,
        project_id: str,
        image_folders: List[Path],
    ) -> Optional[Path]:
        """Generate a thumbnail from the first scene image.

        Args:
            project_id: Project ID.
            image_folders: List of scene folders.

        Returns:
            Path to generated thumbnail or None.
        """
        if not image_folders:
            return None

        # Get first image from first folder
        first_folder = sorted(image_folders)[0]
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = list(first_folder.glob("*.*"))

        for img in images:
            if img.suffix.lower() in image_extensions:
                # Copy image as thumbnail
                thumbnail_path = self.thumbnail_output_dir / f"{project_id}_thumbnail{img.suffix}"
                shutil.copy2(img, thumbnail_path)

                console.print(f"[green]Thumbnail created: {thumbnail_path}[/green]")
                return thumbnail_path

        return None

    def run_full_pipeline(
        self,
        image_folders: List[Path],
        pipeline_config: PipelineConfig,
        project_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """Run the complete video generation pipeline.

        Args:
            image_folders: List of scene folders containing images.
            pipeline_config: Pipeline configuration.
            project_id: Optional project ID (auto-generated if not provided).
            progress_callback: Optional callback(status, progress).

        Returns:
            PipelineResult with all generated files.
        """
        start_time = datetime.now()

        if project_id is None:
            project_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        console.print(Panel.fit(
            f"[bold cyan]Video Generation Pipeline[/bold cyan]\n"
            f"Project: {project_id}\n"
            f"Scenes: {len(image_folders)}\n"
            f"Style: {pipeline_config.style}",
            title="Pipeline",
        ))

        try:
            # Phase 1: Generate video from images
            if progress_callback:
                progress_callback("Generating video from images", 0.1)

            console.print("\n[bold cyan]Phase 1: Generating video from images[/bold cyan]")
            video_path, segments = self.generate_video_from_images(
                image_folders=image_folders,
                output_path=self.video_output_dir / f"{project_id}_silent.mp4",
                pipeline_config=pipeline_config,
            )

            # Get video duration
            video_info = self.video_assembler.get_video_info(video_path)
            video_duration = video_info["duration"]
            total_images = sum(len(s.image_paths) for s in segments)

            console.print(f"[green]Video created: {video_duration:.1f}s, {total_images} images[/green]")

            # Phase 2: Generate audio
            if progress_callback:
                progress_callback("Generating audio", 0.4)

            console.print("\n[bold cyan]Phase 2: Generating audio[/bold cyan]")
            audio_paths = self.generate_audio_for_video(
                video_path=video_path,
                pipeline_config=pipeline_config,
                project_id=project_id,
                target_duration=video_duration,
            )

            # Phase 3: Combine video and audio
            if progress_callback:
                progress_callback("Combining video and audio", 0.6)

            console.print("\n[bold cyan]Phase 3: Creating final videos[/bold cyan]")
            final_video_paths = {}

            for version_type, audio_path in audio_paths.items():
                version_name = "narrated" if version_type == AudioVersionType.NARRATED else "asmr"
                output_path = self.video_output_dir / f"{project_id}_{version_name}.mp4"

                combined_path = self.combine_video_and_audio(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=output_path,
                )

                final_video_paths[version_name] = combined_path
                console.print(f"[green]Video created: {combined_path.name}[/green]")

            # Export YouTube-ready version
            if pipeline_config.export_youtube_ready and final_video_paths:
                main_video = list(final_video_paths.values())[0]
                youtube_path = self.video_output_dir / f"{project_id}_youtube.mp4"
                self.video_assembler.export_for_youtube(main_video, youtube_path)
                final_video_paths["youtube"] = youtube_path

            # Phase 4: Generate metadata
            if progress_callback:
                progress_callback("Generating metadata", 0.8)

            console.print("\n[bold cyan]Phase 4: Generating metadata[/bold cyan]")
            metadata = self.generate_metadata(
                pipeline_config=pipeline_config,
                project_id=project_id,
                segments=segments,
                video_duration=video_duration,
            )

            self.metadata_generator.print_metadata_preview(metadata)

            # Phase 5: Generate thumbnail
            if progress_callback:
                progress_callback("Generating thumbnail", 0.9)

            thumbnail_path = self.generate_thumbnail(
                project_id=project_id,
                image_folders=image_folders,
            )

            # Calculate generation time
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            if progress_callback:
                progress_callback("Complete", 1.0)

            # Create result
            result = PipelineResult(
                success=True,
                project_id=project_id,
                video_paths=final_video_paths,
                audio_paths=audio_paths,
                metadata_path=self.video_output_dir / f"{project_id}_metadata.json",
                thumbnail_path=thumbnail_path,
                duration=video_duration,
                generation_time_seconds=generation_time,
                scenes_processed=len(image_folders),
                total_images_used=total_images,
            )

            # Print summary
            console.print("\n[bold green]Pipeline Complete![/bold green]")
            self._print_result_summary(result)

            return result

        except Exception as e:
            console.print(f"\n[red]Pipeline error: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())

            return PipelineResult(
                success=False,
                project_id=project_id,
                error_message=str(e),
                generation_time_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def _print_result_summary(self, result: PipelineResult) -> None:
        """Print a summary of the pipeline result.

        Args:
            result: PipelineResult to summarize.
        """
        table = Table(title="Pipeline Result Summary")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Project ID", result.project_id)
        table.add_row("Duration", f"{result.duration:.1f} seconds")
        table.add_row("Scenes Processed", str(result.scenes_processed))
        table.add_row("Images Used", str(result.total_images_used))
        table.add_row("Generation Time", f"{result.generation_time_seconds / 60:.1f} minutes")

        console.print(table)

        console.print("\n[bold]Output Files:[/bold]")
        for name, path in result.video_paths.items():
            console.print(f"  [green]Video ({name}):[/green] {path}")

        for name, path in result.audio_paths.items():
            console.print(f"  [cyan]Audio ({name}):[/cyan] {path}")

        if result.thumbnail_path:
            console.print(f"  [yellow]Thumbnail:[/yellow] {result.thumbnail_path}")

        if result.metadata_path:
            console.print(f"  [blue]Metadata:[/blue] {result.metadata_path}")


def run_pipeline_from_output(
    output_dir: Path,
    character_name: str = "Grandmother",
    theme: str = "Garden Adventure",
    style: str = "ghibli",
    project_id: Optional[str] = None,
    config: Optional[AppConfig] = None,
    **kwargs
) -> PipelineResult:
    """Convenience function to run pipeline from output directory.

    Args:
        output_dir: Directory containing scene folders.
        character_name: Main character name.
        theme: Story theme.
        style: Visual style.
        project_id: Optional project ID.
        config: Application configuration.
        **kwargs: Additional PipelineConfig arguments.

    Returns:
        PipelineResult with generated files.
    """
    output_dir = Path(output_dir)

    # Get all scene folders
    scene_folders = sorted([
        f for f in output_dir.iterdir()
        if f.is_dir() and f.name.startswith("scene_")
    ])

    if not scene_folders:
        raise ValueError(f"No scene folders found in {output_dir}")

    # Create pipeline config
    pipeline_config = PipelineConfig(
        character_name=character_name,
        theme=theme,
        style=style,
        **kwargs
    )

    # Run pipeline
    pipeline = VideoPipeline(config)
    return pipeline.run_full_pipeline(
        image_folders=scene_folders,
        pipeline_config=pipeline_config,
        project_id=project_id,
    )
