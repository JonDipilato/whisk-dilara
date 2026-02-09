"""Audio generation module for TTS narration and background music."""

import asyncio
import math
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import edge_tts
    EDGETTS_AVAILABLE = True
except ImportError:
    EDGETTS_AVAILABLE = False

try:
    from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips
    MOVIEPY_AUDIO_AVAILABLE = True
    MOVIEPY_VERSION = 2
except ImportError:
    try:
        from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_audioclips
        from moviepy.audio.fx import volumex
        MOVIEPY_AUDIO_AVAILABLE = True
        MOVIEPY_VERSION = 1
    except ImportError:
        MOVIEPY_AUDIO_AVAILABLE = False
        MOVIEPY_VERSION = 0

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import AppConfig, load_config
from src.models import AudioVersionType


console = Console()


@dataclass
class NarrationSegment:
    """A segment of narration text with timing info."""
    text: str
    scene_id: Optional[int] = None
    start_time: float = 0.0
    duration: float = 0.0


@dataclass
class AudioOutput:
    """Result of audio generation."""
    path: Path
    duration: float
    version_type: AudioVersionType
    has_narration: bool
    has_music: bool
    music_tracks: List[str] = None


class TTSVoice:
    """Text-to-speech voice configuration."""

    VOICES = {
        "aria": "en-US-AriaNeural",
        "guy": "en-US-GuyNeural",
        "jenny": "en-US-JennyNeural",
        "amber": "en-GB-AmberNeural",
        "sonia": "en-GB-SoniaNeural",
        "mia": "en-AU-MiaNeural",
        "natasha": "en-CA-NatashaNeural",
        "clara": "en-IE-ClaraNeural",
    }

    @classmethod
    def get_voice(cls, name: str) -> str:
        """Get voice ID by name."""
        return cls.VOICES.get(name.lower(), cls.VOICES["aria"])

    @classmethod
    def list_voices(cls) -> Dict[str, str]:
        """List all available voices."""
        return cls.VOICES.copy()


class AudioGenerator:
    """Generate audio for videos using TTS and background music."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the audio generator.

        Args:
            config: Application configuration. Uses default if not provided.
        """
        if not EDGETTS_AVAILABLE:
            console.print("[yellow]Warning: edge-tts not installed. Install with: pip install edge-tts[/yellow]")

        if not MOVIEPY_AUDIO_AVAILABLE:
            console.print("[yellow]Warning: moviepy audio functions not available.[/yellow]")

        self.config = config or load_config()
        self.audio_settings = self.config.audio
        self.music_library = Path(self.audio_settings.music_library)

    def text_to_speech(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
    ) -> Path:
        """Convert text to speech using Edge TTS.

        Args:
            text: Text to convert to speech.
            output_path: Output audio file path.
            voice: Voice ID or name. Uses config default if not provided.
            rate: Speaking rate (e.g., "+0%", "+10%", "-10%").
            volume: Volume adjustment (e.g., "+0%", "+10%", "-10%").

        Returns:
            Path to the generated audio file.
        """
        if not EDGETTS_AVAILABLE:
            raise ImportError("edge-tts is not installed")

        voice = voice or self.audio_settings.tts_voice
        rate = rate or self.audio_settings.voice_rate
        volume = volume or self.audio_settings.voice_volume

        # Check if voice is a name or ID
        if voice in TTSVoice.list_voices().values():
            voice_id = voice
        else:
            voice_id = TTSVoice.get_voice(voice)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Generating TTS: {len(text)} chars[/cyan]")

        async def _generate():
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate=rate,
                volume=volume,
            )

            await communicate.save(str(output_path))

        asyncio.run(_generate())

        # Get duration
        duration = self.get_audio_duration(output_path)

        console.print(f"[green]TTS saved: {output_path} ({duration:.1f}s)[/green]")
        return output_path

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get the duration of an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds.
        """
        if not MOVIEPY_AUDIO_AVAILABLE:
            # Fallback: estimate based on file size
            return 0.0

        try:
            clip = AudioFileClip(str(audio_path))
            duration = clip.duration
            clip.close()
            return duration
        except Exception:
            return 0.0

    def generate_narration_from_script(
        self,
        script: str,
        output_path: Path,
        voice: Optional[str] = None,
        segments: Optional[List[NarrationSegment]] = None,
    ) -> Tuple[Path, List[NarrationSegment]]:
        """Generate narration audio from a script.

        Args:
            script: Full narration script text.
            output_path: Output audio file path.
            voice: Voice to use for TTS.
            segments: Optional list of narration segments for timing.

        Returns:
            Tuple of (audio_path, segments_with_durations)
        """
        if segments:
            # Generate separate audio for each segment
            audio_paths = []
            updated_segments = []

            for i, segment in enumerate(segments):
                segment_path = output_path.parent / f"{output_path.stem}_part_{i:03d}{output_path.suffix}"
                self.text_to_speech(segment.text, segment_path, voice)
                duration = self.get_audio_duration(segment_path)
                segment.duration = duration
                updated_segments.append(segment)
                audio_paths.append(segment_path)

            # Combine all segments
            if MOVIEPY_AUDIO_AVAILABLE:
                clips = [AudioFileClip(str(p)) for p in audio_paths]
                combined = concatenate_audioclips(clips)
                combined.write_audiofile(str(output_path))
                for clip in clips:
                    clip.close()
                combined.close()

                # Clean up temp files
                for p in audio_paths:
                    p.unlink(missing_ok=True)

            return output_path, updated_segments
        else:
            # Generate single audio file
            audio_path = self.text_to_speech(script, output_path, voice)
            duration = self.get_audio_duration(audio_path)

            segment = NarrationSegment(text=script, duration=duration)
            return audio_path, [segment]

    def calculate_duration_from_text(self, text: str, words_per_minute: int = 150) -> float:
        """Estimate audio duration from text length.

        Args:
            text: Text to estimate duration for.
            words_per_minute: Average speaking rate.

        Returns:
            Estimated duration in seconds.
        """
        words = len(text.split())
        minutes = words / words_per_minute
        return minutes * 60

    def find_music_track(
        self,
        category: str = "calm",
        min_duration: Optional[float] = None,
    ) -> Optional[Path]:
        """Find a music track in the local library.

        Args:
            category: Music category (ambient, upbeat, dramatic, calm).
            min_duration: Minimum duration in seconds.

        Returns:
            Path to a music track or None if not found.
        """
        category_dir = self.music_library / category

        if not category_dir.exists():
            # Try root music library
            if self.music_library.exists():
                tracks = list(self.music_library.glob("*.mp3")) + list(self.music_library.glob("*.wav"))
                if tracks:
                    return tracks[0]
            return None

        tracks = list(category_dir.glob("*.mp3")) + list(category_dir.glob("*.wav"))

        if not tracks:
            return None

        if min_duration:
            for track in tracks:
                duration = self.get_audio_duration(track)
                if duration >= min_duration:
                    return track
            # Fallback to longest track
            return tracks[0]

        return tracks[0]

    def create_audio_mix(
        self,
        narration_path: Optional[Path],
        music_path: Path,
        output_path: Path,
        music_volume: float = 0.25,
        fade_in: float = 0.5,
        fade_out: float = 1.0,
        target_duration: Optional[float] = None,
    ) -> Path:
        """Create a mixed audio track with narration and music.

        Args:
            narration_path: Path to narration audio (optional).
            music_path: Path to background music.
            output_path: Output audio file path.
            music_volume: Music volume (0-1).
            fade_in: Fade in duration in seconds.
            fade_out: Fade out duration in seconds.
            target_duration: Target duration (loops music if needed).

        Returns:
            Path to the mixed audio file.
        """
        if not MOVIEPY_AUDIO_AVAILABLE:
            raise ImportError("MoviePy audio functions not available")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Creating audio mix...[/cyan]")

        # Load music
        music_clip = AudioFileClip(str(music_path))

        # Loop music to match target duration or narration length
        if target_duration:
            total_duration = target_duration
        elif narration_path:
            total_duration = self.get_audio_duration(narration_path)
        else:
            total_duration = music_clip.duration

        # Loop music if needed
        if music_clip.duration < total_duration:
            loops_needed = math.ceil(total_duration / music_clip.duration)
            music_clips = [music_clip] * loops_needed
            music_clip = concatenate_audioclips(music_clips)

        # Trim to target duration
        # MoviePy 2.x uses subclipped(), 1.x uses subclip()
        if MOVIEPY_VERSION >= 2:
            music_clip = music_clip.subclipped(0, total_duration)
        else:
            music_clip = music_clip.subclip(0, total_duration)

        # Apply volume
        if music_volume != 1.0:
            if MOVIEPY_VERSION >= 2:
                # MoviePy 2.x uses with_effects()
                from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
                music_clip = music_clip.with_effects([MultiplyVolume(music_volume)])
            else:
                music_clip = music_clip.fx(volumex, music_volume)

        # Apply fade in/out
        effects = []
        if fade_in > 0:
            if MOVIEPY_VERSION >= 2:
                from moviepy.audio.fx.AudioFadeIn import AudioFadeIn
                effects.append(AudioFadeIn(fade_in))
            else:
                music_clip = music_clip.audio_fadein(fade_in)
        if fade_out > 0:
            if MOVIEPY_VERSION >= 2:
                from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
                effects.append(AudioFadeOut(fade_out))
            else:
                music_clip = music_clip.audio_fadeout(fade_out)

        if MOVIEPY_VERSION >= 2 and effects:
            music_clip = music_clip.with_effects(effects)

        # Add narration if provided
        if narration_path and Path(narration_path).exists():
            narration_clip = AudioFileClip(str(narration_path))

            # Mix audio tracks
            if MOVIEPY_VERSION >= 2:
                # MoviePy 2.x uses with_start()
                final_audio = CompositeAudioClip([
                    narration_clip,
                    music_clip.with_start(0),
                ])
            else:
                final_audio = CompositeAudioClip([
                    narration_clip,
                    music_clip.set_start(0),
                ])

            narration_clip.close()
        else:
            final_audio = music_clip

        # Write output
        final_audio.write_audiofile(str(output_path))
        final_audio.close()
        music_clip.close()

        console.print(f"[green]Audio mix saved: {output_path}[/green]")
        return output_path

    def generate_dual_audio_versions(
        self,
        narration_text: str,
        music_path: Path,
        output_dir: Path,
        project_name: str,
        voice: Optional[str] = None,
        target_duration: Optional[float] = None,
    ) -> Tuple[AudioOutput, AudioOutput]:
        """Generate both narrated and ASMR versions of the audio.

        Args:
            narration_text: Full narration script.
            music_path: Path to background music.
            output_dir: Output directory for audio files.
            project_name: Name for the project files.
            voice: Voice to use for narration.
            target_duration: Target duration for both versions.

        Returns:
            Tuple of (narrated_output, asmr_output)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Version A: Narrated with ambient music
        narrated_path = output_dir / f"{project_name}_narrated.mp3"
        narration_audio_path = output_dir / f"{project_name}_narration_temp.mp3"

        if narration_text:
            self.text_to_speech(narration_text, narration_audio_path, voice)

        self.create_audio_mix(
            narration_path=narration_audio_path if narration_text else None,
            music_path=music_path,
            output_path=narrated_path,
            music_volume=self.audio_settings.music_volume,
            fade_in=self.audio_settings.fade_in_duration,
            fade_out=self.audio_settings.fade_out_duration,
            target_duration=target_duration,
        )

        narrated_output = AudioOutput(
            path=narrated_path,
            duration=self.get_audio_duration(narrated_path),
            version_type=AudioVersionType.NARRATED,
            has_narration=bool(narration_text),
            has_music=True,
        )

        # Clean up temp narration
        if narration_text:
            narration_audio_path.unlink(missing_ok=True)

        # Version B: ASMR (music only, higher volume)
        asmr_path = output_dir / f"{project_name}_asmr.mp3"

        self.create_audio_mix(
            narration_path=None,
            music_path=music_path,
            output_path=asmr_path,
            music_volume=self.audio_settings.asmr_music_volume,
            fade_in=self.audio_settings.fade_in_duration,
            fade_out=self.audio_settings.fade_out_duration,
            target_duration=target_duration,
        )

        asmr_output = AudioOutput(
            path=asmr_path,
            duration=self.get_audio_duration(asmr_path),
            version_type=AudioVersionType.ASMR,
            has_narration=False,
            has_music=True,
        )

        console.print(f"[green]Generated both audio versions:[/green]")
        console.print(f"  - Narrated: {narrated_path}")
        console.print(f"  - ASMR: {asmr_path}")

        return narrated_output, asmr_output

    def generate_scene_narrations(
        self,
        scenes: List[Dict],
        output_dir: Path,
        voice: Optional[str] = None,
    ) -> Tuple[List[NarrationSegment], Path]:
        """Generate narration audio for individual scenes.

        Args:
            scenes: List of scene dictionaries with 'scene_id' and 'narration' keys.
            output_dir: Output directory for audio files.
            voice: Voice to use for TTS.

        Returns:
            Tuple of (segments_with_timing, combined_audio_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        segments = []
        audio_paths = []

        for scene in scenes:
            scene_id = scene.get("scene_id")
            narration = scene.get("narration", "")

            if not narration:
                continue

            segment_path = output_dir / f"scene_{scene_id:04d}_narration.mp3"
            self.text_to_speech(narration, segment_path, voice)

            duration = self.get_audio_duration(segment_path)
            segment = NarrationSegment(
                text=narration,
                scene_id=scene_id,
                duration=duration,
            )
            segments.append(segment)
            audio_paths.append(segment_path)

        # Combine all narrations
        combined_path = output_dir / "full_narration.mp3"
        if MOVIEPY_AUDIO_AVAILABLE and audio_paths:
            clips = [AudioFileClip(str(p)) for p in audio_paths]
            combined = concatenate_audioclips(clips)
            combined.write_audiofile(str(combined_path))
            for clip in clips:
                clip.close()
            combined.close()

        return segments, combined_path


def generate_narration(
    text: str,
    output_path: str,
    voice: str = "aria",
    config: Optional[AppConfig] = None,
) -> str:
    """Convenience function to generate TTS audio.

    Args:
        text: Text to convert to speech.
        output_path: Output audio file path.
        voice: Voice name (aria, guy, jenny, etc.).
        config: Application configuration.

    Returns:
        Path to the generated audio file.
    """
    generator = AudioGenerator(config)
    voice_id = TTSVoice.get_voice(voice)
    result = generator.text_to_speech(text, Path(output_path), voice_id)
    return str(result)


def list_available_voices() -> Dict[str, str]:
    """List all available TTS voices.

    Returns:
        Dictionary mapping voice names to IDs.
    """
    return TTSVoice.list_voices()
