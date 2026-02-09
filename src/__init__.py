"""Whisk Automation - Automated image generation using Google Whisk."""

__version__ = "1.0.0"

# Core modules
from src.config import AppConfig, load_config, save_config
from src.models import (
    Scene, ImageFormat, QueueStatus, QueueItem, QueueState, GenerationResult,
    AudioVersionType, VideoProject, AudioTrack, VideoMetadata, Chapter,
    StylePreset, VideoConfig,
)
from src.queue_manager import QueueManager
from src.whisk_controller import WhiskController, test_whisk_connection

# Video pipeline modules
from src.video_assembler import VideoAssembler, VideoSegment, create_video_from_output
from src.audio_generator import AudioGenerator, TTSVoice, NarrationSegment, AudioOutput
from src.pipeline import VideoPipeline, PipelineConfig, PipelineResult, run_pipeline_from_output
from src.music_library import MusicLibrary, MusicTrack, MusicCategory, setup_music_library

__all__ = [
    # Core
    "AppConfig", "load_config", "save_config",
    "Scene", "ImageFormat", "QueueStatus", "QueueItem", "QueueState", "GenerationResult",
    "QueueManager", "WhiskController", "test_whisk_connection",
    # Video
    "VideoAssembler", "VideoSegment", "create_video_from_output",
    "AudioGenerator", "TTSVoice", "NarrationSegment", "AudioOutput", "AudioVersionType",
    "VideoPipeline", "PipelineConfig", "PipelineResult", "run_pipeline_from_output",
    "MusicLibrary", "MusicTrack", "MusicCategory", "setup_music_library",
    "VideoProject", "AudioTrack", "VideoMetadata", "Chapter", "StylePreset", "VideoConfig",
]
