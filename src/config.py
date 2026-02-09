"""Configuration management for Whisk Automation."""

import json
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field


class BrowserConfig(BaseModel):
    """Browser configuration."""
    headless: bool = False
    slow_mo: int = 100
    user_data_dir: Optional[str] = None


class PathsConfig(BaseModel):
    """File paths configuration."""
    environments: str = "./data/environments"
    characters: str = "./data/characters"
    output: str = "./output"
    scenes_file: str = "./data/scenes.csv"
    videos: str = "./output/videos"
    audio: str = "./output/audio"
    music_library: str = "./assets/music"
    thumbnails: str = "./output/thumbnails"


class GenerationConfig(BaseModel):
    """Image generation settings."""
    images_per_prompt: int = 4
    batches_per_scene: int = 2
    image_format: str = "landscape"
    download_timeout: int = 30


class QueueConfig(BaseModel):
    """Queue behavior settings."""
    retry_on_failure: bool = True
    max_retries: int = 3
    delay_between_scenes: int = 5


class VideoOutputConfig(BaseModel):
    """Video output settings."""
    resolution: List[int] = Field(default_factory=lambda: [1920, 1080])
    fps: int = 24
    duration_per_image: float = 4.0
    transition_duration: float = 0.5
    codec: str = "libx264"
    crf: int = 23
    audio_codec: str = "aac"
    output_directory: str = "./output/videos"


class AudioConfig(BaseModel):
    """Audio generation settings."""
    tts_voice: str = "en-US-AriaNeural"
    voice_rate: str = "+0%"
    voice_volume: str = "+0%"
    music_volume: float = 0.25
    asmr_music_volume: float = 0.40
    music_library: str = "./assets/music"
    fade_in_duration: float = 0.5
    fade_out_duration: float = 1.0


class YouTubeConfig(BaseModel):
    """YouTube metadata settings."""
    channel_name: str = "Peaceful Stories"
    channel_handle: str = "peacefulstories"
    upload_schedule: str = "Tuesday & Friday"
    default_tags: List[str] = Field(default_factory=lambda: [
        "bedtime stories",
        "sleep stories",
        "ghibli style stories",
        "calming narration",
        "cozy bedtime",
        "relaxing stories"
    ])
    default_category: str = "Education"
    default_privacy: str = "unlisted"


class ContentStrategyConfig(BaseModel):
    """Content strategy settings."""
    target_video_length_start: int = 300
    target_video_length_goal: int = 600
    seconds_per_scene: int = 4
    scenes_per_video_start: int = 75
    scenes_per_video_goal: int = 150
    upload_frequency: str = "3x_per_week"
    target_ctr: float = 5.0
    target_retention_60s: float = 50.0
    target_completion: float = 50.0


class MusicSourcesConfig(BaseModel):
    """Music source configuration."""
    pixabay_enabled: bool = True
    pixabay_api_key: str = ""
    suno_ai_enabled: bool = False
    suno_ai_api_key: str = ""
    local_library: str = "./assets/music"
    default_categories: List[str] = Field(default_factory=lambda: ["ambient", "calm", "upbeat", "dramatic"])


class AppConfig(BaseModel):
    """Main application configuration."""
    whisk_url: str = "https://labs.google.com/fx/tools/whisk"
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    video: VideoOutputConfig = Field(default_factory=VideoOutputConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    content_strategy: ContentStrategyConfig = Field(default_factory=ContentStrategyConfig)
    music_sources: MusicSourcesConfig = Field(default_factory=MusicSourcesConfig)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.json"

    if not config_path.exists():
        config = AppConfig()
        save_config(config, config_path)
        return config

    with open(config_path, "r") as f:
        data = json.load(f)

    return AppConfig(**data)


def save_config(config: AppConfig, config_path: Optional[Path] = None) -> None:
    """Save configuration to JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.json"

    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)
