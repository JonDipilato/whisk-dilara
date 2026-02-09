"""Data models for Whisk Automation."""

from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ImageFormat(str, Enum):
    """Image format options in Whisk."""
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"


class QueueStatus(str, Enum):
    """Status of a queue item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Scene(BaseModel):
    """A scene to generate images for."""
    scene_id: int = Field(..., description="Unique scene identifier")
    environment_id: str = Field(..., description="Environment image filename (without extension)")
    character_ids: list[str] = Field(default_factory=list, description="List of character image filenames")
    prompt: str = Field(..., description="Text prompt for image generation")
    image_format: ImageFormat = Field(default=ImageFormat.LANDSCAPE)

    @property
    def has_characters(self) -> bool:
        return len(self.character_ids) > 0


class QueueItem(BaseModel):
    """A queued job for processing."""
    id: str = Field(..., description="Unique queue item ID")
    scene: Scene
    batch_number: int = Field(default=1, description="Which batch (1, 2, etc.)")
    status: QueueStatus = Field(default=QueueStatus.PENDING)
    output_folder: str = Field(..., description="Folder name for output")
    images_to_generate: int = Field(default=4)
    retry_count: int = Field(default=0)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def total_images_expected(self) -> int:
        return self.images_to_generate


class QueueState(BaseModel):
    """Persistent queue state."""
    items: list[QueueItem] = Field(default_factory=list)
    current_index: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def add_item(self, item: QueueItem) -> None:
        self.items.append(item)
        self.last_updated = datetime.now()

    def get_pending(self) -> list[QueueItem]:
        return [item for item in self.items if item.status == QueueStatus.PENDING]

    def get_in_progress(self) -> list[QueueItem]:
        return [item for item in self.items if item.status == QueueStatus.IN_PROGRESS]

    def get_completed(self) -> list[QueueItem]:
        return [item for item in self.items if item.status == QueueStatus.COMPLETED]

    def get_failed(self) -> list[QueueItem]:
        return [item for item in self.items if item.status == QueueStatus.FAILED]

    @property
    def progress_percent(self) -> float:
        if not self.items:
            return 0.0
        completed = len(self.get_completed())
        return (completed / len(self.items)) * 100


class GenerationResult(BaseModel):
    """Result of a single generation run."""
    queue_item_id: str
    success: bool
    images_generated: int = 0
    output_paths: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class AudioVersionType(str, Enum):
    """Audio output version types."""
    NARRATED = "narrated"
    ASMR = "asmr"


class VideoProject(BaseModel):
    """A video project containing scenes and metadata."""
    project_id: str = Field(..., description="Unique project identifier")
    title: str = Field(..., description="Video title")
    scenes: list[Scene] = Field(default_factory=list)
    style_preset: str = "ghibli"
    character_name: str = "Grandmother"
    theme: str = "Garden Adventure"
    narration_script: Optional[str] = None
    output_path: Optional[str] = None
    video_paths: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class AudioTrack(BaseModel):
    """Audio track metadata."""
    path: str
    duration: float
    type: str
    volume: float = 1.0


class VideoMetadata(BaseModel):
    """YouTube video metadata."""
    title: str
    description: str
    tags: list[str]
    category: str = "Education"
    privacy_status: str = "public"
    thumbnail_path: Optional[str] = None
    chapter_timestamps: dict[str, str] = Field(default_factory=dict)


class Chapter(BaseModel):
    """A chapter/scene in the video."""
    title: str
    start_time: float
    end_time: float
    narration: Optional[str] = None
    image_count: int = 1


class StylePreset(BaseModel):
    """Visual style preset for video generation."""
    id: str
    name: str
    description: str
    default_music: str = "calm"
    tag_suffix: str = ""


class VideoConfig(BaseModel):
    """Configuration for video generation."""
    duration_per_image: float = 4.0
    transition_duration: float = 0.5
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 24
    codec: str = "libx264"
    crf: int = 23
    audio_codec: str = "aac"
    generate_both_versions: bool = True
