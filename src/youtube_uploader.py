"""YouTube upload automation via YouTube Data API v3.

SETUP:
1. Go to Google Cloud Console (console.cloud.google.com)
2. Create a project and enable "YouTube Data API v3"
3. Create OAuth2 credentials (Desktop app type)
4. Download client_secret JSON and save as youtube_client_secret.json in project root
5. Set redirect URIs to: http://localhost:8080/ and http://localhost:8090/

USAGE:
    from src.youtube_uploader import YouTubeUploader

    uploader = YouTubeUploader()
    uploader.upload(
        video_path="output/videos/starfall_valley_narrated.mp4",
        title="Starfall Valley - Bedtime Story",
        description="A peaceful bedtime story...",
        tags=["bedtime stories", "kids", "ghibli"],
        thumbnail_path="output/thumbnails/youtube_thumbnail.png",
        schedule_hours=24,  # publish 24 hours from now
    )

DEPENDENCIES:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import json
import time
import httplib2
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class _Console:
        def print(self, msg): print(msg)
    console = _Console()


SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
           "https://www.googleapis.com/auth/youtube"]

VALID_CATEGORIES = {
    "Film & Animation": "1",
    "Autos & Vehicles": "2",
    "Music": "10",
    "Pets & Animals": "15",
    "Sports": "17",
    "Travel & Events": "19",
    "Gaming": "20",
    "People & Blogs": "22",
    "Comedy": "23",
    "Entertainment": "24",
    "News & Politics": "25",
    "Howto & Style": "26",
    "Education": "27",
    "Science & Technology": "28",
    "Nonprofits & Activism": "29",
}

SCHEDULE_PRESETS = {
    "immediate": 0,
    "1hour": 1,
    "6hours": 6,
    "12hours": 12,
    "24hours": 24,
    "48hours": 48,
    "1week": 168,
}


class YouTubeUploader:
    """Upload videos to YouTube with metadata and scheduling."""

    def __init__(
        self,
        client_secret_path: Optional[str] = None,
        token_path: Optional[str] = None,
    ):
        """Initialize the uploader.

        Args:
            client_secret_path: Path to OAuth2 client secret JSON.
            token_path: Path to store/load OAuth2 token.
        """
        if not HAS_GOOGLE_API:
            console.print("[red]YouTube API dependencies missing. Install with:[/red]")
            console.print("[cyan]pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib[/cyan]")
            self.service = None
            return

        root = Path(__file__).parent.parent
        self.client_secret_path = Path(client_secret_path or root / "youtube_client_secret.json")
        self.token_path = Path(token_path or root / ".youtube_token.json")
        self.service = None

    def authenticate(self) -> bool:
        """Authenticate with YouTube API via OAuth2.

        Opens browser for consent on first run, then uses cached token.

        Returns:
            True if authentication succeeded.
        """
        if not HAS_GOOGLE_API:
            return False

        if not self.client_secret_path.exists():
            console.print(f"[red]Client secret not found: {self.client_secret_path}[/red]")
            console.print("[cyan]Download from Google Cloud Console → APIs & Services → Credentials[/cyan]")
            console.print("[cyan]Save as: youtube_client_secret.json in project root[/cyan]")
            return False

        creds = None

        # Load existing token
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            except Exception:
                pass

        # Refresh or get new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None

            if not creds:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.client_secret_path), SCOPES
                )
                creds = flow.run_local_server(port=8080, open_browser=True)

            # Save token for next time
            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        self.service = build("youtube", "v3", credentials=creds)
        console.print("[green]YouTube API authenticated[/green]")
        return True

    def upload(
        self,
        video_path: str,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        category: str = "Education",
        privacy: str = "private",
        thumbnail_path: Optional[str] = None,
        schedule_hours: Optional[int] = None,
        schedule_datetime: Optional[datetime] = None,
        made_for_kids: bool = False,  # Default False - limits features when True
    ) -> Optional[str]:
        """Upload a video to YouTube.

        Args:
            video_path: Path to the video file.
            title: Video title (max 100 chars).
            description: Video description (max 5000 chars).
            tags: List of tags.
            category: YouTube category name.
            privacy: Privacy status (private, unlisted, public).
            thumbnail_path: Path to custom thumbnail image.
            schedule_hours: Hours from now to publish (sets privacy to private until then).
            schedule_datetime: Specific UTC datetime to publish (overrides schedule_hours).
            made_for_kids: Whether the video is made for kids.

        Returns:
            Video ID if upload succeeded, None otherwise.
        """
        if not self.service:
            if not self.authenticate():
                return None

        video_path = Path(video_path)
        if not video_path.exists():
            console.print(f"[red]Video file not found: {video_path}[/red]")
            return None

        # Determine publish time
        publish_at = None
        if schedule_datetime:
            privacy = "private"
            publish_at = schedule_datetime.strftime("%Y-%m-%dT%H:%M:%S.0Z")
        elif schedule_hours and schedule_hours > 0:
            privacy = "private"
            publish_at = (datetime.now(timezone.utc) + timedelta(hours=schedule_hours)).isoformat()

        # Map category name to ID
        category_id = VALID_CATEGORIES.get(category, "27")  # Default: Education

        # Build request body
        body = {
            "snippet": {
                "title": title[:100],
                "description": (description or "")[:5000],
                "tags": (tags or [])[:500],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": made_for_kids,
            },
        }

        if publish_at:
            body["status"]["publishAt"] = publish_at

        # Upload video
        console.print(f"[cyan]Uploading: {video_path.name} ({video_path.stat().st_size // 1024 // 1024}MB)...[/cyan]")

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=10 * 1024 * 1024,  # 10MB chunks
        )

        try:
            request = self.service.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    console.print(f"  [dim]Upload progress: {progress}%[/dim]")

            video_id = response["id"]
            console.print(f"[green]Upload complete! Video ID: {video_id}[/green]")
            console.print(f"[green]URL: https://www.youtube.com/watch?v={video_id}[/green]")

            if publish_at:
                console.print(f"[cyan]Scheduled to publish: {publish_at}[/cyan]")

            # Set thumbnail if provided
            if thumbnail_path:
                self.set_thumbnail(video_id, thumbnail_path)

            return video_id

        except HttpError as e:
            console.print(f"[red]Upload failed: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Upload error: {e}[/red]")
            return None

    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Set a custom thumbnail for a video.

        Args:
            video_id: YouTube video ID.
            thumbnail_path: Path to thumbnail image (JPEG/PNG, max 2MB).

        Returns:
            True if thumbnail was set.
        """
        thumbnail_path = Path(thumbnail_path)
        if not thumbnail_path.exists():
            console.print(f"[yellow]Thumbnail not found: {thumbnail_path}[/yellow]")
            return False

        try:
            media = MediaFileUpload(str(thumbnail_path), mimetype="image/png")
            self.service.thumbnails().set(
                videoId=video_id,
                media_body=media,
            ).execute()
            console.print(f"[green]Thumbnail set for video {video_id}[/green]")
            return True
        except HttpError as e:
            console.print(f"[yellow]Thumbnail upload failed (may require channel verification): {e}[/yellow]")
            return False

    def update_video_metadata(
        self,
        video_id: str,
        title: str = None,
        description: str = None,
        tags: List[str] = None,
        thumbnail_path: str = None,
    ) -> bool:
        """Update metadata and/or thumbnail for an existing video.

        Args:
            video_id: YouTube video ID to update.
            title: New title (optional).
            description: New description (optional).
            tags: New tags list (optional).
            thumbnail_path: Path to new thumbnail (optional).

        Returns:
            True if update was successful.
        """
        try:
            # Get current video details
            response = self.service.videos().list(
                part="snippet",
                id=video_id
            ).execute()

            if not response.get("items"):
                console.print(f"[red]Video {video_id} not found[/red]")
                return False

            snippet = response["items"][0]["snippet"]

            # Update only provided fields
            if title:
                snippet["title"] = title
            if description:
                snippet["description"] = description
            if tags:
                snippet["tags"] = tags

            # Push metadata update
            self.service.videos().update(
                part="snippet",
                body={
                    "id": video_id,
                    "snippet": snippet
                }
            ).execute()
            console.print(f"[green]Metadata updated for video {video_id}[/green]")

            # Update thumbnail if provided
            if thumbnail_path:
                self.set_thumbnail(video_id, thumbnail_path)

            return True

        except HttpError as e:
            console.print(f"[red]Update failed: {e}[/red]")
            return False

    def list_my_uploads(self, max_results: int = 20) -> List[dict]:
        """List recent uploads from the authenticated channel.

        Args:
            max_results: Maximum number of videos to return.

        Returns:
            List of video info dicts with id, title, publishedAt, description snippet.
        """
        if not self.service:
            if not self.authenticate():
                return []

        try:
            # First get the channel's upload playlist ID
            channels_response = self.service.channels().list(
                part="contentDetails",
                mine=True
            ).execute()

            if not channels_response.get("items"):
                console.print("[red]No channel found[/red]")
                return []

            uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            # Get videos from the uploads playlist
            videos = []
            playlist_response = self.service.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=max_results
            ).execute()

            for item in playlist_response.get("items", []):
                snippet = item["snippet"]
                videos.append({
                    "video_id": snippet["resourceId"]["videoId"],
                    "title": snippet["title"],
                    "published_at": snippet["publishedAt"],
                    "description": snippet["description"][:100] + "..." if len(snippet["description"]) > 100 else snippet["description"]
                })

            return videos

        except HttpError as e:
            console.print(f"[red]Failed to list uploads: {e}[/red]")
            return []

    def get_upload_schedule(
        self,
        frequency: str = "daily",
        start_hour: int = 18,
        timezone_offset: int = -5,
    ) -> List[datetime]:
        """Generate a schedule of publish times.

        Args:
            frequency: Upload frequency ('daily', '3x_week', '2x_week', 'weekly').
            start_hour: Hour of day to publish (0-23).
            timezone_offset: UTC offset hours.

        Returns:
            List of publish datetime objects for next 7 days.
        """
        now = datetime.now(timezone.utc)
        schedule = []

        if frequency == "daily":
            days = range(1, 8)
        elif frequency == "3x_week":
            days = [1, 3, 5]  # Every other day starting tomorrow
        elif frequency == "2x_week":
            days = [2, 5]  # Tue + Fri style
        else:  # weekly
            days = [7]

        for day_offset in days:
            publish_time = now.replace(
                hour=start_hour - timezone_offset,
                minute=0,
                second=0,
                microsecond=0,
            ) + timedelta(days=day_offset)
            schedule.append(publish_time)

        return schedule

    def reschedule_video(self, video_id: str, publish_at: datetime) -> bool:
        """Reschedule a video's publish time.

        Args:
            video_id: YouTube video ID
            publish_at: Datetime in UTC for scheduled publish

        Returns:
            True if successful
        """
        if not self.service:
            if not self.authenticate():
                return False

        try:
            # Format datetime for YouTube API (ISO 8601)
            publish_at_str = publish_at.strftime("%Y-%m-%dT%H:%M:%S.0Z")

            self.service.videos().update(
                part="status",
                body={
                    "id": video_id,
                    "status": {
                        "privacyStatus": "private",
                        "publishAt": publish_at_str
                    }
                }
            ).execute()

            console.print(f"[green]Scheduled {video_id} for {publish_at_str}[/green]")
            return True

        except HttpError as e:
            console.print(f"[red]Reschedule failed: {e}[/red]")
            return False

    def upload_with_metadata_file(
        self,
        video_path: str,
        metadata_path: str,
        thumbnail_path: Optional[str] = None,
        schedule_hours: Optional[int] = None,
        schedule_datetime: Optional[datetime] = None,
    ) -> Optional[str]:
        """Upload a video using a saved metadata JSON file.

        Args:
            video_path: Path to video file.
            metadata_path: Path to youtube_metadata.json.
            thumbnail_path: Optional thumbnail override.
            schedule_hours: Hours from now to publish.
            schedule_datetime: Specific UTC datetime to publish (overrides schedule_hours).

        Returns:
            Video ID if successful.
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            console.print(f"[red]Metadata file not found: {metadata_path}[/red]")
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return self.upload(
            video_path=video_path,
            title=meta.get("title", "Untitled"),
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            category=meta.get("category", "Education"),
            privacy=meta.get("privacy_status", "private"),
            thumbnail_path=thumbnail_path,
            schedule_hours=schedule_hours,
            schedule_datetime=schedule_datetime,
        )
