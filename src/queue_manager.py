"""Queue manager for batch processing scenes."""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from uuid import uuid4
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import AppConfig
from .models import Scene, QueueItem, QueueState, QueueStatus, GenerationResult
from .whisk_controller import WhiskController

console = Console()


class QueueManager:
    """Manages the processing queue for Whisk automation."""

    QUEUE_FILE = "queue_state.json"

    def __init__(self, config: AppConfig):
        self.config = config
        self.state: QueueState = QueueState()
        self.queue_path = Path(config.paths.output) / self.QUEUE_FILE

    def load_state(self) -> None:
        """Load queue state from disk."""
        if self.queue_path.exists():
            with open(self.queue_path, "r") as f:
                data = json.load(f)
                self.state = QueueState(**data)
            console.print(f"[cyan]Loaded queue with {len(self.state.items)} items[/cyan]")
        else:
            self.state = QueueState()

    def save_state(self) -> None:
        """Save queue state to disk."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.queue_path, "w") as f:
            json.dump(self.state.model_dump(mode="json"), f, indent=2, default=str)

    def add_scene(self, scene: Scene, batches: int = None) -> list[QueueItem]:
        """Add a scene to the queue (creates multiple queue items for batches)."""
        if batches is None:
            batches = self.config.generation.batches_per_scene

        items = []
        for batch in range(1, batches + 1):
            item = QueueItem(
                id=str(uuid4())[:8],
                scene=scene,
                batch_number=batch,
                output_folder=f"scene_{scene.scene_id:03d}_batch_{batch}",
                images_to_generate=self.config.generation.images_per_prompt,
            )
            self.state.add_item(item)
            items.append(item)

        self.save_state()
        console.print(f"[green]Added scene {scene.scene_id} to queue ({batches} batches)[/green]")
        return items

    def add_scenes_from_csv(self, csv_path: Path) -> int:
        """Load scenes from CSV and add to queue."""
        import pandas as pd

        if not csv_path.exists():
            console.print(f"[red]CSV file not found: {csv_path}[/red]")
            return 0

        df = pd.read_csv(csv_path)
        count = 0

        for _, row in df.iterrows():
            # Parse character IDs (comma-separated)
            char_ids = []
            if pd.notna(row.get("character_ids", "")) and row["character_ids"]:
                char_ids = [c.strip() for c in str(row["character_ids"]).split(",") if c.strip()]

            scene = Scene(
                scene_id=int(row["scene_id"]),
                environment_id=str(row["environment_id"]),
                character_ids=char_ids,
                prompt=str(row["prompt"]),
            )
            self.add_scene(scene)
            count += 1

        console.print(f"[green]Added {count} scenes from {csv_path.name}[/green]")
        return count

    def get_next_pending(self) -> Optional[QueueItem]:
        """Get the next pending item from the queue."""
        for item in self.state.items:
            if item.status == QueueStatus.PENDING:
                return item
        return None

    def update_item_status(
        self,
        item_id: str,
        status: QueueStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of a queue item."""
        for item in self.state.items:
            if item.id == item_id:
                item.status = status
                if status == QueueStatus.IN_PROGRESS:
                    item.started_at = datetime.now()
                elif status in [QueueStatus.COMPLETED, QueueStatus.FAILED]:
                    item.completed_at = datetime.now()
                if error_message:
                    item.error_message = error_message
                break
        self.save_state()

    def mark_completed(self, item_id: str) -> None:
        """Mark an item as completed."""
        self.update_item_status(item_id, QueueStatus.COMPLETED)

    def mark_failed(self, item_id: str, error: str) -> None:
        """Mark an item as failed."""
        for item in self.state.items:
            if item.id == item_id:
                item.retry_count += 1
                if item.retry_count < self.config.queue.max_retries and self.config.queue.retry_on_failure:
                    # Reset to pending for retry
                    self.update_item_status(item_id, QueueStatus.PENDING, error)
                    console.print(f"[yellow]Item {item_id} will retry ({item.retry_count}/{self.config.queue.max_retries})[/yellow]")
                else:
                    self.update_item_status(item_id, QueueStatus.FAILED, error)
                break
        self.save_state()

    def clear_queue(self) -> None:
        """Clear all items from the queue."""
        self.state = QueueState()
        self.save_state()
        console.print("[yellow]Queue cleared[/yellow]")

    def clear_completed(self) -> None:
        """Remove completed items from the queue."""
        self.state.items = [
            item for item in self.state.items
            if item.status != QueueStatus.COMPLETED
        ]
        self.save_state()
        console.print("[yellow]Completed items removed[/yellow]")

    def reset_failed(self) -> None:
        """Reset failed items to pending."""
        for item in self.state.items:
            if item.status == QueueStatus.FAILED:
                item.status = QueueStatus.PENDING
                item.retry_count = 0
                item.error_message = None
        self.save_state()
        console.print("[yellow]Failed items reset to pending[/yellow]")

    def show_status(self) -> None:
        """Display queue status as a table."""
        table = Table(title="Queue Status")
        table.add_column("ID", style="cyan")
        table.add_column("Scene", style="magenta")
        table.add_column("Batch", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Retries", style="yellow")
        table.add_column("Error", style="red", max_width=30)

        for item in self.state.items:
            status_style = {
                QueueStatus.PENDING: "white",
                QueueStatus.IN_PROGRESS: "cyan",
                QueueStatus.COMPLETED: "green",
                QueueStatus.FAILED: "red",
            }.get(item.status, "white")

            table.add_row(
                item.id,
                str(item.scene.scene_id),
                str(item.batch_number),
                f"[{status_style}]{item.status.value}[/{status_style}]",
                str(item.retry_count),
                (item.error_message or "")[:30],
            )

        console.print(table)

        # Summary
        pending = len(self.state.get_pending())
        in_progress = len(self.state.get_in_progress())
        completed = len(self.state.get_completed())
        failed = len(self.state.get_failed())

        console.print(f"\n[bold]Summary:[/bold] {pending} pending | {in_progress} in progress | {completed} completed | {failed} failed")
        console.print(f"[bold]Progress:[/bold] {self.state.progress_percent:.1f}%")

    def process_queue(self) -> dict:
        """Process all pending items in the queue."""
        results = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
        }

        pending_count = len(self.state.get_pending())
        if pending_count == 0:
            console.print("[yellow]No pending items in queue[/yellow]")
            return results

        console.print(f"\n[bold cyan]Starting queue processing ({pending_count} items)[/bold cyan]\n")

        controller = WhiskController(self.config)

        try:
            controller.start()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Processing queue...", total=pending_count)

                while True:
                    item = self.get_next_pending()
                    if not item:
                        break

                    # Update status
                    self.update_item_status(item.id, QueueStatus.IN_PROGRESS)

                    progress.update(task, description=f"Scene {item.scene.scene_id} (batch {item.batch_number})")

                    # Restart browser for each scene to avoid UI state issues
                    console.print("[cyan]Starting fresh browser session...[/cyan]")
                    controller.stop()
                    time.sleep(2)
                    controller.start()

                    # Process the scene
                    output_folder = Path(self.config.paths.output) / item.output_folder
                    env_path = Path(self.config.paths.environments)
                    char_path = Path(self.config.paths.characters)

                    result = controller.process_scene(
                        scene=item.scene,
                        output_folder=output_folder,
                        env_base_path=env_path,
                        char_base_path=char_path,
                    )

                    results["processed"] += 1

                    if result.success:
                        self.mark_completed(item.id)
                        results["succeeded"] += 1
                        console.print(f"[green][OK] Scene {item.scene.scene_id} batch {item.batch_number} completed[/green]")
                    else:
                        self.mark_failed(item.id, result.error_message or "Unknown error")
                        results["failed"] += 1
                        console.print(f"[red][FAIL] Scene {item.scene.scene_id} batch {item.batch_number} failed[/red]")

                    progress.advance(task)

                    # Delay between scenes (besides browser restart time)
                    if self.config.queue.delay_between_scenes > 0:
                        time.sleep(self.config.queue.delay_between_scenes)

        except Exception as e:
            console.print(f"[red]Queue processing error: {e}[/red]")
        finally:
            controller.stop()

        console.print(f"\n[bold]Queue processing complete![/bold]")
        console.print(f"Processed: {results['processed']} | Succeeded: {results['succeeded']} | Failed: {results['failed']}")

        return results

    def process_one(self) -> dict:
        """Process a single item from the queue (for testing)."""
        item = self.get_next_pending()
        if not item:
            console.print("[yellow]No pending items in queue[/yellow]")
            return {"processed": 0, "succeeded": 0, "failed": 0}

        console.print(f"\n[bold cyan]Processing single item: Scene {item.scene.scene_id} (batch {item.batch_number})[/bold cyan]\n")

        controller = WhiskController(self.config)

        try:
            controller.start()

            # Update status
            self.update_item_status(item.id, QueueStatus.IN_PROGRESS)

            # Process the scene
            output_folder = Path(self.config.paths.output) / item.output_folder
            env_path = Path(self.config.paths.environments)
            char_path = Path(self.config.paths.characters)

            result = controller.process_scene(
                scene=item.scene,
                output_folder=output_folder,
                env_base_path=env_path,
                char_base_path=char_path,
            )

            if result.success:
                self.mark_completed(item.id)
                console.print(f"[green][OK] Scene {item.scene.scene_id} batch {item.batch_number} completed[/green]")
                return {"processed": 1, "succeeded": 1, "failed": 0}
            else:
                self.mark_failed(item.id, result.error_message or "Unknown error")
                console.print(f"[red][FAIL] Scene {item.scene.scene_id} batch {item.batch_number} failed[/red]")
                return {"processed": 1, "succeeded": 0, "failed": 1}

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            self.mark_failed(item.id, str(e))
            return {"processed": 1, "succeeded": 0, "failed": 1}
        finally:
            controller.stop()


def create_sample_csv(output_path: Path) -> None:
    """Create a sample scenes CSV file."""
    import pandas as pd

    sample_data = [
        {
            "scene_id": 1,
            "environment_id": "env_forest",
            "character_ids": "",
            "prompt": "A serene forest at dawn, golden sunlight filtering through the trees, morning mist"
        },
        {
            "scene_id": 2,
            "environment_id": "env_forest",
            "character_ids": "char_girl",
            "prompt": "A young girl walking through the forest, curious expression, discovering nature"
        },
        {
            "scene_id": 3,
            "environment_id": "env_house",
            "character_ids": "char_girl,char_grandmother",
            "prompt": "Grandmother calling gently from the living room doorway, warm inviting expression"
        },
    ]

    df = pd.DataFrame(sample_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Sample CSV created: {output_path}[/green]")
