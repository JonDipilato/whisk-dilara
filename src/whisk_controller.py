"""Selenium-based controller for Google Whisk automation."""

import os
import sys
import platform
import time
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime
from rich.console import Console

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

try:
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WEBDRIVER_MANAGER = True
except ImportError:
    HAS_WEBDRIVER_MANAGER = False

from .config import AppConfig
from .models import Scene, ImageFormat, GenerationResult

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

console = Console()


def crop_letterboxing(image_path: Path, threshold: int = 20) -> bool:
    """Remove black/white letterbox bars from image edges.

    Args:
        image_path: Path to the image file
        threshold: How close to black (0) or white (255) a row must be to be considered a bar

    Returns:
        True if image was cropped, False otherwise
    """
    if not HAS_PIL:
        return False

    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        # Get image dimensions
        height, width = img_array.shape[:2]

        # Function to check if a row is solid black or white
        def is_bar_row(row):
            avg = np.mean(row)
            return avg < threshold or avg > (255 - threshold)

        def is_bar_col(col):
            avg = np.mean(col)
            return avg < threshold or avg > (255 - threshold)

        # Find crop boundaries
        top = 0
        bottom = height
        left = 0
        right = width

        # Scan from top
        for i in range(height // 4):  # Max 25% crop
            if is_bar_row(img_array[i]):
                top = i + 1
            else:
                break

        # Scan from bottom
        for i in range(height - 1, height - height // 4, -1):
            if is_bar_row(img_array[i]):
                bottom = i
            else:
                break

        # Scan from left
        for i in range(width // 4):
            if is_bar_col(img_array[:, i]):
                left = i + 1
            else:
                break

        # Scan from right
        for i in range(width - 1, width - width // 4, -1):
            if is_bar_col(img_array[:, i]):
                right = i
            else:
                break

        # Only crop if we found bars (at least 5 pixels)
        if top > 5 or bottom < height - 5 or left > 5 or right < width - 5:
            cropped = img.crop((left, top, right, bottom))
            cropped.save(image_path)
            console.print(f"[cyan]Cropped letterboxing: {top}px top, {height-bottom}px bottom, {left}px left, {width-right}px right[/cyan]")
            return True

        return False

    except Exception as e:
        console.print(f"[yellow]Could not crop letterboxing: {e}[/yellow]")
        return False


class WhiskController:
    """Controls Google Whisk through browser automation."""

    WHISK_URL = "https://labs.google/fx/tools/whisk/project"

    # Element references from Claude Code web extension analysis
    SELECTORS = {
        # Expand/collapse images panel
        "add_images_toggle": 'button:has-text("ADD IMAGES")',
        "show_hide_toggle": 'ref_34',  # Toggles between "ADD IMAGES" and "HIDE IMAGES"

        # File inputs (hidden, but accessible)
        "subject_file_input": 'ref_69',   # For character/subject images
        "scene_file_input": 'ref_84',     # For environment/scene images
        "style_file_input": 'ref_99',     # For style reference images

        # Prompt and controls
        "prompt_input": 'ref_100',        # Main prompt textarea
        "generate_button": 'ref_43',      # Submit/generate button (arrow_forward)
        "aspect_ratio": 'ref_39',         # Aspect ratio control

        # Section containers
        "subject_section": 'ref_23',
        "scene_section": 'ref_26',
        "style_section": 'ref_29',
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.is_windows = platform.system() == "Windows" or sys.platform == "win32"
        self.is_wsl = os.path.exists("/mnt/c") and not self.is_windows
        self.panels_expanded = False

    def start(self) -> None:
        """Start the browser and navigate to Whisk."""
        console.print("[cyan]Starting browser...[/cyan]")

        # Reset state for fresh session
        self.panels_expanded = False

        options = Options()

        # Use existing Chrome profile for Google login
        if self.config.browser.user_data_dir:
            options.add_argument(f"--user-data-dir={self.config.browser.user_data_dir}")
            console.print(f"[cyan]Using Chrome profile: {self.config.browser.user_data_dir}[/cyan]")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")

        # Configure options
        if self.config.browser.headless:
            options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--window-size=1920,1080")

        # Set user agent
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        # Disable webdriver detection
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # Initialize driver
        if self.is_wsl:
            # WSL can execute Windows .exe directly using /mnt/c/ paths
            console.print("[cyan]Running from WSL - using Windows ChromeDriver[/cyan]")

            # Set Chrome binary location (Windows path for ChromeDriver which runs as Windows process)
            chrome_binary = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            options.binary_location = chrome_binary

            # Find latest chromedriver version dynamically
            win_user = os.environ.get("USER", os.environ.get("USERNAME", ""))
            wdm_base = Path(f"/mnt/c/Users/{win_user}/.wdm/drivers/chromedriver/win64")
            chromedriver_path = None

            if wdm_base.exists():
                versions = sorted(wdm_base.iterdir(), reverse=True)
                for version_dir in versions:
                    candidates = list(version_dir.rglob("chromedriver.exe"))
                    if candidates:
                        # Use WSL path directly - WSL can execute Windows binaries
                        chromedriver_path = str(candidates[0])
                        break

            if chromedriver_path:
                console.print(f"[cyan]Using ChromeDriver: {chromedriver_path}[/cyan]")
                service = Service(executable_path=chromedriver_path)
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                # Try without service - let Selenium find it
                console.print("[cyan]ChromeDriver not found, trying default discovery[/cyan]")
                self.driver = webdriver.Chrome(options=options)
        elif HAS_WEBDRIVER_MANAGER:
            console.print("[cyan]Using webdriver-manager for ChromeDriver[/cyan]")
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            console.print("[cyan]Using system ChromeDriver[/cyan]")
            self.driver = webdriver.Chrome(options=options)

        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, self.config.generation.download_timeout or 60)

        output_path = Path(self.config.paths.output).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Navigating to Whisk...[/cyan]")
        self.driver.get(self.WHISK_URL)
        time.sleep(5)
        console.print("[green]Whisk loaded successfully![/green]")

    def stop(self) -> None:
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            console.print("[yellow]Browser closed.[/yellow]")

    def _ensure_panels_expanded(self) -> None:
        """Ensure the Subject/Scene/Style panels are expanded."""
        if self.panels_expanded:
            return

        try:
            # If file inputs already exist in the DOM, assume panels are available.
            file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if file_inputs:
                self.panels_expanded = True
                return

            # Try to locate a toggle that controls a panel containing file inputs.
            buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[aria-controls], button[aria-expanded]")
            for btn in buttons:
                controls_id = (btn.get_attribute("aria-controls") or "").strip()
                if not controls_id:
                    continue

                try:
                    panel = self.driver.find_element(By.ID, controls_id)
                except Exception:
                    panel = None

                if panel:
                    panel_inputs = panel.find_elements(By.CSS_SELECTOR, "input[type='file']")
                    if panel_inputs:
                        if (btn.get_attribute("aria-expanded") or "").lower() != "true":
                            btn.click()
                            console.print("[cyan]Expanded image panels[/cyan]")
                            time.sleep(1)
                        self.panels_expanded = True
                        return

            # Fallback: click a collapsed toggle and confirm file inputs appear.
            for btn in buttons:
                if (btn.get_attribute("aria-expanded") or "").lower() == "false":
                    btn.click()
                    time.sleep(1)
                    file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
                    if file_inputs:
                        console.print("[cyan]Expanded image panels[/cyan]")
                        self.panels_expanded = True
                        return
        except Exception as e:
            console.print(f"[yellow]Could not expand panels: {e}[/yellow]")

    def _button_has_material_icon(self, button, icon_names: set[str]) -> bool:
        """Check if a button contains a Material icon with one of the given names."""
        try:
            icon_elems = button.find_elements(
                By.CSS_SELECTOR,
                "mat-icon, span.material-icons, i.material-icons, "
                "span.material-symbols-outlined, span.material-symbols-rounded, span.material-symbols-sharp"
            )
            for elem in icon_elems:
                icon_text = (elem.text or "").strip().lower()
                if icon_text in icon_names:
                    return True

            inner_html = (button.get_attribute("innerHTML") or "").lower()
            for icon in icon_names:
                if (f">{icon}<" in inner_html or
                    f"material-icons\">{icon}<" in inner_html or
                    f"material-symbols-outlined\">{icon}<" in inner_html or
                    f"material-symbols-rounded\">{icon}<" in inner_html or
                    f"material-symbols-sharp\">{icon}<" in inner_html):
                    return True
        except Exception:
            return False

        return False

    def _upload_file_by_ref(self, file_path: Path, ref_id: str, upload_type: str) -> bool:
        """Upload a file using the hidden file input by ref ID."""
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return False

        try:
            self._ensure_panels_expanded()

            # Find all file inputs (should be 3: Subject, Scene, Style)
            all_file_inputs = []
            all_inputs = self.driver.find_elements(By.TAG_NAME, "input")
            for inp in all_inputs:
                if inp.get_attribute("type") == "file":
                    all_file_inputs.append(inp)

            console.print(f"[cyan]Found {len(all_file_inputs)} file inputs[/cyan]")

            if not all_file_inputs:
                console.print("[yellow]No file inputs found[/yellow]")
                return False

            # Map ref IDs to input index based on DOM order
            # ref_69 (Subject) = first, ref_84 (Scene) = second, ref_99 (Style) = third
            ref_to_index = {
                "ref_69": 0,  # Subject
                "ref_84": 1,  # Scene
                "ref_99": 2,  # Style
            }

            index = ref_to_index.get(ref_id, 0)
            if index < len(all_file_inputs):
                target_input = all_file_inputs[index]
                abs_path = str(file_path.absolute())
                target_input.send_keys(abs_path)
                console.print(f"[green]Uploaded {upload_type}: {file_path.name}[/green]")
                # Wait for Whisk to analyze/process the uploaded image
                time.sleep(4)
                return True

            console.print(f"[yellow]Could not find file input for {upload_type} (index {index}, total {len(all_file_inputs)})[/yellow]")
            return False

        except Exception as e:
            console.print(f"[red]Error uploading {upload_type}: {e}[/red]")
            return False

    def upload_environment(self, env_path: Path) -> bool:
        """Upload an environment/scene image (uses SCENE section, ref_84)."""
        return self._upload_file_by_ref(env_path, "ref_84", "environment")

    def _click_add_subject_button(self) -> bool:
        """Click the '+' button next to SUBJECT label to add another character slot WITHIN the Subject section."""
        try:
            # Find all visible buttons with an "add" icon, then pick the topmost one.
            all_buttons = self.driver.find_elements(By.TAG_NAME, "button")

            add_icon_names = {"add", "add_circle", "add_box", "add_circle_outline", "add_box_outline"}
            add_category_buttons = []
            for i, btn in enumerate(all_buttons):
                try:
                    if not btn.is_displayed():
                        continue
                except Exception:
                    continue

                if self._button_has_material_icon(btn, add_icon_names):
                    y_pos = btn.location.get('y', 9999)
                    add_category_buttons.append((i, y_pos))

            # Sort by Y position - the first one is SUBJECT section, second is SCENE, third is STYLE
            add_category_buttons.sort(key=lambda x: x[1])

            if len(add_category_buttons) >= 1:
                # Click the first "Add new category" button (SUBJECT section)
                button_index = add_category_buttons[0][0]
                console.print(f"[cyan]Clicking add button #{button_index} (topmost, SUBJECT section)[/cyan]")
                all_buttons[button_index].click()
                console.print("[green]Clicked SUBJECT section's + button (adding character slot)[/green]")
                time.sleep(3)
                return True

            console.print("[yellow]Could not find any add buttons[/yellow]")
            return False

        except Exception as e:
            console.print(f"[yellow]Could not click add button: {e}[/yellow]")
            return False

    def prepare_character_slots(self, num_characters: int) -> bool:
        """Click the SUBJECT + button to create enough slots for all characters.

        Args:
            num_characters: Total number of characters to upload
        """
        if num_characters <= 1:
            return True  # Default slot is enough

        clicks_needed = num_characters - 1  # Already have 1 slot by default
        console.print(f"[cyan]Creating {clicks_needed} additional SUBJECT slots...[/cyan]")

        for i in range(clicks_needed):
            success = self._click_add_subject_button()
            if not success:
                console.print(f"[yellow]Could only create {i} additional slots[/yellow]")
                return False
            time.sleep(2)  # Wait for slot to appear

        console.print(f"[green]Created {clicks_needed} additional SUBJECT slots[/green]")
        return True

    def upload_character(self, char_path: Path, slot_index: int = 0) -> bool:
        """Upload a character/subject image to a specific SUBJECT slot.

        Args:
            char_path: Path to the character image
            slot_index: Which SUBJECT slot to upload to (0 = first/top, 1 = second, etc.)
        """
        if not char_path.exists():
            console.print(f"[red]File not found: {char_path}[/red]")
            return False

        try:
            # Find all file inputs
            all_inputs = self.driver.find_elements(By.TAG_NAME, "input")
            file_inputs = [inp for inp in all_inputs if inp.get_attribute("type") == "file"]

            console.print(f"[cyan]Found {len(file_inputs)} file inputs, targeting slot {slot_index}[/cyan]")

            if slot_index < len(file_inputs):
                target_input = file_inputs[slot_index]
                abs_path = str(char_path.absolute())
                target_input.send_keys(abs_path)
                console.print(f"[green]Uploaded character to slot {slot_index}: {char_path.name}[/green]")
                time.sleep(4)  # Wait for Whisk to process
                return True
            else:
                console.print(f"[yellow]Slot {slot_index} not found (only {len(file_inputs)} inputs)[/yellow]")
                return False

        except Exception as e:
            console.print(f"[red]Error uploading character: {e}[/red]")
            return False

    def upload_all_images(self, char_paths: list[Path], env_path: Optional[Path]) -> bool:
        """Upload all character and environment images in the correct order.

        Workflow:
        1. Expand panels
        2. Click SUBJECT + button (N-1) times to create slots for N characters
        3. Upload each character to its own SUBJECT slot (using slot_index)
        4. Upload environment to SCENE slot (last slot)
        """
        self._ensure_panels_expanded()
        time.sleep(1)

        num_chars = len(char_paths)
        console.print(f"[cyan]Uploading {num_chars} characters + environment[/cyan]")

        # STEP 1: Click + button to create additional SUBJECT slots
        if num_chars > 1:
            console.print(f"[cyan]Creating {num_chars - 1} additional SUBJECT slots...[/cyan]")
            for i in range(num_chars - 1):
                self._click_add_subject_button()
                time.sleep(2)

        time.sleep(2)  # Wait for all slots to be ready

        # STEP 2: Upload characters to SUBJECT slots
        # Always use slot 0 because Whisk removes processed inputs
        for i, char_path in enumerate(char_paths):
            console.print(f"[cyan]Uploading character {i+1}/{num_chars}: {char_path.name}[/cyan]")
            success = self.upload_character(char_path, slot_index=0)
            if not success:
                console.print(f"[red]Failed to upload {char_path.name}[/red]")

        # STEP 3: Upload environment to SCENE slot
        if env_path:
            console.print(f"[cyan]Uploading environment: {env_path.name}[/cyan]")
            time.sleep(2)

            # Find all file inputs
            all_inputs = self.driver.find_elements(By.TAG_NAME, "input")
            file_inputs = [inp for inp in all_inputs if inp.get_attribute("type") == "file"]

            console.print(f"[cyan]Found {len(file_inputs)} total file inputs[/cyan]")

            if file_inputs:
                # After characters uploaded, first remaining input is SCENE, second is STYLE
                scene_index = 0
                console.print(f"[cyan]Targeting SCENE slot at index {scene_index}[/cyan]")
                file_inputs[scene_index].send_keys(str(env_path.absolute()))
                console.print(f"[green]Uploaded environment to SCENE slot: {env_path.name}[/green]")
                time.sleep(4)
            else:
                console.print(f"[yellow]No file input for environment[/yellow]")

        return True

    def set_prompt(self, prompt: str) -> bool:
        """Set the generation prompt (ref_100)."""
        try:
            # Find the first visible textarea (language-agnostic)
            textareas = self.driver.find_elements(By.TAG_NAME, "textarea")
            target = None
            for textarea in textareas:
                try:
                    if textarea.is_displayed():
                        target = textarea
                        break
                except Exception:
                    continue

            if not target and textareas:
                target = textareas[0]

            if target:
                target.clear()
                target.send_keys(prompt)
                console.print(f"[green]Set prompt: {prompt[:50]}...[/green]")
                time.sleep(1)
                return True

            console.print("[yellow]Could not find prompt input[/yellow]")
            return False

        except Exception as e:
            console.print(f"[red]Error setting prompt: {e}[/red]")
            return False

    def set_format(self, format: ImageFormat = ImageFormat.LANDSCAPE) -> bool:
        """Set the image format (landscape 16:9, portrait 9:16, square 1:1).

        Whisk uses a dropdown menu for aspect ratio selection.
        """
        # Map format to possible UI labels
        format_labels = {
            ImageFormat.LANDSCAPE: ["landscape", "16:9", "16 : 9", "wide"],
            ImageFormat.PORTRAIT: ["portrait", "9:16", "9 : 16", "tall"],
            ImageFormat.SQUARE: ["square", "1:1", "1 : 1"],
        }
        target_labels = format_labels.get(format, format_labels[ImageFormat.LANDSCAPE])

        try:
            # First, find and click the aspect ratio toggle/button
            aspect_btn = None
            buttons = self.driver.find_elements(By.TAG_NAME, "button")

            for btn in buttons:
                btn_aria = (btn.get_attribute("aria-label") or "").lower()
                btn_text = (btn.text or "").lower()
                inner_html = (btn.get_attribute("innerHTML") or "").lower()

                # Look for aspect ratio button (may show current ratio like "16:9")
                if ("aspect" in btn_aria or "ratio" in btn_aria or
                    "16:9" in btn_text or "9:16" in btn_text or "1:1" in btn_text or
                    "crop" in inner_html):
                    aspect_btn = btn
                    break

            if not aspect_btn:
                console.print("[yellow]Could not find aspect ratio button[/yellow]")
                return False

            # Click to open dropdown
            aspect_btn.click()
            console.print("[cyan]Opened aspect ratio menu[/cyan]")
            time.sleep(1)

            # Find and click the target format option
            # Look for menu items, buttons, or list items with matching text
            all_elements = self.driver.find_elements(By.CSS_SELECTOR,
                "button, [role='menuitem'], [role='option'], li, mat-option, .menu-item")

            for elem in all_elements:
                elem_text = (elem.text or "").lower().strip()
                elem_aria = (elem.get_attribute("aria-label") or "").lower()

                for label in target_labels:
                    if label in elem_text or label in elem_aria:
                        elem.click()
                        console.print(f"[green]Set aspect ratio to {format.value} ({label})[/green]")
                        time.sleep(0.5)
                        return True

            # If no explicit option found, check if already selected (button may toggle)
            console.print(f"[yellow]Could not find {format.value} option - may already be set[/yellow]")
            # Click elsewhere to close menu
            self.driver.find_element(By.TAG_NAME, "body").click()
            time.sleep(0.3)
            return True

        except Exception as e:
            console.print(f"[yellow]Could not set format: {e}[/yellow]")
            return False

    def generate(self) -> bool:
        """Click the generate button (yellow arrow button near prompt)."""
        try:
            # The generate button is a yellow/orange circular button with arrow icon
            # It's typically at the end of the prompt input bar
            buttons = self.driver.find_elements(By.TAG_NAME, "button")

            for btn in buttons:
                try:
                    # Check for arrow icon inside button (mat-icon or svg)
                    inner_html = btn.get_attribute("innerHTML") or ""
                    aria_label = (btn.get_attribute("aria-label") or "").lower()
                    btn_class = (btn.get_attribute("class") or "").lower()

                    # Look for arrow_forward icon or submit-related attributes
                    if ("arrow_forward" in inner_html or
                        "arrow-forward" in inner_html or
                        "submit" in aria_label or
                        "generate" in aria_label or
                        "send" in aria_label or
                        "submit" in btn_class):
                        btn.click()
                        console.print("[cyan]Generation started (arrow button)...[/cyan]")
                        return True
                except:
                    continue

            # Try finding by button type
            for btn in buttons:
                btn_type = btn.get_attribute("type") or ""
                if btn_type == "submit":
                    btn.click()
                    console.print("[cyan]Generation started (submit button)...[/cyan]")
                    return True

            # Try clicking the last button in the prompt area (often the submit)
            prompt_container = self.driver.find_elements(By.CSS_SELECTOR,
                '[class*="prompt"] button, [class*="input"] button, form button')
            if prompt_container:
                prompt_container[-1].click()
                console.print("[cyan]Generation started (prompt area button)...[/cyan]")
                return True

            # Fallback: Try Enter key on textarea
            prompt_areas = self.driver.find_elements(By.TAG_NAME, "textarea")
            if prompt_areas:
                prompt_areas[0].send_keys(Keys.RETURN)
                console.print("[cyan]Generation started (via Enter key)...[/cyan]")
                return True

            console.print("[yellow]Could not find generate button[/yellow]")
            return False

        except Exception as e:
            console.print(f"[red]Error starting generation: {e}[/red]")
            return False

    def wait_for_generation(self, timeout: int = 30) -> bool:
        """Wait for image generation to complete."""
        try:
            console.print(f"[cyan]Waiting for generation (max {timeout}s)...[/cyan]")

            start_time = datetime.now()
            generation_started = False

            # First, wait for loading/generating indicator to appear
            console.print("[cyan]Waiting for generation to start...[/cyan]")
            while (datetime.now() - start_time).seconds < 15:
                # Look for loading indicators, progress bars, spinners
                loading = self.driver.find_elements(By.CSS_SELECTOR,
                    '[class*="loading"], [class*="progress"], [class*="spinner"], '
                    '[class*="generating"], [role="progressbar"], mat-progress-bar, '
                    '[class*="circular"], svg[class*="animate"]')

                # Also check for disabled generate button (means generation in progress)
                disabled_btns = self.driver.find_elements(By.CSS_SELECTOR, 'button[disabled]')

                if loading or len(disabled_btns) > 0:
                    generation_started = True
                    console.print("[cyan]Generation in progress...[/cyan]")
                    break

                time.sleep(1)

            if not generation_started:
                console.print("[yellow]Could not detect generation start - waiting anyway...[/yellow]")

            # Now wait for generation to complete
            console.print("[cyan]Waiting for AI generation (this takes ~10-30 seconds)...[/cyan]")
            wait_start = datetime.now()
            min_wait_secs = max(5, min(10, timeout // 3))  # Minimum time to wait for AI generation

            while (datetime.now() - start_time).seconds < timeout:
                waited_secs = (datetime.now() - wait_start).seconds

                # Check if loading is still happening
                loading = self.driver.find_elements(By.CSS_SELECTOR,
                    '[class*="loading"], [class*="progress"], [class*="spinner"], [class*="generating"]')

                # Progress indicator for user
                if waited_secs % 10 == 0 and waited_secs > 0:
                    console.print(f"[dim]...waited {waited_secs}s[/dim]")

                # Only check for completion after minimum wait time
                if waited_secs >= min_wait_secs and not loading:
                    # Look for generated result images
                    result_images = self.driver.find_elements(By.CSS_SELECTOR,
                        '[class*="result"] img, [class*="output"] img, [class*="generated"] img, '
                        'main img[src*="storage"], [class*="preview"] img[src*="storage"]')

                    # Also look for a download button with a download/folder icon
                    download_icons = {"download", "file_download", "download_for_offline", "folder", "folder_open"}
                    download_btns = []
                    for btn in self.driver.find_elements(By.TAG_NAME, "button"):
                        if self._button_has_material_icon(btn, download_icons):
                            download_btns.append(btn)

                    if result_images or download_btns:
                        console.print(f"[green]Generation complete! (waited {waited_secs}s)[/green]")
                        time.sleep(3)  # Extra buffer for images to fully render
                        return True

                time.sleep(2)

            console.print("[yellow]Generation timeout - proceeding anyway[/yellow]")
            return True

        except Exception as e:
            console.print(f"[red]Error waiting for generation: {e}[/red]")
            return False

    def download_images(self, output_folder: Path, prefix: str = "image", crop: bool = True) -> list[Path]:
        """Download generated images using Whisk's DOWNLOAD ALL IMAGES button.

        Whisk downloads images as a ZIP file (whisk_images.zip), so we need to:
        1. Detect new ZIP files in Downloads
        2. Extract images from the ZIP
        3. Move them to the output folder
        """
        import shutil
        import zipfile

        output_folder.mkdir(parents=True, exist_ok=True)
        downloaded = []

        # Determine downloads folder
        if self.is_wsl:
            win_user = os.environ.get("USER", os.environ.get("USERNAME", ""))
            downloads_folder = Path("/mnt/c/Users") / win_user / "Downloads"
        else:
            downloads_folder = Path.home() / "Downloads"

        # Get existing ZIP files before download (to detect new ones)
        existing_zips = set(downloads_folder.glob("whisk*.zip"))
        existing_images = set(downloads_folder.glob("*.png")) | set(downloads_folder.glob("*.jpg"))

        try:
            # Scroll to top to ensure download button is visible
            self.driver.execute_script("window.scrollTo(0, 0)")
            time.sleep(1)

            # Click "DOWNLOAD ALL IMAGES" button (yellow button at top)
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            download_clicked = False

            # First try: Look for a download button with download/folder icon
            download_icons = {"download", "file_download", "download_for_offline", "folder", "folder_open"}
            for btn in buttons:
                try:
                    if not btn.is_displayed():
                        continue
                    if self._button_has_material_icon(btn, download_icons):
                        # Ensure button is visible and clickable
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                        time.sleep(0.5)
                        btn.click()
                        download_clicked = True
                        console.print("[cyan]Clicked DOWNLOAD ALL IMAGES button (main)[/cyan]")
                        break
                except:
                    continue

            # Second try: Look for download icon in top toolbar
            if not download_clicked:
                toolbar_buttons = self.driver.find_elements(By.CSS_SELECTOR,
                    'header button, [role="toolbar"] button, .toolbar button')
                for btn in toolbar_buttons:
                    try:
                        if self._button_has_material_icon(btn, download_icons):
                            btn.click()
                            download_clicked = True
                            console.print("[cyan]Clicked download button (toolbar)[/cyan]")
                            break
                    except:
                        continue

            if not download_clicked:
                console.print("[yellow]Could not find download button - trying fallback[/yellow]")
                return self._download_images_fallback(output_folder, prefix, crop=crop)

            # Wait for download to complete
            console.print("[cyan]Waiting for ZIP download to complete...[/cyan]")
            time.sleep(5)  # Initial wait for download to start

            # Check for new ZIP files (Whisk downloads as whisk_images.zip)
            max_wait = 45  # Increased wait time
            waited = 0
            new_zip = None

            while waited < max_wait:
                current_zips = set(downloads_folder.glob("whisk*.zip"))
                new_zips = list(current_zips - existing_zips)

                # Also check for direct image downloads (in case Whisk changes behavior)
                current_images = set(downloads_folder.glob("*.png")) | set(downloads_folder.glob("*.jpg"))
                new_images = [f for f in (current_images - existing_images)
                             if (datetime.now().timestamp() - f.stat().st_mtime) < 60]

                # Progress logging
                if waited % 5 == 0 and waited > 0:
                    console.print(f"[dim]Waiting for download... ({waited}s)[/dim]")

                if new_zips:
                    # Get the most recent ZIP
                    new_zip = max(new_zips, key=lambda f: f.stat().st_mtime)
                    console.print(f"[cyan]Detected ZIP file after {waited}s[/cyan]")
                    # Wait a bit more for ZIP to finish writing
                    time.sleep(3)
                    break
                elif new_images:
                    # Direct image download (no ZIP)
                    console.print(f"[cyan]Found {len(new_images)} direct image downloads[/cyan]")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    for i, src_file in enumerate(sorted(new_images)):
                        dest_filename = f"{prefix}_{timestamp}_{i+1}{src_file.suffix}"
                        dest_path = output_folder / dest_filename
                        shutil.move(str(src_file), str(dest_path))
                        downloaded.append(dest_path)
                        console.print(f"[green]Downloaded: {dest_filename}[/green]")
                    return downloaded

                time.sleep(1)
                waited += 1

            if new_zip:
                console.print(f"[cyan]Found ZIP: {new_zip.name}[/cyan]")

                # Extract ZIP contents
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    with zipfile.ZipFile(new_zip, 'r') as zf:
                        for i, name in enumerate(zf.namelist()):
                            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                # Extract to temp location
                                extracted = zf.extract(name, downloads_folder)
                                extracted_path = Path(extracted)

                                # Move to output with new name
                                ext = extracted_path.suffix
                                dest_filename = f"{prefix}_{timestamp}_{i+1}{ext}"
                                dest_path = output_folder / dest_filename

                                shutil.move(str(extracted_path), str(dest_path))
                                downloaded.append(dest_path)
                                console.print(f"[green]Extracted: {dest_filename}[/green]")

                    # Remove the ZIP after extraction
                    new_zip.unlink()
                    console.print(f"[dim]Cleaned up ZIP file[/dim]")

                except zipfile.BadZipFile:
                    console.print("[yellow]ZIP file still downloading, waiting...[/yellow]")
                    time.sleep(3)
                    # Retry extraction
                    with zipfile.ZipFile(new_zip, 'r') as zf:
                        for i, name in enumerate(zf.namelist()):
                            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                extracted = zf.extract(name, downloads_folder)
                                extracted_path = Path(extracted)
                                ext = extracted_path.suffix
                                dest_filename = f"{prefix}_{timestamp}_{i+1}{ext}"
                                dest_path = output_folder / dest_filename
                                shutil.move(str(extracted_path), str(dest_path))
                                downloaded.append(dest_path)
                    new_zip.unlink()

            if not downloaded:
                console.print("[yellow]No new files detected in Downloads folder[/yellow]")

            return downloaded

        except Exception as e:
            console.print(f"[red]Error downloading images: {e}[/red]")
            return downloaded

    def _download_images_fallback(self, output_folder: Path, prefix: str, crop: bool = True) -> list[Path]:
        """Fallback: Download images directly from URLs or screenshot with UI hidden.

        Strategy: Whisk generates 4 result images. We first try to download directly from
        googleusercontent URLs (clean, no UI). If that fails, we hide UI overlays before
        screenshotting to avoid capturing ANIMATE/REFINE buttons.
        """
        downloaded = []
        temp_files = []

        try:
            # Scroll to ensure all result images are in viewport
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            # Hide ALL UI overlay elements that appear on images (buttons, icons, etc.)
            hide_ui_script = """
                // Hide all buttons that overlay images
                document.querySelectorAll('button').forEach(btn => {
                    const text = (btn.innerText || '').toUpperCase();
                    const html = btn.innerHTML || '';
                    if (text.includes('ANIMATE') || text.includes('REFINE') ||
                        html.includes('thumb_up') || html.includes('thumb_down') ||
                        html.includes('favorite') || html.includes('share') ||
                        html.includes('download') || html.includes('arrow')) {
                        btn.style.display = 'none';
                    }
                });
                // Hide any overlay containers
                document.querySelectorAll('[class*="overlay"], [class*="toolbar"], [class*="action"]').forEach(el => {
                    if (el.querySelector('button')) {
                        el.style.display = 'none';
                    }
                });
                // Hide floating button groups near images
                document.querySelectorAll('img').forEach(img => {
                    const parent = img.parentElement;
                    if (parent) {
                        parent.querySelectorAll('button, [role="button"]').forEach(btn => {
                            btn.style.display = 'none';
                        });
                    }
                });
            """
            self.driver.execute_script(hide_ui_script)
            time.sleep(0.5)
            console.print("[cyan]Hidden UI overlays for clean capture[/cyan]")

            # Find all images - prioritize those in result/output containers
            result_images = self.driver.find_elements(By.CSS_SELECTOR,
                'main img, [class*="result"] img, [class*="output"] img, [class*="generated"] img')

            # If no result-specific images found, fall back to all images
            if not result_images:
                result_images = self.driver.find_elements(By.CSS_SELECTOR, 'img')

            console.print(f"[dim]Found {len(result_images)} image elements, capturing...[/dim]")

            # Capture all candidate images
            for i, img in enumerate(result_images):
                try:
                    src = img.get_attribute("src") or ""
                    width = img.size.get("width", 0)
                    height = img.size.get("height", 0)

                    # Only capture substantial images (likely generated results)
                    if width > 250 and height > 250 and ("blob:" in src or "storage" in src or "googleusercontent" in src):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{prefix}_{timestamp}_{i+1}.png"
                        filepath = output_folder / filename

                        # Try direct download first for googleusercontent URLs (cleaner)
                        if "googleusercontent" in src and not src.startswith("blob:"):
                            try:
                                response = requests.get(src, timeout=30)
                                if response.status_code == 200 and len(response.content) > 50000:
                                    with open(filepath, 'wb') as f:
                                        f.write(response.content)
                                    temp_files.append(filepath)
                                    console.print(f"[dim]Direct download: {filename}[/dim]")
                                    continue
                            except Exception as e:
                                console.print(f"[dim]Direct download failed, using screenshot: {e}[/dim]")

                        # Fallback: Take screenshot (UI should be hidden now)
                        img.screenshot(str(filepath))
                        temp_files.append(filepath)

                except Exception:
                    continue

            # Filter by file size and keep metadata
            large_files = []
            for filepath in temp_files:
                if filepath.exists():
                    file_size = filepath.stat().st_size
                    if file_size > 50_000:  # 50KB threshold (lowered for portrait)
                        large_files.append((filepath, file_size))
                    else:
                        # Delete small UI screenshot
                        filepath.unlink()
                        console.print(f"[dim]Filtered out small file: {filepath.name}[/dim]")

            # Sort by file size (largest = highest quality) and keep only last 2
            # The last images captured are usually the full high-quality renders
            if len(large_files) > 2:
                console.print(f"[cyan]Found {len(large_files)} large images, keeping last 2 highest quality[/cyan]")
                # Sort by file size descending, take top 2
                large_files.sort(key=lambda x: x[1], reverse=True)
                large_files = large_files[:2]

            # Keep the selected files, delete the rest
            selected_paths = {f[0] for f in large_files}
            for filepath in temp_files:
                if filepath.exists() and filepath not in selected_paths:
                    filepath.unlink()
                    console.print(f"[dim]Removed extra capture: {filepath.name}[/dim]")

            # Report final downloads and crop letterboxing
            for filepath, file_size in large_files:
                downloaded.append(filepath)
                size_kb = file_size // 1024
                console.print(f"[green]Downloaded: {filepath.name} ({size_kb}KB)[/green]")
                # Auto-crop any black/white letterbox bars
                if crop:
                    crop_letterboxing(filepath)

            if not downloaded:
                console.print("[yellow]No large generated images found in fallback[/yellow]")

            return downloaded

        except Exception as e:
            console.print(f"[red]Fallback download error: {e}[/red]")
            return downloaded

    def clear_inputs(self) -> None:
        """Clear all inputs for next generation."""
        try:
            # Clear prompt
            textareas = self.driver.find_elements(By.TAG_NAME, "textarea")
            for ta in textareas:
                try:
                    ta.clear()
                except:
                    pass

            time.sleep(0.5)

        except Exception as e:
            console.print(f"[yellow]Could not clear inputs: {e}[/yellow]")

    def process_scene(
        self,
        scene: Scene,
        output_folder: Path,
        env_base_path: Path,
        char_base_path: Path
    ) -> GenerationResult:
        """Process a complete scene - upload images, set prompt, generate, download."""
        start_time = datetime.now()
        result = GenerationResult(
            queue_item_id=str(scene.scene_id),
            success=False
        )

        try:
            console.print(f"\n[bold cyan]Processing Scene {scene.scene_id}[/bold cyan]")

            self.clear_inputs()
            time.sleep(1)

            # Gather all image paths first
            char_paths = []
            for char_id in scene.character_ids:
                char_path = self._find_image(char_base_path, char_id)
                if char_path:
                    char_paths.append(char_path)
                else:
                    console.print(f"[yellow]Character image not found: {char_id}[/yellow]")

            env_path = self._find_image(env_base_path, scene.environment_id) if scene.environment_id else None

            # Use the new upload_all method that handles everything properly
            self.upload_all_images(char_paths, env_path)

            # Set format
            self.set_format(scene.image_format)
            time.sleep(0.5)

            # Set prompt and generate
            self.set_prompt(scene.prompt)
            time.sleep(1)

            self.generate()

            # Wait for generation
            self.wait_for_generation(self.config.generation.download_timeout)

            # Download images
            downloaded = self.download_images(
                output_folder,
                prefix=f"scene_{scene.scene_id}"
            )

            result.success = len(downloaded) > 0
            result.images_generated = len(downloaded)
            result.output_paths = [str(p) for p in downloaded]

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            console.print(f"[red]Error processing scene: {e}[/red]")

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    def _find_image(self, base_path: Path, image_id: str) -> Optional[Path]:
        """Find an image file by ID (tries common extensions)."""
        extensions = [".png", ".jpg", ".jpeg", ".webp", ""]

        for ext in extensions:
            path = base_path / f"{image_id}{ext}"
            if path.exists():
                return path

            for prefix in ["", "env_", "char_", "environment_", "character_"]:
                path = base_path / f"{prefix}{image_id}{ext}"
                if path.exists():
                    return path

        matches = list(base_path.glob(f"*{image_id}*"))
        if matches:
            return matches[0]

        return None


def test_whisk_connection(config: AppConfig) -> bool:
    """Test if we can connect to Whisk."""
    controller = WhiskController(config)
    try:
        controller.start()
        console.print("[green]Successfully connected to Whisk![/green]")

        # Take a screenshot
        screenshot_path = Path(config.paths.output) / "whisk_screenshot.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        controller.driver.save_screenshot(str(screenshot_path))
        console.print(f"[cyan]Screenshot saved to: {screenshot_path}[/cyan]")

        # Debug: List all file inputs found
        console.print("\n[bold cyan]Looking for file inputs...[/bold cyan]")
        controller._ensure_panels_expanded()

        all_inputs = controller.driver.find_elements(By.TAG_NAME, "input")
        file_inputs = [inp for inp in all_inputs if inp.get_attribute("type") == "file"]
        console.print(f"[cyan]Found {len(file_inputs)} file inputs[/cyan]")

        for i, inp in enumerate(file_inputs):
            parent = inp.find_element(By.XPATH, "..").get_attribute("id") or "unknown"
            console.print(f"  File input {i+1}: parent_id={parent}")

        return True
    except Exception as e:
        console.print(f"[red]Connection test failed: {e}[/red]")
        return False
    finally:
        controller.stop()
