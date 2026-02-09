# Whisk Automation

Automated batch image generation using Google Whisk. Upload environment and character images, generate scenes with AI, and download results at scale.

## Features

- **Batch Processing**: Queue multiple scenes and process them automatically
- **Reusable Assets**: Upload environments and characters once, use them infinitely
- **Queue Management**: Track progress, retry failed items, clear completed work
- **CSV Import**: Load hundreds of scenes from a spreadsheet
- **Interactive Mode**: Manual control with live browser debugging
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Quick Start

### Option 1: One-Command Setup

**Windows:**
```bash
# Double-click setup.bat or run:
setup.bat
```

**macOS/Linux:**
```bash
bash setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up the project structure
- Install the Chromium browser for Playwright

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright browser
playwright install chromium
```

## Usage

### Test Connection

First, verify Whisk is accessible:

```bash
# Windows
run.bat test

# macOS/Linux
python run.py test
```

This will open a browser and navigate to Whisk.

### Basic Workflow

1. **Add assets** (environment/character images to `data/environments/` and `data/characters/`)

2. **Create a scene** via CLI:
```bash
python run.py add-scene -s 1 -e env_001 -p "A cozy cat in a sunny window"
```

3. **Or load from CSV**:
```bash
python run.py load-csv data/scenes.csv
```

4. **Process the queue**:
```bash
python run.py process
```

### Commands

| Command | Description |
|---------|-------------|
| `python run.py test` | Test Whisk connection |
| `python run.py status` | Show queue status |
| `python run.py add-scene ...` | Add a single scene |
| `python run.py load-csv <file>` | Load scenes from CSV |
| `python run.py process` | Process all pending |
| `python run.py process-one` | Process one item (testing) |
| `python run.py clear --all` | Clear entire queue |
| `python run.py clear --completed` | Clear completed items |
| `python run.py retry-failed` | Reset failed to pending |
| `python run.py config --show` | Show configuration |
| `python run.py create-sample` | Create sample CSV structure |
| `python run.py interactive` | Interactive mode |

### Adding Scenes

**Single Scene:**
```bash
python run.py add-scene \
  --scene-id 1 \
  --env env_001 \
  --chars char_001,char_002 \
  --prompt "Two friends having coffee" \
  --format landscape \
  --batches 2
```

**From CSV:**
Create `data/scenes.csv`:
```csv
scene_id,environment_id,character_ids,prompt,format,batches
1,env_001,char_001,"A cat on a windowsill",landscape,2
2,env_001,char_001,char_002,"Two cats playing",landscape,2
```

Then:
```bash
python run.py load-csv data/scenes.csv
```

### Configuration

Edit `config.json` or use the CLI:

```bash
# Show current config
python run.py config --show

# Set headless mode
python run.py config --set-headless

# Set images per prompt
python run.py config --images-per-prompt 8

# Set batches per scene
python run.py config --batches 3
```

## Project Structure

```
whisk/
├── data/
│   ├── environments/    # Upload your environment images here
│   ├── characters/      # Upload your character images here
│   └── scenes.csv       # CSV file for batch scene creation
├── output/              # Generated images downloaded here
├── logs/                # Log files
├── src/
│   ├── config.py        # Configuration management
│   ├── models.py        # Data models
│   ├── queue_manager.py # Queue processing logic
│   └── whisk_controller.py # Browser automation
├── config.json          # Main configuration file
├── requirements.txt     # Python dependencies
├── setup.sh            # Linux/macOS setup script
├── setup.bat           # Windows setup script
└── run.py              # Main CLI entry point
```

## Configuration Options

```json
{
  "whisk_url": "https://labs.google/fx/tools/whisk/project",
  "browser": {
    "headless": false,        // Run browser invisible
    "slow_mo": 100,           // Slow down actions (ms)
    "user_data_dir": null     // Chrome user profile (optional)
  },
  "paths": {
    "environments": "./data/environments",
    "characters": "./data/characters",
    "output": "./output",
    "scenes_file": "./data/scenes.csv"
  },
  "generation": {
    "images_per_prompt": 4,   // Images to generate per scene
    "batches_per_scene": 2,   // How many times to repeat the scene
    "image_format": "landscape",
    "download_timeout": 60
  },
  "queue": {
    "retry_on_failure": true,
    "max_retries": 3,
    "delay_between_scenes": 5
  }
}
```

## Requirements

- Python 3.9 or later
- Chromium browser (auto-installed by Playwright)
- Internet connection for Whisk

## Troubleshooting

**Browser doesn't open:**
- Make sure Chromium is installed: `playwright install chromium`

**Uploads fail:**
- Check image paths in `config.json` are correct
- Ensure images are valid formats (PNG, JPG)

**Downloads fail:**
- Increase `download_timeout` in config.json
- Check `output/` directory exists and is writable

**Connection errors:**
- Run `python run.py test` to verify Whisk is accessible
- Check your internet connection

## License

MIT
