# Grandma Episode - Quick Start

## One Command (recommended)

```powershell
python excel_to_config.py --run
```

This reads the Excel, builds the config, and runs the full pipeline (refs + 77 scenes + video).

## Options

```powershell
# Custom title
python excel_to_config.py --run --title "Tea Time with Grandma"

# Set episode number
python excel_to_config.py --run --episode 3

# Add narration from a text file
python excel_to_config.py --run --narration-file narration.txt

# Skip Whisk (reuse existing images)
python excel_to_config.py --run --skip-whisk

# Upload to YouTube after
python excel_to_config.py --run --upload
```

## Manual Steps (if you need to inspect between steps)

```powershell
# 1. Generate config
python excel_to_config.py

# 2. Check config
cat story_config.json | Select-Object -First 20

# 3. Run pipeline with the output dir printed in step 1
python run_story.py --output-dir "output/episodes/grandma_ep1_20260127_100557"
```

**Important:** Always pass `--output-dir` when running `run_story.py` manually. Without it, the pipeline uses the default `output/` folder which may have existing images and will skip scene generation.

## What It Does

1. Generates 3 character refs (Narrator, Grandmother, Miso)
2. Generates 7 environment refs (kitchen, living room, window, garden, fields, hallway, bathroom)
3. Generates 77 scenes with per-scene character + environment switching
4. Assembles music-only video (no narration unless `--narration-file` provided)

## Episode Counter

Grandma episodes have their own counter at `data/episode_counter_grandma.json`.
