# Jet Centerline and Windward Trajectory Detection

Python-based image processing for extracting spray centerline and windward trajectory from jet in crossflow experiments.

## Description

This code performs column-wise trajectory extraction from planar Mie scattering images:

- **Centerline**: Row of maximum intensity per column
- **Windward Edge**: Topmost row above 25% of local column peak per column

Features:
- Background subtraction and temporal averaging
- Savitzky-Golay smoothing (two-region blended)
- Coordinate conversion (pixel → mm → diameter-normalized)
- Per-condition and combined analysis
- Multiple output plots and Excel sheets

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit parameters in `src/spray_trajectory_extraction.py`:

```python
# Origin in image pixel coordinates (nozzle exit position)
ORIGIN_COL = 209     # column (x-direction: rightward)
ORIGIN_ROW = 2327    # row (y-direction: upward from wall)
MM_PER_PX  = 0.0428  # calibration [mm/pixel]
D_MM       = 1.14    # orifice diameter [mm]

# Trajectory thresholds
THRESHOLD_WINDWARD = 0.30   # 30% of local column peak
X_LIMIT_MM = 35.0   # downstream extraction limit

# Smoothing
SG_WINDOW_NEAR = 81    # near-field window
SG_WINDOW_FAR  = 201   # far-field window
X_CROSS_MM     = 5.0   # transition point
```

## Usage

1. **Prepare your data:**
   - Place background images in the background directory
   - Place spray images in condition subdirectories

2. **Update file paths** in `spray_trajectory_extraction.py`:
```python
   BASE_DIR  = "path/to/your/data"
   BG_DIR    = os.path.join(BASE_DIR, "backgrounds")
   SPRAY_DIR = os.path.join(BASE_DIR, "spray_images")
```

3. **Define your test conditions:**
```python
   CONDITIONS = {
       "Condition_Name_1": (We, q),
       "Condition_Name_2": (We, q),
   }
```

4. **Run the extraction:**
```bash
   python src/spray_trajectory_extraction.py
```

## Outputs

### Excel Sheets
- **Raw_mm**: All conditions merged (x_mm, centerline_y_mm, windward_y_mm)
- **Normalized**: All conditions merged (x/d, y/d)
- **Per-condition sheets**: Individual normalized trajectories

### Plots
- Full image + trajectory overlay (per condition)
- Zoomed spray region (per condition)
- Combined centerline plot (all conditions, normalized)
- Combined windward edge plot (all conditions, normalized)

## Coordinate System

Origin at nozzle exit:
- **x** (z): downstream direction (rightward) [mm]
- **y**: wall-normal direction (upward from wall) [mm]

Normalized:
- **x/d**: downstream / orifice diameter
- **y/d**: wall-normal / orifice diameter

## Author

[Abbas Zafar]

## Citation

If using this code in research, please cite:
