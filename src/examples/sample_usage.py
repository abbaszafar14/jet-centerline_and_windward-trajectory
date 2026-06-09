"""
Example usage of spray trajectory extraction

Before running:
  1. Place your background images in: data/backgrounds/
  2. Place your spray images in: data/spray/
  3. Update BASE_DIR and image paths in spray_trajectory_extraction.py
  4. Adjust configuration parameters (ORIGIN_COL, ORIGIN_ROW, MM_PER_PX, D_MM)

Then run:
  python src/spray_trajectory_extraction.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import and run the extraction
from spray_trajectory_extraction import *

print("Trajectory extraction complete!")
print("Check the output Excel file and plots in the output directory.")
