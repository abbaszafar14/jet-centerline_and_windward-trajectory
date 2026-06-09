"""
Spray Trajectory Extraction

Setup:
  Nozzle at bottom wall (y=0), no elevation. Clean images — no acrylic
  scratch artefacts. Standard background subtraction and temporal averaging
  are sufficient without any intensity zeroing pre-processing.

Trajectory extraction (column-wise):
  Centerline   : row of maximum intensity per column (above nozzle exit)
  Windward edge: topmost row above 25% of LOCAL column peak per column

Coordinate system:
  Origin at nozzle exit (ORIGIN_COL, ORIGIN_ROW) in cropped image pixels.
  x_mm = (col - ORIGIN_COL) * MM_PER_PX          downstream (rightward)
  y_mm = (ORIGIN_ROW - row) * MM_PER_PX           wall-normal (upward)

Outputs:
  Excel sheets:
    1. "Raw_mm"       — all conditions merged  (x_mm, y_mm)
    2. "Normalized"   — all conditions merged  (x/d,  y/d)
    3+. One sheet per condition (normalized)
  Plots:
    - Grayscale mean image + trajectory overlay — full (per condition)
    - Grayscale mean image + trajectory overlay — zoomed (per condition)
    - Combined centerline plot  (all conditions, normalized)
    - Combined windward edge plot (all conditions, normalized)
"""

import cv2
import numpy as np
import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR  = "E:/PROJECT LJICF/SPRAY TRAJECTORY PYTHON/Bottom_Y10_Trajectory"
PLOTS_DIR = os.path.join(BASE_DIR, "plots_BI10_1mm")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Origin in CROPPED image pixel coordinates (nozzle exit = x=0, y=0)
# origin_pixel from MATLAB mat file: [x, y] = [col, row]
ORIGIN_COL = 209     # col pixel of nozzle exit  (x-direction: rightward)
ORIGIN_ROW = 2327   # row pixel of nozzle exit  (at bottom wall; rows go top→bottom)
MM_PER_PX  = 0.0428 # mm per pixel
D_MM       = 1.14   # actual orifice diameter [mm]

# Trajectory thresholds
THRESHOLD_WINDWARD = 0.30   # 30% of local column peak -> windward edge
NOISE_FLOOR_FRAC   = 0.01   # relative: 1% of global peak
NOISE_FLOOR_ABS    = 20.0   # absolute floor [DN] — effective = max(relative, absolute)

# Downstream extraction limit [mm]  (set to None for no limit)
X_LIMIT_MM = 35.0

# Savitzky-Golay smoothing
# window=51 px × 0.0472 mm/px ≈ 2.4 mm physical length
# mode='nearest' suppresses boundary oscillations → trajectories start from (0,0)
SG_WINDOW_NEAR = 81   # near-field: short window preserves initial curvature
SG_WINDOW_FAR  = 201   # far-field:  wide window suppresses pixel-staircase noise
X_CROSS_MM     = 5.0   # blend centre — mm downstream where near→far transitions
BLEND_WIDTH_MM = 3.0   # blend width: full far-field beyond X_CROSS_MM + 1.5 mm
SG_POLY   = 3
SG_MODE   = "nearest"

# Zoomed plot padding [mm]
PLOT_PAD_X = 2.0
PLOT_PAD_Y = 2.0

# Condition folders -> (We, q)
CONDITIONS = {
    "BI 1mm 40V 1C": (30, 10),
    "BI 1mm 40V 3C": (30, 20),
    "BI 1mm 50V 1C": (47, 10),
    "BI 1mm 50V 3C": (47, 20),
    "BI 1mm 50V 5C": (47, 30),
    "BI 1mm 60V 1C": (68, 10),
    "BI 1mm 60V 3C": (68, 20),
}

BG_DIR    = os.path.join(BASE_DIR, "backgrounds", "background 1mm", "cropped_red")
SPRAY_DIR = os.path.join(BASE_DIR, "BI 1mm")
OUT_XLSX  = os.path.join(BASE_DIR, "BottomY10_1mm_trajectory.xlsx")

# Plot style
plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.labelsize" : 12,
    "axes.titlesize" : 12,
    "legend.fontsize": 9,
})
CMAP   = plt.colormaps["tab10"]
COLORS = {name: CMAP(i) for i, name in enumerate(CONDITIONS)}

# ============================================================
# STEP 1 — LOAD BACKGROUND
# ============================================================

bg_files = sorted(glob.glob(os.path.join(BG_DIR, "*.png")))
if not bg_files:
    raise FileNotFoundError(f"No background images found in: {BG_DIR}")

bg_stack = np.stack([
    cv2.imread(f)[:, :, 2].astype(np.float32) for f in bg_files
])
mean_bg = bg_stack.mean(axis=0)
print(f"Background: {len(bg_files)} image(s). Mean BG = {mean_bg.mean():.2f} DN")
print(f"Image size (H x W): {mean_bg.shape[0]} x {mean_bg.shape[1]}")
print(f"Origin: col={ORIGIN_COL}, row={ORIGIN_ROW}  |  "
      f"mm/px={MM_PER_PX}  |  d={D_MM} mm")


# STEP 2 — PROCESS EACH CONDITION


raw_rows      = []
norm_rows     = []
per_cond_norm = {}

fig_comb_cl, ax_comb_cl = plt.subplots(figsize=(7, 5))
fig_comb_ue, ax_comb_ue = plt.subplots(figsize=(7, 5))

for folder_name, (We, q) in CONDITIONS.items():
    spray_folder = os.path.join(SPRAY_DIR, folder_name, "cropped_red")
    sp_files = sorted(glob.glob(os.path.join(spray_folder, "*.png")))

    if not sp_files:
        print(f"  WARNING: no images in {spray_folder} — skipping")
        continue

    print(f"\nProcessing: {folder_name}  (We={We}, q={q})  [{len(sp_files)} frames]")

    # --- Load raw frames, background subtract, clip, average ---
    # Clean images — no intensity zeroing needed
    raw_stack = np.stack([
        cv2.imread(f)[:, :, 2].astype(np.float32) for f in sp_files
    ])
    sub_stack = np.clip(raw_stack - mean_bg, 0.0, None)
    I_avg     = sub_stack.mean(axis=0)

    global_peak = float(I_avg.max())
    noise_floor = max(NOISE_FLOOR_FRAC * global_peak, NOISE_FLOOR_ABS)
    H, W        = I_avg.shape

    print(f"  Peak: {global_peak:.2f}  |  noise floor: {noise_floor:.2f}  "
          f"|  windward: 25% of local column peak")

 
    # COLUMN-WISE TRAJECTORY EXTRACTION
    # Scan columns from nozzle exit rightward.
    # For each column, inspect only rows ABOVE the wall (row <= ORIGIN_ROW),
    # i.e. the physical region y >= 0.

    cols_px = []
    cl_rows = []
    ue_rows = []

    row_end = min(ORIGIN_ROW + 1, H)   # clip to valid image height

    for c in range(ORIGIN_COL, W):
        col_profile = I_avg[: row_end, c]   # rows 0 .. ORIGIN_ROW

        local_peak = float(col_profile.max())
        if local_peak < noise_floor:
            continue

        # Centerline: row of maximum intensity
        cl = int(np.argmax(col_profile))

        # Windward edge: topmost row (smallest index = highest y)
        # above 25% of local column peak
        thresh_local = THRESHOLD_WINDWARD * local_peak
        above = np.where(col_profile >= thresh_local)[0]
        if len(above) == 0:
            continue
        ue = int(above.min())

        cols_px.append(c)
        cl_rows.append(cl)
        ue_rows.append(ue)

    n_valid = len(cols_px)
    print(f"  Valid columns: {n_valid}")

    if n_valid == 0:
        print("  WARNING: no valid columns — skipping")
        continue

    # --- Savitzky-Golay smoothing (two-region blended) ---
    # Near-field (0 → X_CROSS_MM): short window preserves initial curvature.
    # Far-field  (X_CROSS_MM →  ): wide  window suppresses pixel-staircase noise.
    # Linear blend across X_CROSS_MM ± BLEND_WIDTH_MM/2.
    if n_valid < SG_WINDOW_NEAR:
        cl_smooth = np.array(cl_rows, dtype=float)
        ue_smooth = np.array(ue_rows, dtype=float)
    else:
        cl_arr = np.array(cl_rows, dtype=float)
        ue_arr = np.array(ue_rows, dtype=float)

        def _sg(arr, win):
            """SG filter with safe odd window clamped to array length."""
            wl = win if win % 2 == 1 else win + 1
            wl = min(wl, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
            wl = max(wl, SG_POLY + 2)
            if wl % 2 == 0:
                wl += 1
            return savgol_filter(arr, wl, SG_POLY, mode=SG_MODE)

        # Per-column blend weight: 0 = near-field window, 1 = far-field window
        x_mm_tmp = (np.array(cols_px) - ORIGIN_COL) * MM_PER_PX
        x0 = X_CROSS_MM - BLEND_WIDTH_MM / 2.0
        x1 = X_CROSS_MM + BLEND_WIDTH_MM / 2.0
        w  = np.clip((x_mm_tmp - x0) / (x1 - x0), 0.0, 1.0)

        cl_smooth = (1.0 - w) * _sg(cl_arr, SG_WINDOW_NEAR) + w * _sg(cl_arr, SG_WINDOW_FAR)
        ue_smooth = (1.0 - w) * _sg(ue_arr, SG_WINDOW_NEAR) + w * _sg(ue_arr, SG_WINDOW_FAR)

    # --- Coordinate conversion: pixel -> mm ---
    cols_arr = np.array(cols_px)
    x_mm    = (cols_arr  - ORIGIN_COL) * MM_PER_PX
    cl_y_mm = (ORIGIN_ROW - cl_smooth) * MM_PER_PX
    ue_y_mm = (ORIGIN_ROW - ue_smooth) * MM_PER_PX

    # --- Apply domain filter: x >= 0, x <= X_LIMIT_MM, clip y >= 0 ---
    mask = x_mm >= 0
    if X_LIMIT_MM is not None:
        mask &= (x_mm <= X_LIMIT_MM)
    x_out  = x_mm[mask]
    cl_out = np.maximum(cl_y_mm[mask], 0.0)
    ue_out = np.maximum(ue_y_mm[mask], 0.0)

    x_d  = x_out  / D_MM
    cl_d = cl_out / D_MM
    ue_d = ue_out / D_MM

    print(f"  x: {x_out.min():.2f}–{x_out.max():.2f} mm  "
          f"({x_d.min():.1f}–{x_d.max():.1f} d)  |  "
          f"CL y max: {cl_out.max():.2f} mm  |  WE y max: {ue_out.max():.2f} mm")

    # --- Accumulate data ---
    label         = f"We={We}, q={q}"
    cond_norm_rows = []

    for x, cl_y, ue_y, xd, cld, ued in zip(x_out, cl_out, ue_out, x_d, cl_d, ue_d):
        raw_rows.append({
            "condition"       : folder_name,
            "label"           : label,
            "We"              : We,
            "q"               : q,
            "x_mm"            : round(float(x),   4),
            "centerline_y_mm" : round(float(cl_y), 4),
            "windward_y_mm"   : round(float(ue_y), 4),
        })
        entry = {
            "condition"      : folder_name,
            "label"          : label,
            "We"             : We,
            "q"              : q,
            "x_d"            : round(float(xd),  4),
            "centerline_y_d" : round(float(cld), 4),
            "windward_y_d"   : round(float(ued), 4),
        }
        norm_rows.append(entry)
        cond_norm_rows.append(entry)

    per_cond_norm[folder_name] = pd.DataFrame(cond_norm_rows)


    # PLOTS — Full image  +  Zoomed spray region
   
    disp_img = cv2.normalize(I_avg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Physical extent of full cropped image [mm]
    x_left_mm  = -ORIGIN_COL * MM_PER_PX
    x_right_mm = (W - 1 - ORIGIN_COL) * MM_PER_PX
    y_top_mm   =  ORIGIN_ROW * MM_PER_PX
    y_bot_mm   = -(H - 1 - ORIGIN_ROW) * MM_PER_PX

    def _draw_traj(ax):
        ax.plot(x_out, cl_out, color="#00BFFF", lw=1.5, label="Centerline")
        ax.plot(x_out, ue_out, color="#FF4500", lw=1.5, label="Windward edge")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper left")

    # --- Plain full image — raw pixels, no overlay (cv2) ---
    safe_name = folder_name.replace(' ', '_')
    cv2.imwrite(os.path.join(PLOTS_DIR, f"plain_mean_raw_{safe_name}.png"), disp_img)

    # --- Plain full image — mm axes, no overlay (matplotlib) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(disp_img, cmap="gray", origin="upper",
              extent=[x_left_mm, x_right_mm, y_bot_mm, y_top_mm], aspect="equal")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Bottom Elevated 1.14 mm — {label}")
    fig.savefig(os.path.join(PLOTS_DIR, f"plain_mean_{safe_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Full image with overlay ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(disp_img, cmap="gray", origin="upper",
              extent=[x_left_mm, x_right_mm, y_bot_mm, y_top_mm], aspect="equal")
    _draw_traj(ax)
    ax.set_title(f"Bottom Elevated 1.14 mm — {label}")
    fig.savefig(os.path.join(PLOTS_DIR,
                f"mean_image_{folder_name.replace(' ', '_')}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Zoomed plot — sized to max windward edge extent + padding
    x_zoom_max = float(x_out.max())  + PLOT_PAD_X
    y_zoom_max = float(ue_out.max()) + PLOT_PAD_Y

    col_min_px = max(0, int(ORIGIN_COL - PLOT_PAD_X / MM_PER_PX))
    col_max_px = min(W, int(ORIGIN_COL + x_zoom_max / MM_PER_PX) + 1)
    row_min_px = max(0, int(ORIGIN_ROW - y_zoom_max / MM_PER_PX) - 1)
    row_max_px = min(H, ORIGIN_ROW + int(PLOT_PAD_Y / MM_PER_PX) + 1)

    img_zoom = disp_img[row_min_px:row_max_px, col_min_px:col_max_px]

    # --- Plain zoomed image — raw pixels, no overlay (cv2) ---
    cv2.imwrite(os.path.join(PLOTS_DIR, f"plain_mean_raw_zoom_{safe_name}.png"), img_zoom)

    xz_left  = (col_min_px - ORIGIN_COL) * MM_PER_PX
    xz_right = (col_max_px - 1 - ORIGIN_COL) * MM_PER_PX
    yz_top   = (ORIGIN_ROW - row_min_px) * MM_PER_PX
    yz_bot   = (ORIGIN_ROW - row_max_px + 1) * MM_PER_PX

    x_span = x_zoom_max + PLOT_PAD_X
    y_span = y_zoom_max
    fig_w  = 7.0
    fig_h  = max(3.0, fig_w * y_span / x_span)

    # --- Plain zoomed image — mm axes, no overlay (matplotlib) ---
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_zoom, cmap="gray", origin="upper",
              extent=[xz_left, xz_right, yz_bot, yz_top], aspect="equal")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0, x_zoom_max)
    ax.set_ylim(0, y_zoom_max)
    ax.set_title(f"Bottom Elevated 1.14 mm — {label}")
    fig.savefig(os.path.join(PLOTS_DIR, f"plain_mean_zoom_{safe_name}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Zoomed image with overlay ---
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_zoom, cmap="gray", origin="upper",
              extent=[xz_left, xz_right, yz_bot, yz_top], aspect="equal")
    _draw_traj(ax)
    ax.set_xlim(0, x_zoom_max)
    ax.set_ylim(0, y_zoom_max)
    ax.set_title(f"Bottom Elevated 1.14 mm — {label}")
    fig.savefig(os.path.join(PLOTS_DIR,
                f"zoomed_{folder_name.replace(' ', '_')}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved full + zoomed plots.")

    # Accumulate combined plots
    color = COLORS[folder_name]
    ax_comb_cl.plot(x_d, cl_d, color=color, lw=1.5, label=label)
    ax_comb_ue.plot(x_d, ue_d, color=color, lw=1.5, label=label)


# COMBINED PLOTS


for ax, fig_obj, title, fname in [
    (ax_comb_cl, fig_comb_cl, "Spray Centerline — Bottom Elevated 1.14 mm",
     os.path.join(PLOTS_DIR, "combined_centerline_normalized.png")),
    (ax_comb_ue, fig_comb_ue, "Windward Edge — Bottom Elevated 1.14 mm",
     os.path.join(PLOTS_DIR, "combined_windward_normalized.png")),
]:
    ax.set_xlabel("$Z/d_o$")
    ax.set_ylabel("$Y/d_o$")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", alpha=0.4)
    fig_obj.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig_obj)

print("\nSaved combined plots.")

# ============================================================
# SAVE EXCEL
# Sheet 1: Raw_mm       — all conditions merged
# Sheet 2: Normalized   — all conditions merged
# Sheet 3+: one per condition (normalized)
# ============================================================

df_raw  = pd.DataFrame(raw_rows)
df_norm = pd.DataFrame(norm_rows)

def _sheet(name):
    return name.replace(" ", "_")[:31]

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df_raw.to_excel(writer,  sheet_name="Raw_mm",    index=False)
    df_norm.to_excel(writer, sheet_name="Normalized", index=False)
    for fn, df_c in per_cond_norm.items():
        df_c.to_excel(writer, sheet_name=_sheet(fn), index=False)

print(f"\n{'='*60}")
print(f"Saved: {OUT_XLSX}")
print(f"  Sheet 'Raw_mm'    : {len(df_raw)} rows")
print(f"  Sheet 'Normalized': {len(df_norm)} rows")
for fn, df_c in per_cond_norm.items():
    print(f"  Sheet '{_sheet(fn)}': {len(df_c)} rows")

print(f"\nConditions summary:")
print(df_raw.groupby(["label", "We", "q"])["x_mm"]
      .agg(n_pts="count", x_max_mm="max")
      .to_string())
