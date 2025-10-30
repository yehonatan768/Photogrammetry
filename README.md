# NerfStudio Photogrammetry Pipeline — Updated & Simplified

Turn a **video** into a trained **NeRF**, export a **dense point cloud**, optionally **clean it**, and generate a **Poisson mesh** — all on Windows (tested on **Python 3.9**, **CUDA 12.1**, RTX‑class GPUs).

> This README matches your current folder layout (see “Project Structure”) and the behavior of the scripts in `scripts/`.
>
> Defaults assume your project folder is `D:\Projects\NerfStudio` (you can change `PROJECT_DIR` at the top of each script).

---

## Table of Contents

- [Quick Start (TL;DR)](#quick-start-tldr)
- [Project Structure](#project-structure)
- [Install & Setup](#install--setup)
- [Nerfstudio Point Cloud & FFmpeg — What’s Going On](#nerfstudio-point-cloud--ffmpeg--whats-going-on)
- [End‑to‑End Workflow (Step‑By‑Step)](#end-to-end-workflow-step-by-step)
- [Script Reference](#script-reference)
- [Tips for Quality / Speed](#tips-for-quality--speed)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (TL;DR)

```powershell
# 0) Activate venv (first time: see Install & Setup)
.venv\Scripts\activate

# 1) Prepare dataset from your video and train nerfacto (Steps 1–2)
python scripts/nerfstudio.py

# 2) Export a dense point cloud from the latest trained run (Step 3)
python scripts/export.py

# 3) (Optional) Prefilter/clean the point cloud (Step 4)
#     - either 'filter.py' (center/density prefilter) or 'cloud_filter.py' (multi-gate cleaner)
python scripts/filter.py
# or
python scripts/cloud_filter.py

# 4) Build a Poisson mesh from any .ply inside an export folder (Step 5)
python scripts/mesh.py
```

The scripts show menus when needed and write outputs under `outputs/…` (see below).

---

## Project Structure

```
NerfStudio/
├── .venv/                      # your virtual environment
├── outputs/
│   ├── dataset/                # datasets from ns-process-data video
│   ├── experiments/            # nerfstudio training runs (nerfacto/*/config.yml, ckpts, etc.)
│   └── exports/                # per‑export folders (point clouds, filtered clouds, meshes)
├── scripts/
│   ├── cloud_filter.py         # cloud/noise reducer (modes: off → ultra)
│   ├── export.py               # export dense point cloud from latest run
│   ├── filter.py               # prefilter base cloud (keep dense center)
│   ├── mesh.py                 # Poisson mesh + (adaptive) smoothing + cleanup
│   ├── nerfstudio.py           # Step 1–2: dataset from video + train nerfacto
│   └── run.py                  # optional launcher/guide (menus & quick flows)
└── videos/                     # put your source videos here (e.g., Barn.mp4)
```

> **Paths**  
> - Datasets: `outputs/dataset/<SceneName>/`  
> - Experiments (training): `outputs/experiments/<EXP_NAME>/nerfacto/<RUN>/`  
> - Exports: `outputs/exports/<PrettyName>_verK-DD-MM-YYYY/`  
> If your local scripts still point to other roots (e.g., `output/experiment`), update the `PROJECT_DIR`/`OUTPUT(S)_DIR` constants at the top of the script(s).

---

## Install & Setup

**Requirements**

- Windows 10/11 (PowerShell) — Linux/macOS are similar
- Python **3.9**
- NVIDIA GPU + CUDA **12.1** drivers (`nvidia-smi`)
- **FFmpeg** on PATH (improves video handling)
- **Nerfstudio** CLIs on PATH: `ns-process-data`, `ns-train`, `ns-export`
- **CloudCompare** (optional, great for visual cleanup/inspection)

**Create venv & install deps**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify CUDA in PyTorch**

```powershell
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA version seen by PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

---

## Nerfstudio Point Cloud & FFmpeg — What’s Going On

- **FFmpeg**: When you run `ns-process-data video`, Nerfstudio uses FFmpeg (if available) to **decode** your input video and **extract frames** reliably, supporting many codecs/containers. Cleaner frame extraction = better COLMAP photogrammetry.
- **Photogrammetry & Poses**: `ns-process-data` runs **COLMAP** to compute camera intrinsics/extrinsics from the frames, producing a dataset Nerfstudio can train on.
- **Training**: `ns-train nerfacto` optimizes a radiance field (NeRF) from the frames + poses. Your trained run lives under `outputs/experiments/<EXP_NAME>/nerfacto/<timestamp>/`.
- **Export Point Cloud**: `ns-export pointcloud --load-config <config.yml>` samples millions of points from the trained field, orienting **normals** (e.g. `--normal-method open3d`) and writing `point_cloud.ply` into a new folder under `outputs/exports/...`. These clouds can be heavy and include **floaters/sky/noise** — the filter scripts help you clean that before meshing.
- **Meshing**: `mesh.py` runs **Poisson surface reconstruction** (Open3D) with helpful presets, plus optional adaptive smoothing & cleanup, producing `mesh.ply`.

---

## End‑to‑End Workflow (Step‑By‑Step)

### 1) Build dataset from video (frames + COLMAP)
Configure at the top of `scripts/nerfstudio.py`: `VIDEO_FILE`, `EXPERIMENT_NAME`, `NUM_FRAMES_TARGET`, `MAX_ITERS` (for training), and `SKIP_*` flags.

```powershell
python scripts/nerfstudio.py
```
- Creates/uses `outputs/dataset/<SceneName>/`  
- Trains to `outputs/experiments/<EXP_NAME>/nerfacto/<RUN>/`

### 2) Export a dense point cloud from the latest run
No arguments needed; a menu will pick the latest run automatically.

```powershell
python scripts/export.py
```
- Writes to a new folder: `outputs/exports/<PrettyName>_verK-DD-MM-YYYY/point_cloud.ply`

### 3) (Optional) Prefilter or cloud‑clean the `.ply`
Choose one (or both, one after the other). Both tools write a new `.ply` beside the original.

```powershell
# Prefilter: keep dense center; light SOR on stronger modes
python scripts/filter.py

# Cloud/noise reducer: center & density gates, optional HSV sky removal + SOR
python scripts/cloud_filter.py
```

### 4) Build a Poisson mesh from any `.ply` in the export folder
You can pick `filtered.ply`, `light_filtered.ply`, or the raw `point_cloud.ply`.

```powershell
python scripts/mesh.py
```
- Result: `outputs/exports/<...>/mesh.ply`

> Tip: You can also open CloudCompare to inspect or manually clean any `.ply`/mesh at any stage.

---

## Script Reference

### `scripts/nerfstudio.py`
- **Does**: `ns-process-data video` → `ns-train nerfacto`  
- **Where**: reads from `videos/<VIDEO_FILE>`, writes to `outputs/dataset/...` and `outputs/experiments/...`  
- **Keys**: `VIDEO_FILE`, `EXPERIMENT_NAME`, `MAX_ITERS`, `NUM_FRAMES_TARGET`, `SKIP_DATASET`, `SKIP_TRAIN`  
- **Run**: `python scripts/nerfstudio.py`

### `scripts/export.py`
- **Does**: invokes `ns-export pointcloud --load-config <.../config.yml>` for the **newest** nerfacto run  
- **Where**: reads `outputs/experiments/<EXP_NAME>/nerfacto/<RUN>/config.yml`, writes `outputs/exports/<PrettyName>_verK-<date>/point_cloud.ply`  
- **Run**: `python scripts/export.py`

### `scripts/filter.py` *(Prefilter)*
- **Does**: keeps the dense center using radius + local‑density thresholds (kNN), with a safe fallback; optional light SOR at stronger presets  
- **Where**: picks an export folder under `outputs/exports/`, loads **`point_cloud.ply`**, writes `<mode>_filtered.ply`  
- **Run**: `python scripts/filter.py`

### `scripts/cloud_filter.py` *(Cloud/noise reducer)*
- **Does**: multi‑stage cleaning (center radius, local density, composite score), optional **HSV sky removal** and **SOR**; supports picking any `.ply` inside an export folder  
- **Where**: reads any `.ply` in `outputs/exports/<…>/`, writes `<mode>_cloudfree.ply`  
- **Run**: `python scripts/cloud_filter.py`

### `scripts/mesh.py`
- **Does**: Poisson mesh + density crop, optional simple & **adaptive** smoothing; optional cloud‑cleanup for “high & low‑density sky” vertices; final mesh cleanup (non‑manifold/duplicates)  
- **Where**: reads any `.ply` in `outputs/exports/<…>/`, writes `mesh.ply` to the same folder  
- **Run**: `python scripts/mesh.py`

### `scripts/run.py` *(optional launcher)*
- **Does**: prints a command guide and provides a small menu to run the common flows  
- **Run**: `python scripts/run.py`

---

## Tips for Quality / Speed

- **Frames**: A good `NUM_FRAMES_TARGET` (e.g., 300–800) balances quality/time; more frames → better poses but longer training.
- **Training iters**: 50k–80k for fast demos; 150k–300k for higher quality if time allows.
- **Export points**: Heavier clouds (4–8M points) mesh better but cost more RAM/time.
- **Filtering**: Start with a **light** mode; only go **hard**/**ultra** if you still see floaters/sky.
- **Meshing presets**: Try `balanced` → `crisp` → `high_detail`; increase Poisson `depth` gradually.

---

## Troubleshooting

- **`ns-*` not found** → Ensure Nerfstudio is installed and its CLIs are on PATH (inside the venv).  
- **FFmpeg missing** → Install FFmpeg and add it to PATH; some videos won’t decode correctly without it.  
- **Wrong folders** → If a script looks in `output/...` instead of `outputs/...`, edit its `PROJECT_DIR`/`OUTPUT(S)_DIR` constants at the top.  
- **Mesh has holes or waves** → Raise Poisson `depth` one step; try `balanced`/`crisp`; reduce smoothing or use adaptive smoothing only.
- **Exported cloud is empty** → Check training quality, try a later checkpoint, or export with more points.
