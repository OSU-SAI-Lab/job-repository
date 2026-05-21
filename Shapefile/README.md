# Custom Shapefile (Weed / Residue)

Build geospatial heatmaps and farm-boundary shapefiles from Harvest-style JSON: either weed/detection bounding boxes plus EXIF GPS (`weed` mode), or residue sample points with lat/lon and residue fraction (`residue` mode). Parallel grid generation uses multiple CPU cores.

## Files

| File | Description |
|---|---|
| `Custom-shapefile.py` | CLI entry point for both modes |
| `Custom-shapefile.def` | Apptainer/Singularity definition |
| `environment.yml` | Conda environment (GeoPandas stack + exifread) |

## What It Does

### Weed mode (`--mode weed`)

1. Reads an inference JSON with records that include image filename (`img`), `bounding_box`, and `label`
2. Loads GPS from each image’s EXIF in `--imagedir` and projects each bbox to WGS84 using configurable FOV footprint and image dimensions
3. Writes an enriched `result.json` in `--outdir` (with lat/lon and bbox corners per detection)
4. For each distinct `label`, builds a feet-based grid over the detection extent, counts points per cell, maps counts to spray decisions, merges polygons, and writes a heatmap shapefile plus a matching farm-boundary polygon

Also writes `missing_images.txt` and `no_gps_data.txt` when some inputs are skipped.

### Residue mode (`--mode residue`)

1. Reads JSON containing points with `lat`, `lon`, and `residue_fraction` (or `residue_frac`)
2. Tiles the extent with a feet-based grid, averages residue fraction per cell, assigns residue levels, unions by level, and writes one stem-named heatmap shapefile plus farm boundary

Both modes name output folders using filesystem-safe stems derived from labels or the input JSON basename.

## Usage

**Weed**

```bash
python Custom-shapefile.py \
  --mode weed \
  --json /path/to/inference.json \
  --imagedir /path/to/images \
  --outdir /path/to/heatmap_output \
  --grid-width 30 \
  --grid-height 30 \
  --spray-mode binary
```

**Residue**

```bash
python Custom-shapefile.py \
  --mode residue \
  --json /path/to/all_results.json \
  --outdir /path/to/heatmap_output \
  --grid-width 15 \
  --grid-height 15
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--mode` | Yes | `weed` or `residue` |
| `--json` | Yes | Input JSON path |
| `--outdir` | Yes | Output directory (shapefiles + sidecar lists / JSON) |
| `--imagedir` | Weed only | Directory of source images (EXIF GPS read from files named in JSON `img`) |
| `--grid-width` | No | Cell width in feet (default `30`) |
| `--grid-height` | No | Cell height in feet (default `30`) |
| `--spray-mode` | No (weed) | `binary` or `custom` (default `binary`) |
| `--spray-levels` | No | Custom mode: names for spray bands |
| `--spray-thresholds` | No | Custom mode: integer thresholds (must match `--spray-levels` count) |
| `--fov-width-ft` | No | Weed: ground footprint width at altitude in feet (default `29`) |
| `--fov-height-ft` | No | Weed: ground footprint height in feet (default `22`) |
| `--gps-offset-x-ft` | No | Weed: east–west GPS vs optical center offset (feet) |
| `--gps-offset-y-ft` | No | Weed: north–south GPS vs optical center offset (feet) |
| `--image-width-px` | No | Weed: image width for bbox math (default `4000`) |
| `--image-height-px` | No | Weed: image height for bbox math (default `3000`) |

## Weed JSON expectations

Each usable record should reference an existing file under `--imagedir` via `img`, include a four-number `bounding_box` `[x1, y1, x2, y2]` in image pixels, and a `label` for per-label heatmaps.

## Residue JSON expectations

Records must include numeric `lat` and `lon`. Residue is read from `residue_fraction` or `residue_frac` (defaults to `0` if missing). Top-level JSON can be a list, or an object with `results`, `data`, `records`, or `items` as a list.

## Output layout (examples)

**Weed** — under `--outdir`:

- `result.json` — enriched detections
- `missing_images.txt`, `no_gps_data.txt` — optional diagnostics
- `{label}_shapefile/{label}_heatmap.shp` — per-label heatmap
- `{label}_farm_boundary/{label}_farm_boundary.shp` — extent polygon

**Residue** — under `--outdir`:

- `{stem}_shapefile/{stem}_heatmap.shp`
- `{stem}_farm_boundary/{stem}_farm_boundary.shp`

(`{stem}` comes from the input JSON filename.)

## Build the Container

```bash
apptainer build custom-shapefile.sif Custom-shapefile.def
```

## Harvest Integration

Typical workflow inputs:

- **Weed:** `--json`, `--imagedir`, `--outdir`, plus grid and FOV parameters as needed for your flight/camera model.
- **Residue:** `--json`, `--outdir`, grid sizing.

`--mode` selects which pipeline runs.
