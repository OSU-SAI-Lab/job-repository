# Image Cropping

Crop a shared region of interest from all images in a source directory using percentage-based ROI coordinates. This is the job used by the Harvest `Crop Images` workflow step.

## Files

| File | Description |
|---|---|
| `Extract_perspective.py` | CLI entry point for cropping images |
| `extract_perspective.def` | Apptainer/Singularity definition |
| `requirements.txt` | Python dependencies for the container |

## What It Does

1. Reads a source image directory
2. Applies one ROI to every supported image
3. Saves cropped images under the destination directory with the prefix `rectified_`
4. Preserves EXIF metadata for JPEG/PNG when available
5. Preserves GPS metadata for TIFF images when available

The job handles both:
- images directly inside the source directory
- one level of subdirectories inside the source directory

## Supported Formats

- `.JPEG`
- `.JPG`
- `.PNG`
- `.TIF`
- `.TIFF`

## Usage

```bash
python Extract_perspective.py \
  --input_path /path/to/source/images \
  --output_path /path/to/output/images \
  --points 10,10:90,10:90,90:10,90
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--input_path` | Yes | Path to the source image directory |
| `--output_path` | Yes | Path to the destination image directory |
| `--points` | Yes | ROI points as `x1,y1:x2,y2:x3,y3:x4,y4` in percentages |

## ROI Format

The crop points are percentage coordinates relative to image width and height.

Example:

```text
10,10:90,10:90,90:10,90
```

This means:
- top-left = `(10%, 10%)`
- top-right = `(90%, 10%)`
- bottom-right = `(90%, 90%)`
- bottom-left = `(10%, 90%)`

The script converts these percentages into a rectangular pixel crop for each image.

## Output Layout

For a source directory like:

```text
input/
  image1.JPG
  image2.PNG
  batch_a/
    image3.TIF
```

The output will look like:

```text
output/
  rectified_image1.JPG
  rectified_image2.PNG
  batch_a/
    rectified_image3.TIF
```

## Build the Container

```bash
apptainer build extract-perspective.sif extract_perspective.def
```

## Harvest Integration

This job matches the Harvest crop workflow interface:

- `--input_path`
- `--output_path`
- `--points`

Those values are submitted from the Harvest UI after the user selects an ROI on the sample images.
