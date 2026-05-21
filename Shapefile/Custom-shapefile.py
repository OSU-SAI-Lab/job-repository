#!/usr/bin/env python3
"""
Unified shapefile generator for Harvest (weed / detection + corn residue).
- weed: inference JSON + image dir + EXIF + bbox → per-label heatmaps + farm boundary
- residue: JSON with lat/lon/residue_fraction → one stem-named layer + farm boundary

Run examples:
  python shapefile_generator.py --mode weed --json /result.json --imagedir /images \\
    --outdir /output/heatmap_output --grid-width 30 --grid-height 30 --spray-mode binary

  python shapefile_generator.py --mode residue --json /data/all_results.json \\
    --outdir /output/heatmap_output --grid-width 15 --grid-height 15
"""

from __future__ import annotations

import argparse
import json
import os
import re
from multiprocessing import Pool

import exifread
import geopandas as gpd
import numpy as np
import shapely
from pyproj import Geod
from shapely.geometry import MultiPolygon, Point, box

# ---------------------------------------------------------------------------
# Defaults (match previous hardcoded behavior)
# ---------------------------------------------------------------------------
HPC_CORES = 10

DEFAULT_FOV_WIDTH_FT = 29.0
DEFAULT_FOV_HEIGHT_FT = 22.0
DEFAULT_GPS_OFFSET_X_FT = 0.0
DEFAULT_GPS_OFFSET_Y_FT = 0.0
DEFAULT_IMAGE_WIDTH_PX = 4000
DEFAULT_IMAGE_HEIGHT_PX = 3000

# Residue level thresholds (fraction 0–1), unchanged from your residue script
RESIDUE_LEVELS = [
    ("Very_Low", 0.2),
    ("Low", 0.4),
    ("Medium", 0.6),
    ("High", 0.8),
    ("Very_High", 1.0),
]


# =============================================================================
# GPS / EXIF (weed mode)
# =============================================================================
def dms_to_decimal(dms, ref):
    try:
        d, m, s = (float(x) for x in dms)
        return (d + m / 60 + s / 3600) * (-1 if ref in ("S", "W") else 1)
    except Exception:
        return None


def ddm_to_decimal(ddm, ref):
    try:
        d, m = (float(x) for x in ddm)
        return (d + m / 60) * (-1 if ref in ("S", "W") else 1)
    except Exception:
        return None


def parse_gps_coordinates(tags):
    try:
        if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
            lat = dms_to_decimal(
                tags["GPS GPSLatitude"].values, str(tags["GPS GPSLatitudeRef"])
            )
            lon = dms_to_decimal(
                tags["GPS GPSLongitude"].values, str(tags["GPS GPSLongitudeRef"])
            )
            if lat is not None and lon is not None:
                return lat, lon

        if "GPS GPSLatitudeDDM" in tags and "GPS GPSLongitudeDDM" in tags:
            lat = ddm_to_decimal(
                tags["GPS GPSLatitudeDDM"].values, str(tags["GPS GPSLatitudeRef"])
            )
            lon = ddm_to_decimal(
                tags["GPS GPSLongitudeDDM"].values, str(tags["GPS GPSLongitudeRef"])
            )
            if lat is not None and lon is not None:
                return lat, lon

        if "GPS GPSLatitudeDecimal" in tags and "GPS GPSLongitudeDecimal" in tags:
            lat = float(tags["GPS GPSLatitudeDecimal"].values[0])
            lon = float(tags["GPS GPSLongitudeDecimal"].values[0])
            if str(tags.get("GPS GPSLatitudeRef", "N")) == "S":
                lat = -lat
            if str(tags.get("GPS GPSLongitudeRef", "E")) == "W":
                lon = -lon
            return lat, lon

        if "GPS GPSLatitudeAlt" in tags and "GPS GPSLongitudeAlt" in tags:
            lat = dms_to_decimal(
                tags["GPS GPSLatitudeAlt"].values, str(tags["GPS GPSLatitudeRef"])
            )
            lon = ddm_to_decimal(
                tags["GPS GPSLongitudeAlt"].values, str(tags["GPS GPSLongitudeRef"])
            )
            if lat is not None and lon is not None:
                return lat, lon

        if "GPS GPSLatitude" in tags:
            lat_values = tags["GPS GPSLatitude"].values
            lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
            if len(lat_values) == 3:
                lat = dms_to_decimal(lat_values, lat_ref)
            elif len(lat_values) == 2:
                lat = ddm_to_decimal(lat_values, lat_ref)
            elif len(lat_values) == 1:
                lat = float(lat_values[0])
                if lat_ref == "S":
                    lat = -lat
            else:
                return None

            if "GPS GPSLongitude" in tags:
                lon_values = tags["GPS GPSLongitude"].values
                lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
                if len(lon_values) == 3:
                    lon = dms_to_decimal(lon_values, lon_ref)
                elif len(lon_values) == 2:
                    lon = ddm_to_decimal(lon_values, lon_ref)
                elif len(lon_values) == 1:
                    lon = float(lon_values[0])
                    if lon_ref == "W":
                        lon = -lon
                else:
                    return None

                if lat is not None and lon is not None:
                    return lat, lon

        return None
    except Exception as e:
        print(f"GPS parsing error: {e}")
        return None


def get_gps_enhanced(path):
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)

        coords = parse_gps_coordinates(tags)
        if coords:
            return coords

        print(f"Standard GPS parsing failed for {path}, attempting fallback...")
        gps_tags = {k: v for k, v in tags.items() if "GPS" in k}
        if gps_tags:
            print(f"Available GPS tags: {list(gps_tags.keys())}")
        return None
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None


def calculate_bounding_box_coordinates(
    image_gps_lat,
    image_gps_lon,
    bounding_box,
    image_width_px,
    image_height_px,
    image_width_ft,
    image_height_ft,
    gps_offset_x_ft,
    gps_offset_y_ft,
):
    """Assumes GPS and camera optical center correspond to the center of the image."""
    x1, y1, x2, y2 = bounding_box

    image_width_m = image_width_ft / 3.28084
    image_height_m = image_height_ft / 3.28084

    meters_per_pixel_x = image_width_m / image_width_px
    meters_per_pixel_y = image_height_m / image_height_px

    bbox_center_x_px = (x1 + x2) / 2 - image_width_px / 2
    bbox_center_y_px = (y1 + y2) / 2 - image_height_px / 2

    bbox_center_x_m = bbox_center_x_px * meters_per_pixel_x
    bbox_center_y_m = bbox_center_y_px * meters_per_pixel_y

    bbox_center_x_m += gps_offset_x_ft / 3.28084
    bbox_center_y_m += gps_offset_y_ft / 3.28084

    bbox_width_m = (x2 - x1) * meters_per_pixel_x
    bbox_height_m = (y2 - y1) * meters_per_pixel_y

    lat_degrees_per_meter = 1.0 / 111320.0
    lon_degrees_per_meter = 1.0 / (111320.0 * np.cos(np.radians(image_gps_lat)))

    bbox_lat = image_gps_lat + bbox_center_y_m * lat_degrees_per_meter
    bbox_lon = image_gps_lon + bbox_center_x_m * lon_degrees_per_meter

    half_width_deg = (bbox_width_m / 2) * lon_degrees_per_meter
    half_height_deg = (bbox_height_m / 2) * lat_degrees_per_meter

    lat1 = bbox_lat - half_height_deg
    lon1 = bbox_lon - half_width_deg
    lat2 = bbox_lat + half_height_deg
    lon2 = bbox_lon + half_width_deg

    return lat1, lon1, lat2, lon2


def safe_folder(name):
    name = name.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", name)


# =============================================================================
# Weed: spray grid
# =============================================================================
def get_spray_decision_and_obj_id(count, spray_levels):
    if count == 0:
        return "No", 0
    for idx, (name, threshold) in enumerate(spray_levels):
        if count <= threshold:
            return name, idx + 1
    return spray_levels[-1][0], len(spray_levels)


def get_target_rate(count):
    return 15 if count > 0 else 0


def process_polygon_chunk_weed(args):
    chunk_indices, minx, miny, dx, dy, group_gdf, spray_levels = args
    results = []

    for i, j in chunk_indices:
        cell = box(
            minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy
        )
        count = int(group_gdf.covered_by(cell).sum())
        spray_decision, obj_id = get_spray_decision_and_obj_id(count, spray_levels)
        tgt_rate = get_target_rate(count)

        results.append(
            {
                "geometry": cell,
                "crop_count": count,
                "spray_deci": spray_decision,
                "Obj__Id": obj_id,
                "Tgt_Rate_g": tgt_rate,
            }
        )

    return results


def create_polygons_parallel_weed(ncol, nrow, minx, miny, dx, dy, group_gdf, spray_levels):
    print(f"  - Creating {ncol * nrow} polygons using {HPC_CORES} cores (parallel)")

    total_polygons = ncol * nrow
    chunk_size = max(1, total_polygons // (HPC_CORES * 2))

    chunks = [(i, j) for i in range(ncol) for j in range(nrow)]

    chunk_args = []
    for i in range(0, len(chunks), chunk_size):
        chunk_indices = chunks[i : i + chunk_size]
        chunk_args.append(
            (chunk_indices, minx, miny, dx, dy, group_gdf, spray_levels)
        )

    print(f"  - Distributed {total_polygons} polygons into {len(chunk_args)} chunks")

    with Pool(processes=HPC_CORES) as pool:
        chunk_results = pool.map(process_polygon_chunk_weed, chunk_args)

    all_results = []
    for chunk_result in chunk_results:
        all_results.extend(chunk_result)

    print(f"  - Successfully created {len(all_results)} polygons using {HPC_CORES} cores")
    return all_results


def create_multipolygon_groups(grid_gdf):
    print("  - Creating MultiPolygon groups by spray decision")

    grouped_results = []
    for spray_decision in grid_gdf["spray_deci"].unique():
        decision_group = grid_gdf[grid_gdf["spray_deci"] == spray_decision]
        if len(decision_group) == 0:
            continue

        geometries = decision_group.geometry.tolist()
        total_count = decision_group["crop_count"].sum()
        obj_id = decision_group["Obj__Id"].iloc[0]
        tgt_rate = decision_group["Tgt_Rate_g"].iloc[0]

        if len(geometries) == 1:
            multipolygon = geometries[0]
        else:
            try:
                combined_geom = shapely.unary_union(geometries)
                if combined_geom.geom_type == "Polygon":
                    multipolygon = combined_geom
                elif combined_geom.geom_type == "MultiPolygon":
                    multipolygon = combined_geom
                else:
                    multipolygon = MultiPolygon(geometries)
            except Exception as e:
                print(f"    - Error combining {spray_decision} polygons: {e}")
                multipolygon = MultiPolygon(geometries)

        grouped_results.append(
            {
                "geometry": multipolygon,
                "crop_count": int(total_count),
                "spray_deci": spray_decision,
                "Obj__Id": obj_id,
                "Tgt_Rate_g": tgt_rate,
                "polygon_count": len(geometries),
            }
        )

    print(f"  - Created {len(grouped_results)} MultiPolygon groups")
    for result in grouped_results:
        print(
            f"    - {result['spray_deci']}: {result['polygon_count']} polygons combined, "
            f"{result['crop_count']} total instances"
        )

    return gpd.GeoDataFrame(grouped_results, crs="EPSG:4326")


def merge_single_level_parallel(args):
    spray_decision, level_gdf = args

    if len(level_gdf) == 0:
        return []

    try:
        geometries = level_gdf.geometry.tolist()
        merged_geometry = shapely.union_all(geometries)

        if merged_geometry.geom_type == "Polygon":
            return [
                {
                    "geometry": merged_geometry,
                    "crop_count": int(level_gdf["crop_count"].sum()),
                    "spray_deci": spray_decision,
                    "Obj__Id": level_gdf["Obj__Id"].iloc[0],
                    "Tgt_Rate_g": level_gdf["Tgt_Rate_g"].iloc[0],
                }
            ]
        if merged_geometry.geom_type == "MultiPolygon":
            results = []
            n = len(merged_geometry.geoms)
            for poly in merged_geometry.geoms:
                results.append(
                    {
                        "geometry": poly,
                        "crop_count": int(level_gdf["crop_count"].sum() / max(n, 1)),
                        "spray_deci": spray_decision,
                        "Obj__Id": level_gdf["Obj__Id"].iloc[0],
                        "Tgt_Rate_g": level_gdf["Tgt_Rate_g"].iloc[0],
                    }
                )
            return results
        return []
    except Exception as e:
        print(f"    - Error merging {spray_decision}: {e}")
        return [
            {
                "geometry": row.geometry,
                "crop_count": row["crop_count"],
                "spray_deci": row["spray_deci"],
                "Obj__Id": row["Obj__Id"],
                "Tgt_Rate_g": row["Tgt_Rate_g"],
            }
            for _, row in level_gdf.iterrows()
        ]


def merge_adjacent_polygons_parallel_weed(grid_gdf):
    print(f"  - Merging polygons using {HPC_CORES} cores (parallel)")

    spray_decisions = grid_gdf["spray_deci"].unique()
    print(f"    - Processing {len(spray_decisions)} spray decisions in parallel")

    level_data = [
        (sd, grid_gdf[grid_gdf["spray_deci"] == sd]) for sd in spray_decisions
    ]

    with Pool(processes=HPC_CORES) as pool:
        level_results = pool.map(merge_single_level_parallel, level_data)

    merged_polygons = []
    merged_crop_counts = []
    merged_spray_decisions = []
    merged_obj_ids = []
    merged_tgt_rates = []

    for level_result in level_results:
        for result in level_result:
            merged_polygons.append(result["geometry"])
            merged_crop_counts.append(result["crop_count"])
            merged_spray_decisions.append(result["spray_deci"])
            merged_obj_ids.append(result["Obj__Id"])
            merged_tgt_rates.append(result["Tgt_Rate_g"])

    print(f"  - Successfully merged using {HPC_CORES} cores")
    print(f"  - Reduced from {len(grid_gdf)} to {len(merged_polygons)} polygons")

    return gpd.GeoDataFrame(
        {
            "crop_count": merged_crop_counts,
            "spray_deci": merged_spray_decisions,
            "Obj__Id": merged_obj_ids,
            "Tgt_Rate_g": merged_tgt_rates,
        },
        geometry=merged_polygons,
        crs="EPSG:4326",
    )


def run_weed(args):
    tile_ft_w = args.grid_width
    tile_ft_h = args.grid_height
    tile_m_w = tile_ft_w / 3.28084
    tile_m_h = tile_ft_h / 3.28084

    fov_w = args.fov_width_ft
    fov_h = args.fov_height_ft
    off_x = args.gps_offset_x_ft
    off_y = args.gps_offset_y_ft
    px_w = args.image_width_px
    px_h = args.image_height_px

    print("=== Weed / detection shapefile (bbox + EXIF) ===")
    print(
        f"Assumes GPS and camera principal point at image center; "
        f"FOV footprint: {fov_w} x {fov_h} ft; offsets (ft): {off_x}, {off_y}; "
        f"image: {px_w}x{px_h} px"
    )
    print(f"Grid: {tile_ft_w} x {tile_ft_h} ft ({tile_m_w:.1f} x {tile_m_h:.1f} m)")

    if args.spray_mode == "binary":
        spray_levels = [("No", 0), ("Yes", 1)]
        print("Spray: binary Yes/No")
    elif args.spray_levels and args.spray_thresholds:
        if len(args.spray_levels) != len(args.spray_thresholds):
            print("Error: spray level names and thresholds must match in count")
            return 1
        spray_levels = list(zip(args.spray_levels, args.spray_thresholds))
        print(f"Spray: custom ({len(spray_levels)} levels)")
    else:
        spray_levels = [("Low", 5), ("Medium", 15), ("High", 30)]
        print("Spray: default Low/Medium/High")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nLoading inference results from: {args.json}")
    with open(args.json, encoding="utf-8") as f:
        raw = json.load(f)

    enriched, missing, no_gps_data = [], [], []
    image_gps_cache = {}

    print(f"Processing {len(raw)} bounding box records...")

    for i, rec in enumerate(raw):
        fname = rec.get("img")
        if not fname:
            continue

        if fname not in image_gps_cache:
            imgpath = os.path.join(args.imagedir, fname)
            if not os.path.isfile(imgpath):
                missing.append(fname)
                continue

            coords = get_gps_enhanced(imgpath)
            if coords:
                image_gps_cache[fname] = coords
            else:
                no_gps_data.append(fname)
                continue

        if fname in image_gps_cache:
            image_gps_lat, image_gps_lon = image_gps_cache[fname]
            bounding_box = rec.get("bounding_box", [])
            if len(bounding_box) == 4:
                lat1, lon1, lat2, lon2 = calculate_bounding_box_coordinates(
                    image_gps_lat,
                    image_gps_lon,
                    bounding_box,
                    px_w,
                    px_h,
                    fov_w,
                    fov_h,
                    off_x,
                    off_y,
                )

                bbox_center_lat = (lat1 + lat2) / 2
                bbox_center_lon = (lon1 + lon2) / 2

                rec2 = rec.copy()
                rec2["latitude"] = bbox_center_lat
                rec2["longitude"] = bbox_center_lon
                rec2["bbox_lat1"] = lat1
                rec2["bbox_lon1"] = lon1
                rec2["bbox_lat2"] = lat2
                rec2["bbox_lon2"] = lon2
                enriched.append(rec2)

        if (i + 1) % 100 == 0:
            print(f"  - Processed {i + 1}/{len(raw)} bounding boxes...")

    with open(os.path.join(args.outdir, "result.json"), "w", encoding="utf-8") as out:
        json.dump(enriched, out, indent=2)

    if missing:
        with open(os.path.join(args.outdir, "missing_images.txt"), "w") as f:
            f.write("\n".join(missing))
        print(f"Warning: {len(missing)} images missing from directory")

    if no_gps_data:
        with open(os.path.join(args.outdir, "no_gps_data.txt"), "w") as f:
            f.write("\n".join(no_gps_data))
        print(f"Warning: {len(no_gps_data)} images missing GPS data")

    if not enriched:
        print("Error: No bounding boxes with GPS data found")
        return 1

    print(f"Successfully processed {len(enriched)} bounding boxes with GPS data")

    gdf_all = gpd.GeoDataFrame(
        enriched,
        geometry=[Point(r["longitude"], r["latitude"]) for r in enriched],
        crs="EPSG:4326",
    )

    for label, group in gdf_all.groupby("label"):
        if group.empty:
            continue

        print(f"\nProcessing label: {label}")
        print(f"  - Data points: {len(group)}")

        minx, miny, maxx, maxy = group.total_bounds
        width_deg = maxx - minx
        height_deg = maxy - miny

        minx -= 0.1 * width_deg
        maxx += 0.1 * width_deg
        miny -= 0.1 * height_deg
        maxy += 0.1 * height_deg

        geod = Geod(ellps="WGS84")
        _, _, field_width_m = geod.inv(minx, miny, maxx, miny)
        _, _, field_height_m = geod.inv(minx, miny, minx, maxy)

        print(f"  - Field dimensions: {field_width_m:.1f}m x {field_height_m:.1f}m")

        if tile_m_w > field_width_m or tile_m_h > field_height_m:
            print("  - Grid larger than field; single cell.")
            ncol, nrow = 1, 1
            dx = maxx - minx
            dy = maxy - miny
        else:
            ncol = max(1, int(np.ceil(field_width_m / tile_m_w)))
            nrow = max(1, int(np.ceil(field_height_m / tile_m_h)))
            dx = (maxx - minx) / ncol
            dy = (maxy - miny) / nrow

        print(f"  - Grid: {ncol} x {nrow} = {ncol * nrow} polygons")
        polygon_results = create_polygons_parallel_weed(
            ncol, nrow, minx, miny, dx, dy, group, spray_levels
        )

        if not polygon_results:
            print(f"  - Warning: No valid polygons for label '{label}'")
            continue

        grid = gpd.GeoDataFrame(
            {
                "geometry": [r["geometry"] for r in polygon_results],
                "crop_count": [r["crop_count"] for r in polygon_results],
                "spray_deci": [r["spray_deci"] for r in polygon_results],
                "Obj__Id": [r["Obj__Id"] for r in polygon_results],
                "Tgt_Rate_g": [r["Tgt_Rate_g"] for r in polygon_results],
            },
            crs="EPSG:4326",
        )

        grid = merge_adjacent_polygons_parallel_weed(grid)
        grid = create_multipolygon_groups(grid)

        print("  - Cleaning geometries (buffer 0)")
        grid["geometry"] = grid["geometry"].buffer(0)

        folder = os.path.join(args.outdir, safe_folder(label) + "_shapefile")
        os.makedirs(folder, exist_ok=True)
        output_file = os.path.join(folder, f"{safe_folder(label)}_heatmap.shp")
        grid.to_file(output_file)
        print(f"  - Wrote {output_file}")

        farm_boundary = box(minx, miny, maxx, maxy)
        farm_gdf = gpd.GeoDataFrame(
            {
                "geometry": [farm_boundary],
                "label": [label],
                "field_width_m": [field_width_m],
                "field_height_m": [field_height_m],
            },
            crs="EPSG:4326",
        )
        farm_gdf["geometry"] = farm_gdf["geometry"].buffer(0)
        farm_folder = os.path.join(args.outdir, safe_folder(label) + "_farm_boundary")
        os.makedirs(farm_folder, exist_ok=True)
        farm_out = os.path.join(farm_folder, f"{safe_folder(label)}_farm_boundary.shp")
        farm_gdf.to_file(farm_out)
        print(f"  - Wrote {farm_out}")

    print("\n=== Weed mode complete ===")
    return 0


# =============================================================================
# Residue: point JSON + fraction
# =============================================================================
def get_residue_level_and_obj_id(avg_residue, point_count, residue_levels):
    if point_count == 0:
        return "No_Data", 0
    for idx, (name, threshold) in enumerate(residue_levels):
        if avg_residue <= threshold:
            return name, idx + 1
    return residue_levels[-1][0], len(residue_levels)


def process_polygon_chunk_residue(args):
    chunk_indices, minx, miny, dx, dy, group_gdf, residue_levels = args
    results = []
    residue_col = "residue_fraction"

    for i, j in chunk_indices:
        cell = box(
            minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy
        )
        mask = group_gdf.covered_by(cell)
        points_in_cell = group_gdf[mask]

        if len(points_in_cell) == 0:
            avg_residue = 0.0
            point_count = 0
        else:
            point_count = len(points_in_cell)
            avg_residue = float(points_in_cell[residue_col].mean())

        residue_level, obj_id = get_residue_level_and_obj_id(
            avg_residue, point_count, residue_levels
        )
        avg_pct = avg_residue * 100.0 if point_count > 0 else 0.0

        results.append(
            {
                "geometry": cell,
                "point_count": point_count,
                "avg_residue": avg_residue,
                "avg_residue_pct": avg_pct,
                "residue_deci": residue_level,
                "Obj__Id": obj_id,
            }
        )

    return results


def create_polygons_parallel_residue(
    ncol, nrow, minx, miny, dx, dy, group_gdf, residue_levels
):
    print(f"Phase 1: Creating {ncol * nrow} polygons using {HPC_CORES} cores...")

    total_polygons = ncol * nrow
    chunk_size = max(1, total_polygons // (HPC_CORES * 2))
    chunks = [(i, j) for i in range(ncol) for j in range(nrow)]
    chunk_args = []
    for i in range(0, len(chunks), chunk_size):
        chunk_indices = chunks[i : i + chunk_size]
        chunk_args.append(
            (chunk_indices, minx, miny, dx, dy, group_gdf, residue_levels)
        )

    all_results = []
    with Pool(processes=HPC_CORES) as pool:
        for chunk_result in pool.imap(process_polygon_chunk_residue, chunk_args):
            all_results.extend(chunk_result)

    print(f"  Phase 1 complete: {len(all_results)} polygons")
    return all_results


def union_by_residue_level(grid_gdf):
    print("Phase 2: Union by residue level...")
    results = []
    for res_deci in grid_gdf["res_deci"].unique():
        group = grid_gdf[grid_gdf["res_deci"] == res_deci]
        if len(group) == 0:
            continue
        geometries = group.geometry.tolist()
        try:
            merged = shapely.union_all(geometries)
        except Exception as e:
            print(f"  union_all failed ({e}), unary_union")
            merged = shapely.unary_union(geometries)

        total_pts = int(group["pt_count"].sum())
        avg_res = float(group["avg_res"].mean())
        res_pct = avg_res * 100.0
        obj_id = group["Obj__Id"].iloc[0]

        results.append(
            {
                "geometry": merged,
                "pt_count": total_pts,
                "avg_res": avg_res,
                "res_pct": res_pct,
                "res_deci": res_deci,
                "Obj__Id": obj_id,
            }
        )
        print(f"  {res_deci}: {len(geometries)} cells -> 1 feature, res_pct~{res_pct:.2f}%")

    print(f"  Phase 2: {len(grid_gdf)} cells -> {len(results)} features")
    return gpd.GeoDataFrame(results, crs="EPSG:4326")


def _to_float(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        return float(s)
    try:
        return float(val)
    except Exception:
        return None


def load_json_records_residue(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("results", "data", "records", "items"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        else:
            records = [data]
    else:
        raise ValueError("JSON must be a list or object with a list field")

    result = []
    for rec in records:
        lat = _to_float(rec.get("lat"))
        lon = _to_float(rec.get("lon"))
        frac = rec.get("residue_fraction")
        if frac is None:
            frac = rec.get("residue_frac")
        frac = _to_float(frac)

        if lat is None or lon is None:
            continue
        if frac is None:
            frac = 0.0
        result.append(
            {
                "latitude": lat,
                "longitude": lon,
                "residue_fraction": frac,
            }
        )
    return result


def run_residue(args):
    tile_ft_w = args.grid_width
    tile_ft_h = args.grid_height
    tile_m_w = tile_ft_w / 3.28084
    tile_m_h = tile_ft_h / 3.28084

    stem = os.path.splitext(os.path.basename(args.json))[0]
    stem_safe = safe_folder(stem) or "output"

    print("=== Residue shapefile (lat/lon + residue_fraction) ===")
    print(f"Stem: {stem_safe}; grid: {tile_ft_w} x {tile_ft_h} ft")

    records = load_json_records_residue(args.json)
    if not records:
        print("Error: no valid records with lat/lon")
        return 1

    print(f"Loaded {len(records)} points")

    os.makedirs(args.outdir, exist_ok=True)

    gdf = gpd.GeoDataFrame(
        records,
        geometry=[Point(r["longitude"], r["latitude"]) for r in records],
        crs="EPSG:4326",
    )

    minx, miny, maxx, maxy = gdf.total_bounds
    geod = Geod(ellps="WGS84")
    _, _, field_width_m = geod.inv(minx, miny, maxx, miny)
    _, _, field_height_m = geod.inv(minx, miny, minx, maxy)

    print(f"Extent: lon [{minx:.6f}, {maxx:.6f}], lat [{miny:.6f}, {maxy:.6f}]")
    print(f"Field: {field_width_m:.1f}m x {field_height_m:.1f}m")

    if tile_m_w > field_width_m or tile_m_h > field_height_m:
        print("Grid larger than field; single cell.")
        ncol, nrow = 1, 1
        dx = maxx - minx
        dy = maxy - miny
    else:
        ncol = max(1, int(np.ceil(field_width_m / tile_m_w)))
        nrow = max(1, int(np.ceil(field_height_m / tile_m_h)))
        dx = (maxx - minx) / ncol
        dy = (maxy - miny) / nrow

    print(f"Grid: {ncol} x {nrow} = {ncol * nrow} cells")

    polygon_results = create_polygons_parallel_residue(
        ncol, nrow, minx, miny, dx, dy, gdf, RESIDUE_LEVELS
    )
    if not polygon_results:
        print("Error: no polygons created")
        return 1

    grid = gpd.GeoDataFrame(
        {
            "geometry": [r["geometry"] for r in polygon_results],
            "pt_count": [r["point_count"] for r in polygon_results],
            "avg_res": [r["avg_residue"] for r in polygon_results],
            "res_pct": [r["avg_residue_pct"] for r in polygon_results],
            "res_deci": [r["residue_deci"] for r in polygon_results],
            "Obj__Id": [r["Obj__Id"] for r in polygon_results],
        },
        crs="EPSG:4326",
    )

    grid["geometry"] = grid["geometry"].buffer(0)
    grid = union_by_residue_level(grid)

    folder = os.path.join(args.outdir, f"{stem_safe}_shapefile")
    os.makedirs(folder, exist_ok=True)
    heatmap_shp = os.path.join(folder, f"{stem_safe}_heatmap.shp")
    grid.to_file(heatmap_shp)
    print(f"Wrote {heatmap_shp}")

    farm_boundary = box(minx, miny, maxx, maxy)
    farm_gdf = gpd.GeoDataFrame(
        {
            "geometry": [farm_boundary],
            "fld_width": [field_width_m],
            "fld_height": [field_height_m],
        },
        crs="EPSG:4326",
    )
    farm_gdf["geometry"] = farm_gdf["geometry"].buffer(0)
    farm_folder = os.path.join(args.outdir, f"{stem_safe}_farm_boundary")
    os.makedirs(farm_folder, exist_ok=True)
    farm_out = os.path.join(farm_folder, f"{stem_safe}_farm_boundary.shp")
    farm_gdf.to_file(farm_out)
    print(f"Wrote {farm_out}")

    print("=== Residue mode complete ===")
    return 0


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(description="Unified Harvest shapefile generator")
    p.add_argument("--mode", choices=("weed", "residue"), required=True)

    p.add_argument("--json", required=True, help="Input JSON path")
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory (e.g. .../heatmap_output)",
    )
    p.add_argument("--grid-width", type=int, default=30, help="Cell width (feet)")
    p.add_argument("--grid-height", type=int, default=30, help="Cell height (feet)")

    p.add_argument(
        "--imagedir",
        default=None,
        help="Image directory (required for --mode weed)",
    )

    p.add_argument(
        "--spray-mode",
        choices=("binary", "custom"),
        default="binary",
        help="Weed mode only",
    )
    p.add_argument(
        "--spray-levels",
        nargs="+",
        default=None,
        help="Weed custom mode: level names",
    )
    p.add_argument(
        "--spray-thresholds",
        nargs="+",
        type=int,
        default=None,
        help="Weed custom mode: thresholds",
    )

    p.add_argument(
        "--fov-width-ft",
        type=float,
        default=DEFAULT_FOV_WIDTH_FT,
        help="Weed: ground footprint width (ft) at altitude",
    )
    p.add_argument(
        "--fov-height-ft",
        type=float,
        default=DEFAULT_FOV_HEIGHT_FT,
        help="Weed: ground footprint height (ft)",
    )
    p.add_argument(
        "--gps-offset-x-ft",
        type=float,
        default=DEFAULT_GPS_OFFSET_X_FT,
        help="Weed: east/west offset of GPS vs image center (ft)",
    )
    p.add_argument(
        "--gps-offset-y-ft",
        type=float,
        default=DEFAULT_GPS_OFFSET_Y_FT,
        help="Weed: north/south offset of GPS vs image center (ft)",
    )
    p.add_argument(
        "--image-width-px",
        type=int,
        default=DEFAULT_IMAGE_WIDTH_PX,
        help="Weed: image width in pixels (bbox coords)",
    )
    p.add_argument(
        "--image-height-px",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT_PX,
        help="Weed: image height in pixels (bbox coords)",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "weed":
        if not args.imagedir:
            parser.error("--mode weed requires --imagedir")
        return run_weed(args)
    return run_residue(args)


if __name__ == "__main__":
    raise SystemExit(main())