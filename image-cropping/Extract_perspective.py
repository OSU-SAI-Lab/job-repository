import argparse
from pathlib import Path

import piexif
from PIL import Image


SUPPORTED_FORMATS = (".JPEG", ".JPG", ".PNG", ".TIF", ".TIFF")


def get_image(path):
    return Image.open(path)


def get_exif_bytes(image_path):
    try:
        exif_dict = piexif.load(str(image_path))
        return piexif.dump(exif_dict)
    except piexif.InvalidImageDataError:
        print(f"No valid EXIF data found for {image_path}")
        return None
    except Exception as exc:
        print(f"Error loading EXIF from {image_path}: {exc}")
        return None


def get_coord_tuple(coord):
    coord_str = str(coord)
    if "." in coord_str:
        whole, frac = coord_str.split(".")
        denominator = 10 ** len(frac)
        numerator = int(whole + frac)
    else:
        numerator = int(coord_str)
        denominator = 1
    return (numerator, denominator)


def convert_gps_coordinates_into_tuple(coords_tuple):
    return (
        get_coord_tuple(coords_tuple[0]),
        get_coord_tuple(coords_tuple[1]),
        get_coord_tuple(coords_tuple[2]),
    )


def get_gps_ifd(image):
    exif_dict = image.getexif()
    gps_info = {}
    try:
        gps_ifd = exif_dict.get_ifd(0x8825)
        if gps_ifd:
            gps_info[piexif.GPSIFD.GPSVersionID] = (2, 0, 0, 0)
            if 1 in gps_ifd:
                gps_info[piexif.GPSIFD.GPSLatitudeRef] = gps_ifd[1].encode("utf-8")
            if 2 in gps_ifd:
                gps_info[piexif.GPSIFD.GPSLatitude] = convert_gps_coordinates_into_tuple(
                    gps_ifd[2]
                )
            if 3 in gps_ifd:
                gps_info[piexif.GPSIFD.GPSLongitudeRef] = gps_ifd[3].encode("utf-8")
            if 4 in gps_ifd:
                gps_info[piexif.GPSIFD.GPSLongitude] = convert_gps_coordinates_into_tuple(
                    gps_ifd[4]
                )
    except Exception as exc:
        print(f"GPS extraction warning for {image.filename}: {exc}")

    return gps_info


def parse_points_argument(points_argument):
    parsed_points = []
    for point in points_argument.split(":"):
        x_str, y_str = point.split(",")
        parsed_points.append((float(x_str), float(y_str)))

    if len(parsed_points) != 4:
        raise ValueError("Expected four x,y percentage pairs in --points")

    return parsed_points


def parse_percentage_points_to_pixels(points, img_width, img_height):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    left = round((min(xs) / 100.0) * img_width)
    top = round((min(ys) / 100.0) * img_height)
    right = round((max(xs) / 100.0) * img_width)
    bottom = round((max(ys) / 100.0) * img_height)

    left = max(0, min(left, img_width))
    top = max(0, min(top, img_height))
    right = max(0, min(right, img_width))
    bottom = max(0, min(bottom, img_height))

    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop box computed from points {points} for image size "
            f"{img_width}x{img_height}"
        )

    return (left, top, right, bottom)


def crop_image(image, crop_box):
    return image.crop(crop_box)


def save_tiff_with_gps(rectified_image, output_path, gps_ifd):
    rectified_image.save(output_path, tiff=True, quality=95)
    if gps_ifd:
        reopened_image = Image.open(output_path)
        exif_dict = piexif.load(str(output_path))
        exif_dict["GPS"] = gps_ifd
        exif_bytes = piexif.dump(exif_dict)
        reopened_image.save(output_path, exif=exif_bytes, tiff=True, quality=95)


def save_non_tiff_with_exif(rectified_image, output_path, exif_bytes):
    if exif_bytes:
        rectified_image.save(output_path, exif=exif_bytes, quality=95)
    else:
        rectified_image.save(output_path, quality=95)


def process_image(image_path, output_path, points):
    image = get_image(image_path)
    pixel_points = parse_percentage_points_to_pixels(points, image.size[0], image.size[1])
    print(f"Processing {image_path} with crop box {pixel_points}")

    rectified_image = crop_image(image, pixel_points)

    if image_path.suffix.upper() in (".TIF", ".TIFF"):
        gps_ifd = get_gps_ifd(image)
        save_tiff_with_gps(rectified_image, output_path, gps_ifd)
    else:
        exif_bytes = get_exif_bytes(image_path)
        save_non_tiff_with_exif(rectified_image, output_path, exif_bytes)

    print(f"Saved {output_path}")


def iter_supported_images(input_path):
    for entry in sorted(input_path.iterdir()):
        if entry.is_dir():
            for sub_entry in sorted(entry.iterdir()):
                if sub_entry.is_file() and sub_entry.suffix.upper() in SUPPORTED_FORMATS:
                    yield sub_entry, input_path / entry.name
        elif entry.is_file() and entry.suffix.upper() in SUPPORTED_FORMATS:
            yield entry, input_path


def extract_perspective(input_path, output_path, points):
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for image_path, parent_dir in iter_supported_images(input_dir):
        relative_parent = parent_dir.relative_to(input_dir)
        target_dir = output_dir / relative_parent
        target_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"rectified_{image_path.name}"
        output_full_path = target_dir / output_filename
        process_image(image_path, output_full_path, points)
        processed_count += 1

    print(f"Finished processing {processed_count} image(s)")


def main():
    parser = argparse.ArgumentParser(description="Crop images using percentage ROI points")
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the source image directory",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to the destination image directory",
    )
    parser.add_argument(
        "--points",
        required=True,
        help="ROI points as x1,y1:x2,y2:x3,y3:x4,y4 in percentages",
    )
    args = parser.parse_args()

    points = parse_points_argument(args.points)
    extract_perspective(args.input_path, args.output_path, points)


if __name__ == "__main__":
    main()
