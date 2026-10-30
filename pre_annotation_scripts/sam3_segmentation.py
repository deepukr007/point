import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


IMAGES_FOLDER = Path("/home/dk/Projects/point/Pointing_Dataset-main/captures/color")
OUTPUT_FOLDER = Path("/home/dk/Projects/point/output")

OBJECT_NAMES = ["apple","banana","bowl","fork","keys","marker","mug","screw-driver","spoon","tenis-ball",]

TARGET_CLASS = "target_obj"
TARGET_CLASS_ID = 0
NO_POINTING = "no-pointing"


CSV_FIELDS = ["image", "object", "expected", "detected", "scores", "status"]


def class_from_filename(image_path):
    """Return the class token from the file name, or None if it is not a known class."""
    parts = image_path.stem.split("_")
    if len(parts) < 2:
        return None
    token = parts[1]
    return token if token in OBJECT_NAMES else None


def boxes_to_yolo(boxes, class_ids, width, height):
    """xyxy pixel boxes -> 'class cx cy w h' lines (normalized)."""
    lines = []
    for (x1, y1, x2, y2), class_id in zip(boxes, class_ids):
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def masks_to_yolo(masks, class_ids, width, height):
    """Binary masks -> 'class x1 y1 x2 y2 ...' polygon lines (normalized).

    YOLO segmentation expects one polygon per instance, so the largest contour is used.
    """
    lines = []
    for mask, class_id in zip(masks, class_ids):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(contour) < 3:
            continue
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in contour)
        lines.append(f"{class_id} {coords}")
    return lines


def write_lines(path, lines):
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def make_output_dirs(output_folder):
    """ YOLO like directory structure:"""
    dirs = {}
    for kind in ("boxes", "masks"):
        for sub in ("images", "labels", "annotated"):
            d = output_folder / kind / sub
            d.mkdir(parents=True, exist_ok=True)
            dirs[(kind, sub)] = d
        # Class list so the labels can be used directly for YOLO training
        write_lines(output_folder / kind / "classes.txt", [TARGET_CLASS])
        yaml_names = f"  {TARGET_CLASS_ID}: {TARGET_CLASS}"
        (output_folder / kind / "data.yaml").write_text(
            f"path: {(output_folder / kind).resolve()}\ntrain: images\nval: images\nnc: 1\nnames:\n{yaml_names}\n"
        )
    return dirs


def detection_summary(records):
    """Summarize how many images had how many detections, overall and per object.

    records: one dict per image with keys image, object, expected, detected, status.
    Returns the summary as a list of lines.
    """
    failures = [r for r in records if r["status"] == "fail"]
    overall = Counter(r["detected"] for r in records)
    per_object = defaultdict(Counter)
    for r in records:
        per_object[r["object"]][r["detected"]] += 1

    lines = [
        f"Total images: {len(records)}",
        f"Pass (detections == expected): {len(records) - len(failures)}",
        f"Fail (detections != expected): {len(failures)}",
        "",
        "Detections per image (overall):",
    ]
    lines += [f"  {n} detections: {overall[n]} images" for n in sorted(overall)]

    lines += ["", "Detections per image (per object):"]
    for object_name in OBJECT_NAMES + [NO_POINTING]:
        if object_name in per_object:
            counts = ", ".join(f"{n} det: {c}" for n, c in sorted(per_object[object_name].items()))
            lines.append(f"  {object_name:<14} {sum(per_object[object_name].values()):>4} images | {counts}")
    return lines


def write_csv(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Run SAM3 on the dataset and export YOLO boxes / masks.")
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER)
    parser.add_argument("--output", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (for testing)")
    args = parser.parse_args()

    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    dirs = make_output_dirs(args.output)

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    images = sorted(args.images.glob("*.png"))
    if args.limit:
        images = images[: args.limit]

    records = []
    for i, image_path in enumerate(images, 1):
        # Original image goes into both datasets; labels are written next to it below
        for kind in ("boxes", "masks"):
            shutil.copy2(image_path, dirs[(kind, "images")] / image_path.name)

        class_name = class_from_filename(image_path)
        if class_name is None:
            # Background image for YOLO: copied into images/ without a label file
            # No object in the name (no-pointing) -> nothing to prompt SAM3 with, expected 0 detections
            records.append({"image": image_path.name, "object": NO_POINTING, "expected": 0,
                            "detected": 0, "scores": "", "status": "pass"})
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Segment using text prompt from the file name
        inputs = processor(images=image, text=[class_name], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        boxes = results["boxes"].cpu().numpy()
        masks = results["masks"].cpu().numpy().astype(bool)
        scores = results["scores"].cpu().numpy()
        class_ids = np.full(len(boxes), TARGET_CLASS_ID, dtype=int)

        # By rule every object image has exactly one target object
        records.append({
            "image": image_path.name,
            "object": class_name,
            "expected": 1,
            "detected": len(boxes),
            "scores": " ".join(f"{score:.3f}" for score in scores),
            "status": "pass" if len(boxes) == 1 else "fail",
        })
        if len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(xyxy=boxes, mask=masks, confidence=scores, class_id=class_ids)
        labels = [f"{TARGET_CLASS} {score:.2f}" for score in scores]

        annotated_box_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_box_image = label_annotator.annotate(scene=annotated_box_image, detections=detections, labels=labels)
        annotated_mask_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_mask_image = label_annotator.annotate(scene=annotated_mask_image, detections=detections, labels=labels)

        annotated_box_image.save(dirs[("boxes", "annotated")] / image_path.name)
        annotated_mask_image.save(dirs[("masks", "annotated")] / image_path.name)

        label_name = image_path.with_suffix(".txt").name
        write_lines(dirs[("boxes", "labels")] / label_name, boxes_to_yolo(boxes, class_ids, width, height))
        write_lines(dirs[("masks", "labels")] / label_name, masks_to_yolo(masks, class_ids, width, height))

        print(f"[{i}/{len(images)}] {image_path.name}: '{class_name}' -> {len(boxes)} detections")

    print(f"\nDone. Output written to {args.output}")

    write_csv(args.output / "detections.csv", records)
    write_csv(args.output / "failures.csv", [r for r in records if r["status"] == "fail"])
    print(f"Wrote {args.output / 'detections.csv'} and {args.output / 'failures.csv'}\n")

    summary = detection_summary(records)
    print("\n".join(summary))
    write_lines(args.output / "summary.txt", summary)


if __name__ == "__main__":
    main()
