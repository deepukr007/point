"""
RF-DETR Feature Visualizer

Runs inference with an RF-DETR model and saves a grid showing what the
backbone produces at each stage — projector feature maps and the fused
output that gets fed into the transformer decoder.

Usage:
    python visualize.py --image bird.jpg
    python visualize.py --image bird.jpg --model medium --threshold 0.4
    python visualize.py --image bird.jpg --no-individual
"""

import argparse
import os
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

PROJ_LAYERS = [
    "backbone.0.projector.stages_sampling.0.0",
    "backbone.0.projector.stages_sampling.0.1",
    "backbone.0.projector.stages_sampling.0.2",
    "backbone.0.projector.stages_sampling.0.3",
]
FUSED_LAYER = "backbone.0.projector.stages.0"


def _attach_hooks(raw_model):
    feats = {}

    def make_hook(key):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.dim() == 4:
                feats[key] = t.detach()
        return hook

    handles = []
    for name, mod in raw_model.named_modules():
        if name in PROJ_LAYERS or name == FUSED_LAYER:
            handles.append(mod.register_forward_hook(make_hook(name)))

    return feats, handles


def _remove_hooks(handles):
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _pca_to_rgb(arr):
    h, w, c = arr.shape
    pixels = arr.reshape(-1, c).astype(np.float32)
    centered = pixels - pixels.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T
    for i in range(3):
        col = projected[:, i]
        mn, mx = col.min(), col.max()
        projected[:, i] = (col - mn) / (mx - mn) * 255 if mx > mn else 0
    return projected.reshape(h, w, 3).astype(np.uint8)


def _feat_to_panel(feat, panel_h, panel_w):
    arr = feat[0].permute(1, 2, 0).cpu().numpy()
    rgb = _pca_to_rgb(arr)
    return np.array(Image.fromarray(rgb).resize((panel_w, panel_h), Image.BILINEAR))


def _scale_detections(detections, orig_hw, panel_hw):
    oh, ow = orig_hw
    ph, pw = panel_hw
    scaled_xyxy = detections.xyxy * np.array([pw / ow, ph / oh, pw / ow, ph / oh])
    return sv.Detections(
        xyxy=scaled_xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id,
        data=detections.data,
    )


def _annotate(img_rgb, detections, orig_hw):
    ph, pw = img_rgb.shape[:2]
    scaled = _scale_detections(detections, orig_hw, (ph, pw))
    labels = [
        f"{cls} {conf:.2f}"
        for cls, conf in zip(scaled.data.get("class_name", [""] * len(scaled)), scaled.confidence)
    ]
    out = sv.BoxAnnotator(thickness=2).annotate(img_rgb.copy(), scaled)
    out = sv.LabelAnnotator(text_scale=0.5, text_thickness=1).annotate(out, scaled, labels=labels)
    return out


def _title_bar(text, width, height=34):
    bar = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(bar)
    tw = draw.textlength(text)
    draw.text(((width - tw) / 2, (height - 12) / 2), text, fill=(220, 220, 220))
    return np.array(bar)


def _make_grid(panels, labels, cols):
    ph, pw = panels[0].shape[:2]
    title_h = 34
    cell_h = ph + title_h
    rows = (len(panels) + cols - 1) // cols
    canvas = np.zeros((rows * cell_h, cols * pw, 3), dtype=np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        r, c = divmod(i, cols)
        y, x = r * cell_h, c * pw
        canvas[y : y + title_h, x : x + pw] = _title_bar(label, pw, title_h)
        canvas[y + title_h : y + cell_h, x : x + pw] = panel
    return canvas


# --------------------------------------------------------------------------- #
# Main function
# --------------------------------------------------------------------------- #

def visualize_features(image, model, threshold=0.5, output_dir="outputs", save_individual=True):
    """Run inference and save backbone feature map visualizations.

    Args:
        image: Path to image file or PIL Image.
        model: An RF-DETR model instance (RFDETRSmall, RFDETRMedium, etc.).
        threshold: Detection confidence threshold.
        output_dir: Directory to save outputs. Created if it doesn't exist.
        save_individual: Whether to also save each panel as a separate PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image, (str, os.PathLike)):
        stem = Path(image).stem
        pil_img = Image.open(image).convert("RGB")
    else:
        stem = "output"
        pil_img = image.convert("RGB")

    orig_w, orig_h = pil_img.size
    panel_w = 512
    panel_h = int(orig_h * panel_w / orig_w)

    raw_model = model.model.model
    feats, handles = _attach_hooks(raw_model)

    try:
        detections = model.predict(pil_img, threshold=threshold)
    finally:
        _remove_hooks(handles)

    orig_np = np.array(pil_img.resize((panel_w, panel_h), Image.BILINEAR))

    def make_panel(feat):
        if feat is None:
            return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        rgb = _feat_to_panel(feat, panel_h, panel_w)
        return _annotate(rgb, detections, (orig_h, orig_w))

    det_panel    = _annotate(orig_np.copy(), detections, (orig_h, orig_w))
    proj_panels  = [make_panel(feats.get(k)) for k in PROJ_LAYERS]
    fused_panel  = make_panel(feats.get(FUSED_LAYER))

    panels = [det_panel] + proj_panels + [fused_panel]
    labels = ["Detection", "Proj 1", "Proj 2", "Proj 3", "Proj 4", "Fused (P4)"]

    grid = _make_grid(panels, labels, cols=3)
    grid_path = output_dir / f"{stem}_grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"Saved grid   → {grid_path}")

    if save_individual:
        names = ["detection", "proj_1", "proj_2", "proj_3", "proj_4", "proj_fused"]
        for panel, name in zip(panels, names):
            path = output_dir / f"{stem}_{name}.png"
            Image.fromarray(panel).save(path)
            print(f"Saved panel  → {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_model(name):
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
    models = {
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models)}")
    return models[name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR backbone feature visualizer")
    parser.add_argument("--image",      required=True,          help="Path to input image")
    parser.add_argument("--model",      default="small",        help="Model size: small / medium / base / large")
    parser.add_argument("--threshold",  type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--output-dir", default="outputs",      help="Output directory (created if missing)")
    parser.add_argument("--no-individual", action="store_true", help="Skip saving individual panel images")
    args = parser.parse_args()

    model = _build_model(args.model)
    visualize_features(
        image=args.image,
        model=model,
        threshold=args.threshold,
        output_dir=args.output_dir,
        save_individual=not args.no_individual,
    )