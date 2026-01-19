#!/usr/bin/env python3
"""
Galaxy Zoo 2 — inference script
--------------------------------
Loads a trained CNN model and classifies galaxy images
from a given directory. Saves predictions to CSV.

Run from repository root:
    python scripts/predict_gz2.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tqdm import tqdm

from gzml.bootstrap import get_paths


# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------

CLASS_NAMES = ["Elliptical", "Spiral", "Artifact"]
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run inference on a folder of galaxy images.")
    ap.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Folder with images. Default: repo/new_galaxy_images (bootstrap paths.galaxies)",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .keras model. Default: outputs/models/gz2_cnn.keras (bootstrap paths.default_model)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: outputs/predictions.csv (bootstrap paths.outputs)",
    )
    ap.add_argument(
        "--img",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("H", "W"),
        help="Image size (H W). Must match training. Default: 128 128",
    )
    return ap.parse_args()


def load_and_prepare_image(path: Path, img_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_size[1], img_size[0]))  # PIL expects (W,H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    return arr


def main() -> None:
    args = parse_args()

    paths = get_paths()

    images_dir = (args.images or paths.galaxies).resolve()
    model_path = (args.model or paths.default_model).resolve()
    out_csv = (args.out or (paths.outputs / "predictions.csv")).resolve()

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train first, or pass --model /path/to/model.keras"
        )

    img_size = (int(args.img[0]), int(args.img[1]))

    print("🧠 Loading model:", model_path)
    model = load_model(model_path)

    print("🔍 Classifying images in:", images_dir)

    image_paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in ALLOWED_EXT]
    if not image_paths:
        print("⚠️ No images found (jpg/jpeg/png). Nothing to do.")
        return

    results: list[tuple[str, str]] = []

    for p in tqdm(image_paths, desc="Classification"):
        try:
            x = load_and_prepare_image(p, img_size)
            preds = model.predict(x, verbose=0)
            label = CLASS_NAMES[int(np.argmax(preds, axis=1)[0])]
            results.append((p.name, label))
        except Exception as e:
            print(f"⚠️ Failed on {p.name}: {e}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class"])
        writer.writerows(results)

    print("✅ Done")
    print("📄 Results saved to:", out_csv)


if __name__ == "__main__":
    main()