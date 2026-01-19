#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm

from gzml.bootstrap import get_paths


# -----------------------------------------------------------------------------
# Defaults (overridden by CLI)
# -----------------------------------------------------------------------------

DEFAULT_IMG_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

NUM_CLASSES = 3


# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------

def preprocess(example, img_size: tuple[int, int]):
    image = tf.image.resize(example["image"], img_size)
    image = tf.cast(image, tf.float32) / 255.0

    table = example["table1"]
    votes = [
        table["t01_smooth_or_features_a01_smooth_fraction"],
        table["t01_smooth_or_features_a02_features_or_disk_fraction"],
        table["t01_smooth_or_features_a03_star_or_artifact_fraction"],
    ]
    label = tf.argmax(tf.stack(votes), axis=0)
    return image, label


def prepare_dataset(
    ds,
    *,
    img_size,
    batch_size,
    autotune,
    shuffle_buffer=1000,
    seed=1,
):
    ds = ds.map(lambda ex: preprocess(ex, img_size), num_parallel_calls=autotune)
    ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(autotune)
    return ds


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def build_model(img_size: tuple[int, int], n_classes: int) -> tf.keras.Model:
    h, w = img_size
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(h, w, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def cardinality_or_count(ds) -> int:
    card = tf.data.experimental.cardinality(ds).numpy()
    if card >= 0:
        return int(card)

    n = 0
    for _ in tqdm(ds, desc="Counting dataset size (fallback)"):
        n += 1
    return n


def save_training_plots(history, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_prediction_grid(model, ds_test, out_png: Path, class_names, n=9) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    for images, labels in ds_test.take(1):
        preds = model.predict(images, verbose=0)
        preds_labels = np.argmax(preds, axis=1)

        plt.figure(figsize=(10, 10))
        k = min(n, images.shape[0])
        for i in range(k):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(
                f"Pred: {class_names[preds_labels[i]]}\n"
                f"True: {class_names[int(labels[i])]}"
            )
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        break


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train CNN on Galaxy Zoo 2 (TFDS).")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--img", type=int, nargs=2, default=DEFAULT_IMG_SIZE, metavar=("H", "W"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="Debug: limit dataset size")
    ap.add_argument("--no-save-model", action="store_true")
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    IMG_SIZE = (int(args.img[0]), int(args.img[1]))
    BATCH_SIZE = int(args.batch)

    paths = get_paths()

    print("[Paths]")
    for k, v in asdict(paths).items():
        print(f" - {k}: {v}")

    print("🔽 Loading Galaxy Zoo 2 from TFDS...")
    ds_full = tfds.load("galaxy_zoo2", split="train", shuffle_files=True)

    if args.limit > 0:
        print(f"⚠️ Using only first {args.limit} samples")
        ds_full = ds_full.take(args.limit)

    n_total = cardinality_or_count(ds_full)
    train_size = int(n_total * args.train_frac)

    print(f"🔢 Total: {n_total} | Train: {train_size} | Test: {n_total - train_size}")

    ds_train = prepare_dataset(
        ds_full.take(train_size),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        autotune=AUTOTUNE,
        seed=args.seed,
    )

    ds_test = prepare_dataset(
        ds_full.skip(train_size),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        autotune=AUTOTUNE,
        seed=args.seed + 1,
    )

    model = build_model(IMG_SIZE, NUM_CLASSES)

    print("🚀 Training...")
    history = model.fit(ds_train, validation_data=ds_test, epochs=args.epochs)

    print("📊 Evaluating...")
    test_loss, test_acc = model.evaluate(ds_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    save_training_plots(history, paths.figures / "gz2_training_curves.png")
    save_prediction_grid(
        model,
        ds_test,
        paths.figures / "gz2_prediction_grid.png",
        ["Elliptical", "Spiral", "Artifact"],
    )

    if not args.no_save_model:
        model_path = paths.models / "gz2_cnn.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print("💾 Saved model:", model_path)

    print("✅ Done. Artifacts in:", paths.outputs)


if __name__ == "__main__":
    main()