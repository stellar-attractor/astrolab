from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# -----------------------------
# Repo discovery
# -----------------------------

REPO_MARKERS = (
    ".git",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
)


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find repository root by walking up from `start` until we hit a marker.
    Works both when called from scripts/ and from notebooks.
    """
    p = (start or Path.cwd()).resolve()

    # If start is a file, use its parent
    if p.is_file():
        p = p.parent

    for cur in (p, *p.parents):
        for m in REPO_MARKERS:
            if (cur / m).exists():
                return cur

    # fallback: current working dir
    return Path.cwd().resolve()


# -----------------------------
# Paths (single source of truth)
# -----------------------------

@dataclass(frozen=True)
class RepoPaths:
    root: Path

    # common dirs
    scripts: Path
    notebooks: Path

    data: Path
    raw: Path
    processed: Path

    outputs: Path
    figures: Path
    logs: Path

    # IMPORTANT: models live under outputs/ (reproducible artifacts)
    models: Path

    # inputs to classify
    galaxies: Path

    # optional: a canonical model file path (so everyone uses the same name)
    default_model: Path

    @staticmethod
    def from_root(root: Path) -> "RepoPaths":
        root = root.resolve()

        data = root / "data"
        outputs = root / "outputs"

        figures = outputs / "figures"
        logs = outputs / "logs"
        models = outputs / "models"            # <-- unified place

        galaxies = root / "new_galaxy_images"

        default_model = models / "gz2_cnn.keras"

        return RepoPaths(
            root=root,
            scripts=root / "scripts",
            notebooks=root / "notebooks",
            data=data,
            raw=data / "raw",
            processed=data / "processed",
            outputs=outputs,
            figures=figures,
            logs=logs,
            models=models,
            galaxies=galaxies,
            default_model=default_model,
        )

    def ensure(self) -> "RepoPaths":
        """
        Create directories that are expected to exist.
        """
        for p in [
            self.data,
            self.raw,
            self.processed,
            self.outputs,
            self.figures,
            self.logs,
            self.models,   # <-- ensure outputs/models exists
        ]:
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(*, start: Path | None = None, ensure: bool = True) -> RepoPaths:
    """
    Main entrypoint: get all repo paths, reliably, from anywhere.
    """
    root = find_repo_root(start=start)
    rp = RepoPaths.from_root(root)
    return rp.ensure() if ensure else rp


# -----------------------------
# Convenience: set env var so other tools can reuse it
# -----------------------------

def export_repo_root_env(root: Path) -> None:
    os.environ["GZML_REPO_ROOT"] = str(root.resolve())