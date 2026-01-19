from __future__ import annotations

import argparse
from pathlib import Path

from gzml.bootstrap import get_paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gzml", description="Galaxy Zoo ML helper CLI")
    ap.add_argument("--paths", action="store_true", help="Print resolved repo paths")
    args = ap.parse_args(argv)

    if args.paths:
        P = get_paths()
        print("root     :", P.root)
        print("data     :", P.data)
        print("raw      :", P.raw)
        print("processed:", P.processed)
        print("models   :", P.models)
        print("outputs  :", P.outputs)
        print("figures  :", P.figures)
        print("logs     :", P.logs)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()