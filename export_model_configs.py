"""Export checkpoint configurations to config.yaml beside each model.pkl."""

import argparse
from dataclasses import asdict
from pathlib import Path
import pickle
import sys

import yaml

from config import Config


def export_config(checkpoint_path: Path) -> Path:
    """Export the stored Config without rerunning its initialization."""
    with checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    config = checkpoint["config"]
    if not isinstance(config, Config):
        raise TypeError("Checkpoint 'config' must be a Config object")

    # Unpickling preserves saved values without invoking __post_init__.
    contents = yaml.safe_dump(asdict(config), sort_keys=False)
    output_path = checkpoint_path.with_name("config.yaml")
    output_path.write_text(contents, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models_folder", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Folder to search recursively (default: models beside this script)",
    )
    args = parser.parse_args()
    if not args.models_folder.is_dir():
        parser.error(f"Not a directory: {args.models_folder}")

    exported = failed = 0
    for checkpoint_path in sorted(args.models_folder.rglob("model.pkl")):
        try:
            output_path = export_config(checkpoint_path)
        except Exception as exc:
            print(f"Failed {checkpoint_path}: {exc}", file=sys.stderr)
            failed += 1
        else:
            print(f"Saved {output_path}")
            exported += 1
    print(f"Exported {exported} configuration(s); {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
