"""Validate published artifacts and package BidPace-RL.

--check validates assets in the working tree and, after ZIP extraction, the
release-wide manifest. Building validates published assets, then fingerprints
the current source/docs separately so intentional documentation edits can ship.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

AGENTS = [Path("agents/comparison") / f"season{s}_{name}.npz"
          for s in (1, 2, 3) for name in ("traffic_dqn", "campaign_dqn", "sampled_q")]
EVIDENCE = [Path("evidence") / name for name in (
    "study.json", "evaluation.csv", "summary.csv", "nominal.csv", "comparisons.csv",
    "result.json", "shared_selection.json", "nominal.png", "validation.csv", "selection.json")]
ASSET_MANIFEST = Path("agents/comparison/manifest.json")
RELEASE_MANIFEST = Path("RELEASE_MANIFEST.json")


def sha(path):
    """Hash in chunks so checking a release does not load every model into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path):
    """Verify portable project-relative names, accepting legacy Windows slashes."""
    entries = json.loads(Path(path).read_text(encoding="utf8"))
    normalized = {}
    for name, expected in entries.items():
        relative = Path(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest path is not project-relative: {name}")
        if sha(relative) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        normalized[relative.as_posix()] = expected
    return normalized


def check_assets(verify_release=True):
    """Require checksums for all nine models and all ten reviewed evidence files."""
    entries = verify_manifest(ASSET_MANIFEST)
    expected = {p.as_posix() for p in AGENTS + EVIDENCE}
    if set(entries) != expected:
        raise ValueError("Published manifest must cover exactly the current agents and evidence")
    if verify_release and RELEASE_MANIFEST.exists():
        entries = verify_manifest(RELEASE_MANIFEST)
        if set(entries) != {p.as_posix() for p in packaged_files()}:
            raise ValueError("Release manifest does not match the packaged file inventory")
    print(f"PASS published assets: {len(AGENTS)} agents, {len(EVIDENCE)} evidence files")


def packaged_files():
    """Explicit release inventory; historical notes and run caches stay separate."""
    files = [Path(name) for name in (
        "README.md", "requirements.txt", "config.json", "build_release.py",
        ".gitignore", ".gitattributes", "docs/methodology.md", "docs/tabular_method.md",
        "docs/outputs.md")]
    for folder in ("bidpace", "experiments", "tests"):
        files.extend(Path(folder).glob("*.py"))
    files.extend(AGENTS + EVIDENCE + [ASSET_MANIFEST])
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(set(files))


def build(target):
    """Build a fresh ZIP with one copy per file and an embedded checksum manifest."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Release already exists: {target}")
    check_assets(verify_release=False)
    files = packaged_files()
    manifest = {p.as_posix(): sha(p) for p in files}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, (Path("BidPace-RL") / path).as_posix())
        archive.writestr("BidPace-RL/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Created {target}: {len(files) + 1} files, {target.stat().st_size / 1024 ** 2:.1f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="BidPace-RL.zip")
    parser.add_argument("--check", action="store_true", help="Check the supplied assets without building a ZIP")
    args = parser.parse_args()
    if args.check:
        check_assets()
    else:
        build(args.out)
