"""Download and safely extract the immutable QM9 extended-XYZ archive."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

URL = "https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/3195389/dsgdb9nsd.xyz.tar.bz2"
SHA256 = "3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c"
EXPECTED_FILES = 133_885


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="QM9 archive") as progress:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                progress.update(len(chunk))
    temporary.replace(destination)


def extract(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with tarfile.open(archive, "r:bz2") as source:
        members = source.getmembers()
        if len(members) != EXPECTED_FILES:
            raise ValueError(f"Expected {EXPECTED_FILES} QM9 files, found {len(members)}")
        for member in members:
            destination = (output_dir / member.name).resolve()
            if root not in destination.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        source.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/qm9_xyz"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/dsgdb9nsd.xyz.tar.bz2"))
    args = parser.parse_args()
    if not args.archive.exists():
        download(URL, args.archive)
    actual = digest(args.archive)
    if actual != SHA256:
        raise ValueError(f"QM9 archive hash mismatch: {actual}")
    existing = len(list(args.output_dir.glob("dsgdb9nsd_*.xyz")))
    if existing == EXPECTED_FILES:
        print(f"QM9 XYZ directory already complete: {args.output_dir}")
        return
    if existing:
        raise ValueError(f"Refusing partial QM9 directory with {existing} files: {args.output_dir}")
    extract(args.archive, args.output_dir)
    print(f"Extracted {EXPECTED_FILES} verified QM9 XYZ records to {args.output_dir}")


if __name__ == "__main__":
    main()
