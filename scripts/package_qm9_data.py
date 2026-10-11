"""Package local data separately from Git, or verify a transferred package and dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Verify archive and extracted data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = args.archive.resolve()
    manifest = args.manifest.resolve()
    if args.verify:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if archive.stat().st_size != payload["archive"]["bytes"]:
            raise ValueError("Archive size mismatch")
        if sha256(archive) != payload["archive"]["sha256"]:
            raise ValueError("Archive SHA-256 mismatch")
        for record in payload["files"]:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "data":
                raise ValueError(f"Invalid dataset path: {relative}")
            local = root / relative
            if local.is_symlink() or not local.resolve().is_relative_to(root / "data"):
                raise ValueError(f"Dataset path escapes data directory: {relative}")
            if local.stat().st_size != record["bytes"] or sha256(local) != record["sha256"]:
                raise ValueError(f"Dataset file mismatch: {relative}")
        print(f"PASS: archive and {len(payload['files'])} dataset files verified")
        return
    if archive.exists() or manifest.exists():
        raise FileExistsError("Refusing to overwrite an archive or manifest")
    files = sorted(p for p in (root / "data").rglob("*") if p.is_file())
    if not files:
        raise ValueError("No dataset files found")
    records = []
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root / "data"):
            raise ValueError(f"Dataset must contain ordinary local files: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
        for path in files:
            bundle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    payload = {
        "schema_version": 1,
        "scope": "Local data directory; includes historical CN subset and diagnostic subsets",
        "uncompressed_bytes": sum(record["bytes"] for record in records),
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "files": records,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    archive.with_name(archive.name + ".sha256").write_text(
        f"{payload['archive']['sha256']}  {archive.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": len(records),
                "archive": payload["archive"],
                "uncompressed_bytes": payload["uncompressed_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
