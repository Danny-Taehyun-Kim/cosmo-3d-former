#!/usr/bin/env python
"""Download pre-trained weights from a GitHub release and verify the checksums.

Usage:
    download_weights.py --out-dir weights [--name M_encoder_ep146.pt]

The expected SHA-256 digests live in scripts/weights_manifest.json next to this
file. A file that is already present and matches its digest is not downloaded
again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).with_name("weights_manifest.json")


def sha256(path, chunk=1 << 20):
    """SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="weights")
    parser.add_argument("--name", default=None, help="download a single file")
    parser.add_argument("--base-url", default=None, help="override the release URL")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    base = args.base_url or manifest["base_url"]
    if "RELEASE_URL" in base:
        raise SystemExit("set the release URL in weights_manifest.json or pass --base-url")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = manifest["files"]
    if args.name:
        entries = [e for e in entries if e["name"] == args.name]
        if not entries:
            raise SystemExit(f"{args.name} is not listed in the manifest")

    for entry in entries:
        target = out_dir / entry["name"]
        if target.exists() and sha256(target) == entry["sha256"]:
            print(f"{entry['name']} already present and verified")
            continue
        url = f"{base.rstrip('/')}/{entry['name']}"
        print(f"downloading {entry['name']} ({entry['size_bytes'] / 1e6:.1f} MB)")
        try:
            urllib.request.urlretrieve(url, target)
        except urllib.error.HTTPError as error:
            target.unlink(missing_ok=True)
            raise SystemExit(f"{url} returned {error.code}; the release may not be "
                             "published yet, or pass --base-url") from None
        digest = sha256(target)
        if digest != entry["sha256"]:
            target.unlink(missing_ok=True)
            raise SystemExit(f"checksum mismatch for {entry['name']}: got {digest}")
        print(f"  verified {digest}")


if __name__ == "__main__":
    main()
