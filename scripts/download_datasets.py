#!/usr/bin/env python3
"""Download official FUNSD and DocLayNet-base releases into a local cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.request import urlopen

CACHE_ENV = "ND_LLM_DATA_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "n-dimensional-llm"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    archive_name: str
    checksum: Optional[str]
    target_subdir: str
    archive_type: str  # "zip" or "tar"


_DATASETS: Dict[str, DatasetSpec] = {
    "funsd": DatasetSpec(
        name="funsd",
        url="https://guillaumejaume.github.io/FUNSD/dataset.zip",
        archive_name="funsd_dataset.zip",
        checksum="7a3736d61b05b0d3181c247ac0b6dcbfb995ad5b9b5cde52d2ed571f25b4360f",
        target_subdir="funsd",
        archive_type="zip",
    ),
    "doclaynet": DatasetSpec(
        name="doclaynet",
        url=(
            "https://huggingface.co/datasets/pierreguillou/DocLayNet-base/resolve/main/"
            "DocLayNet-base.tar.gz?download=1"
        ),
        archive_name="doclaynet-base.tar.gz",
        checksum=None,
        target_subdir="doclaynet",
        archive_type="tar",
    ),
}


class DownloadError(RuntimeError):
    """Raised when a dataset download or extraction fails."""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and unpack the official FUNSD and DocLayNet-base datasets into a "
            "cache directory. The cache location defaults to ~/.cache/n-dimensional-llm "
            "or can be overridden via the ND_LLM_DATA_CACHE environment variable."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(_DATASETS),
        default=sorted(_DATASETS),
        help=(
            "One or more datasets to download. Defaults to both FUNSD and DocLayNet-base."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory for downloaded datasets. Defaults to the value of "
            "the ND_LLM_DATA_CACHE environment variable or ~/.cache/n-dimensional-llm."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing dataset caches.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip checksum validation (not recommended).",
    )
    parser.add_argument(
        "--override-checksum",
        action="append",
        metavar="DATASET=SHA256",
        help="Override the expected checksum for a dataset (e.g. funsd=deadbeef...).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    cache_dir = resolve_cache_dir(args.cache_dir)
    overrides = parse_overrides(args.override_checksum)

    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        spec = _DATASETS[name]
        checksum = overrides.get(name, spec.checksum)
        try:
            download_and_extract(spec, cache_dir, checksum, args.force, args.skip_checksum)
        except DownloadError as error:
            print(f"Error downloading {name}: {error}", file=sys.stderr)
            return 1
    return 0


def resolve_cache_dir(cli_value: Optional[Path]) -> Path:
    if cli_value is not None:
        return cli_value.expanduser().resolve()
    env_value = os.environ.get(CACHE_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_CACHE_DIR


def parse_overrides(values: Optional[Iterable[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not values:
        return overrides
    for entry in values:
        if not entry:
            continue
        if "=" not in entry:
            raise DownloadError(
                f"Invalid checksum override '{entry}'. Expected format DATASET=SHA256."
            )
        dataset, checksum = entry.split("=", 1)
        dataset = dataset.strip().lower()
        checksum = checksum.strip().lower()
        if dataset not in _DATASETS:
            raise DownloadError(
                f"Unknown dataset '{dataset}' in checksum override. Valid options: "
                f"{', '.join(sorted(_DATASETS))}."
            )
        if not checksum or any(char.isspace() for char in checksum):
            raise DownloadError(f"Invalid checksum value for dataset '{dataset}'.")
        overrides[dataset] = checksum
    return overrides


def download_and_extract(
    spec: DatasetSpec,
    cache_dir: Path,
    checksum: Optional[str],
    force: bool,
    skip_checksum: bool,
) -> None:
    target_dir = cache_dir / spec.target_subdir
    if target_dir.exists() and not force:
        print(f"Skipping {spec.name}: already present at {target_dir}")
        return

    if target_dir.exists() and force:
        print(f"Removing existing directory {target_dir}")
        shutil.rmtree(target_dir)

    with tempfile.TemporaryDirectory(prefix=f"{spec.name}-download-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / spec.archive_name
        print(f"Downloading {spec.name} from {spec.url}")
        fetch_to_file(spec.url, archive_path)

        if not skip_checksum:
            if checksum:
                print(f"Validating checksum for {spec.name}")
                digest = sha256_path(archive_path)
                if digest.lower() != checksum.lower():
                    raise DownloadError(
                        "Checksum mismatch: expected "
                        f"{checksum.lower()} but received {digest.lower()}"
                    )
            else:
                print(
                    "No checksum registered for this dataset; skipping validation by default"
                )
        else:
            print("Checksum validation skipped by user request")

        print(f"Extracting {spec.name} to {target_dir}")
        extract_archive(archive_path, target_dir, spec.archive_type)

    print(f"Finished installing {spec.name} into {target_dir}")


def fetch_to_file(url: str, destination: Path) -> None:
    try:
        with urlopen(url) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as error:  # pragma: no cover - network failure
        raise DownloadError(f"Failed to download '{url}': {error}") from error


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive_path: Path, destination: Path, archive_type: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_type == "zip":
        _extract_zip(archive_path, destination)
    elif archive_type == "tar":
        _extract_tar(archive_path, destination)
    else:  # pragma: no cover - defensive
        raise DownloadError(f"Unsupported archive type '{archive_type}'")
    _normalise_layout(destination)


def _extract_zip(path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
    except zipfile.BadZipFile as error:
        raise DownloadError(f"Corrupt ZIP archive: {error}") from error


def _extract_tar(path: Path, destination: Path) -> None:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            safe_extract(archive, destination)
    except tarfile.TarError as error:
        raise DownloadError(f"Corrupt TAR archive: {error}") from error


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        member_path = destination / member.name
        if not is_within_directory(destination, member_path):
            raise DownloadError("Attempted path traversal in tar archive")
    archive.extractall(destination)


def is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve(strict=False)
    return os.path.commonpath([str(directory), str(target)]) == str(directory)


def _normalise_layout(destination: Path) -> None:
    entries = list(destination.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            target = destination / child.name
            if target.exists():
                continue
            child.rename(target)
        inner.rmdir()


if __name__ == "__main__":
    sys.exit(main())
