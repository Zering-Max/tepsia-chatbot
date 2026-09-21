"""Downloads a kDrive (Infomaniak) folder recursively and produces a JSON manifest.

Required environment variables (loaded from .env):
    KDRIVE_TOKEN      -- Infomaniak API Bearer token
    KDRIVE_DRIVE_ID   -- Numeric drive identifier
    KDRIVE_SHARE_UUID -- Public share link UUID (optional)

Usage:
    python scripts/kdrive_downloader.py --folder-id 30609 --dest ./downloads
"""
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import os
import re
import time
import logging
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not available; rely on env vars being set externally

logger = logging.getLogger(__name__)

_API_BASE = "https://api.infomaniak.com"

# Characters forbidden in Windows file names
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Maps file extension to the kDrive preview URL segment.
# Office documents (docx, xlsx…) are previewed as PDF by kDrive, so "pdf" is
# the fallback for unknown types.
_PREVIEW_KIND: dict[str, str] = {
    "pdf": "pdf",
    "jpg": "image", "jpeg": "image", "png": "image", "gif": "image",
    "webp": "image", "bmp": "image", "svg": "image", "heic": "image",
    "mp4": "video", "mov": "video", "avi": "video", "mkv": "video", "webm": "video",
    "mp3": "audio", "wav": "audio", "flac": "audio", "ogg": "audio", "m4a": "audio",
    "txt": "text", "md": "text", "csv": "text", "log": "text",
}


@dataclass
class KdriveDownloader:
    """Downloads files from a kDrive folder and tracks them in a JSON manifest.

    Attributes:
        token: Infomaniak API Bearer token.
        drive_id: Numeric drive identifier.
        share_uuid: UUID of the public share link (used to build preview URLs).
        throttle: Pause in seconds between API requests (rate limit: 60 req/min).
        limit: Items per page for cursor-based pagination.
        manifest: Accumulated list of downloaded-file records.
    """

    token: str
    drive_id: int
    share_uuid: str = ""
    throttle: float = 1.1
    limit: int = 200
    manifest: list = field(default_factory=list)

    # ---------------------------------------------------------------------- #
    # Properties and URL builders
    # ---------------------------------------------------------------------- #

    @property
    def _headers(self) -> dict:
        """Authorization headers for every API request."""
        return {"Authorization": f"Bearer {self.token}"}

    def _list_url(self, folder_id: int) -> str:
        """Returns the v3 API URL to list children of a folder.

        Args:
            folder_id: kDrive folder identifier.

        Returns:
            Absolute URL for the folder listing endpoint.
        """
        return f"{_API_BASE}/3/drive/{self.drive_id}/files/{folder_id}/files"

    def _download_url(self, file_id: int) -> str:
        """Returns the v2 API URL to download a file (returns a 302 redirect).

        Args:
            file_id: kDrive file identifier.

        Returns:
            Absolute URL for the file download endpoint.
        """
        return f"{_API_BASE}/2/drive/{self.drive_id}/files/{file_id}/download"

    def _share_url(self, file_id: int, parent_folder_id: int, kind: str) -> str:
        """Builds the public preview URL for a file via the share link.

        Accessible without authentication; the share UUID acts as the secret.

        Args:
            file_id: kDrive file identifier.
            parent_folder_id: Identifier of the folder containing the file.
            kind: Preview type segment (e.g. ``"pdf"``, ``"image"``).

        Returns:
            Absolute public preview URL.
        """
        return (
            f"https://kdrive.infomaniak.com/app/share/{self.drive_id}/"
            f"{self.share_uuid}/files/{parent_folder_id}/preview/{kind}/{file_id}"
        )

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _preview_kind(name: str) -> str:
        """Infers the kDrive preview URL segment from a file's extension.

        Args:
            name: File name (with extension).

        Returns:
            A preview kind string (e.g. ``"pdf"``, ``"image"``).
            Defaults to ``"pdf"`` for unknown extensions.
        """
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        return _PREVIEW_KIND.get(ext, "pdf")

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitises a file name for use on the local file system.

        Replaces characters forbidden on Windows with underscores and strips
        trailing dots and spaces.

        Args:
            name: Raw file name from the kDrive API.

        Returns:
            A sanitised file name safe for all major platforms.
        """
        cleaned = _ILLEGAL.sub("_", name).strip().rstrip(". ")
        return cleaned or "_unnamed_"

    # ---------------------------------------------------------------------- #
    # Folder listing (cursor-based pagination)
    # ---------------------------------------------------------------------- #

    def list_children(self, folder_id: int) -> list:
        """Lists all children of a kDrive folder, following cursor pagination.

        Args:
            folder_id: kDrive folder identifier.

        Returns:
            Flat list of item dicts returned by the API across all pages.
        """
        items: list = []
        params: dict = {"limit": self.limit}
        while True:
            r = requests.get(
                self._list_url(folder_id),
                headers=self._headers,
                params=params,
            )
            time.sleep(self.throttle)

            if r.status_code != 200:
                logger.error(
                    "Listing %s failed (HTTP %s): %s",
                    folder_id, r.status_code, r.text[:300],
                )
                break

            body = r.json()
            if body.get("result") != "success":
                logger.error("Listing %s non-success: %s", folder_id, body.get("error"))
                break

            items.extend(body.get("data", []))

            cursor = body.get("cursor")
            has_more = body.get("has_more", False)
            if has_more and cursor:
                params = {"limit": self.limit, "cursor": cursor}
            else:
                break
        return items

    # ---------------------------------------------------------------------- #
    # Single file download
    # ---------------------------------------------------------------------- #

    def download_file(self, file_id: int, dest: Path) -> bool:
        """Downloads a single file from kDrive to a local path.

        Skips the download if the file already exists locally.

        Args:
            file_id: kDrive file identifier.
            dest: Local destination path (must not be a directory).

        Returns:
            True if the file is available locally after the call, False on error.
        """
        dest = Path(dest)
        if dest.is_dir():
            logger.warning("%s is a directory, skipping download (id=%s)", dest, file_id)
            return False
        if dest.exists():
            logger.info("File already present locally, skipping: %s", dest)
            return True
        try:
            with requests.get(self._download_url(file_id), headers=self._headers, stream=True) as r:
                time.sleep(self.throttle)
                if r.status_code != 200:
                    logger.error("Download %s failed (HTTP %s)", file_id, r.status_code)
                    return False
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except OSError as e:
            logger.error("Write error on %s: %s", dest, e)
            return False

        logger.info("-> %s", dest)
        return True

    # ---------------------------------------------------------------------- #
    # Recursive folder download
    # ---------------------------------------------------------------------- #

    def download_folder(self, folder_id: int, dest_dir: Path, recursive: bool = True) -> None:
        """Downloads all files in a kDrive folder, optionally descending into sub-folders.

        Appends a manifest entry for each successfully downloaded file.

        Args:
            folder_id: kDrive folder identifier to download.
            dest_dir: Local directory to save files into.
            recursive: When True, descends into sub-folders.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for item in self.list_children(folder_id):
            name = KdriveDownloader._safe_name(item["name"])
            itype = item.get("type")

            if itype == "dir":
                if recursive:
                    self.download_folder(item["id"], dest_dir / name, recursive=True)

            elif itype == "file":
                dest = dest_dir / name
                if self.download_file(item["id"], dest):
                    kind = KdriveDownloader._preview_kind(item["name"])
                    self.manifest.append({
                        "file_id": item["id"],
                        "name": item["name"],
                        "local_path": dest.as_posix(),
                        "folder": dest_dir.as_posix(),
                        "share_url": self._share_url(
                            file_id=item["id"],
                            parent_folder_id=folder_id,
                            kind=kind,
                        ),
                    })

            else:
                logger.warning(
                    "Unexpected type '%s' for %s (id=%s) - skipped",
                    itype, item["name"], item["id"],
                )

    # ---------------------------------------------------------------------- #
    # Manifest persistence
    # ---------------------------------------------------------------------- #

    def save_manifest(self, path: Path) -> None:
        """Writes the accumulated manifest to a JSON file.

        Args:
            path: Destination path for the manifest JSON file.
        """
        path = Path(path)
        path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Manifest written: %s (%d files)", path, len(self.manifest))


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    arg_parser = argparse.ArgumentParser(
        description="Downloads a kDrive folder and produces a JSON manifest.",
    )
    arg_parser.add_argument("--folder-id", type=int, required=True, help="kDrive folder ID to download.")
    arg_parser.add_argument("--dest", type=Path, default=Path("kdrive_downloads"), help="Local destination directory.")
    arg_parser.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON path (default: <dest>/manifest.json).")
    arg_parser.add_argument("--no-recursive", action="store_true", help="Do not descend into sub-folders.")
    args = arg_parser.parse_args()

    token = os.environ.get("KDRIVE_TOKEN")
    drive_id_str = os.environ.get("KDRIVE_DRIVE_ID")
    share_uuid = os.environ.get("KDRIVE_SHARE_UUID", "")

    if not token:
        raise SystemExit("KDRIVE_TOKEN missing from environment / .env")
    if not drive_id_str:
        raise SystemExit("KDRIVE_DRIVE_ID missing from environment / .env")

    dl = KdriveDownloader(token=token, drive_id=int(drive_id_str), share_uuid=share_uuid)
    manifest_path = args.manifest or (args.dest / "manifest.json")

    dl.download_folder(folder_id=args.folder_id, dest_dir=args.dest, recursive=not args.no_recursive)
    dl.save_manifest(manifest_path)
