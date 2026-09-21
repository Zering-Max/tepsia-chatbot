import hashlib
import uuid
from pathlib import Path


class Identity:
    """Utility class for generating deterministic document and chunk identifiers."""

    @staticmethod
    def document_id(file_path: Path) -> str:
        """Computes a SHA-256 hash of a file's binary content.

        Reads in 8-KiB blocks to support large files without loading the whole
        file into memory at once.

        Args:
            file_path: Path to the file to hash.

        Returns:
            Hexadecimal SHA-256 digest of the file.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    @staticmethod
    def chunk_id(doc_id: str, chunk_index: int) -> str:
        """Derives a deterministic UUID5 for a chunk from its parent document.

        The same (doc_id, chunk_index) pair always produces the same UUID, so
        re-ingesting an unchanged document yields identical point IDs in Qdrant.

        Args:
            doc_id: SHA-256 hash of the parent document.
            chunk_index: 0-based position of the chunk within the document.

        Returns:
            A UUID5 string in canonical hyphenated form.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk_index}"))
