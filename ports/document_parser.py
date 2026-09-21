from abc import ABC, abstractmethod
from pathlib import Path

from domain.models import Document


class DocumentParser(ABC):
    """Port: converts a raw file into a structured domain Document."""

    @abstractmethod
    async def parse(self, file_path: Path) -> Document:
        """Parses a file into a Document.

        Args:
            file_path: Path to the source file.

        Returns:
            A Document containing extracted text and metadata.
        """
        ...
