class IngestionStageError(Exception):
    """Raised when a document fails at an identified stage of the ingestion pipeline.

    Wraps the underlying error so batch reports can attribute a failure to a
    precise stage (existence check, parsing, chunking, embedding, upsert) instead
    of only reporting that "the document failed".

    Attributes:
        stage: Name of the pipeline stage that raised.
        file_name: Name of the document being ingested.
        original_error: Underlying exception that triggered the failure.
    """

    def __init__(self, stage: str, file_name: str, original_error: BaseException):
        """Initialises the error with the failing stage and its cause.

        Args:
            stage: Name of the pipeline stage that raised.
            file_name: Name of the document being ingested.
            original_error: Underlying exception that triggered the failure.
        """
        self.stage = stage
        self.file_name = file_name
        self.original_error = original_error

        super().__init__(
            f"[stage={stage}] '{file_name}' failed with "
            f"{type(original_error).__name__}: {original_error}"
        )


class ParsingFileError(Exception):
    """Raised when the document parser fails to process a file.

    Attributes:
        file_path: Path of the file that could not be parsed.
        reason: Human-readable description of what went wrong.
        original_error: Underlying exception that triggered the failure, if any.
    """

    def __init__(self, file_path: str, reason: str, original_error: Exception | None = None):
        """Initialises the error with context about the failed file.

        Args:
            file_path: Path of the file that could not be parsed.
            reason: Human-readable description of what went wrong.
            original_error: Underlying exception that triggered the failure.
        """
        self.file_path = file_path
        self.reason = reason
        self.original_error = original_error

        message = f"Parsing failed for '{file_path}' \n Reason is: {reason}\n"
        if original_error:
            message += f"Technical Error: {original_error}"

        super().__init__(message)


class CollectionNotFoundError(Exception):
    """Raised when the target vector-store collection does not exist.

    Attributes:
        collection_name: Name of the missing collection.
    """

    def __init__(self, collection_name: str):
        """Initialises the error with a hint on how to create the collection.

        Args:
            collection_name: Name of the missing collection.
        """
        self.collection_name = collection_name
        super().__init__(
            f"Qdrant collection '{collection_name}' does not exist. "
            f"Create it first: python main.py --create-collection {collection_name}"
        )
