class ParsingFileError(Exception):

    def __init__(self, file_path: str, reason: str, original_error: Exception | None = None):
        self.file_path = file_path
        self.reason = reason
        self.original_error = original_error

        message = f"Parsing failed for '{file_path}' \n Reason is: {reason}\n"
        if original_error:
            message += f"Technical Error: {original_error}"

        super().__init__(message)
