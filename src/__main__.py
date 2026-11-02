import sys

from fire import Fire
from pydantic import ValidationError

from src.cli import Cli
from src.error import RagError


def format_validation_error(error: ValidationError) -> str:
    messages: list[str] = []
    for detail in error.errors():
        reason = detail["msg"].removeprefix("Value error, ")
        message = f"{reason} (input: {detail['input']!r})"
        location = ".".join(str(part) for part in detail["loc"])
        if any(isinstance(part, str) for part in detail["loc"]):
            message = f"{location}: {message}"
        messages.append(message)
    return "; ".join(messages)


def main() -> None:
    try:
        Fire(Cli)
    except ValidationError as error:
        message = f"Invalid input: {format_validation_error(error)}"
    except RagError as error:
        message = str(error)
    except FileNotFoundError as error:
        message = f"File not found: {error.filename}"
    except OSError as error:
        message = f"Cannot access {error.filename}: {error.strerror}"
    except ValueError as error:
        message = f"Invalid value: {error}"
    except Exception as error:
        message = f"Unexpected error ({type(error).__name__}): {error}"
    else:
        return
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
