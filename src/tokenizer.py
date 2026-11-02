import re


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text)
    tokens: list[str] = []
    for word in words:
        parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", word)
        tokens.extend(parts)
        if len(parts) > 1:
            tokens.append(word)
    return [token.lower() for token in tokens]
