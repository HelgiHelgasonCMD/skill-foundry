"""URL slug helpers."""

import re


def slugify(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text)
