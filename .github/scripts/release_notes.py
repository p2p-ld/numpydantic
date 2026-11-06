"""
Print the changelog section for a version, for use as GitHub release notes.

Prints nothing if the changelog has no section for that version.

Usage: python release_notes.py v2.0.0 docs/changelog.md
"""

import re
import sys
from pathlib import Path

DOCS_URL = "https://numpydantic.readthedocs.io/en/latest/"


def section(version: str, changelog: str) -> str:
    heading = re.compile(rf"^(#+) v?{re.escape(version)}(\s|$)")
    lines = []
    level = None
    in_code = False
    for line in changelog.splitlines():
        if line.startswith(("```", "~~~")):
            in_code = not in_code
        is_heading = not in_code and line.startswith("#")
        if level is None:
            if is_heading and (match := heading.match(line)):
                level = len(match.group(1))
            continue
        if is_heading and len(line) - len(line.lstrip("#")) <= level:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def to_github_markdown(text: str) -> str:
    # links to other docs pages are relative, so point them at the hosted docs
    text = re.sub(
        r"\]\((?:\./)?([\w/-]+)\.md(#[\w-]+)?\)",
        lambda m: f"]({DOCS_URL}{m.group(1)}.html{m.group(2) or ''})",
        text,
    )
    # sphinx roles like {class}`.NDArray` become plain code
    return re.sub(r"\{[\w:]+\}`\.?([^`]*)`", r"`\1`", text)


if __name__ == "__main__":
    tag, path = sys.argv[1:3]
    notes = section(tag.removeprefix("v"), Path(path).read_text(encoding="utf-8"))
    print(to_github_markdown(notes))
