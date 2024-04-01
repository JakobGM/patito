"""A simple script to fix RST headers."""

import sys
from pathlib import Path

if __name__ == "__main__":
    arguments = sys.argv
    directory = Path(arguments[1])

    for file in directory.rglob("*.rst"):
        lines = []
        for line_number, line in enumerate(file.read_text().splitlines()):
            if line.startswith("===="):
                header_length = len(lines[line_number - 1])
                lines.append("=" * header_length)
            else:
                lines.append(line)
        file.write_text("\n".join(lines) + "\n")
