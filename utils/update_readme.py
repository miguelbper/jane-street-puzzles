import os

CHECK = "✔"
GITHUB_LINK = "https://github.com/miguelbper/jane-street-puzzles/blob/main/"
TABLE_START_MARKER = "<!-- TABLE_START -->"
TABLE_END_MARKER = "<!-- TABLE_END -->"


def link(year: int, month: int) -> str:
    year_dir = os.path.join(main_path, str(year))
    files = [f for f in os.listdir(year_dir) if f.startswith(f"{year}-{month:02d}")]
    if not files:
        return ""
    file = files[0]
    return f"[{CHECK}]({GITHUB_LINK}{year}/{file})"


def row(year: int) -> str:
    return f"| {year} | " + " | ".join([link(year, month) for month in months]) + " |"


def generate_table() -> str:
    header = "|      | " + " | ".join([f"{m:2d}" for m in months]) + " |"
    separator = "|------" + "|----" * 12 + "|"
    rows = "\n".join(map(row, years))
    table = f"{header}\n{separator}\n{rows}\n"
    return table


def update_readme_table():
    # Read the current README
    with open("README.md", encoding="utf-8") as f:
        content = f.read()

    # Find the markers
    start_idx = content.find(TABLE_START_MARKER)
    end_idx = content.find(TABLE_END_MARKER)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find table markers in README.md")

    # Generate the new table
    table = generate_table()

    # Combine the parts
    new_content = content[: start_idx + len(TABLE_START_MARKER)] + "\n" + table + "\n" + content[end_idx:]

    # Write the updated README
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(new_content)


if __name__ == "__main__":
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    years = list(sorted(int(f) for f in os.listdir(main_path) if f.isnumeric()))
    months = list(range(1, 13))

    update_readme_table()
