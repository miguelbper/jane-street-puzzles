# Imports
# ----------------------------------------------------------------------
import os


# Base text
# ----------------------------------------------------------------------
markdown = ''
markdown += '# Jane Street Puzzles\n\n'
markdown += 'My solutions to past Jane Street puzzles (see https://www.janestreet.com/puzzles/archive/).\n\n'


# Emojis
# ----------------------------------------------------------------------
check = '✔'


# Table
# ----------------------------------------------------------------------

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
years = list(sorted(int(f) for f in os.listdir(main_path) if f.isnumeric()))
months = list(range(1, 13))
github_link = 'https://github.com/miguelbper/jane-street-puzzles/blob/main/'

header = '|      | ' + ' | '.join([f'{m:2d}' for m in months]) + ' |'
separator = '|------' + '|----' * 12 + '|'

def link(year: int, month: int) -> str:
    year_dir = os.path.join(main_path, str(year))
    files = [f for f in os.listdir(year_dir) if f.startswith(f'{year}-{month:02d}')]
    if not files:
        return ''
    file = files[0]
    return f'[{check}]({github_link}{year}/{file})'

def row(year: int) -> str:
    return f'| {year} | ' + ' | '.join([link(year, month) for month in months]) + ' |'

rows = '\n'.join(map(row, years))
table = header + '\n' + separator + '\n' + rows
markdown += table


# Write to README.md
# ----------------------------------------------------------------------
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(markdown)
