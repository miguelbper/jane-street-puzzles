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
cross = '❌'


# Hardcoded values of table
# ----------------------------------------------------------------------

hardcoded_values = {
    '2023-12': cross,  # hall of mirrors 2
    '2020-03': cross,
}


# Table
# ----------------------------------------------------------------------

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
years = list(sorted(int(f) for f in os.listdir(main_path) if f.isnumeric()))
months = list(range(1, 13))
github_link = 'https://github.com/miguelbper/jane-street-puzzles/blob/main/'

header = '|      | ' + ' | '.join([f'{m:2d}' for m in months]) + ' |'
separator = '|------' + '|----' * 12 + '|'

def link(year: int, month: int) -> str:
    if f'{year}-{month:02d}' in hardcoded_values:
        return hardcoded_values[f'{year}-{month:02d}']
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


# Legend
# ----------------------------------------------------------------------
markdown += '\n\n| Icon | Description |\n'
markdown += '|--------|-------------|\n'
markdown += f'| {check} | Solved |\n'
markdown += f'| {cross} | No puzzle / solved without code / code not required $\\Longrightarrow$ not included in repo |\n'
markdown += f'|         | Unsolved / Haven\'t got to it yet |\n'


# Write to README.md
# ----------------------------------------------------------------------
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(markdown)