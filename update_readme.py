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
done = '✔'
ignored = '➖'


# Hardcoded values of table
# ----------------------------------------------------------------------

hardcoded_values = {
    '2023-12': ignored,  # hall of mirrors 2
    '2024-01': ignored,  # some f squares

}


# Table
# ----------------------------------------------------------------------

current_dir = os.getcwd()
years = [int(f) for f in os.listdir(current_dir) if f.isnumeric()]
months = list(range(1, 13))
github_link = 'https://github.com/miguelbper/jane-street-puzzles/blob/main/'

header = '|      | ' + ' | '.join([f'{m:2d}' for m in months]) + ' |'
separator = '|------' + '|----' * 12 + '|'

def link(year: int, month: int) -> str:
    if f'{year}-{month:02d}' in hardcoded_values:
        return hardcoded_values[f'{year}-{month:02d}']
    year_dir = os.path.join(current_dir, str(year))
    files = [f for f in os.listdir(year_dir) if f.startswith(f'{year}-{month:02d}')]
    if not files:
        return ''
    file = files[0]
    return f'[{done}]({github_link}{year}/{file})'

def row(year: int) -> str:
    return f'| {year} | ' + ' | '.join([link(year, month) for month in months]) + ' |'

rows = '\n'.join(map(row, years))
table = header + '\n' + separator + '\n' + rows
markdown += table


# Legend
# ----------------------------------------------------------------------
# markdown += '\n\n**Legend:**\n'
# Legend
# ----------------------------------------------------------------------

markdown += '\n\n| Icon | Description |\n'
markdown += '|--------|-------------|\n'
markdown += f'| {done}    | Solved      |\n'
markdown += f'| {ignored} | Solved without code, or code not required $\Longrightarrow$ not included in repo |\n'
markdown += f'|           | Unsolved / Haven\'t gotten to it yet |\n'


# Write to README.md
# ----------------------------------------------------------------------
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(markdown)