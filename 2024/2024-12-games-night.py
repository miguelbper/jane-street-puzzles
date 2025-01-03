import pandas as pd
import tabulate

# For each game, there is a letter written with pieces of that game
# List the games and letters in order of appearance
# The letters spell out "Missing Ones"
# -> For each game, find what piece is missing from the shown set

# fmt: off
table = [
    ("Uno"       , "M", "Yellow"),
    ("Catan"     , "I", "Ore"   ),
    ("Risk"      , "S", "Ural"  ),
    ("Monopoly"  , "S", " "     ),
    ("Dominion"  , "I", "Silver"),
    ("Scrabble"  , "N", "U"     ),
    ("Sushi Go"  , "G", "Nigiri"),
    ("Chess"     , "O", "King"  ),
    ("Clue"      , "N", " "     ),
    ("Stratego"  , "E", "Miners"),
    ("Bananagram", "S", "Y"     ),
]
# fmt: on

df = pd.DataFrame(table, columns=["Game", "Letter", "Missing"])
df["First letter of missing"] = df["Missing"].str[0]
print(tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False), end="\n\n")
# | Game       | Letter   | Missing   | First letter of missing   |
# |:-----------|:---------|:----------|:--------------------------|
# | Uno        | M        | Yellow    | Y                         |
# | Catan      | I        | Ore       | O                         |
# | Risk       | S        | Ural      | U                         |
# | Monopoly   | S        |           |                           |
# | Dominion   | I        | Silver    | S                         |
# | Scrabble   | N        | U         | U                         |
# | Sushi Go   | G        | Nigiri    | N                         |
# | Chess      | O        | King      | K                         |
# | Clue       | N        |           |                           |
# | Stratego   | E        | Miners    | M                         |
# | Bananagram | S        | Y         | Y                         |

message = "".join(df["First letter of missing"])
print(f"{message = }")
# message = 'YOU SUNK MY'
# answer: BATTLESHIP
