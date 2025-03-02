import polars as pl
from termcolor import colored

pl.Config.set_tbl_rows(50)
RED = False
BLUE = True


# There are 14 + 14 words on the left
# There are numbers 1 to 14 on the right, in red and blue
# This suggests that there is a correspondence between the two

# "Give or take" suggests that we should "give or take" a letter from the words on the left
# Find "approximate anagrams" of the words on the left

# More precisely,
# top words - blue - remove a letter and create an anagram
# bot words - red - add a letter and create an anagram

# Each row on the right is the title of a movie


# fmt: off
df_top = [
    ("ADMIRERS"  , 7 , "RAIDERS"    , "M"),
    ("ASTIR"     , 2 , "STAR"       , "I"),
    ("BLACK"     , 3 , "BACK"       , "L"),
    ("DITHER"    , 11, "THIRD"      , "E"),
    ("DRINK"     , 12, "KIND"       , "R"),
    ("HOAGIES"   , 9 , "GEISHA"     , "O"),
    ("JOLTS"     , 13, "LOST"       , "J"),
    ("OKRA"      , 6 , "ROCK"       , "O"),
    ("PREMISE"   , 10, "EMPIRE"     , "S"),
    ("STICKERS"  , 5 , "STRIKES"    , "C"),
    ("SURFACED"  , 1 , "CRUSADE"    , "F"),
    ("SWARM"     , 4 , "WARS"       , "M"),
    ("WILTS"     , 14, "LIST"       , "W"),
    ("WRAP"      , 8 , "WAR"        , "P"),
]

df_bot = [
    ("ARK"       , 8 , "PARK"       , "P"),
    ("CHILDREN'S", 10, "SCHINDLER'S", "S"),
    ("CUIRASS"   , 13, "JURASSIC"   , "J"),
    ("FOR"       , 6 , "ROOF"       , "O"),
    ("HOE"       , 4 , "HOME"       , "M"),
    ("ISOMER"    , 7 , "MEMOIRS"    , "M"),
    ("LANE"      , 9 , "ALONE"      , "O"),
    ("LORDS"     , 14, "WORLDS"     , "W"),
    ("NOCTURNES" , 11, "ENCOUNTERS" , "E"),
    ("RIDDLE"    , 1 , "FIDDLER"    , "F"),
    ("SAT"       , 3 , "LAST"       , "L"),
    ("SOLE"      , 5 , "CLOSE"      , "C"),
    ("TRIONYM"   , 2 , "MINORITY"   , "I"),
    ("TROPE"     , 12, "REPORT"     , "R"),
]
# fmt: on

df_top = pl.from_records(df_top, schema=["word", "number", "anagram", "extra letter"], orient="row").sort("number")
df_bot = pl.from_records(df_bot, schema=["word", "number", "anagram", "extra letter"], orient="row").sort("number")


def word(color: bool, index: int) -> str:
    df = df_top if color else df_bot
    color = "blue" if color else "red"
    title = df["anagram"][index - 1]
    return colored(title, color)


titles = [
    f"{word(RED, 1)} on the {word(RED, 6)}",
    f"{word(BLUE, 2)} {word(BLUE, 4)}",
    f"{word(RED, 5)} {word(RED, 11)} of the {word(BLUE, 11)} {word(BLUE, 12)}",
    f"The {word(BLUE, 10)} {word(BLUE, 5)} {word(BLUE, 3)}",
    f"{word(BLUE, 7)} of the {word(BLUE, 13)} {word(BLUE, 6)}",
    f"INDIANA JONES and the {word(RED, 3)} {word(BLUE, 1)}",
    f"{word(RED, 4)} {word(RED, 9)}",
    f"{word(RED, 13)} {word(RED, 8)}",
    f"{word(RED, 10)} {word(BLUE, 14)}",
    f"{word(RED, 2)} {word(RED, 12)}",
    f"{word(BLUE, 8)} of the {word(RED, 14)}",
    f"{word(RED, 7)} of a {word(BLUE, 9)}",
]


print(df_top)
print(df_bot)
print("\nTitles:")
for title in titles:
    print(title)
"""
shape: (14, 4)
┌──────────┬────────┬─────────┬──────────────┐
│ word     ┆ number ┆ anagram ┆ extra letter │
│ ---      ┆ ---    ┆ ---     ┆ ---          │
│ str      ┆ i64    ┆ str     ┆ str          │
╞══════════╪════════╪═════════╪══════════════╡
│ SURFACED ┆ 1      ┆ CRUSADE ┆ F            │
│ ASTIR    ┆ 2      ┆ STAR    ┆ I            │
│ BLACK    ┆ 3      ┆ BACK    ┆ L            │
│ SWARM    ┆ 4      ┆ WARS    ┆ M            │
│ STICKERS ┆ 5      ┆ STRIKES ┆ C            │
│ OKRA     ┆ 6      ┆ ROCK    ┆ O            │
│ ADMIRERS ┆ 7      ┆ RAIDERS ┆ M            │
│ WRAP     ┆ 8      ┆ WAR     ┆ P            │
│ HOAGIES  ┆ 9      ┆ GEISHA  ┆ O            │
│ PREMISE  ┆ 10     ┆ EMPIRE  ┆ S            │
│ DITHER   ┆ 11     ┆ THIRD   ┆ E            │
│ DRINK    ┆ 12     ┆ KIND    ┆ R            │
│ JOLTS    ┆ 13     ┆ LOST    ┆ J            │
│ WILTS    ┆ 14     ┆ LIST    ┆ W            │
└──────────┴────────┴─────────┴──────────────┘

shape: (14, 4)
┌────────────┬────────┬─────────────┬──────────────┐
│ word       ┆ number ┆ anagram     ┆ extra letter │
│ ---        ┆ ---    ┆ ---         ┆ ---          │
│ str        ┆ i64    ┆ str         ┆ str          │
╞════════════╪════════╪═════════════╪══════════════╡
│ RIDDLE     ┆ 1      ┆ FIDDLER     ┆ F            │
│ TRIONYM    ┆ 2      ┆ MINORITY    ┆ I            │
│ SAT        ┆ 3      ┆ LAST        ┆ L            │
│ HOE        ┆ 4      ┆ HOME        ┆ M            │
│ SOLE       ┆ 5      ┆ CLOSE       ┆ C            │
│ FOR        ┆ 6      ┆ ROOF        ┆ O            │
│ ISOMER     ┆ 7      ┆ MEMOIRS     ┆ M            │
│ ARK        ┆ 8      ┆ PARK        ┆ P            │
│ LANE       ┆ 9      ┆ ALONE       ┆ O            │
│ CHILDREN'S ┆ 10     ┆ SCHINDLER'S ┆ S            │
│ NOCTURNES  ┆ 11     ┆ ENCOUNTERS  ┆ E            │
│ TROPE      ┆ 12     ┆ REPORT      ┆ R            │
│ CUIRASS    ┆ 13     ┆ JURASSIC    ┆ J            │
│ LORDS      ┆ 14     ┆ WORLDS      ┆ W            │
└────────────┴────────┴─────────────┴──────────────┘

Titles:
FIDDLER on the ROOF
STAR WARS
CLOSE ENCOUNTERS of the THIRD KIND
The EMPIRE STRIKES BACK
RAIDERS of the LOST ROCK
INDIANA JONES and the LAST CRUSADE
HOME ALONE
JURASSIC PARK
SCHINDLER'S LIST
MINORITY REPORT
WAR of the WORLDS
MEMOIRS of a GEISHA
"""


# After filling out the grid, we conclude
# - All the movies had John Williams as the composer
# - The letters removed / added form the sentence "FILM COMPOSER JW"
# - Suggesting that the answer is...
answer = "John Williams"
print(f"\n{answer = }")
