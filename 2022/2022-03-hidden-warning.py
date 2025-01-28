import pandas as pd
import tabulate

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# fmt: off
table = [
    ("Early stage of an animal"                  , "BABY"    , "AB" , "B" ),
    ("Early stage of a plant"                    , "SEED"    , "CD" , ""  ),
    ("Creepy"                                    , "EERIE"   , "E"  , "R" ),
    ("Reviled image that might fire up a crowd?" , "EFFIGY"  , "FG" , "I" ),
    ("In minesweeper, it means you're surrounded", "EIGHT"   , "H"  , ""  ),
    ("Spiced brew"                               , "CHAI"    , "I"  , ""  ),
    ("Tapper on CNN"                             , "JAKE"    , "JK" , ""  ),
    ("Iron or Bronze"                            , "AGE"     , ""   , "GE"),
    ("Iron but not bronze"                       , "ELEMENT" , "LMN", "T" ),
    ("Boy on 'The Andy Griffith Show'"           , "OPIE"    , "OP" , ""  ),
    ("Signal"                                    , "CUE"     , "Q"  , ""  ),
    ("Suspect, once apprehended"                 , "ARRESTEE", "RST", ""  ),
    ("'__ got mail'"                             , "YOUVE"   , "UV" , ""  ),
    ("Raise the stakes yet again, in backgammon" , "REDOUBLE", ""   , "R" ),
    ("First mammal to be cloned"                 , "EWE"     , "W"  , ""  ),
    ("Outer: prefix"                             , "EXO"     , "X"  , "O" ),
    ("With consideration"                        , "WISELY"  , "YZ" , "L" ),
]
# fmt: on

df = pd.DataFrame(table, columns=["Hint", "Answer", "Letters", "Extra"])
print(tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False), end="\n\n")
message = "".join(df["Extra"])
print(f"{message = }")
# | Hint                                       | Answer   | Letters   | Extra   |
# |:-------------------------------------------|:---------|:----------|:--------|
# | Early stage of an animal                   | BABY     | AB        | B       |
# | Early stage of a plant                     | SEED     | CD        |         |
# | Creepy                                     | EERIE    | E         | R       |
# | Reviled image that might fire up a crowd?  | EFFIGY   | FG        | I       |
# | In minesweeper, it means you're surrounded | EIGHT    | H         |         |
# | Spiced brew                                | CHAI     | I         |         |
# | Tapper on CNN                              | JAKE     | JK        |         |
# | Iron or Bronze                             | AGE      |           | GE      |
# | Iron but not bronze                        | ELEMENT  | LMN       | T       |
# | Boy on 'The Andy Griffith Show'            | OPIE     | OP        |         |
# | Signal                                     | CUE      | Q         |         |
# | Suspect, once apprehended                  | ARRESTEE | RST       |         |
# | '__ got mail'                              | YOUVE    | UV        |         |
# | Raise the stakes yet again, in backgammon  | REDOUBLE |           | R       |
# | First mammal to be cloned                  | EWE      | W         |         |
# | Outer: prefix                              | EXO      | X         | O       |
# | With consideration                         | WISELY   | YZ        | L       |

# message = 'BRIGETROL'
# answer: BRIDGETROLL
