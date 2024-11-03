import numpy as np

# Boards
# ----------------------------------------------------------------------
# fmt: off
board_0 = np.array([
    ['POLO' , 'ENGLAND' , 'SKYSCRAPER', 'DRESS', 'TUXEDO'],
    ['AGENT', 'COMPOUND', 'DECK'      , 'SHOE' , 'SHORTS'],
    ['BOOT' , 'PLANE'   , 'SCHOOL'    , 'CAP'  , 'TEXAS' ],
    ['BOMB' , 'DASH'    , 'TELESCOPE' , 'TIN'  , 'GLOVE' ],
    ['KISS' , 'GOVERNOR', 'SHERLOCK'  , 'SUIT' , 'SUN'   ],
])

board_1 = np.array([
    ['SPACE' , 'MILL'    , 'CIRCLE'  , 'DUCK' , 'POWDER' ],
    ['FEVER' , 'SCORPION', 'OCTOPUS' , 'SILK' , 'WAR'    ],
    ['HOTEL' , 'FOAM'    , 'CUCKOO'  , 'SHEET', 'PENGUIN'],
    ['RABBIT', 'MUD'     , 'GLASSES' , 'SHARK', 'DOG'    ],
    ['TURTLE', 'CLOAK'   , 'REINDEER', 'ICE'  , 'EAGLE'  ],
])

board_2 = np.array([
    ['BANK'    , 'SOUP'  , 'CHEESE'   , 'WELL'     , 'POTATO' ],
    ['MAGAZINE', 'PIE'   , 'SALAD'    , 'CARROT'   , 'PIZZA'  ],
    ['ARMY'    , 'PADDLE', 'HAMBURGER', 'HIMALAYAS', 'COUNTRY'],
    ['CYCLE'   , 'BRIDE' , 'BISCUIT'  , 'PACIFIC'  , 'LAB'    ],
    ['ASH'     , 'KID'   , 'QUEEN'    , 'NOVEL'    , 'JET'    ],
])
# fmt: on

# Answers
# ----------------------------------------------------------------------

board_0_ans = [
    "POLO",
    "DRESS",
    "TUXEDO",
    "SHOE",
    "SHORTS",
    "BOOT",
    "CAP",
    "GLOVE",
    "SUIT",
]

board_1_ans = [
    "DUCK",
    "SCORPION",
    "OCTOPUS",
    "CUCKOO",
    "PENGUIN",
    "RABBIT",
    "SHARK",
    "DOG",
    "TURTLE",
    "REINDEER",
    "EAGLE",
]

board_2_ans = [
    "SOUP",
    "CHEESE",
    "POTATO",
    "PIE",
    "SALAD",
    "CARROT",
    "PIZZA",
    "HAMBURGER",
    "BISCUIT",
]


# Binary
# ----------------------------------------------------------------------


def to_binary(arr: np.ndarray, answers: list[str]) -> np.ndarray:
    return np.vectorize(lambda x: x in answers)(arr).astype(int)


board_0_bin = to_binary(board_0, board_0_ans)
board_1_bin = to_binary(board_1, board_1_ans)
board_2_bin = to_binary(board_2, board_2_ans)
board_bin = np.concatenate([board_0_bin, board_1_bin, board_2_bin])
print(board_bin, end="\n\n")


# Print
# ----------------------------------------------------------------------


def row_to_char(row: np.ndarray) -> str:
    decimal = int("".join(map(str, row)), 2)
    char = chr(decimal - 1 + ord("A"))
    return char


# def to_char

chars = "".join(map(row_to_char, board_bin))
print(chars)  # SCRABBLESUMODD@ -> select words with odd scrabble sum


# Scrabble sum
# ----------------------------------------------------------------------


def scrabble_sum(word: str) -> int:
    #         A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P   Q  R  S  T  U  V  W  X  Y   Z
    points = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10]
    mapping = {chr(i + ord("A")): points[i] for i in range(26)}
    return sum(mapping[char] for char in word) % 2


board = np.concatenate([board_0, board_1, board_2])
board_scrabblesumodd = np.vectorize(scrabble_sum)(board)
chars_scrabblesumodd = "".join(map(row_to_char, board_scrabblesumodd))
print(chars_scrabblesumodd)  # LONGERTHANFIVE@ -> select words longer than five letters


# Longer than five
# ----------------------------------------------------------------------

board_longerthanfive = np.vectorize(lambda x: int(len(x) > 5))(board)
chars_longerthanfive = "".join(map(row_to_char, board_longerthanfive))
print(chars_longerthanfive)  # MIDDLELETTEROF@ -> select middle letter of each word


# Middle letter
# ----------------------------------------------------------------------


def middle_letter(word: str) -> str:
    return word[len(word) // 2]


board_middleletter = np.vectorize(middle_letter)(board)
ans = "".join(board_middleletter[-1])
print(f"{ans = }")  # SIEVE
