# Imports
# ----------------------------------------------------------------------
import pandas as pd
from tabulate import tabulate
from collections import Counter
import os


# Get list of english words
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
words_path = os.path.join(main_path, 'utils', 'words.txt')
with open(words_path) as f:
    words = f.read().splitlines()


# Hints
# ----------------------------------------------------------------------
df = pd.DataFrame({
    'hints': [
        "disgust",
        "mollifies",
        "calling in",
        "it's pickled",
        "staple that's approx 1 to 2 mm",
        "one who will take a leaner, maybe",
        "a 90's spoof (if you insert a space)",
        "guts",
        "like most sets",
        "swinging makes one less impressive",
        "show up again",
        "the player at bat after you, say",
    ],
    'tuples': [
        (3,1,1),
        (4,1,3,2),
        (3,1,3,4,2),
        (4,4,4,4),
        (4,4,4,4),
        (6,6,7,2,4),
        (4,4,4,4),
        (6,6,3,5,5),
        (2,2,3,4,2),
        (5,3,1),
        (7,4,4,1),
        (6,6,3,1),
    ],
    'bold': [1, 0, 4, 0, 0, 0, 0, 3, 0, 0, 1, 2],
})


print(tabulate(df, headers='keys', tablefmt='psql'))
'''
+----+--------------------------------------+-----------------+
|    | hints                                | tuples          |
|----+--------------------------------------+-----------------|
|  0 | disgust                              | (3, 1, 1)       |
|  1 | mollifies                            | (4, 1, 3, 2)    |
|  2 | calling in                           | (3, 1, 3, 4, 2) |
|  3 | it's pickled                         | (4, 4, 4, 4)    |
|  4 | staple that's approx 1 to 2 mm       | (4, 4, 4, 4)    |
|  5 | one who will take a leaner, maybe    | (6, 6, 7, 2, 4) |
|  6 | a 90's spoof (if you insert a space) | (4, 4, 4, 4)    |
|  7 | guts                                 | (6, 6, 3, 5, 5) |
|  8 | like most sets                       | (2, 2, 3, 4, 2) |
|  9 | swinging makes one less impressive   | (5, 3, 1)       |
| 10 | show up again                        | (7, 4, 4, 1)    |
| 11 | the player at bat after you, say     | (6, 6, 3, 1)    |
+----+--------------------------------------+-----------------+
'''

'''
Logic:
- Each hint has an answer which is a single word
- In each answer word, each character appears exactly twice (pair dance)
- The distance between the two appearances of each character is given by
  the tuple.
'''


# Search english dictionary for matching words
# ----------------------------------------------------------------------

def to_tuple(word: str) -> tuple[int, ...]:
    '''Return tuple of distances between pairs of characters in word.
    Assumes that the word has exactly two of each character.'''
    counter = Counter(word)
    distances = []
    for char in counter.keys():
        indices = [i for i, c in enumerate(word) if c == char]

        if len(indices) < 2:
            pass

        distance = indices[1] - indices[0]
        distances.append(distance)
    return tuple(distances)


def valid(word: str, tupl: int) -> bool:
    '''Check if word has n pairs of the same character with the given
    distances.'''
    c = Counter(word)
    n = len(tupl)

    ans = (
        len(c) == n
        and all(v == 2 for v in c.values())
        and tupl == to_tuple(word)
    )

    return ans


for i, row in df.iterrows():
    hint = row['hints']
    tupl = row['tuples']
    valid_words = [word for word in words if valid(word, tupl)]
    num_valid_words = len(valid_words)

    print('\n')
    print(f'{hint = }')
    print(f'{tupl = }')
    print(f'There are {num_valid_words} valid words:')
    for word in valid_words:
        print(word)
'''
hint = 'disgust'
tupl = (3, 1, 1)
There are 4 valid words:
appall <-
immiss
uppuff
uppull


hint = 'mollifies'
tupl = (4, 1, 3, 2)
There are 2 valid words:
appearer
appeases <-


hint = 'calling in'
tupl = (3, 1, 3, 4, 2)
There are 1 valid words:
arraigning <-


hint = "it's pickled"
tupl = (4, 4, 4, 4)
There are 23 valid words:
beriberi
buhlbuhl
bulnbuln
chowchow <-
couscous
froufrou
grisgris
guitguit
hotshots
yariyari
khuskhus
kohekohe
lapulapu
lomilomi
mahimahi
makomako
mokamoka
pincpinc
pioupiou
quiaquia
teruteru
ticktick
tucotuco


hint = "staple that's approx 1 to 2 mm"
tupl = (4, 4, 4, 4)
There are 23 valid words:
beriberi
buhlbuhl
bulnbuln
chowchow
couscous <-
froufrou
grisgris
guitguit
hotshots
yariyari
khuskhus
kohekohe
lapulapu
lomilomi
mahimahi
makomako
mokamoka
pincpinc
pioupiou
quiaquia
teruteru
ticktick
tucotuco


hint = 'one who will take a leaner, maybe'
tupl = (6, 6, 7, 2, 4)
There are 1 valid words:
horseshoer <-


hint = "a 90's spoof (if you insert a space)"
tupl = (4, 4, 4, 4)
There are 23 valid words:
beriberi
buhlbuhl
bulnbuln
chowchow
couscous
froufrou
grisgris
guitguit
hotshots <-
yariyari
khuskhus
kohekohe
lapulapu
lomilomi
mahimahi
makomako
mokamoka
pincpinc
pioupiou
quiaquia
teruteru
ticktick
tucotuco


hint = 'guts'
tupl = (6, 6, 3, 5, 5)
There are 1 valid words:
intestines <-


hint = 'like most sets'
tupl = (2, 2, 3, 4, 2)
There are 1 valid words:
nonordered <-


hint = 'swinging makes one less impressive'
tupl = (5, 3, 1)
There are 14 valid words:
degged
denned
hallah
mallam
marram
pullup <-
redder
retter
selles
succus
tebbet
terret
tibbit
tirrit


hint = 'show up again'
tupl = (7, 4, 4, 1)
There are 1 valid words:
reappear <-


hint = 'the player at bat after you, say'
tupl = (6, 6, 3, 1)
There are 2 valid words:
shammash
teammate <-
'''

df['answers'] = [
    'appall',
    'appeases',
    'arraigning',
    'chowchow',
    'couscous',
    'horseshoer',
    'hotshots',
    'intestines',
    'nonordered',
    'pullup',
    'reappear',
    'teammate',
]


# Check which letter corresponds to the bold index

def bold_letter(word: str, index: int) -> str:
    counter = Counter(word)
    return list(counter.keys())[index]

letters = []
for i, row in df.iterrows():
    word = row['answers']
    index = row['bold']
    letters.append(bold_letter(word, index))
df['letters'] = letters


print(tabulate(df, headers='keys', tablefmt='psql'))
'''
+----+--------------------------------------+-----------------+--------+------------+-----------+
|    | hints                                | tuples          |   bold | answers    | letters   |
|----+--------------------------------------+-----------------+--------+------------+-----------|
|  0 | disgust                              | (3, 1, 1)       |      1 | appall     | p         |
|  1 | mollifies                            | (4, 1, 3, 2)    |      0 | appeases   | a         |
|  2 | calling in                           | (3, 1, 3, 4, 2) |      4 | arraigning | n         |
|  3 | it's pickled                         | (4, 4, 4, 4)    |      0 | chowchow   | c         |
|  4 | staple that's approx 1 to 2 mm       | (4, 4, 4, 4)    |      0 | couscous   | c         |
|  5 | one who will take a leaner, maybe    | (6, 6, 7, 2, 4) |      0 | horseshoer | h         |
|  6 | a 90's spoof (if you insert a space) | (4, 4, 4, 4)    |      0 | hotshots   | h         |
|  7 | guts                                 | (6, 6, 3, 5, 5) |      3 | intestines | e         |
|  8 | like most sets                       | (2, 2, 3, 4, 2) |      0 | nonordered | n         |
|  9 | swinging makes one less impressive   | (5, 3, 1)       |      0 | pullup     | p         |
| 10 | show up again                        | (7, 4, 4, 1)    |      1 | reappear   | e         |
| 11 | the player at bat after you, say     | (6, 6, 3, 1)    |      2 | teammate   | a         |
+----+--------------------------------------+-----------------+--------+------------+-----------+
'''

# 12 hints and [p, a, n, c, c, h, h, e, n, p, e, a]
# What words have 12 characters and contain each letter exactly twice?

for word in words:
    counter = Counter(word)
    if len(counter) == 6 and all(v == 2 for v in counter.values()):
        print(word)
# great-great-
# happenchance <- (anagram of pancchhenpea)

ans = to_tuple('happenchance')
print(f'{ans = }')
# ans = (7, 7, 1, 7, 4, 4)
