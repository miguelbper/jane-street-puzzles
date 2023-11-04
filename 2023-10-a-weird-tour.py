import pandas as pd
from tabulate import tabulate

'''
The vowels in 'A Weird Tour' are 'aeiou'. Moreover,
- The puzzle mentions '120 regions with gradually shifting dialects'.
- Notice that there are 5! = 120 ways to sort 'aeiou'
- It says that '... swapped ends ... using ... methods of transport'
- Each method of transport (bold/italic) has exactly two vowels

This suggests that, we start at 'aeiou', and after taking a method of 
transport with vowels (v0, v1), we should swap v0 and v1 in the current
order of the vowels.

For each clue, we wish to find an answer/region with the 5 vowels aeiou
in the corresponding order.
'''

transports = [
    '', 
    'plane',
    'tour', 
    'big bus', 
    'subway', 
    'tunnel', 
    'train', 
    'boat', 
    'bike', 
    'horse', 
    'limo',
]

orders = []

order = ['a', 'e', 'i', 'o', 'u']
orders.append(''.join(order))

for w in transports[1:]:
    vowels = [c for c in w if c in 'aeiou']
    v0, v1 = tuple(vowels)
    i0, i1 = order.index(v0), order.index(v1)
    order[i0], order[i1] = order[i1], order[i0]
    orders.append(''.join(order))
    print(f'{w = :<7}, vowels = {"".join(vowels)}, order = {"".join(order)}')
'''
w = plane  , vowels = ae, order = eaiou
w = tour   , vowels = ou, order = eaiuo
w = big bus, vowels = iu, order = eauio
w = subway , vowels = ua, order = euaio
w = tunnel , vowels = ue, order = ueaio
w = train  , vowels = ai, order = ueiao
w = boat   , vowels = oa, order = ueioa
w = bike   , vowels = ie, order = uieoa
w = horse  , vowels = oe, order = uioea
w = limo   , vowels = io, order = uoiea
'''

df = pd.DataFrame({
    'num'        : [1          , 2       , 3               , 4         , 5       , 6            , 7                 , 8              , 9     , 10              , 11           ],
    'transport'  : transports,
    'clue'       : ['Unserious', 'Social', 'Learning about', 'Weakness', '???'   , 'Game series', 'Superstar artist', 'Sign of doubt', 'Exec', 'Suddenly emote', 'South Asian'],
    'num_words'  : [1          , 1       , 3               , 1         , 1       , 2            , 2                 , 2              , 1     , 3               , 1            ],
    'tuple'      : [[]         , [1,6,8] , [8,9]           , []        , []      , [4]          , [1]               , [3]            , [11]  , [11]            , [9, 12]      ],
    'permutation': orders,
})
print(tabulate(df, headers='keys', tablefmt='psql'))
'''
+-----+-------------+------------------+-----------+-----------+-------------+
| num | transport   | clue             | num_words | tuple     | permutation |
+-----+-------------+------------------+-----------+-----------+-------------|
|   1 |             | Unserious        |         1 | []        | aeiou       |
|   2 | plane       | Social           |         1 | [1, 6, 8] | eaiou       |
|   3 | tour        | Learning about   |         3 | [8, 9]    | eaiuo       |
|   4 | big bus     | Weakness         |         1 | []        | eauio       |
|   5 | subway      | ???              |         1 | []        | euaio       |
|   6 | tunnel      | Game series      |         2 | [4]       | ueaio       |
|   7 | train       | Superstar artist |         2 | [1]       | ueiao       |
|   8 | boat        | Sign of doubt    |         2 | [3]       | ueioa       |
|   9 | bike        | Exec             |         1 | [11]      | uieoa       |
|  10 | horse       | Suddenly emote   |         3 | [11]      | uioea       |
|  11 | limo        | South Asian      |         1 | [9, 12]   | uoiea       |
+-----+-------------+------------------+-----------+-----------+-------------+
'''

# Filling out the clues
'''
To fill out each answer, it is helpful to search for words in the 
dictionary whose vowels are exactly 'aeiou' in the given permutation.
This can be done programatically and works for the clues whose answer is
one word.
'''

regions = [             # num | clue             | num_words | permutation
    'facetious'       , #   1 | Unserious        |         1 | aeiou      
    'gregarious'      , #   2 | Social           |         1 | eaiou      
    'reading up on'   , #   3 | Learning about   |         3 | eaiuo      
    'exhaustion'      , #   4 | Weakness         |         1 | eauio      
    ''                , #   5 | ???              |         1 | euaio      
    'super mario'     , #   6 | Game series      |         2 | ueaio      
    'lupe fiasco'     , #   7 | Superstar artist |         2 | ueiao      
    'question mark'   , #   8 | Sign of doubt    |         2 | ueioa      
    'businesswoman'   , #   9 | Exec             |         1 | uieoa      
    'burst into tears', #  10 | Suddenly emote   |         3 | uioea      
    'subcontinental'  , #  11 | South Asian      |         1 | uoiea      
]

df['regions'] = regions

'''
Now use the tuples acompanying each clue as indices to find letters in 
the answer/region. Example: gregarious, [1, 6, 8] -> gro
'''

def get_letters(row: pd.DataFrame) -> str:
    w = ''.join(row['regions'].split())
    t = row['tuple']
    ans = []
    for i in t:
        char = w[i - 1] if 0 < i <= len(w) else '_'
        ans.append(char)
    return ''.join(ans)
    
df['letters'] = df.apply(get_letters, axis=1)
print(tabulate(df, headers='keys', tablefmt='psql'))
'''
+-----+-----------+------------------+-----------+-----------+-------------+------------------+---------+
| num | transport | clue             | num_words | tuple     | permutation | regions          | letters |
+-----+-----------+------------------+-----------+-----------+-------------+------------------+---------|
|   1 |           | Unserious        |         1 | []        | aeiou       | facetious        |         |
|   2 | plane     | Social           |         1 | [1, 6, 8] | eaiou       | gregarious       | gro     |
|   3 | tour      | Learning about   |         3 | [8, 9]    | eaiuo       | reading up on    | up      |
|   4 | big bus   | Weakness         |         1 | []        | eauio       | exhaustion       |         |
|   5 | subway    | ???              |         1 | []        | euaio       |                  |         |
|   6 | tunnel    | Game series      |         2 | [4]       | ueaio       | super mario      | e       |
|   7 | train     | Superstar artist |         2 | [1]       | ueiao       | lupe fiasco      | l       |
|   8 | boat      | Sign of doubt    |         2 | [3]       | ueioa       | question mark    | e       |
|   9 | bike      | Exec             |         1 | [11]      | uieoa       | businesswoman    | m       |
|  10 | horse     | Suddenly emote   |         3 | [11]      | uioea       | burst into tears | e       |
|  11 | limo      | South Asian      |         1 | [9, 12]   | uoiea       | subcontinental   | nt      |
+-----+-----------+------------------+-----------+-----------+-------------+------------------+---------+

the letters spell out 'group element', which we can interpret as the 
clue for region 5.

In math, for each set we can form the group of permutations of that set.
This is a group whose elements are permutations. Therefore, the answer
to clue no 5 is 'permutation'.

This is also fitting if we take into account that in this puzzle we 
considered the permutations of the set {a, e, i, o, u}.

Completed table with clues and regions:
+-----+-----------+------------------+-----------+-----------+-------------+------------------+---------+
| num | transport | clue             | num_words | tuple     | permutation | regions          | letters |
+-----+-----------+------------------+-----------+-----------+-------------+------------------+---------|
|   1 |           | Unserious        |         1 | []        | aeiou       | facetious        |         |
|   2 | plane     | Social           |         1 | [1, 6, 8] | eaiou       | gregarious       | gro     |
|   3 | tour      | Learning about   |         3 | [8, 9]    | eaiuo       | reading up on    | up      |
|   4 | big bus   | Weakness         |         1 | []        | eauio       | exhaustion       |         |
|   5 | subway    | GROUP ELEMENT    |         1 | []        | euaio       | PERMUTATION      |         |
|   6 | tunnel    | Game series      |         2 | [4]       | ueaio       | super mario      | e       |
|   7 | train     | Superstar artist |         2 | [1]       | ueiao       | lupe fiasco      | l       |
|   8 | boat      | Sign of doubt    |         2 | [3]       | ueioa       | question mark    | e       |
|   9 | bike      | Exec             |         1 | [11]      | uieoa       | businesswoman    | m       |
|  10 | horse     | Suddenly emote   |         3 | [11]      | uioea       | burst into tears | e       |
|  11 | limo      | South Asian      |         1 | [9, 12]   | uoiea       | subcontinental   | nt      |
+-----+-----------+------------------+-----------+-----------+-------------+------------------+---------+
'''