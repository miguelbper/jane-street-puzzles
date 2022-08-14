import enchant
import numpy as np
from pprint import pprint
from collections import Counter
from string import ascii_lowercase

l_min = 4
i_min = l_min - 1


d = enchant.Dict("en_US")


soup = [
    ['y', 't', 'i', 'm', 'o', 'i', 'c', 'z', 'r', 'y', 'z', 'w', 'b', 'u', 's'],
    ['t', 'e', 'g', 'c', 'i', 'r', 'b', 'p', 'y', 'l', 'k', 'o', 'o', 'r', 'b'],
    ['r', 'e', 't', 'p', 'e', 'm', 'p', 'l', 'o', 'm', 'p', 'i', 'l', 'z', 'y'],
    ['e', 'r', 'h', 'i', 'g', 'h', 'l', 'i', 'p', 'e', 't', 'z', 'x', 'i', 'z'],
    ['b', 't', 'm', 'l', 'o', 'i', 's', 't', 'e', 'r', 's', 'l', 'i', 'c', 'p'],
    ['i', 's', 's', 't', 'o', 'p', 'e', 's', 't', 'r', 'e', 'e', 't', 'r', 'k'],
    ['l', 'l', 't', 'e', 'c', 'z', 'e', 'r', 'e', 'p', 'z', 'u', 'c', 'z', 'e'],
    ['a', 'l', 'm', 'e', 'p', 't', 'r', 'z', 'l', 'f', 'z', 'r', 'k', 'u', 'e'],
    ['o', 'z', 't', 'b', 'r', 'o', 'p', 'x', 'y', 'o', 'o', 'l', 'e', 'g', 's'],
    ['e', 'w', 'j', 'a', 'k', 'o', 'r', 'e', 'z', 't', 'o', 'w', 'p', 'z', 't'],
    ['u', 'i', 'g', 'r', 'z', 'p', 'c', 'm', 'e', 'p', 't', 'r', 'z', 'l', 'z'],
    ['t', 't', 'z', 'l', 'm', 'o', 'p', 'e', 'y', 'i', 's', 'l', 'z', 'p', 'c'],
    ['z', 'e', 'r', 'z', 'u', 'q', 's', 'p', 'o', 's', 'i', 'c', 'z', 'μ', 'i'],
    ['t', 'l', 'e', 'p', 'p', 'u', 't', 'c', 'p', 'z', 'l', 'l', 'o', 'h', 'u'],
    ['s', 'r', 'z', 'b', 'z', 'y', 'y', 'p', 'e', 'w', 'n', 's', 'e', 'u', 'μ']
]


translation = {
    'c': 'd', # 3. yankeestaciuμ -> yankeestadium
    'm': 'c', # 4. radiomity -> radiocity
    'μ': 'm', # 3. yankeestaciuμ -> yankeestadium
    'n': 'μ', # 5. newnseum -> newμseum (newmuseum)
    'p': 'n', # 1. koreztowp -> koreatown
    'f': 'p', # 6. centralfark -> centralpark
    'a': 'f', # 2. statueoaliberty -> statueofliberty
    'z': 'a', # 1. koreztowp -> koreatown 
}


translation_mu = {'μ': 'mu'}


def replace_dict(d, s):
    for k, v in d.items():
        if s.find(k) != -1:
            s = s.replace(k, v)
            break
    return s


def translate(d, xss):
    return [[replace_dict(d, x) for x in xs] for xs in xss]


soup_translated = translate(translation, soup)
pprint(soup_translated)
# soup_translated = 
# y t i c o i d a r y a w b u s
# t e g d i r b n y l k o o r b
# r e t n e c n l o c n i l a y
# e r h i g h l i n e t a x i a
# b t c l o i s t e r s l i d n
# i s s t o n e s t r e e t r k
# l l t e d a e r e n a u d a e
# f l c e n t r a l p a r k u e
# o a t b r o n x y o o l e g s
# e w j f k o r e a t o w n a t
# u i g r a n d c e n t r a l a
# t t a l c o n e y i s l a n d
# a e r a u q s n o s i d a m i
# t l e n n u t d n a l l o h u
# s r a b a y y n e w μ s e u m


# words = 
# bronxyoo
# chinatoon
# laguardia
# jfk
# newμseum
# hollandtunnel
# lincolncenter
# radiocity
# brooklynbridge
# centralpark
# madisonsquare
# coneyisland
# grandcentral
# koreatown
# stonestreet
# cloisters
# highline
# wallstreet
# statueofliberty
# yankeestadium


soup_translated_mu = translate(translation_mu, soup_translated)


def sublists(xs: list[str]) -> list[list[str]]:
    lists = []
    for i in range(i_min, len(xs)):
        for j in range(len(xs) - i):
            lists.append(xs[j: j + i + 1])
    return lists


def columns(xss: list[list[str]]) -> list[list[str]]:
    return list(map(list, zip(*xss)))


def rotated(xss: list[list[str]]) -> list[list[str]]:
    return list(map(list, zip(*reversed(xss))))


def diags_0(xss: list[list[str]]) -> list[list[str]]:
    o = len(xss) - 1
    a = np.array(xss)
    return [list(np.diagonal(a, offset=i)) for i in range(-o, o + 1)]


def diags_1(xss: list[list[str]]) -> list[list[str]]:
    return diags_0(rotated(xss))


def is_word(xs: list[str]) -> bool:
    word = ''.join(xs)
    return d.check(word)


def words_in_str(xs: list[str]) -> list[str]:
    return [''.join(w) for w in sublists(xs) if is_word(w)]


funcs = {
    'rows': lambda xss: xss,
    'columns': columns,
    'diags_0': diags_0,
    'diags_1': diags_1,
    'rows_rev': lambda xss: [list(reversed(xs)) for xs in xss],
    'columns_rev': lambda xss: [list(reversed(xs)) for xs in columns(xss)],
    'diags_0_rev': lambda xss: [list(reversed(xs)) for xs in diags_0(xss)],
    'diags_1_rev': lambda xss: [list(reversed(xs)) for xs in diags_1(xss)],
}


words = {k: [words_in_str(s) for s in f(soup_translated_mu)] for k, f in funcs.items()}
pprint(words)