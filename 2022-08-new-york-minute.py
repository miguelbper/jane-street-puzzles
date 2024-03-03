from pprint import pprint


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
    ['s', 'r', 'z', 'b', 'z', 'y', 'y', 'p', 'e', 'w', 'n', 's', 'e', 'u', 'μ'],
]


translation = {
    'c': 'd',  # 3. yankeestaciuμ -> yankeestadium
    'm': 'c',  # 4. radiomity -> radiocity
    'μ': 'm',  # 3. yankeestaciuμ -> yankeestadium
    'n': 'μ',  # 5. newnseum -> newμseum (newmuseum)
    'p': 'n',  # 1. koreztowp -> koreatown
    'f': 'p',  # 6. centralfark -> centralpark
    'a': 'f',  # 2. statueoaliberty -> statueofliberty
    'z': 'a',  # 1. koreztowp -> koreatown
}


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

'''
soup_translated =
y t i c o i d a r y a w b u s
t e g d i r b n y l k o o r b
r e t n e c n l o c n i l a y
e r h i g h l i n e t a x i a
b t c l o i s t e r s l i d n
i s s t o n e s t r e e t r k
l l t e d a e r e n a u d a e
f l c e n t r a l p a r k u e
o a t b r o n x y o o l e g s
e w j f k o r e a t o w n a t
u i g r a n d c e n t r a l a
t t a l c o n e y i s l a n d
a e r a u q s n o s i d a m i
t l e n n u t d n a l l o h u
s r a b a y y n e w μ s e u m
'''

# ======================================================================
# Sol
# ======================================================================

'''
Answer: Little Italy

Explanation: do the following substitutions in the given grid:

z -> a -> f -> p -> n -> μ -> m -> c -> d.

After that, the following words show up in the grid:

- yabars (zabars)
- bronxyoo (bronx zoo)
- laguardia
- jfk
- newμseum (new museum)
- hollandtunnel
- lincolncenter
- radiocity
- brooklynbridge
- centralpark
- madisonsquare
- coneyisland
- grandcentral
- koreatown
- stonestreet
- cloisters
- highline
- wallstreet
- statueofliberty
- yankeestadium
- taxi
- subway
- duanereade

- chinatoon (china town, 24th entry)

The letters in the grid which are not part of any word are (in order):

LITTLEITALY
'''
