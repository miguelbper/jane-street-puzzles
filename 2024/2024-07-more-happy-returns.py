# Imports
# ----------------------------------------------------------------------
from termcolor import colored
import pandas as pd
import tabulate


# Data & answers
# ----------------------------------------------------------------------

top = [
    (  1, "recipient"                    , "giftee"  , 13,),
    (  1, "popular type of street food"  , "taco"    ,  0,),
    (  1, "type of remark"               , "biting"  ,  9,),
    (  1, "suddenly canceled (agreement)", "reneged" , 16,),
    (  1, "classical dance"              , "ballet"  ,  6,),
    (  1, "graduate-to-be"               , "senior"  ,  7,),
    (  2, "soft touch"                   , "caress"  , 10,),
    (  1, "type of heron"                , "egret"   ,  3,),
    (  2, "soft but unruly drummer"      , "animal"  ,  8,),
    (  2, "group of criminals"           , "cartel"  , 14,),
    ( 10, "type of understanding"        , "tacit"   ,  4,),
    (  9, "BBB-"                         , "rating"  , 11,),
    (  1, "person who wants to quit"     , "seceder" , 18,),
    (  7, "ugly facial expression"       , "sneer"   ,  5,),
    (  6, "warm breakfast dish"          , "omelet"  , 12,),
    ( 10, "brown-coloured ale"           , "draught" , 17,),
    (  4, "plant that grows underground" , "tuber"   ,  2,),
    (  6, "common solvent"               , "acetone" , 15,),
    (  7, "partner dance"                , "tango"   ,  1,),
    (  9, "type of symmetry"             , "rotative", 19,),
    ('Ø', "good luck charms"             , "gnomes"  , 20,),
]


bot = [
    ("important C++ component"                       , 4 , 9 , "allocator"   ),
    ("le supermarché il ristorante and el dinosaurio", 5 , 8 , "cognates"    ),
    ("counterargument"                               , 5 , 8 , "rebuttal"    ),
    ("laundry room staple"                           , 5 , 9 , "detergent"   ),
    ("heavy machine useful in forestry"              , 5 , 10, "masticator"  ),
    ("remnants of an explosion"                      , 5 , 11, "smithereens" ),
    ("(story) that was worth repeating"              , 6 , 8 , "tellable"    ),
    ("Captain Marvel & Black Widow"                  , 6 , 8 , "heroines"    ),
    ("this type of inexpensive flooring"             , 6 , 8 , "laminate"    ),
    ("(fuse) that seemed ready to go off"            , 6 , 9 , "ignitible"   ),
    ("4-dimensional solid"                           , 6 , 9 , "tesseract"   ),
    ("distinguished ambassador"                      , 6 , 9 , "dignitary"   ),
    ("Duel (or High School Musical)"                 , 6 , 9 , "telemovie"   ),
    ("one of the major scenes in West Side Story"    , 6 , 11, "streetfight" ),
    ("narrow bike trail"                             , 6 , 11, "singletrack" ),
    ("cozy wine shop"                                , 7 , 7 , "enoteca"     ),
    ("total reprobate"                               , 7 , 10, "degenerate"  ),
    ("often-wet piece of equipment"                  , 7 , 10, "mouthguard"  ),
    ("person who was here before me"                 , 7 , 11, "predecessor" ),
    ("person hovering a foot off the ground"         , 8 , 9 , "levitator"   ),
    ("dairy producer"                                , 0 , 0 , "cheesemonger"),
]


# Print tables
# ----------------------------------------------------------------------

class Solution:
    def __init__(self, top: list[tuple], bot: list[tuple]):
        self.top = top
        self.bot = bot
        self.length = len(bot)
        self.letters = "saidthedairyproducer."

    def bijection(self) -> list[int]:
        return [self.top[i][3] for i in range(self.length)]

    @staticmethod
    def get_ith_letter(row: pd.Series) -> str:
        if not row["answer_bot"] or row["num"] == 'Ø':
            return ""
        answer = row["answer_bot"]
        i = row["num"] - 1  # Subtract 1 because indexing starts at 0
        return answer[i] if i < len(answer) and i >= 0 else ""

    @staticmethod
    def format_answer(row: pd.Series) -> str:
        answer_top = row["answer_top"]
        answer_bot = row["answer_bot"]
        num = row["num"]

        if not answer_bot:
            return answer_bot
        
        if num == 'Ø':
            return colored(answer_bot, attrs=['bold'])

        reversed_top = answer_top[::-1]
        color = "yellow"
        formatting = "underline"

        i = num - 1
        l = answer_bot.find(reversed_top)
        r = l + len(reversed_top)

        if i < l:
            segment_0 = answer_bot[:i]
            segment_1 = colored(answer_bot[i], None, attrs=[formatting])
            segment_2 = answer_bot[i+1:l]
            segment_3 = colored(answer_bot[l:r], color)
            segment_4 = answer_bot[r:]
        elif i < r:
            segment_0 = answer_bot[:l]
            segment_1 = colored(answer_bot[l:i], color)
            segment_2 = colored(answer_bot[i], color, attrs=[formatting])
            segment_3 = colored(answer_bot[i+1:r], color)
            segment_4 = answer_bot[r:]
        else:
            segment_0 = answer_bot[:l]
            segment_1 = colored(answer_bot[l:r], color)
            segment_2 = answer_bot[r:i]
            segment_3 = colored(answer_bot[i], None, attrs=[formatting])
            segment_4 = answer_bot[i+1:]

        return ''.join([segment_0, segment_1, segment_2, segment_3, segment_4])

    def join(self) -> pd.DataFrame:
        bijec = self.bijection()
        empty = ("", -1, -1, "")

        joined = []
        for i in range(self.length):
            t = top[i][:3]
            b = empty if bijec[i] == -1 else bot[bijec[i]]
            joined.append(t + b)

        cols_top = ["num", "hint_top", "answer_top"]
        cols_bot = ["hint_bot", "dollars", "cents", "answer_bot"]
        colsumns = cols_top + cols_bot

        df = pd.DataFrame(joined, columns=colsumns)
        df["ith_letter"] = df.apply(self.get_ith_letter, axis=1)
        df["answer_bot"] = df.apply(self.format_answer, axis=1)
        return df

    def print(self) -> None:
        df = self.join()
        print(tabulate.tabulate(df, headers='keys', tablefmt='pipe', showindex=False), end='\n\n')


solution = Solution(top, bot)
solution.print()
'''
| num   | hint_top                      | answer_top   | hint_bot                                       |   dollars |   cents | answer_bot   | ith_letter   |
|:------|:------------------------------|:-------------|:-----------------------------------------------|----------:|--------:|:-------------|:-------------|
| 1     | recipient                     | giftee       | one of the major scenes in West Side Story     |         6 |      11 | streetfight  | s            |
| 1     | popular type of street food   | taco         | important C++ component                        |         4 |       9 | allocator    | a            |
| 1     | type of remark                | biting       | (fuse) that seemed ready to go off             |         6 |       9 | ignitible    | i            |
| 1     | suddenly canceled (agreement) | reneged      | total reprobate                                |         7 |      10 | degenerate   | d            |
| 1     | classical dance               | ballet       | (story) that was worth repeating               |         6 |       8 | tellable     | t            |
| 1     | graduate-to-be                | senior       | Captain Marvel & Black Widow                   |         6 |       8 | heroines     | h            |
| 2     | soft touch                    | caress       | 4-dimensional solid                            |         6 |       9 | tesseract    | e            |
| 1     | type of heron                 | egret        | laundry room staple                            |         5 |       9 | detergent    | d            |
| 2     | soft but unruly drummer       | animal       | this type of inexpensive flooring              |         6 |       8 | laminate     | a            |
| 2     | group of criminals            | cartel       | narrow bike trail                              |         6 |      11 | singletrack  | i            |
| 10    | type of understanding         | tacit        | heavy machine useful in forestry               |         5 |      10 | masticator   | r            |
| 9     | BBB-                          | rating       | distinguished ambassador                       |         6 |       9 | dignitary    | y            |
| 1     | person who wants to quit      | seceder      | person who was here before me                  |         7 |      11 | predecessor  | p            |
| 7     | ugly facial expression        | sneer        | remnants of an explosion                       |         5 |      11 | smithereens  | r            |
| 6     | warm breakfast dish           | omelet       | Duel (or High School Musical)                  |         6 |       9 | telemovie    | o            |
| 10    | brown-coloured ale            | draught      | often-wet piece of equipment                   |         7 |      10 | mouthguard   | d            |
| 4     | plant that grows underground  | tuber        | counterargument                                |         5 |       8 | rebuttal     | u            |
| 6     | common solvent                | acetone      | cozy wine shop                                 |         7 |       7 | enoteca      | c            |
| 7     | partner dance                 | tango        | le supermarché il ristorante and el dinosaurio |         5 |       8 | cognates     | e            |
| 9     | type of symmetry              | rotative     | person hovering a foot off the ground          |         8 |       9 | levitator    | r            |
| Ø     | good luck charms              | gnomes       | dairy producer                                 |         0 |       0 | cheesemonger |              |
'''
