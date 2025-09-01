import polars as pl

df = pl.DataFrame(
    schema=["Emoji", "Name", "C1", "C2", "N1", "N2"],
    data=[
        ["😳", "Flushed Face"              , 4, 5, 1        , 0        ],
        ["🤤", "Drooling Face"             , 6, 9, 7 + 7 + 8, 8 + 7 + 8],
        ["🤠", "Cowboy Hat Face"           , 9, 4, 7        , 6 + 6    ],
        ["🥴", "Woozy Face"                , 5, 8, 5 + 5    , 6 + 5    ],
        ["😧", "Anguished Face"            , 7, 5, 8 + 7 + 8, 6        ],
        ["😾", "Pouting Cat"               , 1, 8, 8 + 8    , 0        ],
        ["😖", "Confounded Face"           , 5, 6, 8 + 7 + 8, 0        ],
        ["😝", "Squinting Face with Tongue", 8, 6, 7 + 7    , 9 + 8 + 8],
    ],
)  # fmt: skip


def shift(letter: str, shift: int) -> str:
    return chr((ord(letter) - 65 + shift) % 26 + 65)


df = df.with_columns(
    pl.col("Name").str.to_uppercase().str.replace_all("\\s+", "").alias("Letters"),
).with_columns(
    pl.col("Letters").str.slice(pl.col("C1") - 1, 1).alias("Name[C1]"),
    pl.col("Letters").str.slice(pl.col("C2") - 1, 1).alias("Name[C2]"),
).with_columns(
    pl.concat_str([pl.col("Name[C1]"), pl.col("Name[C2]")]).alias("Name[C1,C2]"),
).with_columns([
    pl.struct(["Name[C1]", "N1"]).map_elements(lambda row: shift(row["Name[C1]"], row["N1"])).alias("Shifted 1"),
    pl.struct(["Name[C2]", "N2"]).map_elements(lambda row: shift(row["Name[C2]"], row["N2"])).alias("Shifted 2")
]).with_columns(
    pl.concat_str([pl.col("Shifted 1"), pl.col("Shifted 2")]).alias("Shifted")
).drop(["Letters", "Name[C1]", "Name[C2]", "Shifted 1", "Shifted 2"])  # fmt: skip


print(df)

sentence = "".join(df["Shifted"])
print(sentence)
# THE CANINE OF CLUBS
# => answer = KC,9C
