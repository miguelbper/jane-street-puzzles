from collections.abc import Generator
from typing import overload

@overload
def divisors(n: int, *, generator: bool = False) -> list[int]: ...
@overload
def divisors(n: int, *, generator: bool = True) -> Generator[int, None, None]: ...
