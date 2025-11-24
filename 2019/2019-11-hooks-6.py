import inspect
from collections.abc import Iterator
from copy import deepcopy
from functools import partial, reduce
from itertools import product
from math import gcd
from operator import mul
from typing import Any, TypeAlias

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from codetiming import Timer
from numpy.typing import NDArray
from scipy.ndimage import label, sum_labels

Int: TypeAlias = np.int32
BitMask: TypeAlias = np.int32
ArrayInt: TypeAlias = NDArray[Int]
ArrayBitMask: TypeAlias = NDArray[BitMask]
Rule: TypeAlias = Any  # Should be Callable[[ArrayBitMask], ArrayBitMask | None], using Any to avoid type checker issues

IS_RULE: str = "is_rule"


def rule(func: Rule) -> Rule:
    """Decorator to mark a method as a rule for the backtracking algorithm.

    Args:
        func (Rule): The function to be marked as a rule.

    Returns:
        Rule: The decorated function.
    """
    setattr(func, IS_RULE, True)  # noqa: B010
    return func


class Backtracking:
    """Represents a backtracking problem.

    Usage:
        1. Define a new class that inherits from this class.

        2. __init__:
            - Override
            - Do super().__init__()

        3. branch -> branch_cell
            Options (from less to more "manual")
            - Leave the methods as is / do nothing
            - Override branch_cell to specify what cell should be chosen
            - Override branch to specify different logic

        4. prune_repeatedly -> prune -> @rule's
            Options (from less to more "manual")
            - Define methods decorated with @rule, they will be called by prune
            - Override prune
            - Override prune_repeatedly

        5. Instantiate and call 'solution' or 'solutions' to find the solution(s).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes a Backtracking object.

        Args:
            Whatever you want to pass as argument.

        Returns:
            None
        """
        self.rules = self.get_rules()

    def solution_generator(self, stack: list[ArrayBitMask]) -> Iterator[ArrayInt]:
        """Generates solutions using backtracking algorithm.

        Generator that is called from the 'solution' and 'solutions'
        methods. A generator is used to prevent code duplication.

        Yields:
            ArrayInt: A valid solution grid.
        """
        stack = deepcopy(stack)

        while stack:
            bm_prev = stack.pop()
            bm = self.prune_repeatedly(bm_prev)

            if bm is None:
                continue
            if self.accept(bm):
                yield self.grid(bm)
            else:
                stack += self.branch(bm)

    def solution(self, stack: list[ArrayBitMask]) -> ArrayInt | None:
        """Finds a solution using a backtracking algorithm.

        Returns:
            ArrayInt | None: The solution grid if found, None otherwise.
        """
        return next(self.solution_generator(stack), None)

    def solutions(self, stack: list[ArrayBitMask]) -> list[ArrayInt]:
        """Returns a list of all possible solutions for the problem.

        Returns:
            A list of ArrayInt objects representing the possible solutions.
        """
        return list(self.solution_generator(stack))

    @staticmethod
    def grid(bm: ArrayBitMask) -> ArrayInt:
        """Convert from a choices matrix to a grid.

        Assumes that all elements of bm are singletons. When used in
        'solution_generator', this is true because of the 'accept'
        function.

        Args:
            bm (ArrayBitMask): The input choices matrix.

        Returns:
            ArrayInt: The resulting grid.
        """
        return np.log2(bm).astype(np.int32)

    @staticmethod
    def reject(bm: ArrayBitMask | None) -> bool:
        """Checks if the choice matrix is invalid.

        Args:
            bm (ArrayBitMask | None): The choice matrix to be checked.

        Returns:
            bool: True if bm is None or contains a 0.
        """
        return bm is None or not np.all(bm)

    @staticmethod
    def accept(bm: ArrayBitMask) -> np.bool:
        """Checks if all elements of the choice matrix are singletons.

        Assumes that bm does not contain a 0, which is true when this
        function is called in 'solution_generator'.

        Args:
            bm (ArrayBitMask): The choice matrix to be checked.

        Returns:
            bool: True if all elements of bm are singletons.
        """
        return np.all(bm & (bm - 1) == 0)

    def branch(self, bm: ArrayBitMask) -> list[ArrayBitMask]:
        """Chooses a cell and lists the possible values for that cell.

        Branches the given choices matrix by selecting the element with
        the fewest possible choices, and creating new choice matrices
        for each possible choice of that element.

        This function may optionally be overridden by the user to make
        more informed guesses for the list of possible choice matrices.

        When overriding, you may make use of the following assumptions:
        - bm was not rejected => np.all(bm), i.e. no zeros in bm
        - bm was not accepted => there exists a cell of bm with c > 1

        When overriding, you must respect the following properties:
        If ems = branch(bm), then
        - Refinement: For all em in ems, em ⊊ bm
        - No solutions are lost: For all solutions xm ⊂ bm, there
          exists em in ems such that xm ⊂ em

        Args:
            bm (ArrayBitMask): The choices matrix to branch.

        Returns:
            list[ArrayBitMask]: A list of new choice matrices, each
                representing a possible choice for the element with the
                fewest possible choices.
        """
        powers_of_two = 1 << np.arange(32)
        multi_index = self.branch_cell(bm)
        powers_present = powers_of_two[bm[multi_index] & powers_of_two > 0]
        bm_copies = np.repeat(bm[np.newaxis, ...], len(powers_present), axis=0)
        bm_copies[:, *multi_index] = powers_present
        return list(bm_copies)

    def branch_cell(self, bm: ArrayBitMask) -> tuple[np.intp, ...]:
        """Find the cell with the fewest possible choices.

        Args:
            bm (ArrayBitMask): The choices matrix to analyze.

        Returns:
            tuple[np.intp, ...]: The multi-dimensional index of the cell with
                fewest possible choices, excluding cells that are already
                determined (have only one choice).
        """
        powers_of_two = 1 << np.arange(32)
        cardinality = np.sum((bm[..., None] & powers_of_two) != 0, axis=-1)
        cardinality_unfilled = np.where(cardinality == 1, np.inf, cardinality)
        multi_index = np.unravel_index(np.argmin(cardinality_unfilled), bm.shape)
        return multi_index

    def prune_repeatedly(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        """Repeatedly calls prune until bm no longer changes.

        Args:
            bm (ArrayBitMask): The choices to be pruned.

        Returns:
            ArrayBitMask | None: The pruned choices, or None if the rules
                are violated.
        """
        prune_again = True
        while prune_again:
            bm_temp = np.copy(bm)
            bm_new = self.prune(bm)
            if self.reject(bm_new):
                return None
            assert bm_new is not None
            bm = bm_new
            prune_again = not np.array_equal(bm, bm_temp)
        return bm

    def prune(self, bm: ArrayBitMask) -> ArrayBitMask | None:
        """Prunes the choices matrix based on the rules of the problem.

        Should be implemented by the user, since it is specific to the
        problem to be solved. Should obey the following properties:

        If om = prune(bm), then
        - Refinement: om ⊂ bm
        - No solutions are lost: xm ⊂ bm satisfies the rule => xm ⊂ om
        - Eventual rejection: If bm is all singletons and does not
          satisfy the rule, then reject(om) is True

        If bm will never lead to a valid solution, may just return None
        in the implementation.

        Args:
            bm (ArrayBitMask): The input choices matrix.

        Returns:
            ArrayBitMask | None: Pruned matrix or None
        """
        bm = np.copy(bm)
        for func in self.rules:
            bm_temp = func(bm)
            if bm_temp is None:
                return None
            bm = bm_temp
            if self.reject(bm):
                return None
        return bm

    def get_rules(self) -> list[Rule]:
        """Get all methods marked with the @rule decorator.

        Returns:
            list[Rule]: A list of bound methods that were decorated with @rule.
        """
        rules: list[Rule] = []
        for name, member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if getattr(member, IS_RULE, False):
                rules.append(getattr(self, name))
        return rules


# fmt: off
red = np.array([
    [9, 0, 6, 5, 0, 0, 0, 0, 0],  # L
    [0, 1, 0, 0, 1, 0, 8, 0, 0],  # D
    [0, 1, 0, 9, 0, 1, 0, 8, 0],  # R
    [0, 0, 1, 0, 2, 0, 1, 0, 8],  # U
], dtype=np.int32)

blk = np.array([
    [0, 0    , 0, 0    , 7**3, 0, 48**2, 0, 0    ], # L
    [0, 0    , 0, 15**2, 0   , 0, 0    , 0, 99**2], # D
    [0, 0    , 0, 0    , 0   , 0, 6**2 , 0, 0    ], # R
    [0, 42**3, 0, 0    , 0   , 0, 0    , 0, 0    ], # U
], dtype=np.int32)
# fmt: on

IntArray = NDArray[np.int32]


class Hooks6(Backtracking):
    # Init and utils
    # ------------------------------------------------------------------
    def __init__(self, red: IntArray, blk: IntArray) -> None:
        super().__init__()
        _, n = red.shape
        self.red = red
        self.blk = blk
        self.nums = np.stack([red, blk])  # red = 0, blk = 1
        self.masks = np.array(list(product(range(2), repeat=n)))

    @staticmethod
    def join(xm: IntArray, os: IntArray) -> IntArray:
        n, _ = xm.shape
        return np.concatenate([xm, os.reshape(n, 1)], axis=1)

    @staticmethod
    def split(cm: IntArray) -> tuple[IntArray, IntArray]:
        n, _ = cm.shape
        xm = cm[:, :n]
        os = cm[:, n]
        return xm, os

    @staticmethod
    def singletons(cm: IntArray, dtype: type = np.int32) -> IntArray:
        power_of_two = (cm > 0) & ((cm & (cm - 1)) == 0)
        ans = -np.ones_like(cm)
        ans[power_of_two] = np.log2(cm[power_of_two])
        return ans.astype(dtype)

    def get_hooks(self, os: IntArray) -> IntArray:
        n = os.shape[0]
        hooks = np.zeros((n, n), dtype=np.int32)
        I = np.arange(0, n)[:, np.newaxis]
        J = np.arange(0, n)[np.newaxis, :]

        i0, j0 = 0, 0

        for k, o in enumerate(self.singletons(os)):
            if o == -1:
                break

            i1 = i0 + n - k
            j1 = j0 + n - k

            row_cond = o == 0 or o == 3
            col_cond = o == 0 or o == 1
            row = i0 if row_cond else i1 - 1
            col = j0 if col_cond else j1 - 1
            in_row = (I == row) & (j0 <= J) & (J < j1)  # noqa: SIM300
            in_col = (J == col) & (i0 <= I) & (I < i1)  # noqa: SIM300
            mask = in_row | in_col
            digit = n - k
            hooks += digit * mask

            i0 += row_cond
            j0 += col_cond

        return hooks

    @staticmethod
    def score_1d(X: IntArray, color: int) -> np.int32:
        if np.array_equal(X, np.zeros_like(X)):
            return 0
        n = X.shape[-1]
        I = np.eye(n, dtype=bool)
        M = X.astype(np.bool)
        starts = np.argwhere(M & (~np.roll(M, 1) | I[0])).reshape(-1)
        ends = (np.argwhere(M & (~np.roll(M, -1) | I[n - 1])) + 1).reshape(-1)
        lists_digits = [X[s:e] for s, e in zip(starts, ends)]
        nums = [int("".join(map(str, digits))) for digits in lists_digits]
        func = np.prod if color else partial(reduce, gcd)
        return func(nums)

    def score(self, X: IntArray, color: int) -> IntArray:
        aux = lambda X: self.score_1d(X, color)
        return np.vectorize(aux, signature="(n)->()")(X)

    # Expand
    # ------------------------------------------------------------------
    def branch_cell(self, cm: IntArray) -> tuple[np.intp, np.intp]:
        isnt_singleton = (cm & (cm - 1)) != 0
        indices = np.argwhere(isnt_singleton)
        sorted_indices = indices[np.lexsort((indices[:, 0], -indices[:, 1]))]
        return tuple(sorted_indices[0])

    # Prune
    # ------------------------------------------------------------------
    @rule
    def connectedness(self, cm: IntArray) -> IntArray | None:
        if not np.all(cm & (cm - 1) == 0):
            return cm
        xm, _ = self.split(cm)
        xm = self.singletons(xm, dtype=np.bool)
        return cm if label(xm)[1] <= 1 else None

    @rule
    def num_digits(self, cm: IntArray) -> IntArray | None:
        n, _ = cm.shape
        xm, os = self.split(cm)
        hooks = self.get_hooks(os)  # (n, n) [i, j]
        hook_indices = n - hooks  # (n, n) [i, j]
        ds = np.arange(2)  # (2,) [d]
        ks = np.arange(n)  # (n,) [k]
        hook_indices_levels = hook_indices == ks.reshape(n, 1, 1)  # (n, n, n) [k, i, j]
        xm_levels = np.where(hook_indices_levels, xm, 0)  # (n, n, n) [k, i, j]
        xm_is_ds = xm_levels == (1 << ds).reshape(2, 1, 1, 1)  # (2, n, n, n) [d, k, i, j]
        num_ds = np.sum(xm_is_ds, axis=(2, 3))  # (2, n) [d, k]
        max_ds = n + ds.reshape(2, 1) - ks - 1  # (2, n) [d, k]
        if np.any(num_ds > max_ds):
            return None

        num_is_max = num_ds == max_ds  # (2, n) [d, k]
        mask = (xm_levels == 3) & num_is_max.reshape(2, n, 1, 1)  # (2, n, n, n) [d, k, i, j]
        xm_levels_masked = np.where(mask, 1 << (1 - ds.reshape(2, 1, 1, 1)), xm_levels)  # noqa: E501  # (2, n, n, n) [d, k, i, j]
        update = np.sum(xm_levels_masked, axis=1)  # (2, n, n) [d, i, j]
        xm_new = np.copy(xm)  # (n, n) [i, j]
        for upd in update:
            xm_new = np.where(upd, xm_new & upd, xm_new)  # (n, n) [i, j]

        cm_new = self.join(xm_new, os)
        return cm_new

    @rule
    def two_by_two(self, cm: IntArray) -> IntArray | None:
        xm, os = self.split(cm)
        n, _ = xm.shape
        mask = np.zeros_like(xm, dtype=np.bool)
        for i, j in product(range(n - 1), repeat=2):
            corners = [(i + ((c + 1) % 4 > 1), j + (c > 1)) for c in range(4)]
            for corner in corners:
                rest_filled = all(xm[a, b] == 1 << 1 for a, b in corners if (a, b) != corner)
                if rest_filled:
                    mask[*corner] = True
        xm_new = np.where(mask, xm & (1 << 0), xm)
        cm_new = self.join(xm_new, os)
        return cm_new

    @rule
    def scores(self, cm: IntArray) -> IntArray | None:
        xm, os = self.split(cm)
        hooks = self.get_hooks(os)  # (n, n) [i, j]
        xm_new = np.copy(xm)

        for color, side in product(range(2), range(4)):
            nums = self.nums[color, side]
            xm_rot = np.rot90(xm_new, k=-side)
            hooks_rot = np.rot90(hooks, k=-side)

            update_rot = self.scores_update(xm_rot, hooks_rot, nums, color)
            if update_rot is None:
                return None

            update = np.rot90(update_rot, k=side)
            xm_new = np.where(update, xm_new & update, xm_new)

        cm_new = self.join(xm_new, os)
        return cm_new

    def scores_update(
        self,
        xm: IntArray,
        hooks: IntArray,
        nums: IntArray,
        color: int,
    ) -> IntArray | None:
        n, _ = xm.shape
        ds = np.arange(2).reshape(2, 1, 1, 1)  # (2, 1, 1, 1) [d, ...]
        xm_is_ds = xm == 1 << ds  # (2, 1, n, n) [d, k, i, j]
        mask_is_ds = self.masks.reshape(2**n, 1, n) == ds  # (2, 2**n, 1, n) [d, k, i, j]
        compatible = ~xm_is_ds | mask_is_ds  # (2, 2**n, n, n) [d, k, i, j]
        compatible_masks = np.all(compatible, axis=(0, 3))  # (2**n, n) [k, i]
        unknown_row = np.any(hooks == 0, axis=1)  # (n,) [i]
        unknown_num = nums == 0  # (n,) [i]
        hooks_masked = hooks * self.masks.reshape(2**n, 1, n)  # (2**n, n, n) [k, i, j]
        score = self.score(hooks_masked, color)  # (2**n, n) [k, i]
        valid_masks = compatible_masks & (unknown_row | unknown_num | (score == nums))  # noqa: E501 (2**n, n) [k, i]

        if not np.all(np.any(valid_masks, axis=0)):
            return None

        update = np.zeros_like(xm, dtype=np.int32)  # (n, n) [i, j]
        for i in range(n):
            row_masks = self.masks[valid_masks[:, i], :]  # (?, n) [k, j]
            masks_agree = np.all(row_masks[0] == row_masks, axis=0)
            update[i] = (1 << row_masks[0]) * masks_agree

        return update

    # Print solution
    # ------------------------------------------------------------------
    @staticmethod
    def answer(sol: IntArray) -> int:
        mat = np.where(sol == 0, 1, 0)
        labels, k = label(mat)
        area = sum_labels(mat, labels, index=range(1, k + 1))
        return int(reduce(mul, area))

    def plot(self, X: IntArray, hooks: IntArray) -> None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        annot = np.array(X).astype("str")
        annot[annot == "0"] = ""

        base_colors = [
            "white",
            "steelblue",
            "cyan",
            "lightgreen",
            "orange",
            "red",
            "purple",
            "olive",
            "dimgray",
            "lightgray",
        ]
        reduced_alpha_colors = [(mcolors.to_rgba(c, alpha=0.6)) for c in base_colors]
        cmap = mcolors.ListedColormap(reduced_alpha_colors)

        ax = sns.heatmap(
            hooks,
            annot=annot,
            cbar=False,
            fmt="",
            linewidths=0.1,
            linecolor="black",
            square=True,
            cmap=cmap,
            vmin=0,
            vmax=9,
        )
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        plt.show()


# State and solve the problem
# ----------------------------------------------------------------------
_, n = red.shape
xm = sum(1 << b for b in range(2)) * np.ones((n, n), dtype=np.int32)
os = sum(1 << b for b in range(4)) * np.ones((n,), dtype=np.int32)
os[n - 1] = 1 << 0
cm = np.concatenate([xm, os.reshape(n, 1)], axis=1)
stack = [cm]

problem = Hooks6(red, blk)
with Timer(initial_text="Solving problem..."):
    sol = problem.solution(stack)

B, O = problem.split(sol)
hooks = problem.get_hooks(1 << O)
X = B * hooks
ans = problem.answer(X)

print(f"{ans = }")
print(f"{X}")
problem.plot(X, hooks)
# Solving problem...
# Elapsed time: 170.0845 seconds
# ans = 10000
# [[0 9 9 9 9 0 0 9 0]
#  [0 0 8 0 8 8 8 8 0]
#  [0 0 6 0 0 0 6 0 0]
#  [8 7 6 5 0 5 5 5 0]
#  [0 7 0 0 0 0 0 4 9]
#  [0 7 6 5 4 3 2 0 9]
#  [8 0 6 0 4 0 1 2 0]
#  [8 7 6 0 4 3 3 0 9]
#  [0 0 7 0 0 0 7 7 9]]
