"""Block decomposition for efficient bulk range updates.

The array is partitioned into B consecutive fixed-size blocks of size ``k``.
When a range update covers an entire block, the block-level delta is updated
in O(1) instead of touching all ``k`` elements. Partially covered head/tail
blocks are repaired in O(k) from the block delta.

Trade-off: reads are O(k) rather than O(1) because they must fold in the
block-level delta. This is the right call when range *updates* dominate and
individual element reads are rare.

Invariant: ``len(data) == n`` is fixed; ``blocks[i]`` holds the aggregate
delta applied wholesale to block ``i``. Element ``data[j]``'s logical value is
``data[j] + blocks[j // k]``. The split is applied lazily on reads; updates
keep both structures consistent.
"""

from __future__ import annotations

from typing import Callable, List, Sequence


class BlockDecomposition:
    """Fixed-size block decomposition over an immutable-length array.

    Args:
        initial: Initial values of the array (copied, not stored by reference).
        block_size: Block size ``k``. Must be a positive integer.
        op: Monotonic aggregation function used to report block totals:
            ``op(a, b)`` merges two updates. Defaults to ``+``. The identity must
            be ``op(x, 0) == x`` for any element, i.e. 0 is a no-op update.

    Raises:
        ValueError: if ``initial`` is empty or ``block_size`` is out of range.
    """

    def __init__(
        self,
        initial: Sequence[float],
        block_size: int,
        *,
        op: Callable[[float, float], float] = lambda a, b: a + b,
    ) -> None:
        if not initial:
            raise ValueError("initial must be non-empty")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._n = len(initial)
        self._k = block_size
        # We COPY so callers cannot mutate our backing storage out of band.
        self._data: List[float] = list(initial)
        # Ceiling division: the final block may be shorter than k.
        self._num_blocks = (self._n + self._k - 1) // self._k
        # Block-level delta accumulator. All zeros means "every element already
        # equals its stored value". We never store the identity anywhere else,
        # so the invariant is: logical[j] == data[j] op blocks[j // k].
        self._blocks: List[float] = [0.0] * self._num_blocks
        self._op = op

    @property
    def n(self) -> int:
        """Fixed array length."""
        return self._n

    @property
    def block_size(self) -> int:
        """Configured block size ``k``."""
        return self._k

    @property
    def num_blocks(self) -> int:
        """Number of blocks (the last may be shorter than ``k``)."""
        return self._num_blocks

    def get(self, i: int) -> float:
        """Logical value at index ``i`` after folding in block delta."""
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for length {self._n}")
        return self._op(self._data[i], self._blocks[i // self._k])

    def range_update(self, l: int, r: int, delta: float) -> None:
        """Apply ``delta`` to every element in the half-open range ``[l, r)``.

        Whole interior blocks get their block-level delta touched once (O(1));
        the partial head and tail blocks are fully repaired from the block
        delta first, then per-element updated (O(k) each). Interior bulk blocks
        are left untouched — their stored values will be folded back in lazily
        on subsequent reads.

        Range convention is half-open ``[l, r)`` to match Python slicing and
        to make empty ranges (``l == r``) well-defined as a no-op.
        """
        if not 0 <= l <= r <= self._n:
            raise IndexError(
                f"range [{l}, {r}) out of bounds for length {self._n}"
            )
        if l == r:
            return

        first_block = l // self._k
        last_block = (r - 1) // self._k

        # Head: partial or full block at the start of the range.
        head_end = min((first_block + 1) * self._k, r)
        self._repair_block(first_block)
        for j in range(l, head_end):
            self._data[j] = self._op(self._data[j], delta)

        # Interior: full blocks updated wholesale via the block delta.
        for b in range(first_block + 1, last_block):
            self._blocks[b] = self._op(self._blocks[b], delta)

        # Tail: partial or full block at the end (distinct from head).
        if last_block != first_block:
            self._repair_block(last_block)
            for j in range(last_block * self._k, r):
                self._data[j] = self._op(self._data[j], delta)

    def _repair_block(self, b: int) -> None:
        """Fold the block-level delta into the stored elements, then reset it.

        After this, the block's elements carry their full logical value and
        the block delta is back to the identity (0). This is needed before any
        per-element edit of a block so the invariant is preserved.
        """
        delta = self._blocks[b]
        if delta == 0.0:
            return
        start = b * self._k
        end = min(start + self._k, self._n)
        for j in range(start, end):
            self._data[j] = self._op(self._data[j], delta)
        self._blocks[b] = 0.0
