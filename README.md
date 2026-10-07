# Block Decomposition

Partitions an array into fixed-size blocks for efficient bulk range updates. Whole interior blocks are updated in O(1) via a block-level delta; only the partially covered head and tail blocks are repaired element-by-element in O(k) each.

```python
from block_decomposition import BlockDecomposition

bd = BlockDecomposition([1, 2, 3, 4, 5, 6], block_size=2)
bd.range_update(1, 5, 10)
assert bd.get(0) == 1
assert bd.get(3) == 14
assert bd.n == 6
assert bd.block_size == 2
assert bd.num_blocks == 3
```

## Why this exists

When range *updates* dominate and individual element reads are rare, touching every element in the range on each update is the bottleneck. Block decomposition pushes the per-update cost from O(range) down to O(k) — the two partial-block repairs — at the cost of making reads O(k) instead of O(1), because a read must fold in the block-level delta. Use this when writes outnumber reads by a wide margin and the array is long enough that O(range) updates are actually painful.

## Edge cases worth knowing

- The range in `range_update(l, r, delta)` is half-open `[l, r)`, matching Python slicing. `l == r` is a valid no-op.
- The last block may be shorter than `block_size`; it is still tracked as its own block.
- `initial` is copied, not stored by reference, so mutating the source list after construction has no effect.
- The aggregation operator defaults to `+` but can be any associative, monotonic binary function with `0` as identity (e.g. `max` with identity `0`, or any group-like operation). Non-identity defaults are not supported.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

