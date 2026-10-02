import math
import unittest

from block_decomposition import BlockDecomposition


class TestConstruction(unittest.TestCase):
    def test_basic_properties(self):
        bd = BlockDecomposition([1, 2, 3, 4, 5], block_size=2)
        self.assertEqual(bd.n, 5)
        self.assertEqual(bd.block_size, 2)
        self.assertEqual(bd.num_blocks, 3)

    def test_block_count_partial_final(self):
        # 5 elements, block size 2 => 3 blocks (last has 1 element).
        bd = BlockDecomposition([0, 0, 0, 0, 0], block_size=2)
        self.assertEqual(bd.num_blocks, 3)

    def test_block_count_exact_division(self):
        bd = BlockDecomposition([0, 0, 0, 0], block_size=2)
        self.assertEqual(bd.num_blocks, 2)

    def test_initial_is_copied(self):
        src = [1, 2, 3]
        bd = BlockDecomposition(src, block_size=1)
        src[0] = 999
        self.assertEqual(bd.get(0), 1)

    def test_empty_initial_rejected(self):
        with self.assertRaises(ValueError):
            BlockDecomposition([], block_size=2)

    def test_nonpositive_block_size_rejected(self):
        with self.assertRaises(ValueError):
            BlockDecomposition([1, 2, 3], block_size=0)
        with self.assertRaises(ValueError):
            BlockDecomposition([1, 2, 3], block_size=-1)


class TestGet(unittest.TestCase):
    def test_get_returns_logical_value(self):
        bd = BlockDecomposition([10, 20, 30], block_size=2)
        self.assertEqual(bd.get(0), 10)
        self.assertEqual(bd.get(1), 20)
        self.assertEqual(bd.get(2), 30)

    def test_get_out_of_range(self):
        bd = BlockDecomposition([1, 2, 3], block_size=2)
        with self.assertRaises(IndexError):
            bd.get(-1)
        with self.assertRaises(IndexError):
            bd.get(3)


class TestRangeUpdate(unittest.TestCase):
    def test_single_element_range(self):
        bd = BlockDecomposition([1, 2, 3, 4, 5], block_size=2)
        bd.range_update(2, 3, 10)
        self.assertEqual([bd.get(i) for i in range(5)], [1, 2, 13, 4, 5])

    def test_full_range_whole_blocks(self):
        bd = BlockDecomposition([1, 2, 3, 4], block_size=2)
        bd.range_update(0, 4, 100)
        self.assertEqual([bd.get(i) for i in range(4)], [101, 102, 103, 104])

    def test_full_range_partial_final_block(self):
        bd = BlockDecomposition([1, 2, 3, 4, 5], block_size=2)
        bd.range_update(0, 5, 100)
        self.assertEqual([bd.get(i) for i in range(5)], [101, 102, 103, 104, 105])

    def test_interior_full_block_bulk_path(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], block_size=2)
        # [2, 6): touches partial head of block 1, all of block 2, partial tail of block 2.
        bd.range_update(2, 6, 5)
        self.assertEqual([bd.get(i) for i in range(6)], [0, 0, 5, 5, 5, 5])

    def test_head_partial_block_only(self):
        bd = BlockDecomposition([1, 2, 3, 4], block_size=2)
        bd.range_update(0, 1, 10)
        self.assertEqual([bd.get(i) for i in range(4)], [11, 2, 3, 4])

    def test_tail_partial_block_only(self):
        bd = BlockDecomposition([1, 2, 3, 4], block_size=2)
        bd.range_update(3, 4, 10)
        self.assertEqual([bd.get(i) for i in range(4)], [1, 2, 3, 14])

    def test_range_spanning_head_and_tail_without_interior(self):
        bd = BlockDecomposition([1, 2, 3, 4], block_size=2)
        bd.range_update(1, 3, 10)
        self.assertEqual([bd.get(i) for i in range(4)], [1, 12, 13, 4])

    def test_overlapping_updates_same_block(self):
        bd = BlockDecomposition([0, 0, 0, 0], block_size=2)
        bd.range_update(0, 4, 5)
        bd.range_update(1, 3, 100)
        self.assertEqual([bd.get(i) for i in range(4)], [5, 105, 105, 5])

    def test_update_then_read_then_update(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0], block_size=2)
        bd.range_update(0, 6, 1)
        # Reads force repair of block 0; block 1 and 2 still carry delta.
        self.assertEqual(bd.get(0), 1)
        self.assertEqual(bd.get(1), 1)
        bd.range_update(0, 6, 1)
        self.assertEqual([bd.get(i) for i in range(6)], [2, 2, 2, 2, 2, 2])

    def test_empty_range_is_noop(self):
        bd = BlockDecomposition([1, 2, 3], block_size=2)
        bd.range_update(1, 1, 99)
        self.assertEqual([bd.get(i) for i in range(3)], [1, 2, 3])

    def test_negative_delta(self):
        bd = BlockDecomposition([10, 20, 30, 40], block_size=2)
        bd.range_update(0, 4, -5)
        self.assertEqual([bd.get(i) for i in range(4)], [5, 15, 25, 35])

    def test_zero_delta_is_noop(self):
        bd = BlockDecomposition([1, 2, 3, 4], block_size=2)
        bd.range_update(0, 4, 0)
        self.assertEqual([bd.get(i) for i in range(4)], [1, 2, 3, 4])

    def test_out_of_range_rejected(self):
        bd = BlockDecomposition([1, 2, 3], block_size=2)
        with self.assertRaises(IndexError):
            bd.range_update(-1, 2, 5)
        with self.assertRaises(IndexError):
            bd.range_update(0, 4, 5)
        with self.assertRaises(IndexError):
            bd.range_update(2, 1, 5)


class TestCustomOp(unittest.TestCase):
    def test_max_op(self):
        bd = BlockDecomposition([1, 5, 3, 2, 4], block_size=2, op=max)
        bd.range_update(0, 4, 10)
        self.assertEqual([bd.get(i) for i in range(5)], [10, 10, 10, 10, 4])

    def test_max_op_then_partial(self):
        bd = BlockDecomposition([1, 5, 3, 2, 4], block_size=2, op=max)
        bd.range_update(0, 5, 10)
        bd.range_update(1, 3, 20)
        self.assertEqual([bd.get(i) for i in range(5)], [10, 20, 20, 10, 10])


class TestRepairSemantics(unittest.TestCase):
    def test_block_delta_resets_after_full_repair(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0, 0, 0], block_size=2)
        # [0, 8): head=block0, interior=block1-2, tail=block3; all repaired.
        bd.range_update(0, 8, 5)
        # Head and tail blocks are repaired element-by-element, so their
        # block-level deltas are folded in and reset to 0.0.
        self.assertEqual(bd._blocks[0], 0.0)
        self.assertEqual(bd._blocks[3], 0.0)

    def test_bulk_block_delta_persists_until_repaired(self):
        bd = BlockDecomposition([0, 0, 0, 0, 0, 0, 0, 0], block_size=2)
        # [2, 8): head=block1 (repaired), interior=block2 (bulk delta), tail=block3 (repaired).
        bd.range_update(2, 8, 5)
        self.assertEqual(bd._blocks[2], 5.0)
        self.assertEqual(bd._blocks[1], 0.0)
        self.assertEqual(bd._blocks[3], 0.0)


if __name__ == "__main__":
    unittest.main()
