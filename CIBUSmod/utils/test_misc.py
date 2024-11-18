import unittest
import pandas as pd

from .misc import extend_index

# Assuming the function `extend_index` has been imported correctly


class TestExtendIndex(unittest.TestCase):
    def setUp(self):
        # Sample initial MultiIndex for testing
        self.initial_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("B", 2)], names=["letter", "number"]
        )

    def test_append_new_levels(self):
        # Test appending new levels
        new_levels = [["X", "Y"], [10, 20]]
        new_names = ["group", "value"]

        result = extend_index(new_levels, new_names, self.initial_index, mode="append")

        # Expected MultiIndex after appending
        expected = pd.MultiIndex.from_tuples(
            [
                ("A", 1, "X", 10),
                ("A", 1, "X", 20),
                ("A", 1, "Y", 10),
                ("A", 1, "Y", 20),
                ("B", 2, "X", 10),
                ("B", 2, "X", 20),
                ("B", 2, "Y", 10),
                ("B", 2, "Y", 20),
            ],
            names=["letter", "number", "group", "value"],
        )

        pd.testing.assert_index_equal(result, expected)

    def test_prepend_new_levels(self):
        # Test prepending new levels
        new_levels = [["X", "Y"], [10, 20]]
        new_names = ["group", "value"]

        result = extend_index(new_levels, new_names, self.initial_index, mode="prepend")

        # Expected MultiIndex after appending
        expected = pd.MultiIndex.from_tuples(
            [
                ("X", 10, "A", 1),
                ("X", 20, "A", 1),
                ("Y", 10, "A", 1),
                ("Y", 20, "A", 1),
                ("X", 10, "B", 2),
                ("X", 20, "B", 2),
                ("Y", 10, "B", 2),
                ("Y", 20, "B", 2),
            ],
            names=["group", "value", "letter", "number"],
        )

        pd.testing.assert_index_equal(result, expected)

    def test_empty_new_levels(self):
        # Test when new levels and names are empty (should return the original index)
        result = extend_index([], [], self.initial_index, mode="append")

        pd.testing.assert_index_equal(result, self.initial_index)

    def test_empty_initial_index(self):
        # Test with an empty initial index and append mode
        empty_index = pd.MultiIndex.from_tuples([], names=["letter", "number"])
        new_levels = [["X", "Y"], [10, 20]]
        new_names = ["group", "value"]

        result = extend_index(new_levels, new_names, empty_index, mode="append")

        # Expected MultiIndex with only the new levels since the initial index is empty
        expected = pd.MultiIndex.from_product(
            [["X", "Y"], [10, 20]], names=["group", "value"]
        )
        pd.testing.assert_index_equal(result, expected)

    def test_single_new_level(self):
        # Test with a single new level to append
        new_levels = [["extra"]]
        new_names = ["new_level"]

        result = extend_index(new_levels, new_names, self.initial_index, mode="append")

        # Expected MultiIndex after appending single new level
        # Expected MultiIndex after appending
        expected = pd.MultiIndex.from_tuples(
            [
                ("A", 1, "extra"),
                ("B", 2, "extra"),
            ],
            names=["letter", "number", "new_level"],
        )

        pd.testing.assert_index_equal(result, expected)

    def test_invalid_mode(self):
        # Test with an invalid mode (should raise an exception)
        new_levels = [["X", "Y"], [10, 20]]
        new_names = ["group", "value"]

        with self.assertRaises(ValueError):
            extend_index(new_levels, new_names, self.initial_index, mode="invalid")

if __name__ == "__main__":
    unittest.main()
