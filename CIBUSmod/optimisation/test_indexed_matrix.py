import unittest
import pandas as pd
import numpy as np
import scipy.sparse as sp

from .indexed_matrix import IndexedMatrix

class TestIndexedMatrix(unittest.TestCase):
    def setUp(self):
        """Set up sample data for testing."""
        # Create a small sparse matrix and corresponding indices for testing:
        #    X Y Z
        # A: 0 1 2
        # B: 3 4 5
        # C: 6 7 8

        self.row_idx = pd.Index(["A", "B", "C"])
        self.col_idx = pd.Index(["X", "Y", "Z"])

        n_rows = len(self.row_idx)
        n_cols = len(self.col_idx)
        val = list(range(n_rows * n_cols))
        rows = [i for i in range(n_rows) for _ in range(n_cols)]
        cols = list(range(n_cols)) * n_rows
        self.matrix_data = sp.coo_array(
            (val, (rows, cols)), shape=(n_rows, n_cols)
        ).tocsc()
        self.indexed_matrix = IndexedMatrix(
            self.matrix_data, self.row_idx, self.col_idx
        )

    def test_init(self):
        """Test that IndexedMatrix initializes correctly."""
        self.assertIsInstance(self.indexed_matrix, IndexedMatrix)
        self.assertTrue(sp.issparse(self.indexed_matrix.M))
        self.assertTrue(self.indexed_matrix.rows.equals(self.row_idx))
        self.assertTrue(self.indexed_matrix.cols.equals(self.col_idx))

    def test_eval(self):
        """Test the eval method for matrix-vector multiplication."""
        x = np.array([1, 2, 3])  # Vector to multiply with the matrix
        result = self.indexed_matrix.eval(x)

        expected_values = self.matrix_data @ x  # Expected result of multiplication
        expected_series = pd.Series(expected_values, index=self.row_idx)

        pd.testing.assert_series_equal(result, expected_series)

    def test_align(self):
        """Test the align method to ensure rows are aligned correctly with another matrix."""
        # Create another matrix with a subset of rows
        other_row_idx = pd.Index(["A", "C"], name="rows")
        other_col_idx = pd.Index(["M", "N"], name="cols")
        # Construct matrix: [5 0; 0 6]
        other_matrix_data = sp.coo_matrix(([5, 6], ([0, 1], [0, 1]))).tocsc()
        other_indexed_matrix = IndexedMatrix(
            other_matrix_data, other_row_idx, other_col_idx
        )

        # Align self.indexed_matrix with other_indexed_matrix
        self.indexed_matrix.align(other_indexed_matrix)

        # Check if rows and matrix data are aligned
        self.assertTrue(self.indexed_matrix.rows.equals(other_row_idx))
        # Row count should match other
        self.assertEqual(self.indexed_matrix.M.shape[0], len(other_row_idx))
        # Column count should remain same
        self.assertEqual(self.indexed_matrix.M.shape[1], self.matrix_data.shape[1])

    def test_todense(self):
        """Test the todense method to ensure it converts to a dense DataFrame correctly."""
        dense_df = self.indexed_matrix.todense()

        # Convert original sparse matrix to dense for comparison
        expected_df = pd.DataFrame(
            self.matrix_data.todense(), index=self.row_idx, columns=self.col_idx
        )

        pd.testing.assert_frame_equal(dense_df, expected_df)

    def test_from_coordinates(self):
        """Test the from_coordinates class method to ensure proper matrix construction."""
        coo_values = [10, 20, 30]
        coo_coords = ([0, 1, 2], [0, 1, 2])  # Coordinates for a 3x3 matrix

        # Create IndexedMatrix from coordinates
        indexed_matrix = IndexedMatrix.from_coordinates(
            (coo_values, coo_coords), self.row_idx, self.col_idx
        )

        # Expected sparse matrix
        expected_matrix = sp.coo_matrix((coo_values, coo_coords), shape=(3, 3)).tocsc()

        self.assertTrue(
            (indexed_matrix.M - expected_matrix).nnz == 0
        )  # Check if matrices are identical
        self.assertTrue(indexed_matrix.rows.equals(self.row_idx))
        self.assertTrue(indexed_matrix.cols.equals(self.col_idx))

    def test_align_rows(self):
        """Test the align_rows class method to ensure row alignment."""
        # Create another matrix with a subset of rows in B
        row_idx_A = pd.Index(["A", "B", "C"])
        row_idx_B = pd.Index(["B", "C"])

        matrix_A = IndexedMatrix(sp.eye(3).tocsc(), row_idx_A, self.col_idx)
        matrix_B = IndexedMatrix(sp.eye(2).tocsc(), row_idx_B, pd.Index(["P", "Q"]))

        # Align B to A's rows
        aligned_matrix = IndexedMatrix.align_rows(matrix_A, matrix_B)

        # Check that rows and matrix are aligned correctly
        self.assertTrue(aligned_matrix.rows.equals(row_idx_B))
        # Check row count matches B's rows
        self.assertEqual(aligned_matrix.M.shape[0], len(row_idx_B))
        # Check column count is unchanged
        self.assertEqual(aligned_matrix.M.shape[1], matrix_A.M.shape[1])


if __name__ == "__main__":
    unittest.main()
