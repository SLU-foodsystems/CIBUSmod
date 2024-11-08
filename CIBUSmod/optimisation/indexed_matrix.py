import pandas as pd
import numpy as np
import scipy

from typing import Self

class IndexedMatrix:
    """Class to store pandas.Index/MultiIndex alongside a sparse
    matrix to keep track of things"""

    type IndexLike = pd.Index | pd.MultiIndex | dict[str, pd.MultiIndex]

    def __init__(
        self, matrix: scipy.sparse.csc_matrix, row_idx: IndexLike, col_idx: IndexLike
    ):
        self.M = matrix
        self.rows = row_idx
        self.cols = col_idx

    @property
    def shape(self):
        return self.M.shape

    def eval(self, x):
        return pd.Series(self.M @ x, index=self.rows)

    def copy(self):
        """
        Create a copy of the IndexedMatrix, copying the data (M) as well as the indices.
        """
        return IndexedMatrix(self.M.copy(), self.rows.copy(), self.cols.copy())

    def align(self, other: Self):
        if isinstance(self.rows, dict) or isinstance(other.rows, dict):
            raise ValueError("IndexedMatrix.align only works with flat indices")

        if not other.rows.isin(self.rows).all():
            raise ValueError(
                "Index of the IndexedMatrix must be a superset (larger) of the other matrix (smaller)."
            )

        rows_indices_other = [i for i, x in enumerate(self.rows) if x in other.rows]

        M_csr = self.M.tocsr()
        self.M = M_csr[rows_indices_other, :].tocsc()
        self.rows = other.rows.copy()

    def prune_rows(self):
        if isinstance(self.rows, dict):
            raise ValueError("IndexedMatrix.prune_rows only works with flat indices.")

        # Convert to CSR format for efficient row-based operations
        csr_mat = self.M.tocsr()

        # Identify rows that contain non-zero entries
        non_empty_rows = np.flatnonzero(csr_mat.getnnz(axis=1))

        # Slice the CSR matrix to keep only non-empty rows
        pruned_csr = csr_mat[non_empty_rows, :]

        # Convert back to CSC format
        self.M = pruned_csr.tocsc()
        self.rows = self.rows[non_empty_rows]
        return self

    def todense(self):
        index = self.rows
        columns = self.cols

        if isinstance(index, dict):
            index = np.concatenate(list(index.values()))
        if isinstance(columns, dict):
            columns = np.concatenate(list(columns.values()))

        return pd.DataFrame(
            self.M.todense(),
            index=index,
            columns=columns,
        )

    @classmethod
    def from_coordinates(
        cls, coo: tuple[list, tuple[list[int], list[int]]], row_idx, col_idx
    ) -> Self:
        """
        Create an IndexedMatrix from a (values, coordinates)-matrix and row- and
        column indices.
        """
        (val, (row_nr, col_nr)) = coo
        return cls(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx=row_idx,
            col_idx=col_idx,
        )

    @classmethod
    def align_rows(cls, A: Self, B: Self):
        _A = A.copy()
        _A.align(B)
        return _A
