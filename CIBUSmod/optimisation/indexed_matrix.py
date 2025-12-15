import pandas as pd
import numpy as np
import scipy

from typing import Self

class IndexedMatrix:
    """Class to store pandas.Index/MultiIndex alongside a sparse
    matrix to keep track of things"""

    def __init__(self, matrix: scipy.sparse.csc_array, row_idx, col_idx):
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
        csr_arr = scipy.sparse.csr_array(self.M)

        # Identify rows that contain non-zero entries
        ## Note that np.diff(.indptr) is the same as .getnnz(axis=1) (was deprecated).
        non_empty_rows = np.flatnonzero(np.diff(csr_arr.indptr))

        # Slice the CSR matrix to keep only non-empty rows
        pruned_csr = csr_arr[non_empty_rows, :]

        # Convert back to CSC format
        self.M = pruned_csr.tocsc()
        self.rows = self.rows[non_empty_rows]
        return self
    
    def extend_cols(
            self,
            target_idx:dict[str, pd.MultiIndex],
            col_key:str|None = None
            ):
        """
        Horizontaly extend matrix with zero-matrices to match target_idx
        """

        # Define function to slice original matrix into sub-matrices
        # per key in column dict
        def slice_horizontally(M, widths:dict):
            slices = dict()
            start = 0
            for k,w in widths.items():
                end = start + w
                slices[k] = M[:, start:end]
                start = end
            return slices

        # Get target keys (i.e. 'crp', 'ani', ...)
        target_keys = list(target_idx.keys())

        # Get width(s) of original matrix(ces)
        if col_key is None:
            try:
                # Get widths of sub-matrices to slice out of original matrix
                orig_widths = {k:len(v) for k,v in self.cols.items()}
            except AttributeError:
                raise ValueError("'col_key' must be specified if self.cols is not a dict")
        elif isinstance(col_key, str):
            if isinstance(self.cols, pd.MultiIndex):
                # Only one sub-matrix with width = length of column index
                orig_widths = {col_key:len(self.cols)}
            else:
                raise ValueError("self.cols should be a pd.MultiIndex if 'col_key' is supplied")
        else:
            raise ValueError("'col_key' must be None or str")

        # Slice original matrix
        orig_matrices = slice_horizontally(self.M, orig_widths)

        # Dict to store output sub-matrices
        matrices = dict()

        # Assign matrices (zero matrix if key not in original matrix)
        for k in target_keys:
            if k in orig_matrices:
                matrices[k] = orig_matrices[k]
            else:
                matrices[k] = scipy.sparse.csc_array((self.shape[0], len(target_idx[k])))

        # Update matrix and cols
        self.M = scipy.sparse.hstack(list(matrices.values()))
        self.cols = target_idx.copy()

        return None

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
    def from_frame(
        cls,
        df: pd.DataFrame,
        row_idx: pd.Index | pd.MultiIndex,
        col_idx: pd.Index | pd.MultiIndex,
        values_name="values",
        row_nr_name="row_i",
        col_nr_name="col_i",
    ) -> Self:
        """
        Create an IndexedMatrix from a (values, coordinates)-matrix and row- and
        column indices.
        """
        assert isinstance(df, pd.DataFrame), "df must be a dataframe"

        val = df[values_name].to_numpy()
        row_nr = df[row_nr_name].to_numpy()
        col_nr = df[col_nr_name].to_numpy()

        return cls(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx=row_idx,
            col_idx=col_idx,
        )
    
    @classmethod
    def zeros(
        cls,
        row_idx: pd.Index | pd.MultiIndex,
        col_idx: pd.Index | pd.MultiIndex,
    ):
        """
        Create an IndexedMatrix of zeros
        """
        return cls(
            scipy.sparse.csc_matrix((len(row_idx), len(col_idx))),
            row_idx=row_idx,
            col_idx=col_idx,
        )
        

    @classmethod
    def align_rows(cls, A: Self, B: Self):
        _A = A.copy()
        _A.align(B)
        return _A
