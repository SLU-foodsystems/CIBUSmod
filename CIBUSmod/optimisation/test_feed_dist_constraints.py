import unittest
from unittest.mock import MagicMock

import pandas as pd
import numpy as np
import scipy

from .feed_dist import FeedDistributor
from .indexed_matrix import IndexedMatrix


class TestMakeA2(unittest.TestCase):
    def setUp(self):
        # Setup mock data and expected output structure for the tests

        # Mock for `self.crops.data_attr.get("production")`
        self.mock_crop_production = pd.DataFrame(
            [
                [100, 0, 50],
                [0, 200, 0],
                [75, 25, 0],
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    ("wheat", "conventional", "A"),
                    ("corn", "organic", "A"),
                    ("wheat", "conventional", "B"),
                ],
                names=["crop", "prod_system", "region"],
            ),
            columns=pd.Index(["product1", "product2", "product3"], name="crop_prod"),
        )

        # Mock x_idx dictionary with expected col_idx
        self.mock_x_idx = {
            "crp": pd.MultiIndex.from_tuples(
                [
                    ("wheat", "conventional", "A"),
                    ("corn", "organic", "A"),
                    ("wheat", "conventional", "B"),
                ],
                names=["crop", "prod_system", "region"],
            ),
            "fds": pd.MultiIndex.from_tuples(
                [
                    ("feed1", "cows", "cattle", "dairy", "conventional", "none", "A"),
                    ("feed2", "cows", "cattle", "dairy", "conventional", "none", "A"),
                ],
                names=[
                    "feed",
                    "animal",
                    "species",
                    "breed",
                    "prod_system",
                    "sub_system",
                    "region",
                ],
            ),
        }

        # Mocked crops attribute and data_attr dictionary
        crops = MagicMock()
        crops.data_attr = {"production": self.mock_crop_production}

        # Factors for `make_A2_2`
        self.factors_with_reg_share = pd.DataFrame(
            {
                "feed": ["feed1", "feed2"],
                "crop_prod": ["product1", "product2"],
                "feed_to_prod": [0.5, 0.8],
                "share_imported": [0.1, 0.2],
                "share_regional": [0.9, 0.7],
            },
        ).set_index(["feed", "crop_prod"])

        self.row_idx = pd.MultiIndex.from_product(
            [
                self.mock_x_idx["fds"].get_level_values("prod_system").unique(),
                self.factors_with_reg_share.index.get_level_values(
                    "crop_prod"
                ).unique(),
                self.mock_x_idx["fds"].get_level_values("region").unique(),
            ],
            names=["prod_system", "crop_prod", "region"],
        )

        self.noop_mock = MagicMock()

        self.feed_dist = FeedDistributor(
            regions=self.noop_mock,
            demand=self.noop_mock,
            crops=crops,
            herds=self.noop_mock,
            feed_mgmt=self.noop_mock,
            par=self.noop_mock,
        )

        self.feed_dist.x_idx = self.mock_x_idx

    def test_make_A2_1(self):
        result = self.feed_dist.make_A2_1(self.row_idx)

        ## Sanity-checks: test the basic properties of the result
        # Check the type of result
        self.assertIsInstance(result, IndexedMatrix)

        # Validate shape
        self.assertEqual(result.shape, (len(self.row_idx), len(self.mock_x_idx["crp"])))

        # Validate non-zero entries based on mock data expectations
        self.assertGreater(result.M.nnz, 0)

        # Ensure all values are non-negative
        self.assertTrue((result.todense() >= 0).all().all())

        # Check row and column indices match expectations
        np.testing.assert_array_equal(result.rows, self.row_idx)
        np.testing.assert_array_equal(result.cols, self.mock_x_idx["crp"])

        ## Data integrity:

        # Ensure we never have two (or more) values in a column, as they are
        # region-specific
        df = result.todense()
        df_nans = df.replace({0: np.nan})
        col_counts = [df_nans.loc[:, c].count() for c in df_nans.columns]
        self.assertTrue(all(map(lambda count: count <= 1, col_counts)))

        self.assertEqual(
            100,
            df.loc[("conventional", "product1", "A"), ("wheat", "conventional", "A")],
        )

    def test_make_A2_2(self):
        result = self.feed_dist.make_A2_2(self.row_idx, self.factors_with_reg_share)

        # Check the type of result
        self.assertIsInstance(result, IndexedMatrix)

        # Validate shape
        self.assertEqual(result.shape, (len(self.row_idx), len(self.mock_x_idx["fds"])))

        # Validate non-zero entries based on expected calculation results
        self.assertGreater(result.M.nnz, 0)

        # Ensure all values are non-positive
        self.assertTrue((result.todense() <= 0).all().all())

        # Check row and column indices match expectations
        np.testing.assert_array_equal(result.rows, self.row_idx)
        np.testing.assert_array_equal(result.cols, self.mock_x_idx["fds"])

        # Ensure we never have two (or more) values in a column, as they are
        # region-specific
        df = result.todense()
        df_nans = df.replace({0: np.nan})
        col_counts = [df_nans.loc[:, c].count() for c in df_nans.columns]
        self.assertTrue(all(map(lambda count: count <= 1, col_counts)))

        self.assertEqual(
            df.loc[
                ("conventional", "product1", "A"),
                ("feed1", "cows", "cattle", "dairy", "conventional", "none", "A"),
            ],
            0.5 * (1 - 0.1) * 0.9 * -1,
        )
        self.assertEqual(
            df.loc[
                ("conventional", "product2", "A"),
                ("feed2", "cows", "cattle", "dairy", "conventional", "none", "A"),
            ],
            0.8 * (1 - 0.2) * 0.7 * -1,
        )


## Constraint 10
class TestMakeA10(unittest.TestCase):
    def setUp(self):
        # Setting up sample data for testing

        # Sample row index (D_idx) for byproducts demand with MultiIndex
        self.row_idx = pd.MultiIndex.from_tuples(
            [("conv", "by_prod1"), ("conv", "by_prod2")],
            names=["prod_system", "by_prod"],
        )

        # Sample col index for feed demands with MultiIndex
        self.col_idx = pd.MultiIndex.from_tuples(
            [
                ("feed1", "ani1", "sp1", "br1", "conv", "ss1", "region1"),
                ("feed2", "ani2", "sp2", "br2", "org", "ss2", "region2"),
            ],
            names=[
                "feed",
                "animal",
                "species",
                "breed",
                "prod_system",
                "sub_system",
                "region",
            ],
        )

        # Mock data for `feed_to_prod` DataFrame
        self.feed_to_prod = pd.DataFrame(
            {"feed_to_prod": [0.8, 0.6, 0.4], "share_imported": [0.2, 0.0, 0.5]},
            index=pd.MultiIndex.from_tuples(
                [("feed1", "by_prod1"), ("feed1", "by_prod2"), ("feed2", "by_prod1")],
                names=["feed", "by_prod"],
            ),
        )

        # Mocked x_idx and D_idx
        self.mock_x_idx = {"fds": self.col_idx}

        self.noop_mock = MagicMock()

        self.feed_dist = FeedDistributor(
            regions=self.noop_mock,
            demand=self.noop_mock,
            crops=self.noop_mock,
            herds=self.noop_mock,
            feed_mgmt=self.noop_mock,
            par=self.noop_mock,
        )

        self.feed_dist.x_idx = self.mock_x_idx
        self.feed_dist.get_feed_to_crop_prod_factors = (
            lambda *args, **kwargs: self.feed_to_prod
        )
        # Mock method to return feed_to_prod when called by make_A10_1

    def test_make_A_basic_functionality(self):
        # Test that make_A10_1 runs and produces output with correct shape
        A10_1 = self.feed_dist.make_A10_1(self.row_idx)
        self.assertEqual(A10_1.shape, (len(self.row_idx), len(self.col_idx)))
        self.assertTrue(
            scipy.sparse.issparse(A10_1.M), "Output M is not a sparse matrix"
        )
        self.assertTrue(
            isinstance(A10_1, IndexedMatrix), "Output is not an IndexedMatrix matrix"
        )

    def test_make_A10_1_value_calculation(self):
        # Test that make_A10_1 correctly calculates values based on feed_to_prod
        A = self.feed_dist.make_A10_1(
            self.row_idx
        ).M.tocoo()  # Convert to COO format to inspect entries

        # Expected values based on `feed_to_prod` and `share_imported`
        # Notice that the organic value should not be present
        expected_values = [
            (0, 0, 0.8 * (1 - 0.2)),  # feed1 -> by_prod1, conv
            (1, 0, 0.6 * (1 - 0.0)),  # feed1 -> by_prod2, conv
        ]

        for row, col, expected_value in expected_values:
            found = False
            for i, j, val in zip(A.row, A.col, A.data):
                if i == row and j == col:
                    self.assertAlmostEqual(val, expected_value, places=5)
                    found = True
            self.assertTrue(found, f"Expected entry at ({row}, {col}) was not found")

    def test_make_A10_1_index_matching(self):
        # Ensure that only matching production systems are included in matrix
        A = self.feed_dist.make_A10_1(self.row_idx).M.tocoo()
        for row, col, val in zip(A.row, A.col, A.data):
            # Only matching prod_system entries should be present
            row_sys = self.row_idx[row][0]
            col_sys = self.col_idx[col][4]
            self.assertEqual(
                row_sys, col_sys, f"Mismatch in prod_system: {row_sys} vs {col_sys}"
            )


class TestMakeA11(unittest.TestCase):
    def setUp(self):
        noop = MagicMock()

        self.feed_dist = FeedDistributor(
            regions=noop,
            demand=noop,
            crops=noop,
            herds=noop,
            feed_mgmt=noop,
            par=noop,
        )

        base_idx = pd.MultiIndex.from_product(
            [
                ["ley silage, 1st cut", "grazing"],
                ["cows", "bulls"],
                ["cattle"],
                ["dairy"],
                ["conventional"],
                ["ley based"],
                ["1001"],
            ],
            names=[
                "feed",
                "animal",
                "species",
                "breed",
                "prod_system",
                "sub_system",
                "region",
            ],
        )

        herd = MagicMock()
        herd.animals = ["cows", "bulls"]
        herd.species = "cattle"
        herd.breed = "dairy"
        herd.prod_system = "conventional"
        herd.sub_system = "ley based"
        herd.index = pd.Index(["1001"], name="Region")

        herd.par = MagicMock()
        _feeds = base_idx.get_level_values("feed").unique().to_list()
        _feeds_mock = MagicMock()
        _feeds_mock.feed.tolist = MagicMock(return_value=_feeds)
        herd.par.get_unique = MagicMock(return_value=_feeds_mock)
        shares_in_ration = pd.DataFrame(
            data=[
                [22.0, 78.0],
                [60.0, 40.0],
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    ("cows", "cattle", "dairy", "conventional", "ley based"),
                    ("bulls", "cattle", "dairy", "conventional", "ley based"),
                ],
                names=["animal", "species", "breed", "prod_system", "sub_system"],
            ),
            columns=pd.Index(_feeds, name="feed"),
            dtype=float,
        )
        herd.par.get_from_frame = MagicMock(return_value=shares_in_ration)
        self.feed_dist.herds = pd.Series([herd])

        self.feed_dist.x_idx = {"fds": base_idx}

        losses = pd.DataFrame(
            data=[0.4, 0.9, 0.5, 1.0],
            index=base_idx.droplevel("region").unique(),
            columns=pd.Index(["losses_factor"]),
        )
        self.feed_dist._get_losses_factors = MagicMock(return_value=losses)

    def test_make_A11_share(self):
        """
        Sanity-test for make_A11, ensuring that we parse the data and distribute it
        into a dataframe nicely. Does, however, not check that we fetch the data
        correctly from the ParameterRetriever, as this is mocked in these tests.
        """
        A11 = self.feed_dist.make_A11("share_in_ration")
        assert A11 is not None
        self.assertIsInstance(A11, IndexedMatrix)
        self.assertEqual(A11.shape, (4, 4))
        res_df = A11.todense()

        ani_1 = ("cows", "cattle", "dairy", "conventional", "ley based", "1001")
        ani_2 = ("bulls", "cattle", "dairy", "conventional", "ley based", "1001")
        f1, f2 = ("ley silage, 1st cut", "grazing")

        f1_ani1 = (f1,) + ani_1
        f2_ani1 = (f2,) + ani_1
        f1_ani2 = (f1,) + ani_2
        f2_ani2 = (f2,) + ani_2

        f1_ani1_row = res_df.loc[f1_ani1, :]
        f2_ani1_row = res_df.loc[f2_ani1, :]
        f1_ani2_row = res_df.loc[f1_ani2, :]
        f2_ani2_row = res_df.loc[f2_ani2, :]

        # Feed 1 (lay), Animal 1 (cows)
        self.assertEqual(f1_ani1_row.loc[f1_ani1], 0.4 * (1 - 0.22))
        self.assertEqual(f1_ani1_row.loc[f2_ani1], 0.4 * (0 - 0.22))
        self.assertEqual(f1_ani1_row.loc[f1_ani2], 0)
        self.assertEqual(f1_ani1_row.loc[f2_ani2], 0)

        # Feed 2 (grazing), Animal 1 (cows)
        self.assertEqual(f2_ani1_row.loc[f1_ani1], 0.5 * (0 - 0.78))
        self.assertEqual(f2_ani1_row.loc[f2_ani1], 0.5 * (1 - 0.78))
        self.assertEqual(f2_ani1_row.loc[f1_ani2], 0)
        self.assertEqual(f2_ani1_row.loc[f2_ani2], 0)

        # Feed 1 (lay), Animal 2 (bulls)
        self.assertEqual(f1_ani2_row.loc[f1_ani1], 0)
        self.assertEqual(f1_ani2_row.loc[f2_ani1], 0)
        self.assertEqual(f1_ani2_row.loc[f1_ani2], 0.9 * (1 - 0.6))
        self.assertEqual(f1_ani2_row.loc[f2_ani2], 0.9 * (0 - 0.6))

        # Feed 2 (grazing), Animal 2 (bulls)
        self.assertEqual(f2_ani2_row.loc[f1_ani1], 0)
        self.assertEqual(f2_ani2_row.loc[f2_ani1], 0)
        self.assertEqual(f2_ani2_row.loc[f1_ani2], 1.0 * (0 - 0.4))
        self.assertEqual(f2_ani2_row.loc[f2_ani2], 1.0 * (1 - 0.4))


if __name__ == "__main__":
    unittest.main()
