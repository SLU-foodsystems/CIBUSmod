import unittest
import pandas as pd
import numpy as np
import scipy.sparse
from unittest.mock import MagicMock

from .feed_dist import FeedDistributor
from .indexed_matrix import IndexedMatrix


class TestMakeA2Methods(unittest.TestCase):
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

        # Ensure all values are non-positive
        self.assertTrue((result.todense() <= 0).all().all())

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
            -100,
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

        # Check row and column indices match expectations
        np.testing.assert_array_equal(result.rows, self.row_idx)
        np.testing.assert_array_equal(result.cols, self.mock_x_idx["fds"])

        # Ensure we never have two (or more) values in a column, as they are
        # region-specific
        df = result.todense()
        df_nans = df.replace({0: np.nan})
        col_counts = [df_nans.loc[:, c].count() for c in df_nans.columns]
        self.assertTrue(all(map(lambda count: count <= 1, col_counts)))

        print("")
        print(result.todense())

        self.assertEqual(
            df.loc[
                ("conventional", "product1", "A"),
                ("feed1", "cows", "cattle", "dairy", "conventional", "none", "A"),
            ],
            0.5 * (1 - 0.1) * 0.9,
        )
        self.assertEqual(
            df.loc[
                ("conventional", "product2", "A"),
                ("feed2", "cows", "cattle", "dairy", "conventional", "none", "A"),
            ],
            0.8 * (1 - 0.2) * 0.7,
        )


if __name__ == "__main__":
    unittest.main()
