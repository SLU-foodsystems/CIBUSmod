import unittest
import pandas as pd
import numpy as np
from .utils import feed_demands_to_crop_demands


class TestFeedDemandsToCropDemands(unittest.TestCase):
    def setUp(self):
        # Basic setup data for `feed_demands` DataFrame
        feed_demands = pd.DataFrame(
            {
                ("conventional", "cattle", "corn"): [100, 200],
                ("conventional", "pig", "soy"): [150, 250],
                ("organic", "chicken", "wheat"): [120, 220],
            },
            index=["Region1", "Region2"],
        )
        feed_demands.columns.names = ["prod_system", "animal", "feed"]
        feed_demands.index.name = "region"
        self.feed_demands = feed_demands.sort_index()

        # Basic setup data for `feed_to_crop_products` DataFrame
        feed_to_crop_products = pd.DataFrame.from_dict(
            {
                ("conventional", "corn", "maize"): {
                    "feed_to_prod": 1.5,
                    "share_domestic": 0.4,
                },
                ("conventional", "soy", "soybeans"): {
                    "feed_to_prod": 1.2,
                    "share_domestic": 0.6,
                },
                ("organic", "wheat", "wheat_grain"): {
                    "feed_to_prod": 0.8,
                    "share_domestic": 0.5,
                },
            },
            orient="index",
        )
        feed_to_crop_products.index.names = ["prod_system", "feed", "crop_prod"]
        self.feed_to_crop_products = feed_to_crop_products.sort_index()

    def test_basic_conversion(self):
        # Test basic conversion with given data
        result = feed_demands_to_crop_demands(
            self.feed_demands, self.feed_to_crop_products
        )

        # Expected data after conversion
        expected_data = {
            ("domestic", "conventional", "cattle", "maize"): [60.0, 120.0],
            ("domestic", "conventional", "pig", "soybeans"): [108.0, 180.0],
            ("domestic", "organic", "chicken", "wheat_grain"): [48.0, 88.0],
            ("imported", "conventional", "cattle", "maize"): [90.0, 180.0],
            ("imported", "conventional", "pig", "soybeans"): [72.0, 120.0],
            ("imported", "organic", "chicken", "wheat_grain"): [48.0, 88.0],
        }
        expected = pd.DataFrame(expected_data, index=["Region1", "Region2"])
        expected.index.name = "region"
        expected.columns.names = ["origin", "prod_system", "animal", "crop_prod"]

        # Check if the result matches expected DataFrame
        pd.testing.assert_frame_equal(result, expected)

    def test_aggregation(self):
        """
        Test if values are correctly aggregated (summed) when multiple feeds map to the
        same crop.

        The goal of this test is to have two demands for the same animal in the same ps,
        but of different crop products that map to the same feed.
        More concretely, for ps=conventional:
        1) cattle -> corn -> maize
        2) cattle -> soy -> maize

        And then we want to ensure that these are summed together correctly, rather than
        e.g. overwriting each other.
        """
        feed_to_crop_products_agg = self.feed_to_crop_products.copy()
        # We pretend that both corn and soy feeds require maize
        # Add (pig, soy) -> maize
        feed_to_crop_products_agg.loc[
            ("conventional", "corn", "maize"), "feed_to_prod"
        ] = 1.0
        feed_to_crop_products_agg.loc[
            ("conventional", "corn", "maize"), "share_domestic"
        ] = 0.5
        feed_to_crop_products_agg.loc[
            ("conventional", "soy", "maize"), "feed_to_prod"
        ] = 1.1
        feed_to_crop_products_agg.loc[
            ("conventional", "soy", "maize"), "share_domestic"
        ] = 0.4

        # Avoid duplicate values
        feed_to_crop_products_agg.loc[("conventional", "soy", "soybeans"), :] = np.nan
        feed_to_crop_products_agg = feed_to_crop_products_agg.dropna()

        # Add another demand for cattle for soy, which also is now mapped to maize
        self.feed_demands.loc[:, ("conventional", "cattle", "soy")] = (
            self.feed_demands.loc[:, ("conventional", "pig", "soy")]
        )
        self.feed_demands.loc[:, ("conventional", "pig", "soy")] = np.nan
        self.feed_demands = self.feed_demands.dropna(how="all", axis=1)

        # Run function
        result = feed_demands_to_crop_demands(
            self.feed_demands, feed_to_crop_products_agg
        )

        # Expected data with aggregated values for maize
        expected_data = {
            ("domestic", "conventional", "cattle", "maize"): [
                100 * 1 * 0.5 + 150 * 1.1 * 0.4,
                200 * 1 * 0.5 + 250 * 1.1 * 0.4,
            ],
            ("domestic", "organic", "chicken", "wheat_grain"): [
                120 * 0.8 * 0.5,
                220 * 0.8 * 0.5,
            ],
            ("imported", "conventional", "cattle", "maize"): [
                100 * 1 * 0.5 + 150 * 1.1 * 0.6,
                200 * 1 * 0.5 + 250 * 1.1 * 0.6,
            ],
            ("imported", "organic", "chicken", "wheat_grain"): [
                120 * 0.8 * 0.5,
                220 * 0.8 * 0.5,
            ],
        }
        expected = pd.DataFrame(expected_data, index=["Region1", "Region2"])
        expected.index.name = "region"
        expected.columns.names = ["origin", "prod_system", "animal", "crop_prod"]

        # Check if the result matches expected DataFrame
        pd.testing.assert_frame_equal(result, expected, atol=0.01)

    def test_missing_feed_in_conversion(self):
        # Test handling of feed items in `feed_demands` without mapping in `feed_to_crop_products`
        self.feed_demands[("conventional", "cattle", "missing_feed")] = [100, 200]

        # Run function
        result = feed_demands_to_crop_demands(
            self.feed_demands, self.feed_to_crop_products
        )

        # Expected data ignoring 'missing_feed' as it has no mapping
        expected_data = {
            ("domestic", "conventional", "cattle", "maize"): [60.0, 120.0],
            ("domestic", "conventional", "pig", "soybeans"): [108.0, 180.0],
            ("domestic", "organic", "chicken", "wheat_grain"): [48.0, 88.0],
            ("imported", "conventional", "cattle", "maize"): [90.0, 180.0],
            ("imported", "conventional", "pig", "soybeans"): [72.0, 120.0],
            ("imported", "organic", "chicken", "wheat_grain"): [48.0, 88.0],
        }
        expected = pd.DataFrame(expected_data, index=["Region1", "Region2"])
        expected.index.name = "region"
        expected.columns.names = ["origin", "prod_system", "animal", "crop_prod"]

        # Check if the result matches expected DataFrame
        pd.testing.assert_frame_equal(result, expected)

    def test_extra_feed_to_crop_mapping(self):
        # Test handling of extra mappings in `feed_to_crop_products` that are not in `feed_demands`
        extra_feed_to_crop_products = self.feed_to_crop_products.copy()
        extra_feed_to_crop_products.loc[("barley", "barley_product")] = {
            "feed_to_prod": 2.0,
            "share_domestic": 0.7,
        }

        # Run function
        result = feed_demands_to_crop_demands(
            self.feed_demands, extra_feed_to_crop_products
        )

        # Expected data should be unaffected by the extra mapping
        expected_data = {
            ("domestic", "conventional", "cattle", "maize"): [60.0, 120.0],
            ("domestic", "conventional", "pig", "soybeans"): [108.0, 180.0],
            ("domestic", "organic", "chicken", "wheat_grain"): [48.0, 88.0],
            ("imported", "conventional", "cattle", "maize"): [90.0, 180.0],
            ("imported", "conventional", "pig", "soybeans"): [72.0, 120.0],
            ("imported", "organic", "chicken", "wheat_grain"): [48.0, 88.0],
        }
        expected = pd.DataFrame(expected_data, index=["Region1", "Region2"])
        expected.index.name = "region"
        expected.columns.names = ["origin", "prod_system", "animal", "crop_prod"]

        # Check if the result matches expected DataFrame
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
