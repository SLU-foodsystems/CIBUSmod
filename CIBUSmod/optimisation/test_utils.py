import unittest
import pandas as pd
import numpy as np
from .utils import feed_demands_to_crop_demands


class TestFeedDemandsToCropDemands(unittest.TestCase):
    def setUp(self):
        # Basic setup data for `feed_demands` DataFrame
        data_feed_demands = {
            ("beef", "cattle", "corn"): [100, 200],
            ("pork", "pig", "soy"): [150, 250],
            ("poultry", "chicken", "wheat"): [120, 220],
        }
        self.feed_demands = pd.DataFrame(
            data_feed_demands, index=["Region1", "Region2"]
        )
        self.feed_demands.columns.names = ["prod_system", "animal", "feed"]
        self.feed_demands.index.name = "region"

        # Basic setup data for `feed_to_crop_products` DataFrame
        data_feed_to_crop = {
            ("corn", "maize"): {"feed_to_prod": 1.5, "share_imported": 0.6},
            ("soy", "soybeans"): {"feed_to_prod": 1.2, "share_imported": 0.4},
            ("wheat", "wheat_grain"): {"feed_to_prod": 0.8, "share_imported": 0.5},
        }
        self.feed_to_crop_products = pd.DataFrame.from_dict(
            data_feed_to_crop, orient="index"
        )
        self.feed_to_crop_products.index.names = ["feed", "crop_prod"]

    def test_basic_conversion(self):
        # Test basic conversion with given data
        result = feed_demands_to_crop_demands(
            self.feed_demands, self.feed_to_crop_products
        )

        # Expected data after conversion
        expected_data = {
            ("domestic", "beef", "cattle", "maize"): [60.0, 120.0],
            ("domestic", "pork", "pig", "soybeans"): [108.0, 180.0],
            ("domestic", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
            ("imported", "beef", "cattle", "maize"): [90.0, 180.0],
            ("imported", "pork", "pig", "soybeans"): [72.0, 120.0],
            ("imported", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
        }
        expected = pd.DataFrame(expected_data, index=["Region1", "Region2"])
        expected.index.name = "region"
        expected.columns.names = ["origin", "prod_system", "animal", "crop_prod"]

        # Check if the result matches expected DataFrame
        pd.testing.assert_frame_equal(result, expected)

    def test_throws_if_duplicate_feeds(self):
        feed_to_crop_products_dup = self.feed_to_crop_products.copy()
        feed_to_crop_products_dup.loc[("soy", "maize"), :] = [1.1, 0.5]

        # Run function
        with self.assertRaises(ValueError):
            feed_demands_to_crop_demands(self.feed_demands, feed_to_crop_products_dup)

    def test_aggregation(self):
        # Test if values are correctly aggregated when multiple feeds map to the same crop
        feed_to_crop_products_agg = self.feed_to_crop_products.copy()
        feed_to_crop_products_agg.loc[("corn", "maize"), "feed_to_prod"] = 1.0
        feed_to_crop_products_agg.loc[("corn", "maize"), "share_imported"] = 0.5
        feed_to_crop_products_agg.loc[("soy", "maize"), "feed_to_prod"] = 1.1
        feed_to_crop_products_agg.loc[("soy", "maize"), "share_imported"] = 0.4

        # Avoid duplicate values
        feed_to_crop_products_agg.loc[("soy", "soybeans"), :] = np.nan
        feed_to_crop_products_agg = feed_to_crop_products_agg.dropna()

        self.feed_demands[("beef", "cattle", "soy")] = [50, 100]

        # Run function
        result = feed_demands_to_crop_demands(
            self.feed_demands, feed_to_crop_products_agg
        )

        # Expected data with aggregated values for maize
        expected_data = {
            ("domestic", "beef", "cattle", "maize"): [83.0, 166.0],
            ("domestic", "pork", "pig", "maize"): [99.0, 165.0],
            ("domestic", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
            ("imported", "beef", "cattle", "maize"): [72.0, 144.0],
            ("imported", "pork", "pig", "maize"): [66.0, 110.0],
            ("imported", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
        }
        expected = pd.DataFrame(expected_data, index=["Region1", "Region2"])
        expected.index.name = "region"
        expected.columns.names = ["origin", "prod_system", "animal", "crop_prod"]

        # Check if the result matches expected DataFrame
        pd.testing.assert_frame_equal(result, expected, atol=0.01)

    def test_missing_feed_in_conversion(self):
        # Test handling of feed items in `feed_demands` without mapping in `feed_to_crop_products`
        self.feed_demands[("beef", "cattle", "missing_feed")] = [100, 200]

        # Run function
        result = feed_demands_to_crop_demands(
            self.feed_demands, self.feed_to_crop_products
        )

        # Expected data ignoring 'missing_feed' as it has no mapping
        expected_data = {
            ("domestic", "beef", "cattle", "maize"): [60.0, 120.0],
            ("domestic", "pork", "pig", "soybeans"): [108.0, 180.0],
            ("domestic", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
            ("imported", "beef", "cattle", "maize"): [90.0, 180.0],
            ("imported", "pork", "pig", "soybeans"): [72.0, 120.0],
            ("imported", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
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
            "share_imported": 0.3,
        }

        # Run function
        result = feed_demands_to_crop_demands(
            self.feed_demands, extra_feed_to_crop_products
        )

        # Expected data should be unaffected by the extra mapping
        expected_data = {
            ("domestic", "beef", "cattle", "maize"): [60.0, 120.0],
            ("domestic", "pork", "pig", "soybeans"): [108.0, 180.0],
            ("domestic", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
            ("imported", "beef", "cattle", "maize"): [90.0, 180.0],
            ("imported", "pork", "pig", "soybeans"): [72.0, 120.0],
            ("imported", "poultry", "chicken", "wheat_grain"): [48.0, 88.0],
        }
        expected = pd.DataFrame(expected_data, index=["Region1", "Region2"])
        expected.index.name = "region"
        expected.columns.names = ["origin", "prod_system", "animal", "crop_prod"]

        # Check if the result matches expected DataFrame
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
