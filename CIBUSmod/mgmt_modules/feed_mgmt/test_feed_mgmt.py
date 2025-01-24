import unittest
from unittest.mock import MagicMock
import pandas as pd

from .feed_mgmt_feeddist import FeedDistFeedMgmt as FeedMgmt


class TestFeedMgmt(unittest.TestCase):
    def setUp(self):
        # Mock the ParameterRetriever object
        self.mock_par = MagicMock()
        # Create a mock AnimalHerd object with expected attributes and methods
        self.mock_herd = MagicMock()
        self.mock_herd.species = "cattle"
        self.mock_herd.breed = "dairy"
        # Setting up 'data_attr' mock, which should be a callable object with 'add' method
        self.mock_herd.data_attr = MagicMock()
        # Feed demand of 1000 kg
        self.mock_herd.data_attr.get = MagicMock(return_value=1000)
        # Mock the method get_from_frame to return specific percentages for losses
        # 10% for feeding, 5% for storage
        self.mock_par.get_from_frame.side_effect = (
            lambda x, _df: 10 if x == "feeding_losses" else 5
        )

        # Set up the herds as a pd.Series to pass to FeedMgmt
        herds_series = pd.Series([self.mock_herd])

        # Instantiate FeedMgmt with the mocked objects
        self.feed_mgmt = FeedMgmt(herds=herds_series, par=self.mock_par)

    def test_calculate_consumption_and_losses(self):
        # Run the method to calculate consumption and losses
        self.feed_mgmt.calculate_consumption_and_losses()

        # Check if ParameterRetriever's set method was called with correct species and breed
        self.mock_par.set.assert_called_once_with(species="cattle", breed="dairy")

        # Expected calculations
        expected_storage_losses = 1000 * 0.05
        expected_feeding_losses = (1000 - expected_storage_losses) * 0.1
        expected_consumption = 1000 * 0.95 * 0.9

        # Verify that data_attr.add was called with the expected values
        self.mock_herd.data_attr.add.assert_any_call(
            expected_consumption,
            name="feed.consumption",
            unit="kg DM/year",
            orig="FeedMgmt",
            desc="Demand for feed after accounting for storage and feeding losses",
        )
        self.mock_herd.data_attr.add.assert_any_call(
            expected_storage_losses,
            name="feed.storage_losses",
            unit="kg DM/year",
            orig="FeedMgmt",
            desc="Losses of feed during storage",
        )
        self.mock_herd.data_attr.add.assert_any_call(
            expected_feeding_losses,
            name="feed.feeding_losses",
            unit="kg DM/year",
            orig="FeedMgmt",
            desc="Losses of feed during feeding",
        )


# Run the tests
if __name__ == "__main__":
    unittest.main()
