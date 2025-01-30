# Enable easier importing of modules
from .feed_mgmt import FeedMgmt as FeedMgmtABC
from .feed_mgmt_feeddist import FeedDistFeedMgmt
from .feed_mgmt_geodist import GeoDistFeedMgmt


class FeedMgmt(FeedMgmtABC):
    """
    Factory-class to instantiate a concrete implementation of FeedMgmt. mainly added to
    maintain backwards compatibility after splitting feed_mgmt into two classes.

    This is a bit 'hacky', one option is for this to raise a warning so that we can
    phase it out.
    """

    def __new__(cls, herds, par, type="GeoDist"):
        if type == "GeoDist":
            return GeoDistFeedMgmt(herds, par)
        elif type == "FeedDist":
            return FeedDistFeedMgmt(herds, par)
        else:
            raise ValueError(
                "Received unexpected type {type}, expected one of 'FeedDist' and 'GeoDist'."
            )
