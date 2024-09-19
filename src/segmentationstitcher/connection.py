"""
A connection between segments in the segmentation data.
"""
from cmlibs.utils.zinc.general import ChangeManager
from segmentationstitcher.annotation import AnnotationCategory


class Connection:
    """
    A connection between segments in the segmentation data.
    """
    _separator = " - "

    def __init__(self, segments, root_region):
        """
        :param segments: List of 2 Stitcher Segment objects.
        :param root_region: Zinc root region to create segment region under.
        """
        assert len(segments) == 2, "Only supports connections between 2 segments"
        self._name = self._separator.join(segment.get_name() for segment in segments)
        self._segments = segments
        self._region = root_region.createChild(self._name)
        assert self._region.isValid(), \
            "Cannot create connection region " + self._name + ". Name may already be in use?"
        # ensure category groups exist:
        fieldmodule = self._region.getFieldmodule()
        with ChangeManager(fieldmodule):
            for category in AnnotationCategory:
                group_name = category.get_group_name()
                group = fieldmodule.createFieldGroup()
                group.setName(group_name)
                group.setManaged(True)
        self._linked_nodes = []  # (segment0_node_identifier, segment1_node_identifier)

    def decode_settings(self, settings_in: dict):
        """
        Update segment settings from JSON dict containing serialised settings.
        :param settings_in: Dictionary of settings as produced by encode_settings().
        :param all_segments: List of all segments in Stitcher.
        """
        settings_name = self._separator.join(settings_in["segments"])
        assert settings_name == self._name
        # update current settings to gain new ones and override old ones
        settings = self.encode_settings()
        settings.update(settings_in)
        self._linked_nodes = settings["linked nodes"]

    def encode_settings(self) -> dict:
        """
        Encode segment data in a dictionary to serialize.
        :return: Settings in a dict ready for passing to json.dump.
        """
        settings = {
            "segments": [segment.get_name() for segment in self._segments],
            "linked nodes": self._linked_nodes
        }
        return settings

    def get_name(self):
        return self._name

    def get_region(self):
        """
        Get the region containing any UI visualisation data for connection.
        :return: Zinc Region.
        """
        return self._region

    def get_segments(self):
        """
        :return: List of segments joined by this connection.
        """
        return self._segments
