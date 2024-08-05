"""
A segment of the segmentation data, generally from a separate image block.
"""
from cmlibs.zinc.result import RESULT_OK


class Segment:
    """
    A segment of the segmentation data, generally from a separate image block.
    """

    def __init__(self, name, segmentation_file_name, root_region):
        """
        :param name: Unique name of segment, usually derived from the file name.
        :param segmentation_file_name: Path and file name of raw segmentation file, in Zinc format.
        :param root_region: Zinc root region to create segment region under.
        """
        self._name = name
        self._segmentationFileName = segmentation_file_name
        # print("Create segment", self._name)
        self._base_region = root_region.createChild(self._name)
        assert self._base_region.isValid(), \
            "Cannot create segment region " + self._name + ". Name may already be in use?"
        # the raw region contains the original segment data which is not modified apart from building
        # groups to categorise data for stitching a visualisation, including selecting for display.
        self._raw_region = self._base_region.createChild("raw")
        result = self._raw_region.readFile(segmentation_file_name)
        assert result == RESULT_OK, \
            "Could not read segmentation file " + segmentation_file_name
        self._rotation = [0.0, 0.0, 0.0]
        self._translation = [0.0, 0.0, 0.0]

    def decode_settings(self, settings_in: dict):
        """
        Update segment settings from JSON dict containing serialised settings.
        :param settings_in: Dictionary of settings as produced by encode_settings().
        """
        assert settings_in.get("name") == self._name
        # update current settings to gain new ones and override old ones
        settings = self.encode_settings()
        settings.update(settings_in)
        self._rotation = settings["rotation"]
        self._translation = settings["translation"]

    def encode_settings(self) -> dict:
        """
        Encode segment data in a dictionary to serialize.
        :return: Settings in a dict ready for passing to json.dump.
        """
        settings = {
            "name": self._name,
            "rotation": self._rotation,
            "translation": self._translation
        }
        return settings

    def get_base_region(self):
        """
        Get the base region for all segmentation and auxiliary data for this segment.
        :return: Zinc Region.
        """
        return self._base_region

    def get_name(self):
        return self._name

    def get_raw_region(self):
        """
        Get the raw region, a child of base region, into which the raw segmentation was loaded.
        :return: Zinc Region.
        """
        return self._raw_region

    def get_rotation(self):
        return self._rotation

    def set_rotation(self, rotation):
        assert len(rotation) == 3
        self._rotation = rotation

    def get_translation(self):
        return self._translation

    def set_translation(self, translation):
        assert len(translation) == 3
        self._translation = translation
