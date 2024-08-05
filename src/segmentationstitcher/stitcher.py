"""
Interface for stitching segmentation data from and calculating transformations between adjacent image blocks.
"""
from cmlibs.zinc.context import Context
from segmentationstitcher.segment import Segment
from segmentationstitcher.annotation import region_get_annotations
from pathlib import Path


class Stitcher:
    """
    Interface for stitching segmentation data from and calculating transformations between adjacent image blocks.
    """

    def __init__(self, segmentation_file_names: list):
        """
        :param segmentation_file_names: List of filenames containing raw segmentations in Zinc format.
        """
        self._context = Context("Scaffoldfitter")
        self._root_region = self._context.getDefaultRegion()
        self._annotations = []
        self._segments = []
        self._version = 1  # increment when new settings added to migrate older serialised settings
        for segmentation_file_name in segmentation_file_names:
            name = Path(segmentation_file_name).stem
            segment = Segment(name, segmentation_file_name, self._root_region)
            self._segments.append(segment)
            segment_annotations = region_get_annotations(
                segment.get_raw_region(), simple_network_keywords=["vagus", "nerve", "trunk", "branch"],
                complex_network_keywords=["fascicle"], term_keyword="http")
            for segment_annotation in segment_annotations:
                name = segment_annotation.get_name()
                term = segment_annotation.get_term()
                index = 0
                for annotation in self._annotations:
                    if (annotation.get_name() == name) and (annotation.get_term() == term):
                        # print("Found annotation name", name, "term", term)
                        break  # exists already
                    if name > annotation.get_name():
                        index += 1
                else:
                    # print("Add annoation name", name, "term", term, "dim", segment_annotation.get_dimension(),
                    #       "category", segment_annotation.get_category())
                    self._annotations.insert(index, segment_annotation)

    def decode_settings(self, settings_in: dict):
        """
        Update stitcher settings from dictionary of serialised settings.
        :param settings_in: Dictionary of settings as produced by encode_settings().
        """
        assert settings_in.get("annotations") and settings_in.get("segments") and settings_in.get("version"), \
            "Stitcher.decode_settings: Invalid settings dictionary"
        # settings_version = settings_in["version"]

        # update annotations and warn about differences
        processed_count = 0
        for annotation_settings in settings_in["annotations"]:
            name = annotation_settings["name"]
            term = annotation_settings["term"]
            for annotation in self._annotations:
                if (annotation.get_name() == name) and (annotation.get_term() == term):
                    annotation.decode_settings(annotation_settings)
                    processed_count += 1
                    break
            else:
                print("WARNING: Segmentation Stitcher.  Annotation with name", name, "term", term,
                      "in settings not found; ignoring. Have input files changed?")
        if processed_count != len(self._annotations):
            for annotation in self._annotations:
                name = annotation.get_name()
                term = annotation.get_term()
                for annotation_settings in settings_in["annotations"]:
                    if (annotation_settings["name"] == name) and (annotation_settings["term"] == term):
                        break
                else:
                    print("WARNING: Segmentation Stitcher.  Annotation with name", name, "term", term,
                          "not found in settings; using defaults. Have input files changed?")

        # update segment settings and warn about differences
        processed_count = 0
        for segment_settings in settings_in["segments"]:
            name = segment_settings["name"]
            for segment in self._segments:
                if segment.get_name() == name:
                    segment.decode_settings(segment_settings)
                    processed_count += 1
                    break
            else:
                print("WARNING: Segmentation Stitcher.  Segment with name", name,
                      "in settings not found; ignoring. Have input files changed?")
        if processed_count != len(self._segments):
            for segment in self._segments:
                name = segment.get_name()
                for segment_settings in settings_in["segments"]:
                    if segment_settings["name"] == name:
                        break
                else:
                    print("WARNING: Segmentation Stitcher.  Segment with name", name,
                          "not found in settings; using defaults. Have input files changed?")

    def encode_settings(self) -> dict:
        """
        :return: Dictionary of Stitcher settings ready to serialise to JSON.
        """
        settings = {
            "annotations": [annotation.encode_settings() for annotation in self._annotations],
            "segments": [segment.encode_settings() for segment in self._segments],
            "version": self._version
        }
        return settings

    def get_annotations(self):
        return self._annotations
    def get_segments(self):
        return self._segments

    def get_version(self):
        return self._version
