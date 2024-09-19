"""
Interface for stitching segmentation data from and calculating transformations between adjacent image blocks.
"""
from cmlibs.utils.zinc.general import HierarchicalChangeManager
from cmlibs.zinc.context import Context
from segmentationstitcher.connection import Connection
from segmentationstitcher.segment import Segment
from segmentationstitcher.annotation import region_get_annotations

import copy
from pathlib import Path


class Stitcher:
    """
    Interface for stitching segmentation data from and calculating transformations between adjacent image blocks.
    """

    def __init__(self, segmentation_file_names: list, network_group1_keywords, network_group2_keywords):
        """
        :param segmentation_file_names: List of filenames containing raw segmentations in Zinc format.
        :param network_group1_keywords: List of keywords. Segmented networks annotated with any of these keywords are
        initially assigned to network group 1, allowing them to be stitched together.
        :param network_group2_keywords: List of keywords. Segmented networks annotated with any of these keywords are
        initially assigned to network group 2, allowing them to be stitched together.
        """
        self._context = Context("Segmentation Stitcher")
        self._root_region = self._context.getDefaultRegion()
        self._annotations = []
        self._network_group1_keywords = copy.deepcopy(network_group1_keywords)
        self._network_group2_keywords = copy.deepcopy(network_group2_keywords)
        self._term_keywords = ['fma:', 'fma_', 'ilx:', 'ilx_', 'uberon:', 'uberon_']
        self._segments = []
        self._connections = []
        self._version = 1  # increment when new settings added to migrate older serialised settings
        for segmentation_file_name in segmentation_file_names:
            name = Path(segmentation_file_name).stem
            segment = Segment(name, segmentation_file_name, self._root_region)
            self._segments.append(segment)
            segment_annotations = region_get_annotations(
                segment.get_raw_region(), self._network_group1_keywords, self._network_group2_keywords,
                self._term_keywords)
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
        with HierarchicalChangeManager(self._root_region):
            for segment in self._segments:
                segment.reset_annotation_category_groups(self._annotations)
        for annotation in self._annotations:
            annotation.set_category_change_callback(self._annotation_change)

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
        with HierarchicalChangeManager(self._root_region):
            for segment in self._segments:
                segment.reset_annotation_category_groups(self._annotations)

        # create connections from stitcher settings' connection serialisations
        assert len(self._connections) == 0, "Cannot decode connections after any exist"
        for connection_settings in settings_in["connections"]:
            connection_segments = []
            for segment_name in connection_settings["segments"]:
                for segment in self._segments:
                    if segment.get_name() == segment_name:
                        connection_segments.append(segment)
                        break
                else:
                    print("WARNING: Segmentation Stitcher.  Segment with name", segment_name,
                          "in connection settings not found; ignoring. Have input files changed?")
            if len(connection_segments) >= 2:
                connection = self.create_connection(connection_segments)
                connection.decode_settings(connection_settings)


    def encode_settings(self) -> dict:
        """
        :return: Dictionary of Stitcher settings ready to serialise to JSON.
        """
        settings = {
            "annotations": [annotation.encode_settings() for annotation in self._annotations],
            "connections": [connection.encode_settings() for connection in self._connections],
            "segments": [segment.encode_settings() for segment in self._segments],
            "version": self._version
        }
        return settings

    def _annotation_change(self, annotation, old_category):
        """
        Callback from annotation that its category has changed.
        Update segment category groups.
        :param annotation: Annotation that has changed category.
        :param old_category: The old category to remove segmentations with annotation from.
        """
        with HierarchicalChangeManager(self._root_region):
            for segment in self._segments:
                segment.update_annotation_category(annotation, old_category)

    def get_annotations(self):
        return self._annotations

    def create_connection(self, segments):
        """
        :param segments: List of 2 Stitcher Segment objects to connect.
        :return: Connection object or None if invalid segments or connection between segments already exists
        """
        if len(segments) != 2:
            print("Only supports connections between 2 segments")
            return None
        for connection in self._connections:
            if all(segment in connection.get_segments() for segment in segments):
                print("Stitcher.create_connection:  Already have a connection between segments")
                return None
        connection = Connection(segments, self._root_region)
        self._connections.append(connection)
        return connection

    def get_connections(self):
        return self._connections

    def remove_connection(self, connection):
        self._connections.remove(connection)

    def get_context(self):
        return self._context

    def get_root_region(self):
        return self._root_region

    def get_segments(self):
        return self._segments

    def get_version(self):
        return self._version

    def write_output_segmentation_file(self, file_name):
        pass
