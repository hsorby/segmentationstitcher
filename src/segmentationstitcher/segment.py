"""
A segment of the segmentation data, generally from a separate image block.
"""
from cmlibs.utils.zinc.group import group_add_group_local_contents, group_remove_group_local_contents
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK
from segmentationstitcher.annotation import AnnotationCategory


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
        self._segmentation_file_name = segmentation_file_name
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
        # ensure category groups exist:
        fieldmodule = self._raw_region.getFieldmodule()
        with ChangeManager(fieldmodule):
            for category in AnnotationCategory:
                group_name = category.get_group_name()
                group = fieldmodule.createFieldGroup()
                group.setName(group_name)
                group.setManaged(True)
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

    def get_annotation_group(self, annotation):
        """
        Get Zinc group containing segmentations for the supplied annotation
        :param annotation: An Annotation object.
        :return: Zinc FieldGroup in the segment's raw region, or None if not present in segment.
        """
        fieldmodule = self._raw_region.getFieldmodule()
        annotation_group = fieldmodule.findFieldByName(annotation.get_name()).castGroup()
        if annotation_group.isValid():
            return annotation_group
        return None

    def get_category_group(self, category):
        """
        Get Zinc group in which segmentations with the supplied annotation category are maintained
        for visualisation.
        :param category: The AnnotationCategory to query.
        :return: Zinc FieldGroup in the segment's raw region.
        """
        fieldmodule = self._raw_region.getFieldmodule()
        group_name = category.get_group_name()
        group = fieldmodule.findFieldByName(group_name).castGroup()
        return group

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

    def update_annotation_category(self, annotation, old_category=AnnotationCategory.EXCLUDE):
        """
        Ensures special groups representing annotion categories contain via addition or removal the
        correct contents for this annotation.
        :param annotation: The annotation to update category group for. Ensures its local contents
        (elements, nodes, datapoints) are in its category group.
        :param old_category: The old category for this annotation, i.e. category group to remove from.
        """
        new_category = annotation.get_category()
        if new_category == old_category:
            return
        annotation_group = self.get_annotation_group(annotation)
        if not annotation_group:
            return  # not present in this segment
        fieldmodule = self._raw_region.getFieldmodule()
        with ChangeManager(fieldmodule):
            old_category_group = self.get_category_group(old_category)
            group_remove_group_local_contents(old_category_group, annotation_group)
            new_category_group = self.get_category_group(new_category)
            group_add_group_local_contents(new_category_group, annotation_group)

    def reset_annotation_category_groups(self, annotations):
        """
        Rebuild all annotation category groups e.g. after loading settings.
        :param annotations: List of all annotations from stitcher.
        """
        fieldmodule = self._raw_region.getFieldmodule()
        with ChangeManager(fieldmodule):
            # clear all category groups
            for category in AnnotationCategory:
                category_group = self.get_category_group(category)
                category_group.clear()
            for annotation in annotations:
                annotation_group = self.get_annotation_group(annotation)
                if annotation_group:
                    category_group = self.get_category_group(annotation.get_category())
                    group_add_group_local_contents(category_group, annotation_group)
