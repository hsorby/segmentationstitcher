import os
import unittest
from segmentationstitcher.annotation import AnnotationCategory
from segmentationstitcher.stitcher import Stitcher
from tests.testutils import assertAlmostEqualList

here = os.path.abspath(os.path.dirname(__file__))


class StitchVagusTestCase(unittest.TestCase):

    def test_io_vagus1(self):
        """
        Test loading, modifying and serialising synthetic vagus nerve/fascicle segmentations.
        """
        resource_names = [
            "vagus-segment1.exf",
            "vagus-segment2.exf",
            "vagus-segment3.exf",
        ]
        TOL = 1.0E-7
        zero = [0.0, 0.0, 0.0]
        new_translation = [5.0, 0.5, 0.1]
        segmentation_file_names = [os.path.join(here, "resources", resource_name) for resource_name in resource_names]
        network_group1_keywords = ["vagus", "nerve", "trunk", "branch"]
        network_group2_keywords = ["fascicle"]
        stitcher1 = Stitcher(segmentation_file_names, network_group1_keywords, network_group2_keywords)
        segments1 = stitcher1.get_segments()
        self.assertEqual(3, len(segments1))
        segment12 = segments1[1]
        self.assertEqual("vagus-segment2", segment12.get_name())
        assertAlmostEqualList(self, zero, segment12.get_translation(), delta=TOL)
        segment12.set_translation(new_translation)
        annotations1 = stitcher1.get_annotations()
        self.assertEqual(7, len(annotations1))
        self.assertEqual(1, stitcher1.get_version())
        annotation11 = annotations1[0]
        self.assertEqual("Epineurium", annotation11.get_name())
        self.assertEqual("http://purl.obolibrary.org/obo/UBERON_0000124", annotation11.get_term())
        self.assertEqual(AnnotationCategory.GENERAL, annotation11.get_category())
        annotation12 = annotations1[1]
        self.assertEqual("Fascicle", annotation12.get_name())
        self.assertEqual("http://uri.interlex.org/base/ilx_0738426", annotation12.get_term())
        self.assertEqual(AnnotationCategory.NETWORK_GROUP_2, annotation12.get_category())
        annotation15 = annotations1[4]
        self.assertEqual("left vagus X nerve trunk", annotation15.get_name())
        self.assertEqual('http://purl.obolibrary.org/obo/UBERON_0035020', annotation15.get_term())
        self.assertEqual(AnnotationCategory.NETWORK_GROUP_1, annotation15.get_category())
        annotation17 = annotations1[6]
        self.assertEqual("unknown", annotation17.get_name())
        self.assertEqual(AnnotationCategory.GENERAL, annotation17.get_category())

        # test changing category and that category groups are updated
        segment13 = segments1[2]
        mesh1d = segment13.get_raw_region().getFieldmodule().findMeshByDimension(1)
        exclude13_group = segment13.get_category_group(AnnotationCategory.EXCLUDE)
        exclude13_mesh_group = exclude13_group.getMeshGroup(mesh1d)
        general13_group = segment13.get_category_group(AnnotationCategory.GENERAL)
        general13_mesh_group = general13_group.getMeshGroup(mesh1d)
        self.assertFalse(exclude13_mesh_group.isValid())
        self.assertEqual(27, general13_mesh_group.getSize())
        annotation17_group = segment13.get_annotation_group(annotation17)
        annotation17_mesh_group = annotation17_group.getMeshGroup(mesh1d)
        self.assertEqual(1, annotation17_mesh_group.getSize())
        annotation17.set_category(AnnotationCategory.EXCLUDE)
        exclude13_mesh_group = exclude13_group.getMeshGroup(mesh1d)
        self.assertEqual(1, exclude13_mesh_group.getSize())
        self.assertEqual(26, general13_mesh_group.getSize())

        settings = stitcher1.encode_settings()
        self.assertEqual(3, len(settings["segments"]))
        self.assertEqual(7, len(settings["annotations"]))
        self.assertEqual(1, settings["version"])
        assertAlmostEqualList(self, new_translation, settings["segments"][1]["translation"], delta=TOL)
        self.assertEqual(AnnotationCategory.EXCLUDE.name, settings["annotations"][6]["category"])

        stitcher2 = Stitcher(segmentation_file_names, network_group1_keywords, network_group2_keywords)
        stitcher2.decode_settings(settings)
        segments2 = stitcher2.get_segments()
        segment22 = segments2[1]
        assertAlmostEqualList(self, new_translation, segment22.get_translation(), delta=TOL)
        annotations2 = stitcher2.get_annotations()
        annotation27 = annotations2[6]
        self.assertEqual(AnnotationCategory.EXCLUDE, annotation27.get_category())


if __name__ == "__main__":
    unittest.main()
