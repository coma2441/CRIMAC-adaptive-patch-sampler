import unittest
import numpy as np

from cruise.base import Cruise, CruiseConfig
from utils.cropping import crop_data, crop_annotations, crop_bbox
from test.test_constants import *

class TestCropUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.short_cruise = Cruise(CruiseConfig(path=SHORT_SURVEY_PATH,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_pings = self.short_cruise.num_pings()
        self.num_ranges = self.short_cruise.num_ranges()

    def test_edge_cases_data(self):
        self.assertFalse(np.all(np.isnan(crop_data(self.short_cruise, [0, 0], [256, 256]))))
        self.assertFalse(np.all(np.isnan(crop_data(self.short_cruise, [self.num_pings, 0], [256, 256]))))

    def test_output_data(self):
        data = crop_data(self.short_cruise, [1000, 0], [256, 256])
        self.assertTrue(np.all(np.isnan(data[:, :, :128])))

        data = crop_data(self.short_cruise, [0, 500], [256, 256])
        self.assertTrue(np.all(np.isnan(data[:, :128, :])))

        data = crop_data(self.short_cruise, [0, 500], [256, 256], boundary_val=-100)
        self.assertTrue(np.all(data[:, :128, :] == -100))

    def test_output_shape_data(self):
        data = crop_data(self.short_cruise, [1000, 500], [256, 512])
        self.assertEqual(data.shape, (6, 256, 512))

        data = crop_data(self.short_cruise, [1000, 500], [512, 256])
        self.assertEqual(data.shape, (6, 512, 256))

        data = crop_data(self.short_cruise, [1000, 500], [256, 256], frequencies=[18000, 120000])
        self.assertEqual(data.shape, (2, 256, 256))

    def test_edge_cases_labels(self):
        self.assertFalse(np.all(np.isnan(crop_annotations(self.short_cruise, [0, 0], [256, 256]))))
        self.assertFalse(np.all(np.isnan(crop_annotations(self.short_cruise, [self.num_pings, 0], [256, 256]))))

    def test_output_annotations(self):
        annotations = crop_annotations(self.short_cruise, [1000, 0], [256, 256])
        self.assertTrue(np.all(np.isnan(annotations[:, :, :128])))

        annotations = crop_annotations(self.short_cruise, [0, 500], [256, 256])
        self.assertTrue(np.all(np.isnan(annotations[:, :128, :])))

        annotations = crop_annotations(self.short_cruise, [0, 500], [256, 256], boundary_val=-100)
        self.assertTrue(np.all(annotations[:, :128, :] == -100))

    def test_output_shape_annotations(self):
        annotations = crop_annotations(self.short_cruise, [1000, 500], [256, 512])
        self.assertEqual(annotations.shape, (3, 256, 512))

        annotations = crop_annotations(self.short_cruise, [1000, 500], [512, 256])
        self.assertEqual(annotations.shape, (3, 512, 256))

        annotations = crop_annotations(self.short_cruise, [1000, 500], [256, 256], categories=[1, 27])
        self.assertEqual(annotations.shape, (2, 256, 256))

    def test_output_boxes(self):
        boxes, labels = crop_bbox(self.short_cruise, [1000, 0], [256, 256])
        self.assertEqual(len(labels), 0)
        self.assertEqual(boxes.shape, (0, 4))

        boxes, labels = crop_bbox(self.short_cruise, [78268, 578], [256, 256]) # patch with three fish schools
        self.assertEqual(len(labels), 3)
        self.assertEqual(boxes.shape, (3, 4))

        boxes, labels = crop_bbox(self.short_cruise, [78268, 578], [256, 256], categories=[1]) # patch with 2 "other" schools
        self.assertEqual(len(labels), 2)
        self.assertEqual(boxes.shape, (2, 4))
