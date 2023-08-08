import unittest
import numpy as np
import pandas as pd
import os

from cruise.base import Cruise, CruiseConfig
from samplers.random import Random
from samplers.gridded import Gridded
from samplers.indexed import Indexed
from utils.cropping import crop_data, crop_annotations, crop_bbox

TEST_SURVEY = "/lokal_uten_backup/pro/COGMAR/zarr_data_feb23/2019/S2019847_0511"

class TestRandomSampler(unittest.TestCase):
    def setUp(self) -> None:
        cruise_path = TEST_SURVEY
        self.cruise = Cruise(CruiseConfig(path=cruise_path,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_samples = 1000
        self.sampler = Random(cruise_list=[self.cruise], num_samples=self.num_samples)

    def test_length(self):
        self.assertEqual(len(self.sampler), self.num_samples)

    def test_output(self):
        output = self.sampler()
        self.assertIn('cruise', output.keys())
        self.assertIn('center_ping', output.keys())
        self.assertIn('center_range', output.keys())

    def test_random(self):
        output1 = self.sampler()
        output2 = self.sampler()

        self.assertNotEqual(output1, output2)

class TestGriddedSampler(unittest.TestCase):
    def setUp(self) -> None:
        cruise_path = TEST_SURVEY
        self.cruise = Cruise(CruiseConfig(path=cruise_path,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_samples = 1000
        self.patch_size = [256, 256]
        self.sampler = Gridded(cruise_list=[self.cruise], patch_size=self.patch_size, patch_overlap=0)


class TestCropUtils(unittest.TestCase):
    def setUp(self) -> None:
        cruise_path = TEST_SURVEY
        self.cruise = Cruise(CruiseConfig(path=cruise_path,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_pings = self.cruise.num_pings()
        self.num_ranges = self.cruise.num_ranges()

    def test_edge_cases_data(self):
        self.assertFalse(np.all(np.isnan(crop_data(self.cruise, [0, 0], [256, 256]))))
        self.assertFalse(np.all(np.isnan(crop_data(self.cruise, [self.num_pings, 0], [256, 256]))))

    def test_output_data(self):
        data = crop_data(self.cruise, [1000, 0], [256, 256])
        self.assertTrue(np.all(np.isnan(data[:, :, :128])))

        data = crop_data(self.cruise, [0, 500], [256, 256])
        self.assertTrue(np.all(np.isnan(data[:, :128, :])))

        data = crop_data(self.cruise, [0, 500], [256, 256], boundary_val=-100)
        self.assertTrue(np.all(data[:, :128, :] == -100))

    def test_output_shape_data(self):
        data = crop_data(self.cruise, [1000, 500], [256, 512])
        self.assertEqual(data.shape, (6, 256, 512))

        data = crop_data(self.cruise, [1000, 500], [512, 256])
        self.assertEqual(data.shape, (6, 512, 256))

        data = crop_data(self.cruise, [1000, 500], [256, 256], frequencies=[18000, 120000])
        self.assertEqual(data.shape, (2, 256, 256))

    def test_edge_cases_labels(self):
        self.assertFalse(np.all(np.isnan(crop_annotations(self.cruise, [0, 0], [256, 256]))))
        self.assertFalse(np.all(np.isnan(crop_annotations(self.cruise, [self.num_pings, 0], [256, 256]))))

    def test_output_annotations(self):
        annotations = crop_annotations(self.cruise, [1000, 0], [256, 256])
        self.assertTrue(np.all(np.isnan(annotations[:, :, :128])))

        annotations = crop_annotations(self.cruise, [0, 500], [256, 256])
        self.assertTrue(np.all(np.isnan(annotations[:, :128, :])))

        annotations = crop_annotations(self.cruise, [0, 500], [256, 256], boundary_val=-100)
        self.assertTrue(np.all(annotations[:, :128, :] == -100))

    def test_output_shape_annotations(self):
        annotations = crop_annotations(self.cruise, [1000, 500], [256, 512])
        self.assertEqual(annotations.shape, (3, 256, 512))

        annotations = crop_annotations(self.cruise, [1000, 500], [512, 256])
        self.assertEqual(annotations.shape, (3, 512, 256))

        annotations = crop_annotations(self.cruise, [1000, 500], [256, 256], categories=[1, 27])
        self.assertEqual(annotations.shape, (2, 256, 256))

    def test_output_boxes(self):
        boxes, labels = crop_bbox(self.cruise, [1000, 0], [256, 256])
        self.assertEqual(len(labels), 0)
        self.assertEqual(boxes.shape, (0, 4))

        boxes, labels = crop_bbox(self.cruise, [78268, 578], [256, 256]) # patch with three fish schools
        self.assertEqual(len(labels), 3)
        self.assertEqual(boxes.shape, (3, 4))

        boxes, labels = crop_bbox(self.cruise, [78268, 578], [256, 256], categories=[1]) # patch with 2 "other" schools
        self.assertEqual(len(labels), 2)
        self.assertEqual(boxes.shape, (2, 4))

class TestIndexSampler(unittest.TestCase):
    def setUp(self) -> None:
        cruise_path = TEST_SURVEY
        self.cruise = Cruise(CruiseConfig(path=cruise_path,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_pings = self.cruise.num_pings()
        self.num_ranges = self.cruise.num_ranges()

        # Create and save dummy index sampler file
        np.random.seed(42)
        random_pings = np.random.randint(0, self.num_pings, 1000)
        random_ranges = np.random.randint(0, self.num_ranges, 1000)

        self.df = pd.DataFrame({'ping_index': random_pings, 'range_index': random_ranges,
                           'cruise_name': [self.cruise.name]*1000})
        self.df.to_csv('test_index_sampler.csv', index=True)

        # Create sampler
        self.sampler = Indexed([self.cruise], 'test_index_sampler.csv')

    def test_index_sampler(self):
        # Assert sampler returns the expected indices
        out_0 = self.sampler(0)
        self.assertEqual(out_0["cruise"].name, self.cruise.name)
        self.assertEqual(out_0["center_ping"], self.df.iloc[0].ping_index)
        self.assertEqual(out_0["center_range"], self.df.iloc[0].range_index)

        out_100 = self.sampler(100)
        self.assertEqual(out_100["cruise"].name, self.cruise.name)
        self.assertEqual(out_100["center_ping"], self.df.iloc[100].ping_index)
        self.assertEqual(out_100["center_range"], self.df.iloc[100].range_index)

        self.assertEqual(len(self.sampler), 1000)
        out_last = self.sampler(999)
        self.assertEqual(out_last["cruise"].name, self.cruise.name)
        self.assertEqual(out_last["center_ping"], self.df.iloc[999].ping_index)
        self.assertEqual(out_last["center_range"], self.df.iloc[999].range_index)

        # assert error is raised if sampler is called with an index larger than the number of samples
        self.assertRaises(IndexError, self.sampler, 1000)

    def tearDown(self):
        # Delete sampler after test
        os.remove('test_index_sampler.csv')



