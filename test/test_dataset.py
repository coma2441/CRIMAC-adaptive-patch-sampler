import unittest

from cruise.base import Cruise, CruiseConfig
from samplers.random import Random
from samplers.gridded import Gridded
from dataset import DatasetSegmentation, DatasetBoundingBox
from test.test_constants import *

import numpy as np


class TestDatasetSegmentation(unittest.TestCase):
    def setUp(self) -> None:
        self.short_cruise = Cruise(CruiseConfig(path=SHORT_SURVEY_PATH,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_samples = 1000
        self.random_sampler = Random(cruise_list=[self.short_cruise], num_samples=self.num_samples)

        self.dataset = DatasetSegmentation([self.random_sampler], patch_size=[256, 256],
                                           frequencies=[18000, 38000, 120000, 200000])

    def test_categories(self):
        dataset = DatasetSegmentation([self.random_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=None)
        out = dataset[0]
        self.assertEqual(out['labels'].shape, (3, 256, 256))

        dataset = DatasetSegmentation([self.random_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=[1, 27])
        out = dataset[0]
        self.assertEqual(out['labels'].shape, (2, 256, 256))

    def test_length(self):
        self.assertEqual(len(self.dataset), self.num_samples)

        small_dataset = DatasetSegmentation([self.random_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], num_samples=100)
        self.assertEqual(len(small_dataset), 100)

        gridded_sampler = Gridded([self.short_cruise], patch_size=(256, 256), patch_overlap=0)
        gridded_dataset = DatasetSegmentation([gridded_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=None)
        self.assertEqual(len(gridded_dataset), (self.short_cruise.num_pings()//256)*(self.short_cruise.num_ranges()//256))

    def test_school_output(self):
        gridded_sampler = Gridded([self.short_cruise], patch_size=(256, 256))
        gridded_dataset = DatasetSegmentation([gridded_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=None)

        # Select area with fish schools
        out = gridded_dataset[3051]
        self.assertListEqual(list(np.unique(out['mask'])), [0, 1, 2])

        gridded_dataset2 = DatasetSegmentation([gridded_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=[1, 27])
        out = gridded_dataset2[3051]
        self.assertListEqual(list(np.unique(out['mask'])), [0, 1, 27])


class TestDatasetBoundingBox(unittest.TestCase):
    def setUp(self) -> None:
        self.short_cruise = Cruise(CruiseConfig(path=SHORT_SURVEY_PATH,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_samples = 1000
        self.random_sampler = Random(cruise_list=[self.short_cruise], num_samples=self.num_samples)

        self.dataset = DatasetBoundingBox([self.random_sampler], patch_size=[256, 256],
                                           frequencies=[18000, 38000, 120000, 200000])


    def test_output_shape(self):
        out = self.dataset[0]
        self.assertEqual(out['data'].shape, (4, 256, 256))
        self.assertEqual(out['boxes'].shape[1], 4)

        datasetRectangle = DatasetBoundingBox([self.random_sampler], patch_size=[512, 256], frequencies=[18000, 38000, 120000, 200000])
        out = datasetRectangle[0]
        self.assertEqual(out['data'].shape, (4, 512, 256))
        self.assertEqual(out['boxes'].shape[1], 4)

        datasetRectangle2 = DatasetBoundingBox([self.random_sampler], patch_size=[256, 1028], frequencies=[18000, 38000, 120000, 200000])
        out = datasetRectangle2[0]
        self.assertEqual(out['data'].shape, (4, 256, 1028))
        self.assertEqual(out['boxes'].shape[1], 4)

        datasetOdd = DatasetBoundingBox([self.random_sampler], patch_size=[257, 257], frequencies=[18000, 38000, 120000, 200000])
        out = datasetOdd[0]
        self.assertEqual(out['data'].shape, (4, 257, 257))
        self.assertEqual(out['boxes'].shape[1], 4)

    def test_output_targets(self):
        gridded_sampler = Gridded([self.short_cruise], patch_size=(256, 256))
        gridded_dataset = DatasetBoundingBox([gridded_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000])

        # Select area with fish schools
        out = gridded_dataset[3051]

        self.assertTrue(out['boxes'].shape[0] > 0)
        self.assertTrue(np.all(out['boxes']) >= 0)
        self.assertTrue(np.all(out['boxes']) < 256)

        self.assertEqual(list(np.unique(out['labels'])), [1, 27])


