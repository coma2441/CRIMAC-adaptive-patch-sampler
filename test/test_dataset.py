import unittest

from cruise.base import Cruise, CruiseConfig
from samplers.random import Random
from samplers.gridded import Gridded
from dataset import DatasetSegmentation

import numpy as np

TEST_SURVEY = "/lokal_uten_backup/pro/COGMAR/zarr_data_feb23/2019/S2019847_0511"

class TestDatasetSegmentation(unittest.TestCase):
    def setUp(self) -> None:
        cruise_path = TEST_SURVEY
        self.cruise = Cruise(CruiseConfig(path=cruise_path,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))
        self.num_samples = 1000
        self.random_sampler = Random(cruise_list=[self.cruise], num_samples=self.num_samples)

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

        gridded_sampler = Gridded([self.cruise], patch_size=(256, 256), patch_overlap=0)
        gridded_dataset = DatasetSegmentation([gridded_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=None)
        self.assertEqual(len(gridded_dataset), (self.cruise.num_pings()//256)*(self.cruise.num_ranges()//256))

    def test_school_output(self):
        gridded_sampler = Gridded([self.cruise], patch_size=(256, 256))
        gridded_dataset = DatasetSegmentation([gridded_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=None)

        # Select area with fish schools
        out = gridded_dataset[3051]
        self.assertListEqual(list(np.unique(out['mask'])), [0, 1, 2])

        gridded_dataset2 = DatasetSegmentation([gridded_sampler], patch_size=[256, 256],
                                      frequencies=[18000, 38000, 120000, 200000], categories=[1, 27])
        out = gridded_dataset2[3051]
        self.assertListEqual(list(np.unique(out['mask'])), [0, 1, 27])




