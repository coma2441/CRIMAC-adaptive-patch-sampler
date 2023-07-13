import unittest

from cruise.base import Cruise, CruiseConfig
from samplers.random import Random
from dataset import DatasetSegmentation

TEST_SURVEY = "/lokal_uten_backup/pro/COGMAR/zarr_data_feb23/2019/S2019847_0511"

class TestDatasetSegmentation(unittest.TestCase):
    def setUp(self) -> None:
        cruise_path = TEST_SURVEY
        self.cruise = Cruise(CruiseConfig(path=cruise_path,
                                          require_annotations=True,
                                          require_bottom=True))
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
