import unittest
import numpy as np
from cruise.base import Cruise, CruiseConfig

TEST_SURVEY = "/lokal_uten_backup/pro/COGMAR/zarr_data_feb23/2019/S2019847_0511"

class TestGriddedSampler(unittest.TestCase):
    def setUp(self) -> None:
        cruise_path = TEST_SURVEY
        self.cruise = Cruise(CruiseConfig(path=cruise_path,
                                          require_annotations=True,
                                          require_bottom=True))


    def test_frequences(self):
        frequencies = np.array([38000, 18000, 70000, 120000, 200000, 333000])
        self.assertTrue(np.all(frequencies == np.array(self.cruise.frequencies())))

    def test_name(self):
        self.assertEqual('S2019847_0511', self.cruise.name)

    def test_size(self):
        self.assertEqual(249415, self.cruise.num_pings())
        self.assertEqual(2634, self.cruise.num_ranges())

    def test_categories(self):
        self.assertEqual([1, 27, 6009], self.cruise.categories())

