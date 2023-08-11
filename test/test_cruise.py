import unittest
import numpy as np
from cruise.base import Cruise, CruiseConfig
from test.test_constants import *

class TestGriddedSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.short_cruise = Cruise(CruiseConfig(path=SHORT_SURVEY_PATH,
                                          require_annotations=True,
                                          require_bottom=True,
                                          require_school_boxes=True))


    def test_frequences(self):
        frequencies = np.array([38000, 18000, 70000, 120000, 200000, 333000])
        self.assertTrue(np.all(frequencies == np.array(self.short_cruise.frequencies())))

    def test_name(self):
        self.assertEqual('S2019847_0511', self.short_cruise.name)

    def test_size(self):
        self.assertEqual(249415, self.short_cruise.num_pings())
        self.assertEqual(2634, self.short_cruise.num_ranges())

    def test_categories(self):
        self.assertEqual([1, 27, 6009], self.short_cruise.categories())

