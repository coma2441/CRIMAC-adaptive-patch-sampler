import unittest
import numpy as np
import pandas as pd
import os

from cruise.base import Cruise, CruiseConfig
from samplers.random import Random
from samplers.regular import Regular
from samplers.indexed import Indexed
from test.test_constants import *


class TestRandomSampler(unittest.TestCase):
    def setUp(self) -> None:
        if self.test_surveys_exist():
            self.short_cruise = Cruise(CruiseConfig(path=SHORT_SURVEY_PATH,
                                                    require_annotations=True,
                                                    require_bottom=True,
                                                    require_school_boxes=True))
            self.num_samples = 1000
            self.sampler = Random(cruise_list=[self.short_cruise], num_samples=self.num_samples)
        else:
            self.skipTest("Short survey path not found")

    def test_surveys_exist(self):
        return os.path.isdir(SHORT_SURVEY_PATH)

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

    # def test_random_multiple_cruises(self):
    #     cruise_2016 = Cruise(CruiseConfig(path=SANDEEL_2016_PATH,
    #                                       require_annotations=True,
    #                                       require_bottom=True,
    #                                       require_school_boxes=True))
    #     cruise_2017 = Cruise(CruiseConfig(path=SANDEEL_2017_PATH,
    #                                       require_annotations=True,
    #                                       require_bottom=True,
    #                                       require_school_boxes=True))
    #
    #     np.random.seed(42)
    #     sampler = Random(cruise_list=[cruise_2016, cruise_2017], num_samples=self.num_samples)
    #     outputs = []
    #
    #     for i in range(100):
    #         outputs.append(sampler())
    #
    #     # Assert both cruises are sampled from
    #     self.assertTrue(np.any([output['cruise'].name == cruise_2016.name for output in outputs]))
    #     self.assertTrue(np.any([output['cruise'].name == cruise_2017.name for output in outputs]))


class TestGriddedSampler(unittest.TestCase):
    def setUp(self) -> None:
        if self.test_surveys_exist():
            self.short_cruise = Cruise(CruiseConfig(path=SHORT_SURVEY_PATH,
                                                    require_annotations=True,
                                                    require_bottom=True,
                                                    require_school_boxes=True))
            self.num_samples = 1000
            self.patch_size = (256, 256)
            self.sampler = Regular(cruise_list=[self.short_cruise], stride=(256, 256))
        else:
            self.skipTest("Short survey path not found")

    def test_surveys_exist(self):
        return os.path.isdir(SHORT_SURVEY_PATH)


class TestIndexSampler(unittest.TestCase):
    def setUp(self) -> None:
        if self.test_surveys_exist():
            self.short_cruise = Cruise(CruiseConfig(path=SHORT_SURVEY_PATH,
                                                    require_annotations=True,
                                                    require_bottom=True,
                                                    require_school_boxes=True))
            self.num_pings = self.short_cruise.num_pings()
            self.num_ranges = self.short_cruise.num_ranges()

            # Create and save dummy index sampler file
            np.random.seed(42)
            random_pings = np.random.randint(0, self.num_pings, 1000)
            random_ranges = np.random.randint(0, self.num_ranges, 1000)

            self.df = pd.DataFrame({'ping_index': random_pings, 'range_index': random_ranges,
                                    'cruise_name': [self.short_cruise.name] * 1000})
            self.df.to_csv('test_index_sampler.csv', index=True)

            # Create sampler
            self.sampler = Indexed([self.short_cruise], 'test_index_sampler.csv')
        else:
            self.skipTest("Short survey path not found")

    def test_surveys_exist(self):
        return os.path.isdir(SHORT_SURVEY_PATH)

    def test_index_sampler(self):
        # Assert sampler returns the expected indices
        out_0 = self.sampler(0)
        self.assertEqual(out_0["cruise"].name, self.short_cruise.name)
        self.assertEqual(out_0["center_ping"], self.df.iloc[0].ping_index)
        self.assertEqual(out_0["center_range"], self.df.iloc[0].range_index)

        out_100 = self.sampler(100)
        self.assertEqual(out_100["cruise"].name, self.short_cruise.name)
        self.assertEqual(out_100["center_ping"], self.df.iloc[100].ping_index)
        self.assertEqual(out_100["center_range"], self.df.iloc[100].range_index)

        self.assertEqual(len(self.sampler), 1000)
        out_last = self.sampler(999)
        self.assertEqual(out_last["cruise"].name, self.short_cruise.name)
        self.assertEqual(out_last["center_ping"], self.df.iloc[999].ping_index)
        self.assertEqual(out_last["center_range"], self.df.iloc[999].range_index)

        # assert error is raised if sampler is called with an index larger than the number of samples
        self.assertRaises(IndexError, self.sampler, 1000)

    def tearDown(self):
        # Delete sampler after test
        os.remove('test_index_sampler.csv')


if __name__ == '__main__':
    unittest.main()
