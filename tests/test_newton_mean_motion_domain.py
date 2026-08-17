import unittest

import numpy as np

from dsgp4.newton_method import _MIN_TLE_MEAN_MOTION, _bound_mean_motion_step


class NewtonMeanMotionDomainTestCase(unittest.TestCase):
    def test_in_domain_step_is_unchanged(self):
        current = 0.06
        step = -0.01
        self.assertEqual(_bound_mean_motion_step(current, step), step)

    def test_negative_candidate_is_projected_to_positive_tle_resolution(self):
        current = 0.01
        step = -0.02
        bounded = _bound_mean_motion_step(current, step)
        candidate = current + bounded
        self.assertAlmostEqual(candidate, _MIN_TLE_MEAN_MOTION, places=18)
        self.assertGreater(candidate, 0.0)

    def test_zero_candidate_is_projected_to_positive_tle_resolution(self):
        current = 0.01
        bounded = _bound_mean_motion_step(current, -current)
        self.assertAlmostEqual(current + bounded, _MIN_TLE_MEAN_MOTION, places=18)

    def test_lower_bound_matches_one_unit_of_tle_mean_motion_precision(self):
        rev_per_day = _MIN_TLE_MEAN_MOTION * 1440.0 / (2.0 * np.pi)
        self.assertAlmostEqual(rev_per_day, 1e-8, places=18)


if __name__ == "__main__":
    unittest.main()
