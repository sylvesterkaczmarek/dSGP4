import unittest

import dsgp4
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch


BASE_TLE = (
    "1 43437U 18100A   20143.90384230  .00041418  00000-0  10000-3 0 99968",
    "2 43437  97.8268 249.9127 0221000 123.9136 259.1144 15.12608579563539",
)


def _tle_with_rev_per_day(rev_per_day):
    tle = dsgp4.TLE(list(BASE_TLE))
    tle.update({"mean_motion": rev_per_day * 2.0 * np.pi / 86400.0})
    return tle


class Sgp4RegimeBoundaryTestCase(unittest.TestCase):
    def test_reference_and_dsgp4_select_same_regime(self):
        # 6.45 rev/day is safely below the 225-minute period threshold;
        # 6.35 rev/day is safely above it while remaining close to the switch.
        cases = [(6.45, "n"), (6.35, "d")]

        for rev_per_day, expected_method in cases:
            tle = _tle_with_rev_per_day(rev_per_day)
            reference = sgp4.io.twoline2rv(
                tle.line1,
                tle.line2,
                whichconst=sgp4.earth_gravity.wgs84,
            )
            dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

            self.assertEqual(reference.method, expected_method)
            self.assertEqual(tle._method, expected_method)

    def test_states_remain_reference_consistent_across_switch(self):
        times = torch.tensor([-720.0, -180.0, 0.0, 180.0, 720.0])

        for rev_per_day in (6.45, 6.35):
            tle = _tle_with_rev_per_day(rev_per_day)
            reference = sgp4.io.twoline2rv(
                tle.line1,
                tle.line2,
                whichconst=sgp4.earth_gravity.wgs84,
            )
            dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

            actual_states = dsgp4.propagate(tle, times).detach().numpy()
            reference_states = np.stack(
                [
                    np.asarray(sgp4.propagation.sgp4(reference, float(t)))
                    for t in times
                ]
            )

            np.testing.assert_allclose(
                actual_states,
                reference_states,
                rtol=0.0,
                atol=1e-8,
            )

    def test_regime_selection_is_stable_under_small_period_offsets(self):
        near = _tle_with_rev_per_day(6.45)
        deep = _tle_with_rev_per_day(6.35)
        dsgp4.initialize_tle(near, gravity_constant_name="wgs-84")
        dsgp4.initialize_tle(deep, gravity_constant_name="wgs-84")

        near_period = 2.0 * np.pi / float(near._no_unkozai)
        deep_period = 2.0 * np.pi / float(deep._no_unkozai)

        self.assertLess(near_period, 225.0)
        self.assertGreaterEqual(deep_period, 225.0)


if __name__ == "__main__":
    unittest.main()
