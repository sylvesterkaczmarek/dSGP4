import unittest

import dsgp4
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch


GEO_RESONANT_TLE = (
    "1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
    "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093",
)


class GeosynchronousResonanceLongHorizonTestCase(unittest.TestCase):
    def test_synchronous_resonance_matches_reference_over_sixty_days(self):
        line1, line2 = GEO_RESONANT_TLE
        tle = dsgp4.TLE([line1, line2])
        dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

        self.assertEqual(tle._method, "d")
        self.assertEqual(tle._irez, 1)

        days = torch.tensor([-30.0, -14.0, -7.0, 0.0, 7.0, 14.0, 30.0])
        times = days * 1440.0
        states = dsgp4.propagate(tle, times).detach().numpy()

        reference = sgp4.io.twoline2rv(
            line1,
            line2,
            whichconst=sgp4.earth_gravity.wgs84,
        )
        expected = np.stack(
            [
                np.array(sgp4.propagation.sgp4(reference, float(t)))
                for t in times.numpy()
            ]
        )

        np.testing.assert_allclose(
            states[:, 0, :],
            expected[:, 0, :],
            rtol=0.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            states[:, 1, :],
            expected[:, 1, :],
            rtol=0.0,
            atol=1e-8,
        )

    def test_long_horizon_states_remain_finite_and_geosynchronous_scale(self):
        tle = dsgp4.TLE(list(GEO_RESONANT_TLE))
        dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

        times = torch.linspace(-30.0 * 1440.0, 30.0 * 1440.0, 25)
        states = dsgp4.propagate(tle, times)
        radii = torch.linalg.vector_norm(states[:, 0, :], dim=1)

        self.assertTrue(torch.isfinite(states).all())
        self.assertTrue(torch.all(radii > 35000.0))
        self.assertTrue(torch.all(radii < 50000.0))


if __name__ == "__main__":
    unittest.main()
