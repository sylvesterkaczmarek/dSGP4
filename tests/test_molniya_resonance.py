import unittest

import dsgp4
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch


MOLNIYA_TLE = (
    "1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0  813",
    "2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383 13900",
)


class MolniyaResonanceValidationTestCase(unittest.TestCase):
    def test_two_to_one_resonance_matches_reference_propagator(self):
        line1, line2 = MOLNIYA_TLE
        tle = dsgp4.TLE([line1, line2])
        dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

        self.assertEqual(tle._method, "d")
        self.assertEqual(tle._irez, 2)

        reference = sgp4.io.twoline2rv(
            line1,
            line2,
            whichconst=sgp4.earth_gravity.wgs84,
        )

        times = torch.tensor(
            [-4320.0, -1440.0, -720.0, 0.0, 720.0, 1440.0, 4320.0]
        )
        actual = dsgp4.propagate(tle, times, initialized=True).detach().numpy()
        expected = np.stack(
            [
                np.asarray(sgp4.propagation.sgp4(reference, float(tsince)))
                for tsince in times
            ]
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-8)

    def test_resonant_propagation_remains_finite_over_one_week(self):
        tle = dsgp4.TLE(list(MOLNIYA_TLE))
        dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

        times = torch.linspace(-7.0 * 1440.0, 7.0 * 1440.0, 57)
        states = dsgp4.propagate(tle, times, initialized=True)

        self.assertTrue(torch.isfinite(states).all())
        radii = torch.linalg.vector_norm(states[:, 0, :], dim=1)
        self.assertTrue(torch.all(radii > 6378.137))


if __name__ == "__main__":
    unittest.main()
