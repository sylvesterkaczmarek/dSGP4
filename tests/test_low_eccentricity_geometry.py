import unittest

import dsgp4
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch


# Low-eccentricity cases exercise geometries where classical angular elements
# become increasingly ill-conditioned. The GEO-like case is also close to
# equatorial, while the LEO case retains a substantial inclination.
CASES = [
    (
        "1 40967U 15058A   24060.50000000  .00000033  00000-0  00000+0 0  9992",
        "2 40967   0.0187  89.2881 0002035  82.1068 220.3980  1.00270014 30754",
    ),
    (
        "1 25544U 98067A   24087.49097222  .00016717  00000+0  10270-3 0  9993",
        "2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000000",
    ),
]


class LowEccentricityGeometryTestCase(unittest.TestCase):
    def test_low_eccentricity_states_match_reference(self):
        times = torch.tensor([-1440.0, -360.0, 0.0, 360.0, 1440.0])

        for line1, line2 in CASES:
            tle = dsgp4.TLE([line1, line2])
            dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")
            actual = dsgp4.propagate(tle, times).detach().numpy()

            reference_sat = sgp4.io.twoline2rv(
                line1,
                line2,
                whichconst=sgp4.earth_gravity.wgs84,
            )
            reference = np.stack(
                [
                    np.asarray(sgp4.propagation.sgp4(reference_sat, float(t)))
                    for t in times
                ]
            )

            np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-8)
            self.assertTrue(np.isfinite(actual).all())

    def test_near_equatorial_case_keeps_finite_orientation(self):
        line1, line2 = CASES[0]
        tle = dsgp4.TLE([line1, line2])
        dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

        states = dsgp4.propagate(
            tle,
            torch.linspace(-3.0 * 1440.0, 3.0 * 1440.0, 25),
        )

        self.assertTrue(torch.isfinite(states).all())
        radii = torch.linalg.vector_norm(states[:, 0, :], dim=1)
        self.assertTrue(torch.all(radii > 6000.0))


if __name__ == "__main__":
    unittest.main()
