import unittest

import dsgp4
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch


TLES = [
    (
        "1 25544U 98067A   24087.49097222  .00016717  00000+0  10270-3 0  9993",
        "2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000000",
    ),
    (
        "1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
        "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093",
    ),
]

GRAVITY_MODELS = [
    ("wgs-72old", sgp4.earth_gravity.wgs72old),
    ("wgs-72", sgp4.earth_gravity.wgs72),
    ("wgs-84", sgp4.earth_gravity.wgs84),
]


class GravityModelParityTestCase(unittest.TestCase):
    def test_all_supported_gravity_models_match_python_sgp4(self):
        times = torch.tensor([-1440.0, -180.0, 0.0, 180.0, 1440.0])

        for gravity_name, reference_constants in GRAVITY_MODELS:
            for line1, line2 in TLES:
                with self.subTest(gravity_model=gravity_name, satellite=line1[2:7]):
                    tle = dsgp4.TLE([line1, line2])
                    dsgp4.initialize_tle(
                        tle,
                        gravity_constant_name=gravity_name,
                    )
                    actual = dsgp4.propagate(
                        tle,
                        times,
                        initialized=True,
                    ).detach().numpy()

                    reference = sgp4.io.twoline2rv(
                        line1,
                        line2,
                        whichconst=reference_constants,
                    )
                    expected = np.stack(
                        [
                            np.asarray(
                                sgp4.propagation.sgp4(
                                    reference,
                                    float(tsince),
                                )
                            )
                            for tsince in times
                        ]
                    )

                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=0.0,
                        atol=1e-8,
                    )


if __name__ == "__main__":
    unittest.main()
