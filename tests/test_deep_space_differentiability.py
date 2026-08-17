import unittest

import dsgp4
import torch


DEEP_SPACE_TLES = [
    (
        "1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
        "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093",
    ),
    (
        "1 09998U 74033F   05148.79417928 -.00000112  00000-0  00000+0 0  4480",
        "2 09998   9.4958 313.1750 0270971 327.5225  30.8097  1.16186785 45878",
    ),
]


def _state_from_elements(line1, line2, elements, tsince):
    tle = dsgp4.tle.TLE([line1, line2])
    whichconst = dsgp4.util.get_gravity_constants("wgs-84")
    dsgp4.sgp4init(
        whichconst=whichconst,
        opsmode="i",
        satn=tle.satellite_catalog_number,
        epoch=(tle._jdsatepoch + tle._jdsatepochF) - 2433281.5,
        xbstar=elements[0],
        xndot=elements[1],
        xnddot=elements[2],
        xecco=elements[3],
        xargpo=elements[4],
        xinclo=elements[5],
        xmo=elements[6],
        xno_kozai=elements[7],
        xnodeo=elements[8],
        satellite=tle,
    )
    if tle._method != "d":
        raise AssertionError("test fixture must initialize in deep-space mode")
    return dsgp4.sgp4(tle, torch.as_tensor(tsince))


class DeepSpaceDifferentiabilityTestCase(unittest.TestCase):
    def test_position_time_gradient_matches_reported_velocity(self):
        for line1, line2 in DEEP_SPACE_TLES:
            tle = dsgp4.tle.TLE([line1, line2])
            tsince = torch.tensor(720.0, requires_grad=True)
            state = dsgp4.propagate(tle, tsince, initialized=False)

            dr_dt_minutes = torch.stack(
                [
                    torch.autograd.grad(
                        state[0, component],
                        tsince,
                        retain_graph=True,
                    )[0]
                    for component in range(3)
                ]
            )

            torch.testing.assert_close(
                dr_dt_minutes,
                60.0 * state[1],
                rtol=2e-3,
                atol=1e-1,
            )

    def test_deep_space_element_gradients_match_central_differences(self):
        # The first three TLE terms (BSTAR, ndot, nddot) are not the primary
        # geometric state variables. Validate the six SGP4 mean elements that
        # directly define the orbit.
        element_indices = (3, 4, 5, 6, 7, 8)

        for line1, line2 in DEEP_SPACE_TLES:
            tle = dsgp4.tle.TLE([line1, line2])
            elements = dsgp4.initialize_tle(
                tle,
                gravity_constant_name="wgs-84",
                with_grad=True,
            )
            self.assertEqual(tle._method, "d")

            tsince = torch.tensor(720.0)
            state = dsgp4.propagate(tle, tsince, initialized=True)
            state_flat = state.reshape(-1)

            jacobian = torch.stack(
                [
                    torch.autograd.grad(
                        state_flat[component],
                        elements,
                        retain_graph=True,
                    )[0]
                    for component in range(6)
                ]
            )

            base = elements.detach()
            for element_index in element_indices:
                step = max(
                    1e-6,
                    1e-6 * abs(float(base[element_index])),
                )
                plus = base.clone()
                minus = base.clone()
                plus[element_index] += step
                minus[element_index] -= step

                finite_difference = (
                    _state_from_elements(line1, line2, plus, tsince).reshape(-1)
                    - _state_from_elements(line1, line2, minus, tsince).reshape(-1)
                ) / (2.0 * step)

                torch.testing.assert_close(
                    jacobian[:, element_index],
                    finite_difference,
                    rtol=3e-3,
                    atol=5e-1,
                )


if __name__ == "__main__":
    unittest.main()
