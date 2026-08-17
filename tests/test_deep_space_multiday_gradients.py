import unittest

import dsgp4
import torch


DEEP_SPACE_CASES = [
    (
        (
            "1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
            "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093",
        ),
        4680.0,  # +3.25 days, synchronous 1:1 resonance
        1,
    ),
    (
        (
            "1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0  813",
            "2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383 13900",
        ),
        -3960.0,  # -2.75 days, Molniya 2:1 resonance
        2,
    ),
]


def _position_at(lines, tsince):
    tle = dsgp4.TLE(list(lines))
    dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")
    return dsgp4.propagate(tle, torch.tensor(float(tsince))).detach()[0]


class DeepSpaceMultiDayGradientTestCase(unittest.TestCase):
    def test_time_gradients_match_velocity_and_finite_differences(self):
        for lines, epoch_offset, expected_resonance in DEEP_SPACE_CASES:
            tle = dsgp4.TLE(list(lines))
            dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")
            self.assertEqual(tle._method, "d")
            self.assertEqual(tle._irez, expected_resonance)

            time = torch.tensor(epoch_offset, requires_grad=True)
            state = dsgp4.propagate(tle, time, initialized=True)

            dr_dt = torch.stack(
                [
                    torch.autograd.grad(
                        state[0, component],
                        time,
                        retain_graph=True,
                    )[0]
                    for component in range(3)
                ]
            )

            # tsince is measured in minutes, while SGP4 reports km/s.
            torch.testing.assert_close(
                dr_dt,
                60.0 * state[1],
                rtol=5e-3,
                atol=2e-1,
            )

            step = 1e-2  # minutes = 0.6 seconds
            finite_difference = (
                _position_at(lines, epoch_offset + step)
                - _position_at(lines, epoch_offset - step)
            ) / (2.0 * step)

            torch.testing.assert_close(
                dr_dt.detach(),
                finite_difference,
                rtol=5e-3,
                atol=2e-1,
            )

    def test_gradients_are_finite_away_from_resonance_step_boundaries(self):
        for lines, epoch_offset, _ in DEEP_SPACE_CASES:
            tle = dsgp4.TLE(list(lines))
            time = torch.tensor(epoch_offset, requires_grad=True)
            state = dsgp4.propagate(tle, time, initialized=False)
            scalar = torch.linalg.vector_norm(state[0])
            gradient = torch.autograd.grad(scalar, time)[0]

            self.assertTrue(torch.isfinite(state).all())
            self.assertTrue(torch.isfinite(gradient))


if __name__ == "__main__":
    unittest.main()
