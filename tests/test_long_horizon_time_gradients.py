import unittest

import dsgp4
import torch


LEO_TLES = [
    (
        "1 25544U 98067A   24087.49097222  .00016717  00000+0  10270-3 0  9993",
        "2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000000",
    ),
    (
        "1 34427U 93036RU  22068.94647328  .00008100  00000-0  11455-2 0  9999",
        "2 34427  74.0145 306.8269 0033346  13.0723 347.1308 14.76870515693886",
    ),
]


def _state(lines, minutes):
    tle = dsgp4.TLE(list(lines))
    return dsgp4.propagate(tle, torch.tensor(float(minutes)), initialized=False)


class LongHorizonTimeGradientTestCase(unittest.TestCase):
    def test_autograd_position_derivative_tracks_velocity_over_one_day(self):
        times = (-1440.0, -720.0, -60.0, 0.0, 60.0, 720.0, 1440.0)

        for lines in LEO_TLES:
            for minutes in times:
                tle = dsgp4.TLE(list(lines))
                time = torch.tensor(minutes, requires_grad=True)
                state = dsgp4.propagate(tle, time, initialized=False)

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
                    rtol=1e-3,
                    atol=2e-1,
                )

    def test_autograd_time_gradient_matches_central_difference(self):
        times = (-1440.0, -360.0, 0.0, 360.0, 1440.0)
        step_minutes = 1e-3

        for lines in LEO_TLES:
            for minutes in times:
                tle = dsgp4.TLE(list(lines))
                time = torch.tensor(minutes, requires_grad=True)
                state = dsgp4.propagate(tle, time, initialized=False)

                autograd_derivative = torch.stack(
                    [
                        torch.autograd.grad(
                            state[0, component],
                            time,
                            retain_graph=True,
                        )[0]
                        for component in range(3)
                    ]
                )

                plus = _state(lines, minutes + step_minutes)[0]
                minus = _state(lines, minutes - step_minutes)[0]
                finite_difference = (plus - minus) / (2.0 * step_minutes)

                torch.testing.assert_close(
                    autograd_derivative,
                    finite_difference,
                    rtol=2e-4,
                    atol=5e-3,
                )


if __name__ == "__main__":
    unittest.main()
