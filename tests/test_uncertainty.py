import unittest

import dsgp4
import torch


ISS_TLE = [
    "1 25544U 98067A   24060.50000000  .00016717  00000-0  30134-3 0  9990",
    "2 25544  51.6403 124.7938 0005102 220.2782 248.4427 15.50010353440289",
]


class UncertaintyTestCase(unittest.TestCase):
    def test_state_jacobian_matches_direct_autograd(self):
        tle = dsgp4.TLE(ISS_TLE)
        times = torch.tensor([0.0, 20.0])
        state, jacobian = dsgp4.state_jacobian(tle, times)

        self.assertEqual(state.shape, (2, 2, 3))
        self.assertEqual(jacobian.shape, (2, 2, 3, 9))

        reference_tle = tle.copy()
        elements = dsgp4.initialize_tle(reference_tle, with_grad=True)
        reference_state = dsgp4.propagate(
            reference_tle,
            times,
            initialized=True,
        )
        reference_gradient = torch.autograd.grad(
            reference_state[1, 0, 0],
            elements,
        )[0]

        torch.testing.assert_close(
            jacobian[1, 0, 0],
            reference_gradient,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_covariance_matches_first_order_mapping(self):
        tle = dsgp4.TLE(ISS_TLE)
        covariance = torch.diag(
            torch.tensor(
                [
                    1e-12,
                    1e-16,
                    1e-20,
                    1e-10,
                    1e-10,
                    1e-10,
                    1e-10,
                    1e-12,
                    1e-10,
                ]
            )
        )

        state, propagated, jacobian = dsgp4.propagate_covariance(
            tle,
            torch.tensor(30.0),
            covariance,
        )
        jacobian_2d = jacobian.reshape(6, 9)
        expected = jacobian_2d @ covariance @ jacobian_2d.T

        self.assertEqual(state.shape, (2, 3))
        self.assertEqual(propagated.shape, (6, 6))
        torch.testing.assert_close(propagated, expected)
        torch.testing.assert_close(propagated, propagated.T)

    def test_covariance_supports_multiple_epochs(self):
        tle = dsgp4.TLE(ISS_TLE)
        covariance = torch.eye(9) * 1e-12
        _, propagated, _ = dsgp4.propagate_covariance(
            tle,
            torch.tensor([0.0, 10.0, 60.0]),
            covariance,
        )
        self.assertEqual(propagated.shape, (3, 6, 6))

    def test_covariance_shape_is_checked(self):
        tle = dsgp4.TLE(ISS_TLE)
        with self.assertRaisesRegex(ValueError, r"shape \(9, 9\)"):
            dsgp4.propagate_covariance(
                tle,
                torch.tensor(0.0),
                torch.eye(6),
            )


if __name__ == "__main__":
    unittest.main()
