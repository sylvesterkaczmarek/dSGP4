import unittest

import dsgp4
import torch


class RtnFrameTestCase(unittest.TestCase):
    def test_identity_orbit_geometry(self):
        state = torch.tensor(
            [
                [7000.0, 0.0, 0.0],
                [0.0, 7.5, 0.0],
            ]
        )
        rotation = dsgp4.rtn_rotation_matrix(state)
        torch.testing.assert_close(rotation, torch.eye(3))

    def test_rotation_is_orthonormal_and_right_handed(self):
        state = torch.tensor(
            [
                [5102.5, 6123.0, 6378.0],
                [-4.743, 0.790, 5.533],
            ]
        )
        rotation = dsgp4.rtn_rotation_matrix(state)

        torch.testing.assert_close(
            rotation @ rotation.T,
            torch.eye(3),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            torch.linalg.det(rotation),
            torch.tensor(1.0),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_covariance_transform_matches_block_rotation(self):
        state = torch.tensor(
            [
                [0.0, 7000.0, 0.0],
                [-7.5, 0.0, 0.0],
            ]
        )
        covariance = torch.diag(
            torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
        )

        transformed, rotation = dsgp4.cartesian_covariance_to_rtn(
            state,
            covariance,
        )
        zero = torch.zeros_like(rotation)
        transform = torch.cat(
            (
                torch.cat((rotation, zero), dim=-1),
                torch.cat((zero, rotation), dim=-1),
            ),
            dim=-2,
        )
        expected = transform @ covariance @ transform.T

        torch.testing.assert_close(transformed, expected)
        torch.testing.assert_close(transformed, transformed.T)

    def test_batched_states_are_supported(self):
        states = torch.tensor(
            [
                [[7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]],
                [[0.0, 7000.0, 0.0], [-7.5, 0.0, 0.0]],
            ]
        )
        covariance = torch.eye(6).expand(2, 6, 6).clone()
        transformed, rotation = dsgp4.cartesian_covariance_to_rtn(
            states,
            covariance,
        )

        self.assertEqual(rotation.shape, (2, 3, 3))
        self.assertEqual(transformed.shape, (2, 6, 6))

    def test_degenerate_orbit_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero position"):
            dsgp4.rtn_rotation_matrix(torch.zeros((2, 3)))

        with self.assertRaisesRegex(ValueError, "collinear"):
            dsgp4.rtn_rotation_matrix(
                torch.tensor(
                    [
                        [7000.0, 0.0, 0.0],
                        [7.5, 0.0, 0.0],
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
