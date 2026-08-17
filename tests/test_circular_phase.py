import unittest

import numpy as np

from dsgp4.newton_method import _circular_phase_angles


def _angle_error(actual, expected):
    return ((actual - expected + np.pi) % (2.0 * np.pi)) - np.pi


def _rot_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class CircularPhaseTestCase(unittest.TestCase):
    def test_inclined_circular_orbit_preserves_argument_of_latitude(self):
        mu = 3.986004418e14
        radius = 7.0e6
        inclination = np.deg2rad(50.0)
        expected_raan = np.deg2rad(40.0)
        expected_u = np.deg2rad(70.0)

        transform = _rot_z(expected_raan) @ _rot_x(inclination)
        r_perifocal = radius * np.array([np.cos(expected_u), np.sin(expected_u), 0.0])
        speed = np.sqrt(mu / radius)
        v_perifocal = speed * np.array([-np.sin(expected_u), np.cos(expected_u), 0.0])
        r_vec = transform @ r_perifocal
        v_vec = transform @ v_perifocal

        raan, argp, mean_anomaly = _circular_phase_angles(r_vec, v_vec)

        self.assertAlmostEqual(_angle_error(raan, expected_raan), 0.0, places=12)
        self.assertEqual(argp, 0.0)
        self.assertAlmostEqual(_angle_error(mean_anomaly, expected_u), 0.0, places=12)

    def test_equatorial_circular_orbit_preserves_true_longitude(self):
        mu = 3.986004418e14
        radius = 7.0e6
        longitude = np.deg2rad(123.0)
        speed = np.sqrt(mu / radius)
        r_vec = radius * np.array([np.cos(longitude), np.sin(longitude), 0.0])
        v_vec = speed * np.array([-np.sin(longitude), np.cos(longitude), 0.0])

        raan, argp, mean_anomaly = _circular_phase_angles(r_vec, v_vec)

        self.assertEqual(raan, 0.0)
        self.assertEqual(argp, 0.0)
        self.assertAlmostEqual(_angle_error(mean_anomaly, longitude), 0.0, places=12)

    def test_degenerate_state_is_rejected(self):
        with self.assertRaises(ValueError):
            _circular_phase_angles(np.zeros(3), np.ones(3))


if __name__ == "__main__":
    unittest.main()
