import unittest

import dsgp4
import torch


LEO_TLES = [
    (
        "1 25544U 98067A   24087.49097222  .00016717  00000-0  10270-3 0  9993",
        "2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000000",
    ),
    (
        "1 40967U 15058A   24060.50000000  .00000033  00000-0  00000+0 0  9992",
        "2 40967  97.4440 125.0000 0010000  25.0000 335.0000 14.90000000000000",
    ),
]


class Mldsgp4InputPurityTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = dsgp4.mldsgp4(hidden_size=12)
        self.model.eval()

    def _assert_tle_unchanged(self, tle, before_dict, before_instance_keys):
        self.assertEqual(set(tle.__dict__.keys()), before_instance_keys)
        self.assertEqual(set(tle._data.keys()), set(before_dict.keys()))
        for key, expected in before_dict.items():
            actual = tle._data[key]
            if torch.is_tensor(expected):
                self.assertTrue(torch.equal(actual, expected))
            else:
                self.assertEqual(actual, expected)

    def test_single_tle_forward_is_repeatable_and_does_not_mutate_input(self):
        tle = dsgp4.TLE(list(LEO_TLES[0]))
        before_dict = dsgp4.tle.copy_data(tle._data)
        before_instance_keys = set(tle.__dict__.keys())
        times = torch.tensor([0.0, 10.0, 30.0])

        with torch.no_grad():
            first = self.model(tle, times)
            second = self.model(tle, times)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self._assert_tle_unchanged(tle, before_dict, before_instance_keys)

    def test_batched_forward_does_not_initialize_or_rewrite_input_tles(self):
        tles = [dsgp4.TLE(list(lines)) for lines in LEO_TLES]
        snapshots = [dsgp4.tle.copy_data(tle._data) for tle in tles]
        instance_keys = [set(tle.__dict__.keys()) for tle in tles]
        times = torch.tensor([5.0, 25.0])

        with torch.no_grad():
            first = self.model(tles, times)
            second = self.model(tles, times)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        for tle, before_dict, before_keys in zip(tles, snapshots, instance_keys):
            self._assert_tle_unchanged(tle, before_dict, before_keys)


if __name__ == "__main__":
    unittest.main()
