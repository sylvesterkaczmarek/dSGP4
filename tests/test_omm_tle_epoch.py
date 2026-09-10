import unittest

from dsgp4.omm import OMM, dumps
from dsgp4.tle import TLE


class OMMTLEEpochTestCase(unittest.TestCase):
    def make_omm(self, epoch):
        return OMM({
            'EPOCH': epoch,
            'MEAN_MOTION': '14.59199732',
            'ECCENTRICITY': '0.0001341',
            'INCLINATION': '98.1819',
            'RA_OF_ASC_NODE': '68.1874',
            'ARG_OF_PERICENTER': '82.4703',
            'MEAN_ANOMALY': '277.6657',
            'NORAD_CAT_ID': '39634',
        })

    def test_to_tle_rejects_unrepresentable_epoch_year(self):
        for year in (1956, 2057, 2100):
            with self.subTest(year=year):
                omm = self.make_omm('{}-01-01T12:00:00.000000'.format(year))
                original_fields = dict(omm._fields)
                with self.assertRaisesRegex(ValueError, '1957.*2056'):
                    omm.to_tle()
                self.assertEqual(omm.epoch_year, year)
                self.assertEqual(omm._fields, original_fields)
                self.assertEqual(omm.copy().epoch_year, year)
                self.assertEqual(OMM(dumps(omm)).epoch_year, year)

    def test_to_tle_preserves_epoch_at_year_boundaries(self):
        for epoch in ('1957-01-01T12:00:00.000000', '1999-12-31T12:00:00.000000',
                      '2000-01-01T12:00:00.000000', '2056-12-31T12:00:00.000000'):
            with self.subTest(epoch=epoch):
                omm = self.make_omm(epoch)
                tle = omm.to_tle()
                reloaded = TLE([tle.line1, tle.line2])
                self.assertEqual(tle.date_string, omm.date_string)
                self.assertEqual(reloaded.date_string, omm.date_string)
                self.assertEqual(float(reloaded._jdsatepoch), float(omm._jdsatepoch))
                self.assertEqual(float(reloaded._jdsatepochF), float(omm._jdsatepochF))
