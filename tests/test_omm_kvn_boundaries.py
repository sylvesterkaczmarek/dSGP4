import unittest

from dsgp4.omm import OMM, dumps, loads


class OMMKVNBoundariesTestCase(unittest.TestCase):
    def test_kvn_round_trip_preserves_record_boundaries(self):
        fields = {
            'EPOCH': '2022-02-28T01:57:54.918432',
            'MEAN_MOTION': '14.59199732',
            'ECCENTRICITY': '0.0001341',
            'INCLINATION': '98.1819',
            'RA_OF_ASC_NODE': '68.1874',
            'ARG_OF_PERICENTER': '82.4703',
            'MEAN_ANOMALY': '277.6657',
        }
        cases = [('missing', None, '3.0'), ('none', None, '3.0'),
                 ('empty', '', '3.0'), ('blank', ' ', '3.0'),
                 ('last', '2.0', '2.0'), ('first', '3.0', '3.0')]
        for position, version, expected_version in cases:
            for record_type in (dict, OMM):
                for count in (1, 2):
                    with self.subTest(position=position, record_type=record_type, count=count):
                        records = [dict(fields, NORAD_CAT_ID=str(39634+i)) for i in range(count)]
                        if position != 'missing':
                            for record in records:
                                record['CCSDS_OMM_VERS'] = version
                        if position == 'first':
                            records = [dict(CCSDS_OMM_VERS=record.pop('CCSDS_OMM_VERS'),
                                            **record) for record in records]
                        originals = [dict(record) for record in records]
                        objects = [record_type(record) for record in records]
                        text = dumps(objects[0] if count == 1 else objects, file_format='kvn')
                        restored = loads(text)
                        self.assertEqual(text.count('CCSDS_OMM_VERS = '), count)
                        self.assertTrue(text.startswith('CCSDS_OMM_VERS = '+expected_version+'\n'))
                        self.assertEqual(restored, [dict(record, CCSDS_OMM_VERS=expected_version)
                                                    for record in originals])
                        self.assertEqual(records, originals)
                        for obj, original in zip(objects, originals):
                            self.assertEqual(obj._fields if isinstance(obj, OMM) else obj, original)
                        for record in restored:
                            self.assertEqual(OMM(record).satellite_catalog_number,
                                             int(record['NORAD_CAT_ID']))

    def test_empty_kvn_output(self):
        self.assertEqual(dumps([], file_format='kvn'), '')
