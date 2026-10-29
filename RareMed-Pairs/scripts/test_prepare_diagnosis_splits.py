import unittest
from prepare_diagnosis_splits import diagnosis_family


class Labels(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(diagnosis_family('CADASIL (NOTCH3 pathogenic variant)'), 'CADASIL_author_reported')

    def test_negated(self):
        self.assertEqual(diagnosis_family('Leukoencephalopathy (CADASIL not confirmed)'), 'needs_label_review')

    def test_control_not_automatic(self):
        self.assertEqual(diagnosis_family('Fabry disease'), 'other_reported_diagnosis_needs_control_selection')

    def test_sporadic(self):
        self.assertEqual(diagnosis_family('sporadic cerebral small vessel disease'), 'sporadic_CSVD_author_reported')


if __name__ == '__main__':
    unittest.main()
