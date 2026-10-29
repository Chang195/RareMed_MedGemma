"""Offline failure-path checks for acquisition invariants."""
from io import StringIO
import unittest
from collect import Figures, license_allowed, parse_bioc, caption_signals


class AcquisitionChecks(unittest.TestCase):
    def test_mri_and_ct_are_separately_flagged(self):
        self.assertFalse(caption_signals('Axial CT head with yellow arrows')['mri'])
        signals = caption_signals('(A) MRI FLAIR and (B) CT head, arrow')
        self.assertTrue(signals['mri'])
        self.assertTrue(signals['other_modality'])
        self.assertTrue(signals['panel'])
        self.assertTrue(signals['annotation'])

    def test_license_does_not_accept_restricted_or_ambiguous_license(self):
        self.assertTrue(license_allowed('Creative Commons Attribution CC-BY 4.0.'))
        self.assertTrue(license_allowed('Creative Commons Attribution 4.0 International License'))
        self.assertTrue(license_allowed('http://creativecommons.org/licenses/by/4.0/'))
        self.assertFalse(license_allowed('Creative Commons Attribution-NonCommercial-NoDerivatives 4.0'))
        self.assertFalse(license_allowed('http://creativecommons.org/licenses/by-nc/4.0/'))
        for text in ['CC BY', 'CC-BY-NC 4.0', 'CC-BY-SA 4.0', 'all rights reserved']:
            self.assertFalse(license_allowed(text))

    def test_multiple_assets_remain_distinct_for_review(self):
        parser = Figures()
        parser.feed('<figure id="F1"><img src="a.jpg"><img src="b.jpg"></figure>')
        self.assertEqual(parser.figures['F1'], ['a.jpg', 'b.jpg'])

    def test_wrong_article_rejected(self):
        source = StringIO('<collection><document><id>PMC1</id></document></collection>')
        with self.assertRaisesRegex(ValueError, 'identity'):
            parse_bioc(source, 'PMC2')


if __name__ == '__main__':
    unittest.main()
