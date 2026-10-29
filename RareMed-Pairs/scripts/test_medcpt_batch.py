import unittest
from medcpt_batch import substantive, figure_mentions


class EvidenceCandidateTests(unittest.TestCase):
    def test_heading_is_filtered(self):
        self.assertFalse(substantive({'text': 'Case presentation'}))
        self.assertTrue(substantive({'text': 'The patient was found to carry a pathogenic NOTCH3 variant.'}))

    def test_figure_number_boundaries(self):
        passages = [{'section': 'CASE', 'text': 'The scan is shown in Figure 2A.'},
                    {'section': 'CASE', 'text': 'Other findings are in Figure 20.'}]
        self.assertEqual(figure_mentions({'figure_id': 'FIG2'}, passages), [passages[0]])

    def test_opaque_figure_id_not_guessed(self):
        self.assertEqual(figure_mentions({'figure_id': 'publisher_unknown'}, []), [])


if __name__ == '__main__':
    unittest.main()
