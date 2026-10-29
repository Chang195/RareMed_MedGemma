"""Synthetic fixtures only. These are not model outputs or clinical labels."""
import copy
import unittest
from link_cases import validate


class LinkageChecks(unittest.TestCase):
    def setUp(self):
        self.request = {'article_id':'PMC1', 'candidate_figures':[{'figure_id':'F1'}],
            'passages':[{'passage_id':'p1', 'source_offset':100,
                         'text':'Patient 1 has suspected CADASIL. MRI of patient 1 is shown in Figure 1.'}]}
        self.response = {'article_id':'PMC1', 'figures':[{
            'figure_id':'F1', 'case_label':'Patient 1', 'link_status':'explicit',
            'diagnosis_name':'CADASIL', 'diagnosis_status':'suspected', 'diagnosis_basis':None,
            'link_evidence':[{'passage_id':'p1','quote':'MRI of patient 1 is shown in Figure 1.'}],
            'diagnosis_evidence':[{'passage_id':'p1','quote':'Patient 1 has suspected CADASIL.'}],
            'basis_evidence':[], 'uncertainties':[]}]}

    def test_exact_quotes_are_anchored(self):
        errors, anchors = validate(self.response, self.request)
        self.assertEqual(errors, [])
        self.assertEqual(len(anchors), 2)
        self.assertEqual(anchors[0]['quote_start_in_passage'], 33)

    def test_invented_quote_rejected(self):
        self.response['figures'][0]['diagnosis_evidence'][0]['quote'] = 'CADASIL genetically confirmed.'
        self.assertTrue(any('not found' in e for e in validate(self.response,self.request)[0]))

    def test_missing_or_invented_figure_rejected(self):
        self.response['figures'][0]['figure_id'] = 'F9'
        self.assertTrue(any('coverage' in e for e in validate(self.response,self.request)[0]))

    def test_duplicate_rejected(self):
        self.response['figures'].append(copy.deepcopy(self.response['figures'][0]))
        self.assertTrue(any('Duplicate' in e for e in validate(self.response,self.request)[0]))

    def test_unresolved_cannot_assign_diagnosis(self):
        self.response['figures'][0]['link_status'] = 'unresolved'
        self.assertTrue(any('cannot receive' in e for e in validate(self.response,self.request)[0]))

    def test_quote_check_does_not_claim_semantic_truth(self):
        # Real quote + incorrect interpretation cannot be detected by substring matching.
        self.response['figures'][0]['diagnosis_status'] = 'author_reported_confirmed'
        self.assertEqual(validate(self.response,self.request)[0], [])

    def test_bad_state_type_reports_error(self):
        self.response['figures'][0]['link_status'] = []
        self.assertTrue(validate(self.response,self.request)[0])


if __name__ == '__main__':
    unittest.main()
