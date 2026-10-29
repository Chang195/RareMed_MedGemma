"""Synthetic review decisions; never written as real human reviews."""
import copy
import unittest
from review_links import validate_review, VERSION


class ReviewChecks(unittest.TestCase):
    def setUp(self):
        self.binding={'input_sha256':'test-only'}
        self.review={'review_version':VERSION,'article_id':'PMC1','binding':self.binding,
                     'reviewer':'synthetic-test-fixture','reviewed_at':'2026-10-26T00:00:00Z',
                     'figures':[{'figure_id':'F1','decision':'hold',
                                 'checks':{},'arrows':'yes','multi_panel':'uncertain',
                                 'visible_identifiers':'uncertain','notes':'Test fixture only.'}]}

    def test_hold_is_accepted_without_claiming_approval(self):
        self.assertEqual(validate_review(self.review,'PMC1',['F1'],self.binding),[])

    def test_approval_requires_three_checks(self):
        self.review['figures'][0]['decision']='approve_source_link'
        self.assertTrue(any('three' in e for e in validate_review(self.review,'PMC1',['F1'],self.binding)))

    def test_modified_model_input_invalidates_review(self):
        self.assertTrue(any('changed' in e for e in validate_review(self.review,'PMC1',['F1'],{})))

    def test_no_default_decision(self):
        self.review['figures'][0]['decision']=''
        self.assertTrue(validate_review(self.review,'PMC1',['F1'],self.binding))

    def test_all_figures_once(self):
        self.review['figures'].append(copy.deepcopy(self.review['figures'][0]))
        self.assertTrue(any('exactly once' in e for e in validate_review(self.review,'PMC1',['F1'],self.binding)))


if __name__=='__main__':
    unittest.main()
