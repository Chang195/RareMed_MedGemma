import unittest
from license_rules import collection_decision
from run_collection_v3 import build_candidates


class CollectionV3Checks(unittest.TestCase):
    def test_only_nc_nd_excluded_from_collection(self):
        for name in ['CC BY 3.0','CC BY 4.0','CC BY-NC 4.0','CC BY-NC-SA 4.0','CC BY-SA 4.0','CC BY-ND 4.0','']:
            with self.subTest(name=name):
                self.assertTrue(collection_decision(name)['collect'])
        self.assertFalse(collection_decision('CC BY-NC-ND 4.0')['collect'])

    def test_long_name_nc_nd_is_excluded(self):
        self.assertFalse(collection_decision('Creative Commons Attribution-NonCommercial-NoDerivatives 3.0')['collect'])

    def test_metadata_nc_nd_is_not_lost(self):
        self.assertFalse(collection_decision('', 'cc by-nc-nd')['collect'])

    def test_reviews_and_missing_metadata_are_retained(self):
        candidates=[{'article_id':'PMC1','title':'CADASIL literature review'},
                    {'article_id':'PMC2','title':'Meta-analysis of MRI'},
                    {'article_id':'PMC3','title':'Case report'}]
        metadata={'PMC1':{'license':'cc by-nc'},'PMC2':{},'PMC3':{'license':'cc by-nc-nd'}}
        rows=build_candidates(candidates,metadata)
        self.assertEqual([r['queue'] for r in rows],['fulltext_required','fulltext_required','excluded_license_by_nc_nd'])
        self.assertTrue(all(not r['review_exclusion'] for r in rows))

    def test_collecting_unknown_does_not_certify_reuse(self):
        d=collection_decision('')
        self.assertEqual(d['status'],'collection_only_rights_pending')


if __name__=='__main__':
    unittest.main()
