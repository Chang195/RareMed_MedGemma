import unittest
from four_disease_eval import disease_matches, load, CONFIG, relation_metrics, native_medpix


class FourDisease(unittest.TestCase):
    def test_three_strata(self):
        self.assertEqual(set(load(CONFIG)['diseases']),{'CADASIL','multiple_sclerosis','NMOSD'})

    def test_sampling_is_mention_not_diagnosis(self):
        self.assertIn('CADASIL',disease_matches('CADASIL was excluded.',load(CONFIG)))

    def test_ambiguous_ms_not_matched(self):
        self.assertNotIn('multiple_sclerosis',disease_matches('20 ms echo time',load(CONFIG)))

    def test_wrong_relation_counts_twice(self):
        r=relation_metrics(1,1,1)
        self.assertEqual(r['precision'],.5)
        self.assertEqual(r['recall'],.5)

    def test_empty_no_fake_zero(self):
        self.assertIsNone(relation_metrics(0,0,0)['f1'])


if __name__=='__main__':unittest.main()
