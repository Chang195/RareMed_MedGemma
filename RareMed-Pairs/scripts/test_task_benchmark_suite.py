import unittest
from task_benchmark_suite import text_scores, actual_disease


class Tasks(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(text_scores('Multiple Sclerosis','multiple sclerosis')['word_f1'],1)

    def test_missing_word(self):
        r=text_scores('frontal lobe','left frontal lobe')
        self.assertAlmostEqual(r['word_recall'],2/3)
        self.assertAlmostEqual(r['rouge_l_f1'],.8)

    def test_order_matters_for_rouge(self):
        r=text_scores('lobe frontal left','left frontal lobe')
        self.assertEqual(r['word_f1'],1)
        self.assertLess(r['rouge_l_f1'],1)

    def test_differential_not_disease(self):
        self.assertEqual(actual_disease('CADASIL was excluded; glioblastoma confirmed'),'other_or_not_normalized')

    def test_empty(self):
        self.assertEqual(text_scores('','MRI')['word_f1'],0)


if __name__=='__main__':unittest.main()
