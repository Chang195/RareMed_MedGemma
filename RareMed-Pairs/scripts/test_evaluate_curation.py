import unittest
from evaluate_curation import metrics, predict


class Metrics(unittest.TestCase):
    def test_confusion(self):
        r=metrics([True,True,False,False],[True,False,True,False])
        self.assertEqual([r[k] for k in ['TP','FP','FN','TN']], [1,1,1,1])
        self.assertEqual(r['f1'],.5)

    def test_no_negative(self):
        self.assertIsNone(metrics([True],[True])['specificity'])

    def test_no_predicted_positive(self):
        self.assertIsNone(metrics([True,False],[False,False])['precision'])

    def test_existing_rules(self):
        self.assertTrue(predict('(A) MRI (B) CT')['multipanel_caption_flag'])
        self.assertFalse(predict('Scatter plot of MRI volumes')['medical_text_proxy'])


if __name__=='__main__': unittest.main()
