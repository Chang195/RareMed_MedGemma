import unittest
from run_deepseek_batch import usage_total


class BatchChecks(unittest.TestCase):
    def test_usage_includes_reused_calls_once_per_receipt(self):
        receipts=[{'usage':{'prompt_tokens':5788,'completion_tokens':980,'total_tokens':6768}},
                  {'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}},
                  {'status':'network_error'}]
        self.assertEqual(usage_total(receipts),{'prompt_tokens':5888,'completion_tokens':1000,'total_tokens':6888})

    def test_bad_usage_does_not_inflate_totals(self):
        self.assertEqual(usage_total([{'usage':{'total_tokens':-1,'prompt_tokens':'unknown'}}])['total_tokens'],0)


if __name__=='__main__':
    unittest.main()
