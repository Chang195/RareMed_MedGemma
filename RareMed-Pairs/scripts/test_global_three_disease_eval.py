import unittest
from global_three_disease_eval import mentions, predicted_cases, relation_metrics, source_passages

class GlobalEvaluation(unittest.TestCase):
    def test_abbreviation_and_long_names(self):
        self.assertEqual(mentions('CADASIL, MS and neuromyelitis optica spectrum disorder'),['CADASIL','MS','NMOSD'])
        self.assertEqual(mentions('multiple sclerosis'),['MS'])
        self.assertEqual(mentions('items were normal'),[])

    def test_no_gold_case_fallback(self):
        ref={'source_case_ids':['gold'],'case_identities':{'Case 1':'a','Case 2':'b'},'case_passage_identity':{'p0':'a','p1':'b'}}
        self.assertEqual(predicted_cases({'case_label':None}, {'passages':[]},ref),[])
        self.assertEqual(predicted_cases({'case_label':'Case 2'}, {'passages':[]},ref),['b'])

    def test_multiple_case_abstention(self):
        ref={'case_identities':{'Case 1':'a'},'case_passage_identity':{}}
        self.assertEqual(predicted_cases({'case_label':'Case 1','link_status':'multiple_cases'}, {'passages':[]},ref),[])

    def test_unique_evidence_only(self):
        quote='The patient had progressive visual loss over two weeks.'
        source={'passages':[{'passage_id':'p0','section':'CASE','text':quote},{'passage_id':'p1','section':'CASE','text':quote}]}
        ref={'case_identities':{},'case_passage_identity':{'p0':'a','p1':'b'}}
        self.assertEqual(predicted_cases({'link_evidence':[{'quote':quote}]},source,ref),[])

    def test_missing_predictions_count_as_misses(self):
        data=[{'source_case_ids':['a','b'],'predicted_case_ids':[],'inference_status':'not_run'}]
        m=relation_metrics(data)
        self.assertEqual(m['FN'],2)
        self.assertEqual(m['recall'],0)
        self.assertEqual(m['validated_figures'],0)

    def test_case_text_not_truncated(self):
        text='x'*130000
        passages,_=source_passages([{'case_id':'a','text':text}])
        self.assertIn(text,passages[0]['text'])

if __name__=='__main__':unittest.main()
