"""API response fixtures only; no real network or credentials."""
import json
import unittest
from run_deepseek import check_completion, build_request
from pathlib import Path
from unittest.mock import patch


class DeepSeekChecks(unittest.TestCase):
    def setUp(self):
        self.source={'article_id':'PMC1','candidate_figures':[],'passages':[]}

    def test_truncated_response_fails_even_with_parseable_json(self):
        raw={'choices':[{'finish_reason':'length','message':{'content':json.dumps({'article_id':'PMC1','figures':[]})}}]}
        self.assertTrue(any('finish' in e for e in check_completion(raw,self.source)[1]))

    def test_actual_entry_point_repairs_redundant_type(self):
        raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(
            {'type':'json_object','article_id':'PMC1','figures':[]})}}]}
        parsed,errors,_=check_completion(raw,self.source)
        self.assertEqual(errors,[])
        self.assertNotIn('type',parsed)

    def test_empty_content_is_not_success(self):
        raw={'choices':[{'finish_reason':'stop','message':{'content':''}}]}
        self.assertTrue(check_completion(raw,self.source)[1])

    def test_malformed_json_is_not_success(self):
        raw={'choices':[{'finish_reason':'stop','message':{'content':'```json\n{}\n```'}}]}
        self.assertTrue(check_completion(raw,self.source)[1])

    def test_request_disables_thinking(self):
        with patch.object(Path,'read_text',return_value='JSON extraction task'):
            payload=build_request(Path('unused'),8192)
        self.assertEqual(payload['thinking']['type'],'disabled')
        self.assertEqual(payload['response_format']['type'],'json_object')
        self.assertNotIn('api_key',payload)


if __name__=='__main__':
    unittest.main()
