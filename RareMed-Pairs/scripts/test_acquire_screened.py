import unittest
import json
from pathlib import Path
from unittest.mock import patch
from acquire_screened import select_candidates
from run_deepseek_batch import article_ids


class IncrementalChecks(unittest.TestCase):
    def test_unicode_line_separator_inside_json_record_is_not_record_break(self):
        text=json.dumps({'article_id':'PMC1','caption':'a\u2028b\u2029c'},ensure_ascii=False)+'\n'
        with patch.object(Path,'read_text',return_value=text):
            self.assertEqual(article_ids(Path('unused'),10),['PMC1'])

    def test_excludes_every_previous_attempt_including_failures(self):
        rows=[{'article_id':'PMC1','queue':'first'},{'article_id':'PMC2','queue':'first'},
              {'article_id':'PMC2','queue':'first'},{'article_id':'PMC3','queue':'second'}]
        self.assertEqual([r['article_id'] for r in select_candidates(rows,['first','second'],{'PMC1'},20)],['PMC2','PMC3'])

    def test_preserves_priority_order(self):
        rows=[{'article_id':'PMC1','queue':'second'},{'article_id':'PMC2','queue':'first'}]
        self.assertEqual(select_candidates(rows,['first','second'],set(),1)[0]['article_id'],'PMC2')


if __name__=='__main__':
    unittest.main()
