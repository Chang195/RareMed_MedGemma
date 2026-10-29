import unittest
from plot_article_flow import build_flow


class FlowTests(unittest.TestCase):
    def test_article_units_and_conservation(self):
        initial = [{'article_id': 'A', 'queue': 'fulltext_required'},
                   {'article_id': 'B', 'queue': 'excluded_license_by_nc_nd'},
                   {'article_id': 'C', 'queue': 'fulltext_required'}]
        full = [{'article_id': 'A', 'queue': 'fulltext_passed'}, initial[1],
                {'article_id': 'C', 'queue': 'held_fulltext_error'}]
        edges, paths = build_flow(initial, full, [{'article_id': 'A', 'status': 'candidate', 'figures': 12}])
        self.assertEqual(edges['fulltext_passed', 'acq_candidate'], 1)
        self.assertEqual(len(paths), 3)

    def test_missing_result_rejected(self):
        with self.assertRaises(ValueError):
            build_flow([{'article_id': 'A', 'queue': 'fulltext_required'}],
                       [{'article_id': 'A', 'queue': 'fulltext_passed'}], [])

    def test_duplicate_id_rejected(self):
        row = {'article_id': 'A', 'queue': 'fulltext_required'}
        with self.assertRaises(ValueError):
            build_flow([row, row], [], [])

    def test_unfinished_fulltext_rejected(self):
        row = {'article_id': 'A', 'queue': 'fulltext_required'}
        with self.assertRaises(ValueError):
            build_flow([row], [row], [])


if __name__ == '__main__':
    unittest.main()
