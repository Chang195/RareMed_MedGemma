import unittest
from plot_dataset_flow import flow,record_paths


class DatasetFlowTests(unittest.TestCase):
    def test_record_units_conserved_and_missing_asset_kept_separate(self):
        figures=[{'sample_id':'a','article_id':'P1','figure_id':'F1','assets':[{}]},
                 {'sample_id':'b','article_id':'P1','figure_id':'F2','assets':[]}]
        inventory=[{'sample_id':'a','inventory_queue':'candidate_single_figure_caption_task'}]
        dedup=[{'sample_id':'a','proposed_split':'train'}]
        paths,rows=record_paths(figures,inventory,dedup)
        edges,inc,out=flow(paths)
        self.assertEqual(out['figures_raw'],2)
        self.assertEqual(inc['records_with_images'],1)
        self.assertEqual(inc['no_asset'],1)
        self.assertEqual(inc['draft_train'],1)

    def test_missing_inventory_record_rejected(self):
        with self.assertRaises(ValueError):
            record_paths([{'sample_id':'a','assets':[{}]}],[],[])

    def test_duplicate_route_node_rejected(self):
        with self.assertRaises(ValueError):flow([['a','b','a']])


if __name__=='__main__':unittest.main()
