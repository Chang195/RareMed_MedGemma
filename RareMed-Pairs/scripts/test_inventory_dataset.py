import unittest
from inventory_dataset import screen


class InventoryTests(unittest.TestCase):
    def row(self):
        return {'caption': 'Axial brain MRI shows a small infarct.', 'proposed_linkage': {
            'link_status': 'explicit', 'case_label': 'Index case', 'diagnosis_name': 'CADASIL', 'diagnosis_status': 'suspected'}}

    def test_caption_task_does_not_require_confirmed_disease(self):
        self.assertEqual(screen(self.row(), [{'size': [750, 750]}])[0], 'candidate_single_figure_caption_task')

    def test_cadasil_mouse_not_human_candidate(self):
        row = self.row(); row['proposed_linkage']['case_label'] = 'CADASIL mouse model'
        self.assertEqual(screen(row, [{'size': [750, 750]}])[0], 'held_animal_signal')

    def test_composite_is_kept_separately(self):
        row = self.row(); row['caption'] = 'MRI (A) and SWI (B) show lesions.'
        self.assertEqual(screen(row, [{'size': [750, 750]}])[0], 'candidate_composite_or_serial_caption_task')

    def test_no_case_label_not_assigned(self):
        row = self.row(); row['proposed_linkage']['case_label'] = None
        self.assertEqual(screen(row, [{'size': [750, 750]}])[0], 'held_case_linkage_uncertain')

    def test_bad_file_blocks_candidate(self):
        self.assertEqual(screen(self.row(), [{'quality_error': 'bad file'}])[0], 'held_file_or_caption_problem')


if __name__ == '__main__':
    unittest.main()
