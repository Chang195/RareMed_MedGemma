import unittest
from evidence_rules import sentence_spans, route, figure_numbers, attach_figure_sentences, ground_existing


class RuleRoutingTests(unittest.TestCase):
    def test_age_and_family_keep_exact_text(self):
        text = 'A 69-year-old woman presented with slurred speech. Her mother had a stroke at 40.'
        spans = list(sentence_spans({'text': text, 'start_in_passage': 20}))
        for span in spans:
            self.assertEqual(span['text'], text[span['start_in_passage']-20:span['end_in_passage']-20])
        self.assertTrue(route(spans[0]['text'], 'demographics')[0])
        self.assertTrue(route(spans[1]['text'], 'family_history')[0])

    def test_genetic_result_not_clinical_context(self):
        text = 'The patient with a family history of stroke had a pathogenic NOTCH3 variant.'
        self.assertFalse(route(text, 'family_history')[0])
        self.assertTrue(route(text, 'genetic_results')[0])

    def test_treatment_not_diagnostic_evidence(self):
        self.assertFalse(route('She was given 300 mg of aspirin after stroke.', 'diagnostic_results')[0])

    def test_inheritance_risk_not_patient_test_result(self):
        self.assertFalse(route('Any offspring would have a 50% chance of inheriting the pathogenic variant.', 'genetic_results')[0])
        self.assertTrue(route('The centre identified the patient to be heterozygous for a pathogenic NOTCH3 gene variant.', 'genetic_results')[0])

    def test_clinical_diagnosis_not_mri_finding(self):
        self.assertFalse(route('A clinical diagnosis of right hemispheric lacunar infarct was made.', 'imaging_description')[0])

    def test_ct_not_mri_and_method_not_finding(self):
        self.assertFalse(route('CT showed white matter lesions.', 'imaging_description')[0])
        self.assertFalse(route('MRI findings were discussed at a meeting.', 'imaging_description')[0])
        self.assertTrue(route('FLAIR showed no white matter lesions.', 'imaging_description')[0])

    def test_cohort_age_not_individual(self):
        self.assertFalse(route('The patients were aged 45 years.', 'demographics')[0])

    def test_reference_ranges_and_number_boundary(self):
        self.assertEqual(figure_numbers('Figures 2-3 and Figure 4 show lesions.'), {2, 3, 4})
        self.assertEqual(figure_numbers('Figure 20A.'), {20})
        self.assertEqual(figure_numbers('eFigure 2 is supplementary.'), set())

    def test_multiple_figures_never_auto_assigned(self):
        p = {'passage_id': 'p1', 'section': 'CASE', 'source_offset': 100,
             'text': 'MRI showed lesions (Figures 2-3) and signal changes (Figure 4).'}
        found = attach_figure_sentences({'figure_id': 'FIG3'}, [p])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['routing_status'], 'ambiguous_or_non_mri_reference')

    def test_unmatched_old_quote_fails(self):
        with self.assertRaises(ValueError):
            ground_existing([{'passage_id': 'p1', 'quote': 'invented'}],
                            [{'passage_id': 'p1', 'text': 'real', 'source_offset': 0}])

    def test_fig_abbreviation_not_split(self):
        text = 'Fig. A shows an abnormality. The patient had headache.'
        self.assertEqual(len(list(sentence_spans({'text': text}))), 2)


if __name__ == '__main__':
    unittest.main()
