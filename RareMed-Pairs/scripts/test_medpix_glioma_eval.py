import unittest
from medpix_glioma_eval import OUT, canonical, clean, load, rows, check, validate, UNCERTAIN

class MedPixGlioma(unittest.TestCase):
    def test_subtypes_included(self):
        for value in ['Pilocytic astrocytoma','Subependymal Giant Cell Astrocytoma','Desmoplastic infantile Astrocytoma']:
            self.assertEqual(canonical(value),'astrocytoma')
        self.assertEqual(canonical('Glioblastoma multiforme'),'glioblastoma')

    def test_uncertainty_and_missingness(self):
        self.assertTrue(UNCERTAIN.search('Pontine glioma probably astrocytoma'))
        self.assertIsNone(clean('N/A'))
        self.assertIsNone(clean(None))
        self.assertEqual(clean('  original text  '),'  original text  ')

    def test_frozen_and_full_counts(self):
        check();m=load(OUT/'manifest.json')
        self.assertEqual(m['selected_cases'],14)
        self.assertEqual(m['selected_images'],37)
        self.assertEqual(sum(m['cases_by_disease'].values()),14)

    def test_no_source_identifier_leak(self):
        cases=rows(OUT/'case_references.jsonl');figs=rows(OUT/'figure_references.jsonl')
        for job in load(OUT/'jobs.json'):
            prompt=(OUT/'inputs'/job/'prompt.txt').read_text(encoding='utf-8')
            for c in cases:self.assertNotIn(c['source_case_id'],prompt)
            for f in figs:self.assertNotIn(f['source_image'],prompt)
            source=load(OUT/'inputs'/job/'input.json')
            self.assertEqual(len(source['candidate_cases']),3)
            self.assertEqual({f['figure_id'] for f in figs if f['job']==job},set(source['candidate_figures']))

    def test_validator_rejects_invented_text(self):
        source={'candidate_cases':['Case 1'],'candidate_figures':['F0001'],
            'passages':[{'passage_id':'p0','section':'CASE','text':'Case 1\nHistory: Original history.'}]}
        output={'cases':[{'case_label':'Case 1','History':'invented history','Exam':None,'Findings':None}],
            'figures':[{'figure_id':'F0001','caption_quote':None,'case_label':None,'diagnosis_quote':None,'disease_class':'unclear','evidence':[]}]}
        self.assertTrue(any('Non-verbatim' in e for e in validate(output,source)))

    def test_not_run_is_not_success(self):
        m=load(OUT/'manifest.json')
        # A non-run manifest contains no model-derived reference fields.
        self.assertNotIn('predicted_diagnosis',m)
        self.assertEqual(len(rows(OUT/'figure_references.jsonl')),37)

if __name__=='__main__':unittest.main()
