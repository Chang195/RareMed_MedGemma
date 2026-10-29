import unittest
from evaluate_requested_tasks import diagnosis_class, mentions, edge_matrix, bin_record

class RequestedTasks(unittest.TestCase):
    def test_all_explicit_astrocytoma_subtypes_included(self):
        for text in ['Pilocytic astrocytoma','Subependymal Giant Cell Astrocytoma','Anaplastic astrocytoma - WHO Grade 3','Astrocytoma, IDH-mutant']:
            self.assertEqual(diagnosis_class(text),'Astrocytoma')

    def test_gbm_priority_and_uncertainty(self):
        self.assertEqual(diagnosis_class('Glioblastoma (grade IV astrocytoma)'),'GBM')
        self.assertEqual(diagnosis_class('Pontine glioma probably astrocytoma'),'Unresolved')
        self.assertEqual(diagnosis_class('CADASIL'),'CADASIL')
        self.assertEqual(diagnosis_class(None),'Unresolved')

    def test_no_old_disease_strata(self):
        self.assertEqual(mentions('MS NMOSD multiple sclerosis'),[])
        self.assertEqual(mentions('Glioblastoma versus astrocytoma and CADASIL'),['Astrocytoma','CADASIL','GBM'])

    def test_binary_matrix_orientation(self):
        r=bin_record('x','x','ALL',[False,False,True,True],[False,True,False,True],'edge','')
        self.assertEqual(r['matrix'],[[1,1],[1,1]])
        self.assertEqual(r['metrics']['TP'],1)

    def test_patient_relation_confusion(self):
        data=[{'source_case_ids':['a'],'predicted_case_ids':['b'],'case_identities':{'Case 1':'a','Case 2':'b','Case 3':'c'}}]
        r=edge_matrix(data,'case_association','MedPix2','ALL')
        self.assertEqual(r['matrix'],[[1,1],[1,0]])

    def test_description_wrong_pair_produces_fp_and_fn(self):
        data=[{'source_document':'doc','source_case_ids':['a'],'source_caption':'caption A','extracted_caption':'caption B'},
              {'source_document':'doc','source_case_ids':['b'],'source_caption':'caption B','extracted_caption':'caption B'}]
        r=edge_matrix(data,'original_description','MultiCaRe','ALL',caption=True)
        self.assertEqual(r['matrix'],[[1,1],[1,1]])

    def test_missing_description_is_missed_positive(self):
        data=[{'source_document':'doc','source_case_ids':['a'],'source_caption':'source caption','extracted_caption':None}]
        r=edge_matrix(data,'original_description','MultiCaRe','ALL',caption=True)
        self.assertEqual(r['metrics']['FN'],1)
        self.assertEqual(r['metrics']['TP'],0)

if __name__=='__main__':unittest.main()
