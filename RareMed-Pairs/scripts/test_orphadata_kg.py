"""Guard against clinically misleading source interpretation and cache misuse."""
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import orphadata_kg as k

def phenotype_xml(frequency='',count=1):
    f='<HPOFrequency id="7"><Name>'+frequency+'</Name></HPOFrequency>' if frequency else ''
    return ET.fromstring('<Disorder><OrphaCode>136</OrphaCode><Name>CADASIL</Name><HPODisorderAssociationList count="'+str(count)+'"><HPODisorderAssociation id="1"><HPO><HPOId>HP:0000001</HPOId><HPOTerm>Example</HPOTerm></HPO>'+f+'<DiagnosticCriteria/></HPODisorderAssociation></HPODisorderAssociationList></Disorder>')

class SourceSemanticsTests(unittest.TestCase):
    def test_unknown_frequency_is_not_zero_or_negative(self):
        r=k.parse_phenotypes(phenotype_xml())[0]
        self.assertIsNone(r['frequency_label']);self.assertEqual(r['relation'],'has_phenotype')
        self.assertNotIn('probability',r)

    def test_frequency_range_remains_verbatim(self):
        r=k.parse_phenotypes(phenotype_xml('Occasional (29-5%)'))[0]
        self.assertEqual(r['frequency_label'],'Occasional (29-5%)')
        self.assertNotIn('probability',r)

    def test_explicit_exclusion_not_sampled_as_positive(self):
        self.assertEqual(k.parse_phenotypes(phenotype_xml('Excluded (0%)'))[0]['relation'],'excludes_phenotype')

    def test_truncated_association_list_rejected(self):
        with self.assertRaisesRegex(ValueError,'count mismatch'):k.parse_phenotypes(phenotype_xml(count=2))

    def test_html_and_partial_xml_not_accepted_as_download(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.xml';p.write_text('<html>Error</html>')
            with self.assertRaises(ValueError):k.inspect_xml(p)
            p.write_text('<JDBOR><Disorder>')
            with self.assertRaises(ET.ParseError):k.inspect_xml(p)

    def test_ambiguous_disease_requires_identity_selection(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.xml';p.write_text('<JDBOR><HPODisorderSetStatus><Disorder><OrphaCode>1</OrphaCode><Name>Example one</Name></Disorder></HPODisorderSetStatus><HPODisorderSetStatus><Disorder><OrphaCode>2</OrphaCode><Name>Example two</Name></Disorder></HPODisorderSetStatus></JDBOR>')
            with self.assertRaisesRegex(ValueError,'Ambiguous'):k.resolve(p,'Example')
            self.assertEqual(k.resolve(p,'Example',code='ORPHA:2')['orpha_code'],'2')

    def test_corrupted_cache_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.xml';p.write_text('modified')
            k.save(p.with_suffix('.source.json'),dict(sha256='not_the_hash',source_url='https://example.org/source.xml'))
            with self.assertRaisesRegex(ValueError,'hash changed'):
                k.fetch_source(dict(file='source.xml',url='https://example.org/source.xml'),Path(t),offline=True)

    def test_gene_association_type_not_upgraded_to_causality(self):
        d=ET.fromstring('<Disorder><DisorderGeneAssociationList><DisorderGeneAssociation><Gene id="10"><Symbol>X</Symbol><Name>Example gene</Name></Gene><DisorderGeneAssociationType><Name>Candidate gene tested in</Name></DisorderGeneAssociationType><DisorderGeneAssociationStatus><Name>Not yet assessed</Name></DisorderGeneAssociationStatus></DisorderGeneAssociation></DisorderGeneAssociationList></Disorder>')
        r=k.parse_genes(d)[0]
        self.assertEqual(r['association_type'],'Candidate gene tested in');self.assertEqual(r['status'],'Not yet assessed')

if __name__=='__main__':unittest.main()
