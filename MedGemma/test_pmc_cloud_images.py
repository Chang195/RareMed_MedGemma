import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import pmc_cloud_images as cloud


class CloudTests(unittest.TestCase):
    def test_only_official_host_and_bucket(self):
        self.assertEqual(cloud.https_url('s3://pmc-oa-opendata/PMC1.1/a.jpg?md5=x'),
                         cloud.BASE+'PMC1.1/a.jpg?md5=x')
        for url in ['https://example.com/a.jpg','s3://other/PMC1.1/a.jpg']:
            with self.assertRaises(ValueError):
                cloud.https_url(url)

    def test_exact_figure_filename_mapping_without_extension_guessing(self):
        xml=ET.fromstring('<article xmlns:xlink="http://www.w3.org/1999/xlink"><fig id="F1"><graphic xlink:href="one.jpg"/></fig>'
                          '<fig id="F2"><graphic xlink:href="one"/></fig></article>')
        url='s3://pmc-oa-opendata/PMC1.1/one.jpg?md5=x'
        result=cloud.media_mapping(xml,[url],'PMC1.1')
        self.assertEqual(result['F1'],[cloud.https_url(url)])
        self.assertEqual(result['F2'],[])
        with self.assertRaises(ValueError):
            cloud.media_mapping(xml,['s3://pmc-oa-opendata/PMC2.1/one.jpg'],'PMC1.1')

    def test_md5_mismatch_rejects_download(self):
        with tempfile.TemporaryDirectory() as td:
            client=cloud.CloudClient(Path(td))
            def fake(command,**kw):
                Path(command[command.index('--output')+1]).write_bytes(b'actual')
                return types.SimpleNamespace(returncode=0,stderr='')
            with patch.object(cloud.subprocess,'run',side_effect=fake):
                with self.assertRaisesRegex(ValueError,'MD5 mismatch'):
                    client.get(cloud.BASE+'PMC1.1/a.jpg?md5='+'0'*32)
            self.assertEqual(list(Path(td).glob('*.bin')),[])

    def test_successful_cache_is_revalidated_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            client=cloud.CloudClient(Path(td))
            data=b'fixture-original-bytes'
            def fake(command,**kw):
                Path(command[command.index('--output')+1]).write_bytes(data)
                return types.SimpleNamespace(returncode=0,stderr='')
            url=cloud.BASE+'PMC1.1/a.jpg?md5='+hashlib.md5(data).hexdigest()
            with patch.object(cloud.subprocess,'run',side_effect=fake) as network,patch.object(cloud.time,'sleep'):
                first=client.get(url)
                second=client.get(url)
                self.assertEqual(first.read_bytes(),data)
                self.assertEqual(first,second)
                self.assertEqual(network.call_count,1)


if __name__=='__main__':
    unittest.main()
