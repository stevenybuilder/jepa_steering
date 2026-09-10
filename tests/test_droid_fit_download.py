import base64
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline_study.droid_fit_download import download_object


class DownloadTests(unittest.TestCase):
    def test_generation_bound_complete_hash_and_no_overwrite(self):
        data=b'native source bytes'
        obj={'name':'native/trajectory.h5','size':str(len(data)),'generation':'123',
            'md5Hash':base64.b64encode(hashlib.md5(data).digest()).decode()}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('urllib.request.urlopen',return_value=io.BytesIO(data)) as request:
                result=download_object(obj,root)
                self.assertIn('generation=123',request.call_args.args[0])
            self.assertEqual(result['sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual((root/obj['name']).read_bytes(),data)
            with self.assertRaises(ValueError):download_object(obj,root)

    def test_wrong_bytes_never_publish_completed_object(self):
        obj={'name':'native/trajectory.h5','size':'3','generation':'123',
            'md5Hash':base64.b64encode(hashlib.md5(b'yes').digest()).decode()}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):download_object(obj,root)
            self.assertFalse((root/obj['name']).exists())
            self.assertEqual((root/'native/trajectory.h5.partial-attempt-0').read_bytes(),b'bad')


if __name__=='__main__':unittest.main()
