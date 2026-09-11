import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/vast'
sys.path.insert(0,str(SCRIPTS))
import manage_confirmation as manager


class ConfirmationManagementTests(unittest.TestCase):
    def test_identity_refuses_any_other_lease(self):
        lease={'id':123,'label':'owned','geolocation':'Florida, US','num_gpus':8}
        row={**lease,'gpu_name':'RTX 4090'}
        manager.identity(lease,row)
        for key,value in [('id',124),('label','other'),('geolocation','Shanghai, CN'),
                          ('num_gpus',4),('gpu_name','RTX 5090')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                manager.identity(lease,{**row,key:value})

    def test_full_archive_readback_rejects_corruption_and_extra_members(self):
        for kind in ('valid','corrupt','extra'):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'archive.tar.gz'
                payload=b'complete research record'
                manifest={'members':{'result.json':{'bytes':len(payload),
                                                     'sha256':hashlib.sha256(payload).hexdigest()}}}
                with tarfile.open(path,'w:gz') as out:
                    for name,data in [('PRESERVATION_MANIFEST.json',json.dumps(manifest).encode()),
                                      ('result.json',payload if kind!='corrupt' else b'changed')]:
                        entry=tarfile.TarInfo(name);entry.size=len(data);out.addfile(entry,io.BytesIO(data))
                    if kind=='extra':
                        entry=tarfile.TarInfo('unexpected');entry.size=1;out.addfile(entry,io.BytesIO(b'x'))
                if kind=='valid':self.assertEqual(manager.verify_archive(path)['members'],1)
                else:
                    with self.assertRaises(ValueError):manager.verify_archive(path)


if __name__=='__main__':unittest.main()
