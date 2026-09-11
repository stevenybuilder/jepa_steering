import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/vast'
sys.path.insert(0,str(SCRIPTS))
import manage_confirmation as manager


class ConfirmationManagementTests(unittest.TestCase):
    def test_user_pause_prevents_new_rental_and_staging_before_api_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'USER_PAUSE.json').write_text('{}')
            with patch.object(manager,'ROOT',root),patch.object(manager,'command') as command:
                with self.assertRaisesRegex(ValueError,'User paused'):manager.reserve({})
                with self.assertRaisesRegex(ValueError,'User paused'):manager.stage({})
                command.assert_not_called()

    def test_incomplete_worker_cannot_be_parked_or_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            local=Path(temp)
            (local/'WORKER_DONE.json').write_text(json.dumps({'task':'reach','episodes':120}))
            with patch.object(manager,'provider') as provider, patch.object(manager,'preserve') as preserve:
                with self.assertRaises(ValueError):
                    manager.park_completed({'id':123,'task':'reach','num_gpus':2},local)
                provider.assert_not_called();preserve.assert_not_called()

    def test_drive_preservation_failure_prevents_completed_worker_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            local=Path(temp)/'worker';local.mkdir()
            (local/'WORKER_DONE.json').write_text(json.dumps({'task':'reach','episodes':60}))
            shard=local/'results/reach/native/rank-0';shard.mkdir(parents=True)
            for i in range(60):(shard/f'episode-{i}.json').write_text('{}')
            lease={'id':123,'task':'reach','num_gpus':1,'label':'owned','geolocation':'Florida, US'}
            row={**lease,'gpu_name':'RTX 4090'}
            with patch.object(manager,'ROOT',Path(temp)), patch.object(manager,'provider',return_value={123:row}), \
                 patch.object(manager,'preserve',side_effect=ValueError('Drive unavailable')), \
                 patch.object(manager,'command') as command:
                with self.assertRaisesRegex(ValueError,'Drive unavailable'):manager.park_completed(lease,local)
                command.assert_not_called()

    def test_completed_worker_parks_only_after_archive_and_retains_disk(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);local=root/'worker';local.mkdir()
            (local/'WORKER_DONE.json').write_text(json.dumps({'task':'reach','episodes':60}))
            for arm in ('native','fixed_rank4','matched_random_fixed_rank4','coupling_only','matched_random_coupling'):
                shard=local/'results/reach'/arm/'rank-0';shard.mkdir(parents=True)
                for i in range(12):(shard/f'episode-{i}.json').write_text('{}')
            lease={'id':123,'task':'reach','num_gpus':1,'label':'owned','geolocation':'Florida, US'}
            running={**lease,'gpu_name':'RTX 4090','actual_status':'running'}
            order=[]
            def preserved(*args):
                order.append('full_drive_readback')
                manager.write(root/'closeout/123/DRIVE_VERIFIED.json',{'file_id':'verified'})
            def stopped(argv,timeout):
                self.assertEqual(order,['full_drive_readback'])
                self.assertEqual(argv[1:4],['stop','instance','123'])
                order.append('stop');return 'success'
            with patch.object(manager,'ROOT',root), \
                 patch.object(manager,'provider',side_effect=[{123:running},{123:running},
                     {123:{**running,'actual_status':'exited'}}]), \
                 patch.object(manager,'preserve',side_effect=preserved), \
                 patch.object(manager,'command',side_effect=stopped):
                manager.park_completed(lease,local)
            proof=json.loads((root/'closeout/123/PARKED.json').read_text())
            self.assertTrue(proof['disk_retained']);self.assertTrue(proof['drive_verified_first'])
            self.assertEqual(order,['full_drive_readback','stop'])

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
