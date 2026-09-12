"""CPU-only preservation/assembly safety; no network or scientific launches."""
import hashlib
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/vast'))
from analyze_completed_navigation import copy_tree, extract_verified


class NavigationTableAssemblyTests(unittest.TestCase):
    def fixture(self, root, names):
        archive = root/'archive.tar.gz'
        raw = b'preserved data'
        with tarfile.open(archive, 'w:gz') as stream:
            for name in names:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                stream.addfile(member, io.BytesIO(raw))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        receipt = {'archive_sha256': digest, 'verified': True,
            'full_byte_readback': True,
            'metadata': {'sha256Checksum': digest, 'size': str(archive.stat().st_size)},
            'manifest': {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                         for name in names}}
        return archive, receipt

    def test_exact_preserved_bytes_extract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, proof = self.fixture(root, ['a/report.json'])
            extract_verified(archive, proof, root/'out')
            self.assertEqual((root/'out/a/report.json').read_bytes(), b'preserved data')

    def test_member_tampering_fails_before_member_write(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, proof = self.fixture(root, ['report.json'])
            proof['manifest']['report.json']['sha256'] = 'wrong'
            with self.assertRaises(AssertionError):
                extract_verified(archive, proof, root/'out')
            self.assertFalse((root/'out/report.json').exists())

    def test_duplicate_or_escaping_names_rejected(self):
        for names in (['../escape'], ['/absolute'], ['same', 'same']):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                archive, proof = self.fixture(root, names)
                with self.assertRaises((ValueError, AssertionError)):
                    extract_verified(archive, proof, root/'out')

    def test_conflicting_copy_never_overwrites_target(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root/'source', root/'target'
            source.mkdir(); target.mkdir()
            (source/'record').write_text('original')
            (target/'record').write_text('different')
            with self.assertRaises(AssertionError):
                copy_tree(source, target)
            self.assertEqual((target/'record').read_text(), 'different')


if __name__ == '__main__':
    unittest.main()
