import io
import stat
import unittest
from zipfile import ZipFile, ZipInfo

from offline_study.navigation_assets import safe_members


class NavigationAssetTests(unittest.TestCase):
    def test_archive_path_and_symlink_guards(self):
        for path in ("../escape", "/absolute", "dir/../../escape", "dir\\file"):
            with ZipFile(io.BytesIO(), "w") as archive:
                archive.writestr(path, "data")
                with self.assertRaises(ValueError):
                    safe_members(archive)
        with ZipFile(io.BytesIO(), "w") as archive:
            entry = ZipInfo("link")
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, "target")
            with self.assertRaises(ValueError):
                safe_members(archive)
        with ZipFile(io.BytesIO(), "w") as archive:
            archive.writestr("point_maze/train/obs.pt", "data")
            self.assertEqual(len(safe_members(archive)), 1)
