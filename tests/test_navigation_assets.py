import io
import stat
import unittest
from types import SimpleNamespace
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

    def test_actual_maze_size_fits_but_budget_is_still_enforced(self):
        entry = ZipInfo("point_maze/obses/episode_000.pth")
        entry.file_size = 30117624491  # Verified complete official expansion size.
        archive = SimpleNamespace(infolist=lambda: [entry])
        self.assertEqual(safe_members(archive), [entry])
        with self.assertRaises(ValueError):
            safe_members(archive, max_bytes=20 * 1024**3)
        entry.file_size = 65 * 1024**3
        with self.assertRaises(ValueError):
            safe_members(archive)
