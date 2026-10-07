import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare_release import prepare


class ReleaseTests(unittest.TestCase):
    def test_public_snapshot_excludes_local_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = prepare(Path(__file__).resolve().parent, Path(temporary))
            with zipfile.ZipFile(result["zip"]) as archive:
                names = archive.namelist()
                self.assertTrue(any(name.endswith("/LICENSE") for name in names))
                self.assertTrue(any(name.endswith("/web/app.js") for name in names))
                self.assertFalse(
                    any(
                        "/exports/" in name
                        or "/references/" in name
                        or name.endswith("reference.jpg")
                        for name in names
                    )
                )
                self.assertIsNone(archive.testzip())
            manifest = json.loads((Path(result["folder"]) / "SOURCE_MANIFEST.json").read_text())
            self.assertEqual(manifest["version"], "2.1.0")
            # A source-only installation serves no personal reference image.
            self.assertFalse((Path(result["folder"]) / "web/reference.jpg").exists())


if __name__ == "__main__":
    unittest.main()
