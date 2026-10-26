"""Release inventory, corruption detection and extracted-copy checks."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import build_release as release


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        for path in release.AGENTS + release.EVIDENCE:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture artifact: " + path.as_posix().encode())
        manifest = {p.as_posix(): hashlib.sha256((root / p).read_bytes()).hexdigest()
                    for p in release.AGENTS + release.EVIDENCE}
        (root / release.ASSET_MANIFEST).write_text(json.dumps(manifest))
        for name in ("README.md", "requirements.txt", "config.json",
                      ".gitignore", ".gitattributes", "docs/methodology.md",
                     "docs/tabular_method.md", "docs/outputs.md"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        (root / "build_release.py").write_bytes(Path(release.__file__).read_bytes())
        (root / "docs/history").mkdir()
        (root / "docs/history/old.md").write_text("historical notes")

    def run_builder(self, root, *args):
        return subprocess.run([sys.executable, "build_release.py", *args], cwd=root,
                              capture_output=True, text=True, timeout=30)

    def test_archive_is_unique_and_extracted_source_checksums_are_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            result = self.run_builder(root, "--out", "release.zip")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with zipfile.ZipFile(root / "release.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
                self.assertNotIn("BidPace-RL/docs/history/old.md", archive.namelist())
                manifest = json.loads(archive.read("BidPace-RL/RELEASE_MANIFEST.json"))
                self.assertEqual(len(manifest) + 1, len(archive.namelist()))
                archive.extractall(root / "extracted")
            extracted = root / "extracted/BidPace-RL"
            result = self.run_builder(extracted, "--check")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (extracted / "README.md").write_text("modified")
            result = self.run_builder(extracted, "--check")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Checksum mismatch", result.stderr)

    def test_model_corruption_prevents_release_creation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            (root / release.AGENTS[0]).write_bytes(b"modified model")
            result = self.run_builder(root, "--out", "bad.zip")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "bad.zip").exists())


if __name__ == "__main__":
    unittest.main()
