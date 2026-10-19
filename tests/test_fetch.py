"""Downloader contracts, tested offline with provider-shaped fixtures."""

import hashlib
import io
import tempfile
import unittest
from unittest.mock import patch
from bidpace.fetch import selected, download_file, PREFIX


class DownloadTests(unittest.TestCase):
    def test_scope(self):
        self.assertTrue(selected(PREFIX + "training2nd/imp.20130606.txt.bz2"))
        self.assertFalse(selected(PREFIX + "testing2nd/leaderboard.test.data.20130613_15.txt.bz2"))
        self.assertFalse(selected(PREFIX + "training3rd/imp.20130606.txt.bz2"))
        self.assertFalse(selected(PREFIX + "training2nd/imp.20130616.txt.bz2"))

    def test_checksums_atomic_install_and_reuse(self):
        payload = b"raw fixture"
        digest = hashlib.md5(payload).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            name = PREFIX + "training2nd/imp.20130606.txt.bz2"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as network:
                path = download_file(name, td, len(payload), digest)
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(download_file(name, td, len(payload), digest), path)
                self.assertEqual(network.call_count, 1)
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"broken")):
                with self.assertRaises(ValueError):
                    download_file(name, td, len(payload), "0" * 32)
            self.assertEqual(path.read_bytes(), payload)

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                download_file(PREFIX + "../unsafe", td)
