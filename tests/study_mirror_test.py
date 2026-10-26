"""Archive integrity, transport interruption, and normal-closure boundaries."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/exp/lib"))
import study_mirror as mirror


class StudyMirrorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.registration = {
            "maximum_formal_runs": 1,
            "automatic_retries": 0,
            "server_package": "/registered-study",
            "rows": [{"id": "E01", "state": "/state/E01", "output": "/out/E01"}],
        }
        (self.root / "registration.json").write_text(json.dumps(self.registration))

    def archive(self, extra=None):
        payload = b'{"passed": true}\n'
        members = {
            "audit/completion-audit.json": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        }
        manifest = json.dumps(members).encode()
        path = self.root / "source.tar.gz"
        with tarfile.open(path, "w:gz") as bundle:
            for name, content in [
                ("audit/completion-audit.json", payload),
                ("manifest.json", manifest),
            ]:
                item = tarfile.TarInfo(name)
                item.size = len(content)
                bundle.addfile(item, io.BytesIO(content))
            if extra:
                item = tarfile.TarInfo(extra)
                bundle.addfile(item, io.BytesIO(b""))
        receipt = dict(
            passed=True,
            stage="E01",
            path="/registered-study/archives/E01-raw.tar.gz",
            sha256=mirror.digest(path),
            bytes=path.stat().st_size,
            members=2,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        )
        return path, receipt

    def test_every_member_is_verified(self):
        archive, receipt = self.archive()
        target = self.root / "verified"
        target.mkdir()
        mirror.extract_verified(archive, target, receipt)
        self.assertTrue(
            json.loads((target / "audit/completion-audit.json").read_text())["passed"]
        )

    def acknowledgement_fixture(self):
        """Execute the actual remote source against an isolated local study."""
        server = self.root / "server"
        (server / "archives").mkdir(parents=True)
        (server / "evidence").mkdir()
        registration = dict(self.registration, server_package=str(server))
        (server / "registration.json").write_text(json.dumps(registration))
        archive, receipt = self.archive()
        (server / "archives/E01-raw.tar.gz").write_bytes(archive.read_bytes())
        remote = mirror.Remote("unused-test-host")

        def local_python(source, payload):
            result = subprocess.run(
                [sys.executable, "-B", "-c", source],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=10,
            )
            mirror.require(result.returncode == 0, result.stderr)
            return json.loads(result.stdout)

        remote.python = local_python
        return remote, registration, receipt, server

    def test_remote_acknowledgement_source_writes_exact_receipt(self):
        remote, registration, receipt, server = self.acknowledgement_fixture()
        result = remote.acknowledge(registration, "E01", receipt)
        self.assertEqual(result, dict(passed=True, sha256=receipt["sha256"]))
        path = server / "evidence/E01-local-mirror.json"
        self.assertEqual(json.loads(path.read_text()), receipt)
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "FileExistsError"):
            remote.acknowledge(registration, "E01", receipt)
        self.assertEqual(path.read_bytes(), original)

    def test_remote_acknowledgement_source_rejects_tampered_archive(self):
        remote, registration, receipt, server = self.acknowledgement_fixture()
        (server / "archives/E01-raw.tar.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Archive changed"):
            remote.acknowledge(registration, "E01", receipt)
        self.assertFalse((server / "evidence/E01-local-mirror.json").exists())

    def test_tampered_archive_is_rejected_before_extraction(self):
        archive, receipt = self.archive()
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            mirror.extract_verified(archive, self.root / "verified", receipt)
        self.assertFalse((self.root / "verified").exists())

    def test_extra_member_is_rejected(self):
        archive, receipt = self.archive("../outside")
        with self.assertRaisesRegex(ValueError, "extra"):
            mirror.extract_verified(archive, self.root / "verified", receipt)
        self.assertFalse((self.root / "outside").exists())

    def test_partial_transfer_is_preserved_and_never_retried(self):
        archive, receipt = self.archive()
        remote = mock.Mock()

        def partial(source, destination):
            destination.write_bytes(archive.read_bytes()[:12])
            raise TimeoutError("transport interrupted")

        remote.transfer.side_effect = partial
        with self.assertRaises(TimeoutError):
            mirror.mirror_stage(self.root, self.registration, remote, "E01", receipt)
        with self.assertRaises(FileExistsError):
            mirror.mirror_stage(self.root, self.registration, remote, "E01", receipt)
        self.assertEqual(remote.transfer.call_count, 1)
        remote.acknowledge.assert_not_called()
        self.assertEqual((self.root / "rows/E01/E01-raw.tar.gz").stat().st_size, 12)

    def test_completed_transfer_is_not_repeated(self):
        archive, receipt = self.archive()
        remote = mock.Mock()
        remote.transfer.side_effect = lambda source, dest: dest.write_bytes(
            archive.read_bytes()
        )
        mirror.mirror_stage(self.root, self.registration, remote, "E01", receipt)
        mirror.mirror_stage(self.root, self.registration, remote, "E01", receipt)
        self.assertEqual(remote.transfer.call_count, 1)
        self.assertEqual(remote.acknowledge.call_count, 1)

    def test_outside_namespace_is_rejected(self):
        _, receipt = self.archive()
        receipt["path"] = "/another-study/archives/E01-raw.tar.gz"
        remote = mock.Mock()
        with self.assertRaisesRegex(ValueError, "namespace"):
            mirror.mirror_stage(self.root, self.registration, remote, "E01", receipt)
        remote.transfer.assert_not_called()

    def test_final_post_archive_receipt_is_required(self):
        archive, receipt = self.archive()
        remote = mock.Mock()
        remote.transfer.side_effect = lambda source, dest: dest.write_bytes(
            archive.read_bytes()
        )
        records = {
            "E01-archive.json": json.dumps(receipt),
            "controller-complete.json": json.dumps(
                dict(passed=True, failed=False, launched=1, all_children_reaped=True)
            ),
        }
        remote.snapshot.return_value = dict(records=records, final_states={})
        with self.assertRaisesRegex(ValueError, "post-archive"):
            mirror.tick(self.root, self.registration, remote)
        self.assertFalse((self.root / "complete.json").exists())
        remote.snapshot.return_value["final_states"] = {"E01": '{"passed":true}\n'}
        self.assertTrue(mirror.tick(self.root, self.registration, remote)["complete"])
        self.assertEqual(
            (self.root / "server-final/E01-final-state.json").read_text(),
            '{"passed":true}\n',
        )
        self.assertEqual(remote.transfer.call_count, 1)

    def test_server_failure_prevents_all_normal_closure(self):
        remote = mock.Mock()
        remote.snapshot.return_value = dict(
            records={"failure-E01.json": '{"error":"exit 1"}'}, final_states={}
        )
        with self.assertRaisesRegex(ValueError, "Server failure"):
            mirror.tick(self.root, self.registration, remote)
        remote.transfer.assert_not_called()
        self.assertFalse((self.root / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
