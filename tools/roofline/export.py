"""Create an immutable, hash-indexed evidence archive after the owner exits."""

from pathlib import Path
import tarfile
from common import digest, freeze_check, load, require, save


def export(root):
    require(
        load(root / "execution/complete.json")["status"] == "passed",
        "Scan is incomplete",
    )
    require(
        load(root / "automated-qa.json")["status"] == "passed",
        "Automated QA is incomplete",
    )
    require(
        not (root / "execution/failure.json").exists(),
        "Failure evidence requires review",
    )
    owner = load(root / "execution/start.json")["pid"]
    proc = Path(f"/proc/{owner}/cmdline")
    require(
        not proc.exists()
        or str(root / "control/scan.py").encode() not in proc.read_bytes(),
        "Owner is still active",
    )
    freeze_check(root)
    destination = root / "exports"
    destination.mkdir(exist_ok=False)
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and destination not in p.parents
    ]
    require(
        not any(p.is_symlink() for p in files), "Symlinks are not portable evidence"
    )
    manifest = {
        str(p.relative_to(root)): dict(
            sha256=digest(p), bytes=p.stat().st_size, mode=p.stat().st_mode & 0o7777
        )
        for p in files
    }
    save(destination / "manifest.json", manifest)
    archive = destination / "sdot-roofline-evidence.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(destination / "manifest.json", arcname="manifest.json")
        for path in files:
            stream.add(
                path, arcname="files/" + str(path.relative_to(root)), recursive=False
            )
    for path in files:
        require(
            digest(path) == manifest[str(path.relative_to(root))]["sha256"],
            "Evidence changed during export",
        )
    receipt = dict(
        status="passed",
        archive=archive.name,
        archive_sha256=digest(archive),
        manifest_sha256=digest(destination / "manifest.json"),
        files=len(files),
    )
    save(destination / "receipt.json", receipt)
    return receipt
