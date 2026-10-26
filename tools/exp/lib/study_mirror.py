"""Preserve completed run archives without simulation, parsing, or retries."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time

from study_records import digest, load, require, save, study_budget


class Remote:
    """Bound SSH calls to one registered study and the selected account."""

    def __init__(self, host, identity=None, socket=None):
        self.command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if identity:
            self.command += ["-i", str(identity)]
        if socket:
            self.command += ["-S", str(socket)]
        self.command.append(host)

    def python(self, source, payload):
        result = subprocess.run(
            self.command + ["python3 -B -c " + shlex.quote(source)],
            input=json.dumps(payload).encode(),
            capture_output=True,
            timeout=55,
        )
        require(result.returncode == 0, result.stderr.decode(errors="replace"))
        return json.loads(result.stdout)

    def snapshot(self, registration):
        return self.python(
            """
import json,pathlib,sys
request=json.load(sys.stdin);root=pathlib.Path(request['root'])
reg=json.loads((root/'registration.json').read_text())
if reg!=request['registration']: raise ValueError('Registration mismatch')
records={p.name:p.read_text() for p in (root/'evidence').glob('*.json')
         if not p.name.endswith('-current.json')}
states={row['id']:(pathlib.Path(row['state'])/'complete.json').read_text()
        for row in reg['rows'] if (pathlib.Path(row['state'])/'complete.json').exists()}
print(json.dumps(dict(records=records,final_states=states)))
""",
            {"root": registration["server_package"], "registration": registration},
        )

    def transfer(self, source, destination):
        with destination.open("xb") as output:
            result = subprocess.run(
                self.command + ["cat " + shlex.quote(source)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=55,
            )
        require(result.returncode == 0, result.stderr.decode(errors="replace"))

    def acknowledge(self, registration, stage, receipt):
        return self.python(
            r"""
import hashlib,json,pathlib,sys
r=json.load(sys.stdin);root=pathlib.Path(r['root']);stage=r['stage']
reg=json.loads((root/'registration.json').read_text())
if stage not in [row['id'] for row in reg['rows']]: raise ValueError(stage)
archive=root/'archives'/(stage+'-raw.tar.gz');h=hashlib.sha256()
with archive.open('rb') as f:
 for block in iter(lambda:f.read(1024**2),b''):h.update(block)
if h.hexdigest()!=r['receipt']['sha256']: raise ValueError('Archive changed')
path=root/'evidence'/(stage+'-local-mirror.json')
with path.open('x') as f: json.dump(r['receipt'],f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=h.hexdigest())))
""",
            {
                "root": registration["server_package"],
                "stage": stage,
                "receipt": receipt,
            },
        )


def extract_verified(archive, destination, receipt):
    """Check the entire member set before writing regular files exclusively."""
    require(digest(archive) == receipt["sha256"], "Archive checksum mismatch")
    require(archive.stat().st_size == receipt["bytes"], "Archive size mismatch")
    with tarfile.open(archive) as bundle:
        data = bundle.extractfile("manifest.json").read()
        manifest = json.loads(data)
        names = bundle.getnames()
        require(
            hashlib.sha256(data).hexdigest() == receipt["manifest_sha256"],
            "Member manifest checksum mismatch",
        )
        require(
            len(names) == len(manifest) + 1 == receipt["members"]
            and set(names) == set(manifest) | {"manifest.json"},
            "Archive contains missing, extra, or duplicate members",
        )
        for name in names:
            path = PurePosixPath(name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and bundle.getmember(name).isfile(),
                "Unsafe archive member: " + name,
            )
        for name, metadata in manifest.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            with bundle.extractfile(name) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024**2), b""):
                    checksum.update(block)
                    output.write(block)
            require(
                checksum.hexdigest() == metadata["sha256"]
                and target.stat().st_size == metadata["bytes"],
                "Member checksum or size mismatch: " + name,
            )
    with (destination / "member-manifest.json").open("xb") as output:
        output.write(data)


def mirror_stage(root, registration, remote, stage, receipt):
    row = root / "rows" / stage
    expected_path = str(
        Path(registration["server_package"]) / "archives" / (stage + "-raw.tar.gz")
    )
    require(receipt["passed"] and receipt["stage"] == stage, "Invalid archive receipt")
    require(receipt["path"] == expected_path, "Archive outside registered namespace")
    if (row / "both-host.json").exists():
        saved = load(row / "both-host.json")
        require(saved["passed"] and saved["sha256"] == receipt["sha256"], stage)
        return
    # An incomplete row is never retried. Inspect its preserved bytes separately.
    row.mkdir(parents=True, exist_ok=False)
    save(row / "server-archive.json", receipt)
    archive = row / (stage + "-raw.tar.gz")
    remote.transfer(receipt["path"], archive)
    extract_verified(archive, row, receipt)
    result = dict(
        passed=True,
        stage=stage,
        sha256=receipt["sha256"],
        bytes=receipt["bytes"],
        members=receipt["members"],
        all_member_hashes_verified=True,
        local_archive=str(archive),
        server_archive=receipt["path"],
        epoch=time.time(),
    )
    remote.acknowledge(registration, stage, result)
    save(row / "both-host.json", result)


def tick(root, registration, remote):
    budget = study_budget(registration)
    snapshot = remote.snapshot(registration)
    records = {name: json.loads(text) for name, text in snapshot["records"].items()}
    failures = {
        name: value for name, value in records.items() if name.startswith("failure-")
    }
    require(not failures, "Server failure requires inspection: " + json.dumps(failures))
    for row in registration["rows"]:
        name = row["id"] + "-archive.json"
        if name in records:
            mirror_stage(root, registration, remote, row["id"], records[name])
    mirrored = sum(
        (root / "rows" / row["id"] / "both-host.json").exists()
        for row in registration["rows"]
    )
    controller = records.get("controller-complete.json")
    if mirrored != budget or controller is None:
        return dict(complete=False, mirrored=mirrored)
    require(
        controller["passed"]
        and not controller["failed"]
        and controller["launched"] == budget
        and controller["all_children_reaped"],
        "Controller did not close normally",
    )
    final_states = {
        stage: json.loads(text) for stage, text in snapshot["final_states"].items()
    }
    require(
        set(final_states) == {row["id"] for row in registration["rows"]}
        and all(value["passed"] for value in final_states.values()),
        "Final post-archive stage receipts are incomplete",
    )
    final = root / "server-final"
    final.mkdir(exist_ok=False)
    for name, text in snapshot["records"].items():
        require(Path(name).name == name, "Invalid remote receipt name")
        with (final / name).open("x") as output:
            output.write(text)
    for stage, text in snapshot["final_states"].items():
        with (final / (stage + "-final-state.json")).open("x") as output:
            output.write(text)
    save(
        root / "complete.json",
        dict(
            passed=True,
            mirrored=mirrored,
            stage_final_receipts=final_states,
            controller_complete=controller,
            registration_sha256=digest(root / "registration.json"),
            epoch=time.time(),
        ),
    )
    return dict(complete=True, mirrored=mirrored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    remote = Remote(args.host, args.identity, args.socket)
    with (root / "mirror.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "complete.json").exists():
            print(json.dumps(dict(already_complete=True)))
            return
        if args.watch:
            save(
                root / "mirror-owner-start.json",
                dict(
                    pid=os.getpid(), epoch=time.time(), helper_sha256=digest(__file__)
                ),
            )
        try:
            while True:
                result = tick(root, load(root / "registration.json"), remote)
                if result["complete"] or not args.watch:
                    print(json.dumps(result), flush=True)
                    return
                time.sleep(60)
        except Exception as error:
            path = root / "mirror-attention.json"
            if not path.exists():
                save(
                    path,
                    dict(error=repr(error), epoch=time.time(), automatic_retry=False),
                )
            raise


if __name__ == "__main__":
    main()
