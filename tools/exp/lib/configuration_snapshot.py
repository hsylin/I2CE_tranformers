"""Freeze gem5 configuration imports and external ARM boot resources.

The simulator embeds m5 itself; local configuration modules (devices, common,
and their imports) live under GEM5_CWD/configs. Keep that entire tree, including
non-Python assets, so relative imports and file reads retain their layout.
Bytecode caches are excluded at creation and disabled during execution.
"""

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import sys

ARM_BOOTLOADERS = ("boot.arm64", "boot.arm")


def regular_file_record(path):
    """Hash a regular file without accepting a symlink or special-file endpoint."""
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Expected a regular snapshot source: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def freeze_bootloaders(m5_path, working_directory, destination):
    """Snapshot the supported ARM platform's two loaders using M5_PATH order.

    Resolve source aliases once, then use only the private resources directory.
    Absolute kernel/disk arguments do not require access to the original roots.
    The adapter rejects any instantiated loader outside this declared pair.
    """
    roots = []
    for entry in m5_path.split(":"):
        if not entry:
            continue
        root = Path(entry).expanduser()
        if not root.is_absolute():
            root = Path(working_directory) / root
        if root.is_dir():
            roots.append(root)
    sources = {}
    for name in ARM_BOOTLOADERS:
        source = next(
            (
                root / "binaries" / name
                for root in roots
                if (root / "binaries" / name).exists()
            ),
            None,
        )
        if source is None:
            raise ValueError("Cannot resolve ARM bootloader from M5_PATH: " + name)
        source = source.resolve(strict=True)
        sources[name] = dict(path=str(source), **regular_file_record(source))
    destination = Path(destination)
    destination.mkdir()  # Never replace a previous or partial snapshot.
    (destination / "binaries").mkdir()
    for name, source in sources.items():
        shutil.copyfile(source["path"], destination / "binaries" / name)
        if regular_file_record(source["path"]) != {
            key: source[key] for key in ("bytes", "sha256")
        }:
            raise ValueError("Bootloader changed during snapshot creation: " + name)
    snapshot = dict(
        path=str(destination.resolve()),
        source_m5_path=m5_path,
        sources=sources,
        members=[
            dict(path="binaries/" + name, bytes=item["bytes"], sha256=item["sha256"])
            for name, item in sorted(sources.items())
        ],
    )
    verify_bootloaders(snapshot)
    return snapshot


def verify_bootloaders(snapshot):
    """Return the ordered private loaders only after checking all snapshot files."""
    if not isinstance(snapshot, dict):
        raise ValueError("Missing frozen ARM bootloader snapshot")
    root = Path(snapshot["path"])
    expected = {"binaries/" + name for name in ARM_BOOTLOADERS}
    if (
        root.resolve() != root
        or {item["path"] for item in snapshot["members"]} != expected
        or tree_manifest(root) != snapshot["members"]
    ):
        raise ValueError("Frozen ARM bootloader snapshot changed")
    return [str(root / "binaries" / name) for name in ARM_BOOTLOADERS]


def tree_manifest(root, *, ignore_bytecode=False):
    """List every regular file; refuse links and special files without following."""
    root = Path(root)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Configuration root must be a real directory")
    members = []

    def visit(directory):
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("Unsupported configuration entry: " + str(path))
            if ignore_bytecode and (
                path.name == "__pycache__" or path.suffix in (".pyc", ".pyo")
            ):
                continue
            if stat.S_ISDIR(mode):
                visit(path)
            else:
                members.append(
                    dict(
                        path=path.relative_to(root).as_posix(),
                        **regular_file_record(path),
                    )
                )

    visit(root)
    if not members:
        raise ValueError("Empty configuration tree")
    return members


def freeze_configuration(source_root, starter, destination):
    """Create a private copy; reject source changes during copying or partial copies."""
    source_root = Path(source_root).resolve(strict=True)
    starter = Path(starter).resolve(strict=True)
    try:
        relative_starter = starter.relative_to(source_root)
    except ValueError as error:
        raise ValueError("GEM5_CFG must be inside GEM5_CWD/configs") from error
    before = tree_manifest(source_root, ignore_bytecode=True)
    if relative_starter.as_posix() not in {item["path"] for item in before}:
        raise ValueError("Starter is not a regular configuration file")
    destination = Path(destination)
    destination.mkdir()  # A failed or completed snapshot is never overwritten.
    for member in before:
        target = destination / member["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / member["path"], target)
    if before != tree_manifest(source_root, ignore_bytecode=True):
        raise ValueError("Source configuration changed during snapshot creation")
    snapshot = dict(
        source_root=str(source_root),
        source_starter=str(starter),
        path=str(destination.resolve()),
        starter=relative_starter.as_posix(),
        members=before,
    )
    verify_snapshot(snapshot)
    return snapshot


def verify_snapshot(snapshot):
    """Revalidate the complete private tree, including added and removed files."""
    root = Path(snapshot["path"])
    if root.resolve() != root or tree_manifest(root) != snapshot["members"]:
        raise ValueError("Frozen configuration tree changed")
    starter = root / snapshot["starter"]
    if starter.relative_to(root).as_posix() not in {
        item["path"] for item in snapshot["members"]
    }:
        raise ValueError("Frozen starter is missing from the configuration manifest")
    return starter


def configuration_names(snapshot):
    """Identify import namespaces owned by this tree and the starter's directory."""
    starter_parent = Path(snapshot["starter"]).parent
    names = set()
    for item in snapshot["members"]:
        path = Path(item["path"])
        if path.suffix != ".py":
            continue
        names.add(path.parts[0] if len(path.parts) > 1 else path.stem)
        if path.parent == starter_parent:
            names.add(path.stem)
    return names - {"__init__"}


def within(path, root):
    """Check containment without requiring Python 3.9 in embedded gem5 Python."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def verify_configuration_imports(snapshot):
    """Reject preloaded or shadowing modules that bypass the private configuration."""
    root = Path(snapshot["path"])
    source = Path(snapshot["source_root"])
    names = configuration_names(snapshot)
    members = {item["path"] for item in snapshot["members"]}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        owned_name = name.split(".", 1)[0] in names
        locations = [filename] if filename else list(getattr(module, "__path__", ()))
        if owned_name and not locations:
            raise ValueError("Configuration module has no verifiable origin: " + name)
        for location in locations:
            path = Path(location).resolve()
            if within(path, source) or owned_name:
                if not within(path, root):
                    raise ValueError("Configuration import escaped snapshot: " + name)
            if within(path, root) and filename:
                if path.relative_to(root).as_posix() not in members:
                    raise ValueError("Unrecorded configuration import: " + name)


def activate_configuration(snapshot, starter):
    """Select private imports and reject previously loaded configuration modules."""
    expected = verify_snapshot(snapshot)
    if Path(starter) != expected:
        raise ValueError("Starter path differs from frozen configuration")
    verify_configuration_imports(snapshot)
    source = Path(snapshot["source_root"])
    sys.path[:] = [
        entry for entry in sys.path if not within(Path(entry or ".").resolve(), source)
    ]
    sys.path[:0] = [str(expected.parent), snapshot["path"]]
    sys.dont_write_bytecode = True


def freeze_diod(path):
    """Resolve one executable once; later aliases cannot redirect queued runs.

    DIOD must be an absolute path. The executable remains in place and is
    rehashed at every execution boundary; it is not invoked while planning.
    """
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("DIOD must be an absolute executable path")
    resolved = path.resolve(strict=True)
    record = dict(path=str(resolved), **regular_file_record(resolved))
    record["mode"] = stat.S_IMODE(resolved.stat().st_mode)
    verify_diod(record)
    return record


def verify_diod(record):
    """Refuse missing, replaced, aliased or nonexecutable frozen 9P helpers."""
    if not isinstance(record, dict):
        raise ValueError("Missing frozen diod executable identity")
    path = Path(record["path"])
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("Frozen diod path changed")
    actual = dict(path=str(path), **regular_file_record(path))
    actual["mode"] = stat.S_IMODE(path.stat().st_mode)
    if actual != record or not os.access(path, os.X_OK):
        raise ValueError("Frozen diod executable changed or is not executable")
    return str(path)


def runner_manifest(root):
    """Record runner sources/templates, excluding regular derived bytecode caches."""
    return tree_manifest(root, ignore_bytecode=True)


class RunnerSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile the verified source bytes directly; never read or write a .pyc."""

    def __init__(self, name, path, snapshot):
        super().__init__(name, str(path))
        self.snapshot = snapshot

    def get_code(self, fullname):
        data = self.get_data(self.path)
        expected = self.snapshot.sources[fullname]
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("Runner source changed before import: " + fullname)
        return self.source_to_code(data, self.path)


class RunnerSources(importlib.abc.MetaPathFinder):
    """Bind the flat runner helper library before its first import.

    CLI scripts bootstrap this class with runpy.run_path, which executes the
    bootstrap .py directly. Only this library's names are intercepted; standard
    library and gem5 imports keep their normal loaders. Existing bytecode is
    harmless because every owned helper is compiled from verified source.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.members = runner_manifest(self.root)
        self.sources = {
            Path(item["path"]).stem: item
            for item in self.members
            if Path(item["path"]).suffix == ".py" and len(Path(item["path"]).parts) == 1
        }
        for name in self.sources:
            if name in sys.modules:
                raise ValueError(
                    "Runner helper was loaded before source bootstrap: " + name
                )

    def find_spec(self, fullname, path=None, target=None):
        item = self.sources.get(fullname)
        if item is None:
            # A removed .py must not silently become an unrecorded sourceless
            # module. This also refuses packages: this helper library is flat.
            candidate = importlib.machinery.PathFinder.find_spec(fullname, path)
            if candidate is not None:
                locations = list(candidate.submodule_search_locations or ())
                if candidate.origin:
                    locations.append(candidate.origin)
                if any(
                    within(Path(location).absolute(), self.root)
                    or within(Path(location).resolve(), self.root)
                    for location in locations
                ):
                    raise ValueError("Unrecorded runner module: " + fullname)
            return None
        source = self.root / item["path"]
        loader = RunnerSourceLoader(fullname, source, self)
        return importlib.util.spec_from_file_location(fullname, source, loader=loader)

    def verify(self):
        """Reject source drift or helper replacement after this process loaded code."""
        if runner_manifest(self.root) != self.members:
            raise ValueError("Runner source tree changed after source bootstrap")
        for name in self.sources:
            module = sys.modules.get(name)
            if module is None:
                continue
            loader = getattr(module, "__loader__", None)
            if (
                not isinstance(loader, RunnerSourceLoader)
                or loader.snapshot is not self
            ):
                raise ValueError(
                    "Runner helper bypassed verified source loading: " + name
                )
        return self.members


def activate_runner_sources(root):
    """Install source-only helper imports before an owner/collector/adapter starts."""
    snapshot = RunnerSources(root)
    sys.meta_path.insert(0, snapshot)
    sys.dont_write_bytecode = True
    return snapshot
