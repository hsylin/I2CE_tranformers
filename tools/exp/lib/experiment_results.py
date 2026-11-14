"""Collect integer cumulative counters using the run's actual phase markers."""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import stat
from experiment_spec import require
from study_counters import raw_blocks_text, deltas

# Bump when the counter/phase aggregation contract changes. Source hashes also
# identify implementation revisions that retain the same metrics schema.
METRICS_SCHEMA = "i2ce.phase-metrics.v1"

COUNTERS = {
    "sim_ticks": "simTicks",
    "instructions": "simInsts",
    "l1d_accesses": "system.cpu_cluster.cpus.dcache.demandAccesses::total",
    "l1d_misses": "system.cpu_cluster.cpus.dcache.demandMisses::total",
    "l2_accesses": "system.cpu_cluster.l2.demandAccesses::total",
    "l2_misses": "system.cpu_cluster.l2.demandMisses::total",
}
CODEBOOK_PHASES = {"MHA_QKV", "Projection", "FF1", "FF2"}


def file_version(info):
    """Detect replacement or writes while a file is being captured."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True)
class CapturedFile:
    """Keep the bytes used for parsing and their identity together.

    Inode and timestamps guard the read, but are not part of the durable
    identity: copying an unchanged run must not invalidate its evidence.
    """

    path: Path
    data: bytes
    mode: int

    @classmethod
    def read(cls, path):
        """Capture one stable, nonempty regular file; never follow a final symlink."""
        path = Path(path).absolute()
        try:
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            with os.fdopen(os.open(path, flags), "rb") as stream:
                before = os.fstat(stream.fileno())
                require(
                    stat.S_ISREG(before.st_mode),
                    "Not a regular result file: " + str(path),
                )
                data = stream.read()
                after = os.fstat(stream.fileno())
                current = path.lstat()
        except OSError as error:
            raise ValueError("Cannot read result file: " + str(path)) from error
        require(
            file_version(before) == file_version(after) == file_version(current)
            and len(data) == before.st_size,
            "Result file changed while reading: " + str(path),
        )
        require(data, "Empty result file: " + str(path))
        return cls(path, data, stat.S_IMODE(before.st_mode))

    @property
    def identity(self):
        """Describe exactly the captured bytes, without another filesystem read."""
        return {
            "path": str(self.path),
            "bytes": len(self.data),
            "mode": self.mode,
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }

    def verify_unchanged(self):
        """Refuse a result whose source changed during collection."""
        require(
            self.read(self.path).identity == self.identity,
            "Result file changed during collection: " + str(self.path),
        )


def extract(stats, mapping):
    """Read files for callers that do not need completion-bound output evidence."""
    return extract_bytes(stats.read_bytes(), mapping.read_bytes())


def extract_bytes(stats_data, mapping_data):
    """Convert fixed bytes into ROI deltas and explicit BLOCK/CODEBOOK aggregates."""
    labels = []
    for line in mapping_data.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[0].isdigit():
            require(
                int(fields[0]) == len(labels) + 1,
                "ROI dump indices must be consecutive from one",
            )
            require(fields[1] and fields[2], "Empty phase marker")
            labels.append(fields[2])
    require(labels, "Missing phase/ROI markers")
    blocks = raw_blocks_text(
        stats_data.decode("utf-8"), expected_blocks=len(labels) + 1
    )
    require(
        all("simTicks" in block and "simFreq" in block for block in blocks),
        "Missing runtime counters",
    )
    frequency = Decimal(blocks[0]["simFreq"])
    require(
        frequency == 10**12
        and all(Decimal(block["simFreq"]) == frequency for block in blocks),
        "Unsupported or inconsistent simulation tick frequency",
    )
    available = {
        name: key
        for name, key in COUNTERS.items()
        if all(key in block for block in blocks)
    }
    values = deltas(blocks, available)
    require(
        all(
            type(value["sim_ticks"]) is int and value["sim_ticks"] >= 0
            for value in values
        ),
        "Runtime must be nonnegative integer ticks",
    )
    phases = {}
    for label, interval in zip(labels, values):
        target = phases.setdefault(label, {name: 0 for name in available})
        for name, value in interval.items():
            target[name] += value
    phases["BLOCK"] = {
        name: sum(interval[name] for interval in values[:-1]) for name in available
    }
    require(phases["BLOCK"]["sim_ticks"] > 0, "Empty BLOCK interval")
    phases["CODEBOOK"] = {
        name: sum(
            interval[name]
            for label, interval in zip(labels, values)
            if label in CODEBOOK_PHASES
        )
        for name in available
    }
    return {
        "tick_frequency": int(frequency),
        "raw_dumps": len(blocks),
        "roi_intervals": len(labels),
        "phases": phases,
        "missing_counters": sorted(set(COUNTERS) - set(available)),
        "intervals": [
            {"index": i + 1, "phase": label, "counters": value}
            for i, (label, value) in enumerate(zip(labels, values))
        ],
        "post_roi": values[-1],
    }
