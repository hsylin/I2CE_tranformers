"""Fail closed on legacy runs without frozen labels or verified cache policy."""

import csv
import json
from pathlib import Path
import sys

from experiment_spec import digest, git, require, save_new
from false_cache_config import verify_configuration


def bind(output):
    """Freeze row and build identities before launching a newly managed run."""
    row = next(
        csv.reader(
            (output / "experiment-row.tsv").read_text().splitlines(), delimiter="\t"
        )
    )
    require(len(row) >= 6, "Incomplete frozen experiment row")
    config = dict(
        csv.reader(
            (output / "build_config.tsv").read_text().splitlines(), delimiter="\t"
        )
    )
    for field, value in zip(
        ("exp_id", "codebook_size", "n_learners", "sve_bits", "impl", "overrides"), row
    ):
        require(
            config.get(field) == value, "Built configuration differs from row: " + field
        )
    share = Path((output / "build_dir").read_text().strip())
    require(
        digest(share / "transformer.o") == config["binary_sha256"],
        "Legacy binary changed",
    )
    require(
        len(config.get("repo_commit_full", "")) == 40
        and "+dirty" not in config.get("repo_commit", ""),
        "Legacy formal execution requires a clean, full source commit identity",
    )
    save_new(
        output / "frozen-labels.json",
        {
            "row_sha256": digest(output / "experiment-row.tsv"),
            "build_config_sha256": digest(output / "build_config.tsv"),
            "source_commit": config["repo_commit_full"],
        },
    )


def check(output, formal=False):
    """Check the actual generated policy and the recorded configuration hash."""
    receipt = json.loads((output / "cache-policy.json").read_text())
    require(receipt["passed"], "Missing successful cache policy receipt")
    require(
        receipt.get("simulate_called") is formal,
        "Incorrect configuration/formal receipt",
    )
    actual = verify_configuration(
        output / "config.ini", allow_uncached=receipt.get("uncached_boot", False)
    )
    require(
        actual["config_sha256"] == receipt["config_sha256"],
        "Actual configuration changed after execution",
    )


def collect(output):
    """Verify all immutable labels for post-gate legacy runs."""
    frozen = json.loads((output / "frozen-labels.json").read_text())
    require(
        digest(output / "experiment-row.tsv") == frozen["row_sha256"],
        "Frozen row changed",
    )
    require(
        digest(output / "build_config.tsv") == frozen["build_config_sha256"],
        "Frozen build metadata changed",
    )
    check(output, formal=True)


def dryrun(repo, ref, table, identifier):
    """Resolve a legacy source and row without creating a build or result."""
    sha = git(repo, "rev-parse", "--verify", ref + "^{commit}").decode().strip()
    with Path(table).open() as stream:
        matches = [
            row
            for row in csv.reader(stream, delimiter="\t")
            if row and row[0] == identifier
        ]
    require(len(matches) == 1, "Unknown or duplicate legacy ID")
    require(len(matches[0]) >= 6, "Malformed legacy row")
    print(
        json.dumps(
            {
                "source_commit": sha,
                "legacy_row": matches[0],
                "build": [
                    "bash",
                    "<isolated source at " + sha + ">/compile_transformer.sh",
                ],
                "configuration_adapter": "false_cache_config.py",
                "configuration_only_required": True,
                "formal_policy": {"L1D.writeback_clean": False},
                "files_created": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        action = sys.argv[1]
        if action == "dryrun":
            dryrun(*sys.argv[2:])
        elif action == "bind":
            bind(Path(sys.argv[2]))
        elif action == "collect":
            collect(Path(sys.argv[2]))
        elif action == "check":
            check(Path(sys.argv[2]))
        else:
            raise ValueError("Unknown legacy gate action")
    except (ValueError, OSError) as error:
        sys.exit(str(error))
