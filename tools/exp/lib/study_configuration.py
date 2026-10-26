"""Initialize the pinned gem5 machine and gate it before simulation."""

import importlib.util
from pathlib import Path
import sys
import m5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_records import require, load, save, verify_contract, verify_files
from study_config_gate import verify


def take_option(name):
    values = [item for item in sys.argv if item.startswith(name + "=")]
    require(len(values) == 1, "Study check failed: len(values) == 1")
    sys.argv.remove(values[0])
    return values[0].split("=", 1)[1]


def main():
    root = Path(take_option("--study-root"))
    stage = take_option("--study-stage")
    contract_sha = take_option("--study-contract")
    check = "--configuration-check" in sys.argv
    if check:
        sys.argv.remove("--configuration-check")
    registration = verify_contract(root, contract_sha)
    row = next(item for item in registration["rows"] if item["id"] == stage)
    state, inputs = Path(row["state"]), Path(row["inputs"])
    bound = load(state / "input-binding.json")
    verify_files(bound["files"])
    verify_files(registration["runtime_pins"])
    tree = Path(registration["gem5_root"])
    original_path = tree / "configs/example/arm/starter_fs.py"
    sys.path[:0] = [str(tree / "configs"), str(original_path.parent)]
    specification = importlib.util.spec_from_file_location(
        "_reconstruction_original_config", original_path
    )
    original = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(original)
    create, run = original.create, original.run

    def checked_create(args):
        require(
            args.restore == registration["checkpoint"],
            "Study check failed: args.restore == registration['checkpoint']",
        )
        require(
            Path(args.script) == inputs / "run.rcS" and Path(args.vio_9p) == inputs,
            "Study check failed: Path(args.script) == inputs/'run.rcS' and Path(args.vio_9p) == inputs",
        )
        require(
            args.l1d_size == "32KiB"
            and args.l2_size == "1MiB"
            and args.l1i_size is None,
            "Study check failed: args.l1d_size == '32KiB' and args.l2_size == '1MiB' and args.l1i_size is None",
        )
        return create(args)

    def checked_run(args):
        output = Path(m5.options.outdir)
        verify(root, row, inputs, output)
        require(
            int(m5.curTick()) == registration["restored_tick"],
            "Study check failed: int(m5.curTick()) == registration['restored_tick']",
        )
        if check:
            save(
                output / "initialization-check.json",
                dict(
                    passed=True,
                    stage=stage,
                    restored_tick=int(m5.curTick()),
                    simulate_called=False,
                    guest_instructions_executed=0,
                    performance_sample=False,
                ),
            )
            return
        gate = load(state / "prelaunch-gate.json")
        require(
            gate["passed"] and gate["stage"] == stage,
            "Study check failed: gate['passed'] and gate['stage'] == stage",
        )
        require(
            load(state / "formal-reservation.json")["stage"] == stage,
            "Study check failed: load(state/'formal-reservation.json')['stage'] == stage",
        )
        return run(args)

    original.create, original.run = checked_create, checked_run
    original.main()


if __name__ == "__m5_main__":
    main()
