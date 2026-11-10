"""Restore a registered machine, optionally observe traffic, and simulate once.

Run with gem5, not host Python. The registration and reference config must be
prepared and reviewed before launch; this module never creates checkpoints.
"""

import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import m5
from m5.objects import CommMonitor


def option(name):
    values = [arg for arg in sys.argv if arg.startswith(name + "=")]
    if len(values) != 1:
        raise ValueError("Expected exactly one " + name)
    sys.argv.remove(values[0])
    return values[0].split("=", 1)[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def read_ini(path):
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(path)
    return {section: dict(parser[section]) for section in parser.sections()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def check_config(reference, output, inputs, monitored):
    """Compare every INI parameter, allowing only explicit paths and monitors."""
    expected, actual = read_ini(reference), read_ini(output / "config.ini")
    for section, key, value in [
        ("system", "readfile", str(inputs / "run.rcS")),
        ("system.pci_devices1.vio", "root", str(inputs)),
        ("system.pci_devices1.vio", "socketPath", str(output / "9p.sock")),
        ("system.workload", "dtb_filename", str(output / "system.dtb")),
    ]:
        expected[section][key] = value
    if monitored:
        connections = [
            (
                "system.cpu_cluster.cpus",
                "dcache_port",
                "system.roofline_l1.cpu_side_port",
            ),
            (
                "system.cpu_cluster.cpus.dcache",
                "cpu_side",
                "system.roofline_l1.mem_side_port",
            ),
            (
                "system.cpu_cluster.cpus.dcache",
                "mem_side",
                "system.roofline_l2.cpu_side_port",
            ),
        ]
        for section, key, value in connections:
            expected[section][key] = value
        section = "system.cpu_cluster.toL2Bus"
        expected[section]["cpu_side_ports"] = expected[section][
            "cpu_side_ports"
        ].replace(
            "system.cpu_cluster.cpus.dcache.mem_side",
            "system.roofline_l2.mem_side_port",
        )
        # Monitor parameters are checked separately, never ignored wholesale.
        for name, cpu_peer, mem_peer in [
            (
                "roofline_l1",
                "system.cpu_cluster.cpus.dcache_port",
                "system.cpu_cluster.cpus.dcache.cpu_side",
            ),
            (
                "roofline_l2",
                "system.cpu_cluster.cpus.dcache.mem_side",
                "system.cpu_cluster.toL2Bus.cpu_side_ports[1]",
            ),
        ]:
            monitor = actual.pop("system." + name)
            if (
                monitor["type"] != "CommMonitor"
                or monitor["cpu_side_port"] != cpu_peer
                or monitor["mem_side_port"] != mem_peer
            ):
                raise ValueError("Unexpected monitor connection: " + name)
            if (
                monitor["disable_bandwidth_hists"] != "false"
                or monitor["sample_period"] != "1000000000000"
            ):
                raise ValueError("Unexpected monitor byte accounting")
            for key in (
                "burst_length_hists",
                "latency_hists",
                "itt_dists",
                "outstanding_hists",
                "transaction_hists",
                "addr_dists",
            ):
                if monitor["disable_" + key] != "true":
                    raise ValueError("Unexpected monitor instrumentation")
        actual["system"]["children"] = " ".join(
            name
            for name in actual["system"]["children"].split()
            if name not in ("roofline_l1", "roofline_l2")
        )
    differences = []
    for section in sorted(set(expected) | set(actual)):
        for key in sorted(
            set(expected.get(section, {})) | set(actual.get(section, {}))
        ):
            a, b = expected.get(section, {}).get(key), actual.get(section, {}).get(key)
            if a != b:
                differences.append(
                    dict(section=section, parameter=key, expected=a, actual=b)
                )
    save(
        output / "config-gate.json",
        dict(passed=not differences, differences=differences),
    )
    if differences:
        raise ValueError("Hardware config differs; simulation was not entered")


def main():
    registration_path = Path(option("--roofline-registration"))
    registration_sha = option("--roofline-registration-sha256")
    if digest(registration_path) != registration_sha:
        raise ValueError("Registration identity changed")
    registration = json.loads(registration_path.read_text())
    name = option("--roofline-run")
    row = registration["runs"][name]
    check = "--configuration-check" in sys.argv
    if check:
        sys.argv.remove("--configuration-check")
    for filename, expected in {
        **registration["runtime_pins"],
        **registration.get("checkpoint_pins", {}),
        **row["input_pins"],
    }.items():
        if digest(filename) != expected:
            raise ValueError("Input identity changed: " + filename)
    tree = Path(registration["gem5_root"])
    path = tree / "configs/example/arm/starter_fs.py"
    sys.path[:0] = [str(tree / "configs"), str(path.parent)]
    spec = importlib.util.spec_from_file_location("_roofline_machine", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    create = original.create

    def instrumented_create(args):
        if (
            args.restore != registration["checkpoint"]
            or args.vio_9p != row["inputs"]
            or args.script != row["inputs"] + "/run.rcS"
        ):
            raise ValueError("Unregistered checkpoint or guest inputs")
        system = create(args)
        if row["monitored"]:
            for name in ("roofline_l1", "roofline_l2"):
                setattr(
                    system,
                    name,
                    CommMonitor(
                        sample_period="1s",
                        disable_burst_length_hists=True,
                        disable_bandwidth_hists=False,
                        disable_latency_hists=True,
                        disable_itt_dists=True,
                        disable_outstanding_hists=True,
                        disable_transaction_hists=True,
                        disable_addr_dists=True,
                    ),
                )
            cpu = system.cpu_cluster[0].cpus[0]
            cpu.dcache_port.splice(
                system.roofline_l1.cpu_side_port, system.roofline_l1.mem_side_port
            )
            cpu.dcache.mem_side.splice(
                system.roofline_l2.cpu_side_port, system.roofline_l2.mem_side_port
            )
        return system

    def bounded_run(args):
        output = Path(m5.options.outdir)
        check_config(
            registration["reference_ini"], output, Path(row["inputs"]), row["monitored"]
        )
        start = int(m5.curTick())
        if start != registration["restored_tick"]:
            raise ValueError("Unexpected restored tick")
        if check:
            save(
                output / "initialization-check.json",
                dict(passed=True, simulate_called=False, guest_instructions_executed=0),
            )
            return
        save(
            output / "simulation-start.json",
            dict(
                start_tick=start,
                max_ticks=row["max_ticks"],
                registration_sha256=registration_sha,
            ),
        )
        event = m5.simulate(row["max_ticks"])
        result = dict(
            start_tick=start,
            end_tick=int(m5.curTick()),
            cause=event.getCause(),
            code=event.getCode(),
        )
        save(output / "simulation-exit.json", result)
        if result["cause"] != "m5_exit instruction encountered" or result["code"] != 0:
            raise RuntimeError("Simulation did not complete normally: " + str(result))

    original.create, original.run = instrumented_create, bounded_run
    original.main()


if __name__ == "__m5_main__":
    main()
