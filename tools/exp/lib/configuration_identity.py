"""Compare instantiated machines while normalizing only their output directories."""

import configparser
import json
import os
from pathlib import Path

from configuration_snapshot import verify_bootloaders, verify_diod
from experiment_results import CapturedFile


# The structured runner supports one fixed machine, with variable L1D/L2 sizes.
# Clock and DRAM timing values in config.ini are ticks at this explicit frequency.
TICKS_PER_SECOND = 10**12
HARDWARE_OPTIONS = (
    "--cpu-freq=4GHz",
    "--mem-size=2GiB",
    "--mem-type=DDR3_1600_8x8",
    "--mem-channels=1",
)
MACHINE_OPTIONS = (
    "--cpu=minor",
    "--num-cores=1",
    *HARDWARE_OPTIONS,
    "--sve-vl=1",
    "--l1i-size=49152B",
)
DRAM_PARAMETERS = {
    "type": "DRAMInterface",
    "device_size": "536870912",
    "device_bus_width": "8",
    "burst_length": "8",
    "device_rowbuffer_size": "1024",
    "devices_per_rank": "8",
    "ranks_per_channel": "2",
    "banks_per_rank": "8",
    "tck": "1250",
    "tburst": "5000",
    "trcd": "13750",
    "tcl": "13750",
    "trp": "13750",
    "tras": "35000",
    "trrd": "6000",
    "txaw": "30000",
    "activation_limit": "4",
    "trfc": "260000",
    "twr": "15000",
    "twtr": "7500",
    "trtp": "7500",
    "trtw": "2500",
    "tcs": "2500",
    "trefi": "7800000",
    "txp": "6000",
    "txs": "270000",
}
# Linux sockaddr_un.sun_path has 108 bytes, including its terminating NUL.
MAX_SOCKET_PATH_BYTES = 107


def socket_path(output):
    """Check the pathname gem5 will bind, using bytes rather than characters."""
    path = Path(output).resolve() / "9p.sock"
    length = len(os.fsencode(path))
    if length > MAX_SOCKET_PATH_BYTES:
        raise ValueError(
            f"9P socket path is {length} bytes; Linux allows at most "
            f"{MAX_SOCKET_PATH_BYTES}. Shorten EXP_ROOT or the experiment ID: {path}"
        )
    return str(path)


def configuration_file(source):
    """Accept an existing byte snapshot or capture a live configuration once."""
    return source if isinstance(source, CapturedFile) else CapturedFile.read(source)


def read_configuration(source):
    """Parse captured bytes without interpolation or another path read."""
    captured = configuration_file(source)
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(captured.data.decode("utf-8"), source=str(captured.path))
    return config


def require_fields(config, section, expected):
    """Report the exact instantiated field that violates the fixed contract."""
    if section not in config:
        raise ValueError("Missing actual machine section: " + section)
    for field, value in expected.items():
        if config[section].get(field) != str(value):
            raise ValueError(
                f"Actual machine {section}.{field} must be {value}; "
                f"got {config[section].get(field)!r}"
            )


def single_section(candidates, label):
    """Require one concrete object rather than accepting an empty/extra set."""
    names = list(candidates)
    if len(names) != 1:
        raise ValueError("Expected exactly one actual " + label)
    return names[0]


def verify_machine_configuration(path, parameters):
    """Check instantiated hardware, independently of preflight/formal equality.

    The supported ARM starter exposes a source CPU clock and one MemCtrl/DRAM
    channel. Reject other layouts explicitly; do not infer them from CLI flags.
    DDR3 geometry and timings are checked because its serialized type is the
    generic DRAMInterface, not the Python preset name DDR3_1600_8x8.
    """
    config = read_configuration(path)
    require_fields(config, "root", {"full_system": "true"})
    require_fields(
        config,
        "system",
        {
            "type": "ArmSystem",
            "mem_mode": "timing",
            "sve_vl": "1",
            "cache_line_size": "64",
            "mem_ranges": "2147483648:4294967296",
        },
    )
    cpu = single_section((s for s in config if "cpu_id" in config[s]), "CPU")
    require_fields(
        config,
        cpu,
        {
            "type": "BaseMinorCPU",
            "cpu_id": "0",
            "numthreads": "1",
            "switched_out": "false",
        },
    )
    clock = config[cpu].get("clk_domain", "")
    require_fields(config, clock, {"type": "SrcClockDomain", "clock": "250"})
    require_fields(config, cpu + ".icache", {"type": "Cache", "size": "49152"})
    require_fields(
        config,
        cpu + ".dcache",
        {
            "type": "Cache",
            "size": str(parameters["l1d_size"]),
            "writeback_clean": "false",
        },
    )
    outer = single_section((s for s in config if s.endswith(".l2")), "L2")
    require_fields(config, outer, {"type": "Cache", "size": str(parameters["l2_size"])})
    controller = single_section(
        (s for s in config if config[s].get("type") == "MemCtrl"),
        "memory channel",
    )
    dram = config[controller].get("dram", "")
    require_fields(config, dram, dict(DRAM_PARAMETERS, range="2147483648:4294967296"))
    return config


def bind_diod(system, record, share, output):
    """Pin the real 9P object's executable before gem5 instantiates the system."""
    helper = verify_diod(record)
    devices = [
        obj
        for obj in system.descendants()
        if getattr(obj, "type", None) == "VirtIO9PDiod"
    ]
    if len(devices) != 1 or str(devices[0].root) != share:
        raise ValueError("Expected one 9P device exporting the frozen private share")
    devices[0].diod = helper
    if str(devices[0].socketPath) != socket_path(output):
        raise ValueError("Constructed 9P socket differs from the planned path")


def verify_run_configuration(path, plan):
    """Apply the same hardware and bound-resource contract at every boundary."""
    path = configuration_file(path)
    config = verify_machine_configuration(path, plan["parameters"])
    verify_bootloader_configuration(path, plan["identities"].get("bootloaders"))
    helper = verify_diod(plan["identities"].get("diod"))
    device = single_section(
        (s for s in config if config[s].get("type") == "VirtIO9PDiod"),
        "9P device",
    )
    require_fields(
        config,
        device,
        {
            "diod": helper,
            "root": plan["paths"]["share"],
            "socketpath": socket_path(path.path.parent),
        },
    )
    return {
        "l1d_sections": [section for section in config if section.endswith(".dcache")],
        "config_sha256": path.identity["sha256"],
        "writeback_clean": False,
        "uncached_boot": False,
    }


def verify_captured_policy(config, policy, simulated, plan=None):
    """Validate policy and machine from the exact bytes included in the seal."""
    gate = json.loads(policy.data)
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or gate.get("simulate_called") is not simulated
        or gate.get("config_sha256") != config.identity["sha256"]
    ):
        raise ValueError("Invalid or changed configuration-only/formal policy receipt")
    if plan is not None:
        if gate.get("tick_frequency") != TICKS_PER_SECOND:
            raise ValueError("Actual machine tick frequency must be 1 THz")
        return verify_run_configuration(config, plan)
    return gate


def verify_bootloader_configuration(path, snapshot):
    """Bind actual workload loader paths to the content-verified private pair."""
    expected = " ".join(verify_bootloaders(snapshot))
    config = read_configuration(path)
    actual = [
        config[section]["boot_loader"]
        for section in config
        if "boot_loader" in config[section]
    ]
    if actual != [expected]:
        raise ValueError("Actual ARM bootloaders differ from the frozen snapshot")


def machine_configuration(path):
    """Retain every section and value; stage-local output paths share one token."""
    path = configuration_file(path)
    config = read_configuration(path)
    output = str(path.path.parent.resolve()) + "/"
    return {
        section: {
            key: value.replace(output, "<stage-output>/")
            for key, value in config[section].items()
        }
        for section in config.sections()
    }


def capture_stage_configuration(directory, prefix):
    """Capture a completed stage's configuration and policy as one pair."""
    return {
        prefix + "_config": CapturedFile.read(Path(directory) / "config.ini"),
        prefix + "_policy": CapturedFile.read(Path(directory) / "cache-policy.json"),
    }


def capture_configurations(check_directory, formal_directory, *, completed=True):
    """Enumerate configuration evidence; the live formal policy does not yet exist."""
    files = capture_stage_configuration(check_directory, "check")
    if completed:
        files.update(capture_stage_configuration(formal_directory, "formal"))
    else:
        files["formal_config"] = CapturedFile.read(
            Path(formal_directory) / "config.ini"
        )
    return files


def verify_matching_configuration(
    check_directory, formal_directory, plan=None, *, captured=None
):
    """Check preflight and the live formal machine, before a formal receipt exists."""
    files = (
        captured
        if captured is not None
        else capture_configurations(check_directory, formal_directory, completed=False)
    )
    verify_captured_policy(files["check_config"], files["check_policy"], False, plan)
    actual = None
    if plan is not None:
        actual = verify_run_configuration(files["formal_config"], plan)
    compare_captured_configurations(files["check_config"], files["formal_config"])
    for captured_file in files.values():
        captured_file.verify_unchanged()
    return actual


def compare_captured_configurations(checked, actual):
    """Compare every instantiated value, normalizing stage-local output paths only."""
    checked = machine_configuration(checked)
    actual = machine_configuration(actual)
    if checked != actual:
        raise ValueError("Formal machine differs from configuration-only gate")


def verify_completed_configuration(
    check_directory, formal_directory, plan, *, captured=None
):
    """Validate both completed stages once and return the formal configuration.

    Each policy receipt binds its config hash and tick frequency. Each machine
    independently satisfies the frozen contract before their values are compared.
    """
    files = (
        captured
        if captured is not None
        else capture_configurations(check_directory, formal_directory)
    )
    verify_captured_policy(files["check_config"], files["check_policy"], False, plan)
    actual = verify_captured_policy(
        files["formal_config"], files["formal_policy"], True, plan
    )
    compare_captured_configurations(files["check_config"], files["formal_config"])
    return actual
