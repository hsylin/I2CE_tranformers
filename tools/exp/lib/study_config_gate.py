"""Require full hardware equality, allowing only the four host path fields."""

import configparser
from pathlib import Path
from study_records import require, digest, load, save


def read_ini(path):
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    with Path(path).open() as stream:
        parser.read_file(stream)
    return {section: dict(parser[section]) for section in parser.sections()}


def differences(expected, actual, path=""):
    if type(expected) is not type(actual):
        return [dict(path=path, expected=expected, actual=actual)]
    if isinstance(expected, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append(
                    dict(
                        path=path + "/" + key,
                        expected=expected.get(key),
                        actual=actual.get(key),
                    )
                )
            else:
                result.extend(differences(expected[key], actual[key], path + "/" + key))
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [
                dict(
                    path=path, expected_length=len(expected), actual_length=len(actual)
                )
            ]
        return [
            item
            for i, (a, b) in enumerate(zip(expected, actual))
            for item in differences(a, b, path + "/" + str(i))
        ]
    return (
        []
        if expected == actual
        else [dict(path=path, expected=expected, actual=actual)]
    )


def expected_config(root, inputs, output):
    paths = {
        ("system", "readfile"): str(inputs / "run.rcS"),
        ("system.pci_devices1.vio", "root"): str(inputs),
        ("system.pci_devices1.vio", "socketPath"): str(output / "9p.sock"),
        ("system.workload", "dtb_filename"): str(output / "system.dtb"),
    }
    ini = read_ini(root / "reference/config.ini")
    serialized = load(root / "reference/config.json")
    for (section, key), value in paths.items():
        require(key in ini[section], "Study check failed: key in ini[section]")
        ini[section][key] = value
    seen = set()

    def replace_fields(node):
        if isinstance(node, dict):
            for (section, key), value in paths.items():
                if node.get("path") == section:
                    require(
                        key in node and isinstance(node[key], str),
                        "Study check failed: key in node and isinstance(node[key], str)",
                    )
                    node[key] = value
                    seen.add((section, key))
            for value in node.values():
                replace_fields(value)
        elif isinstance(node, list):
            for value in node:
                replace_fields(value)

    replace_fields(serialized)
    require(seen == set(paths), "Study check failed: seen == set(paths)")
    return ini, serialized, paths


def verify(root, row, inputs, output):
    expected_ini, expected_json, allowed = expected_config(root, inputs, output)
    ini_diff = differences(expected_ini, read_ini(output / "config.ini"))
    json_diff = differences(expected_json, load(output / "config.json"))
    result = dict(
        passed=not ini_diff and not json_diff,
        stage=row["id"],
        allowed_host_paths=[
            dict(section=s, key=k, value=v) for (s, k), v in allowed.items()
        ],
        unexpected_ini=ini_diff,
        unexpected_json=json_diff,
        config_ini_sha256=digest(output / "config.ini"),
        config_json_sha256=digest(output / "config.json"),
    )
    save(output / "config-gate.json", result)
    require(
        result["passed"],
        "Actual hardware configuration differs; simulation is not entered",
    )
    return result
