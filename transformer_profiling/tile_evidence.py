"""Embed verified historical tile evidence into the existing offline report."""

import hashlib
import json
from pathlib import Path


def compact_payload(data):
    """Remove repeated object keys without browser codecs or network requests."""
    points = list(data["points"].values())
    point_fields = [key for key in points[0] if key not in ("id", "phases")]
    first_phase = next(iter(points[0]["phases"].values()))
    phase_fields = list(first_phase)
    rows = []
    for point in points:
        if set(point) != set(point_fields) | {"id", "phases"}:
            return data
        phases = []
        for name, values in point["phases"].items():
            if set(values) != set(phase_fields):
                return data
            phases.append([name, [values[key] for key in phase_fields]])
        rows.append([point["id"], [point[key] for key in point_fields], phases])
    return {
        "encoding": "columnar-v1",
        "metadata": {key: value for key, value in data.items() if key != "points"},
        "point_fields": point_fields,
        "phase_fields": phase_fields,
        "rows": rows,
    }


def render(path):
    """Embed one verified snapshot, with shared field names to keep offline HTML small."""
    path = Path(path)
    content = path.read_bytes()
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if (
        not manifest.get("passed")
        or hashlib.sha256(content).hexdigest() != manifest["data_sha256"]
    ):
        raise ValueError("Tile evidence does not match its verification manifest")
    data = json.loads(content)
    if data.get("schema") != "i2ce.false-grid-v9.v1" or data["counts"]["grid"] != 1728:
        raise ValueError("Unsupported or incomplete tile evidence")
    payload = json.dumps(compact_payload(data), separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c")
    assets = Path(__file__).with_name("tile_evidence")
    return (
        assets.with_suffix(".html").read_text()
        + '<script type="application/json" id="tile-evidence-data">'
        + payload
        + "</script>"
        + "<script>"
        + assets.with_suffix(".js").read_text()
        + "</script>"
    )
