import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/exp/prepare_stage.py"
SPEC = importlib.util.spec_from_file_location("prepare_stage", MODULE_PATH)
stage_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_tool)


class PrepareStageTests(unittest.TestCase):
    def test_rejects_path_escape(self):
        for name in ["", "/tmp/input", "../input", "a/../../input"]:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                stage_tool.relative_path(name)

    def test_accepts_nested_source_path(self):
        self.assertEqual(
            stage_tool.relative_path("Full_NN/src/gemm_exec.c"),
            Path("Full_NN/src/gemm_exec.c"),
        )

    def test_rejects_source_hash_mismatch(self):
        with mock.patch.object(stage_tool, "git_bytes", return_value=b"changed"):
            with self.assertRaisesRegex(RuntimeError, "Source hash mismatch"):
                stage_tool.checked_bytes("revision", "source.c", "0" * 64)

    def test_unknown_stage_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for stage in ("E11", "E12", "E13", "E99"):
                with self.subTest(stage=stage), self.assertRaises(RuntimeError):
                    stage_tool.prepare(stage, "HEAD", output)
                self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "keep"
            sentinel.write_bytes(b"original")
            with self.assertRaisesRegex(RuntimeError, "Output already exists"):
                stage_tool.prepare("E01", "HEAD", output)
            self.assertEqual(sentinel.read_bytes(), b"original")

    def fixture(self, directory):
        root = Path(directory)
        profiles = root / "transformer_profiling"
        inputs = {
            "headers/config.h": b"canonical dimensions\n",
            "weights/input.bin": b"fixed weights",
        }
        inputs_manifest = json.dumps(
            {"files": {name: stage_tool.sha256(data) for name, data in inputs.items()}}
        ).encode()
        catalog, snapshots, revisions = [], {}, {}
        for stage, marker in (("E09", "9"), ("E10", "a")):
            source = {
                "attention.cc": ("implementation " + stage).encode(),
                "Full_NN/gemm_definitions/config.h": b"default dimensions\n",
            }
            manifest = {
                "stage": stage,
                "source_files": {
                    name: stage_tool.sha256(data) for name, data in source.items()
                },
                "binary_sha256": marker * 64,
                "scientific_inputs_manifest_sha256": stage_tool.sha256(inputs_manifest),
            }
            folder = profiles / "experiments" / stage
            folder.mkdir(parents=True)
            (folder / "source.json").write_text(json.dumps(manifest))
            catalog.append({"id": stage, "binary_sha256": marker * 64})
            revisions[stage] = marker * 40
            # Model immutable source revisions without the renamed manifests.
            snapshots[marker * 40] = {
                **source,
                "transformer_profiling/experiments/"
                + stage
                + "/source.json": json.dumps(manifest).encode(),
                "transformer_profiling/inputs/manifest.json": inputs_manifest,
                **{
                    "transformer_profiling/inputs/workload/" + name: data
                    for name, data in inputs.items()
                },
            }
        (profiles / "sequence.json").write_text(json.dumps({"experiments": catalog}))
        return root, revisions, snapshots

    def test_public_attention_stages_use_matching_tag_manifests_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root, revisions, snapshots = self.fixture(directory)
            for stage in ("E09", "E10"):
                commit = revisions[stage]
                output = root / (stage + "-prepared")
                with (
                    self.subTest(stage=stage),
                    mock.patch.object(stage_tool, "ROOT", root),
                    mock.patch.object(
                        stage_tool.subprocess, "check_output", return_value=commit
                    ),
                    mock.patch.object(
                        stage_tool,
                        "git_bytes",
                        side_effect=lambda revision, name: snapshots[revision][name],
                    ),
                ):
                    source, manifest, record = stage_tool.prepare(stage, stage, output)
                self.assertEqual(
                    (source / "attention.cc").read_bytes(),
                    snapshots[commit]["attention.cc"],
                )
                self.assertEqual(
                    (source / "Full_NN/gemm_definitions/config.h").read_bytes(),
                    b"canonical dimensions\n",
                )
                self.assertEqual(
                    (source / "weights/input.bin").read_bytes(), b"fixed weights"
                )
                self.assertEqual(record["stage"], stage)
                self.assertEqual(record["public_commit"], commit)
                self.assertEqual(record["reference_binary_sha256"], commit[0] * 64)
                self.assertEqual(manifest["binary_sha256"], commit[0] * 64)
                self.assertFalse(record["binary_built"])
                self.assertFalse(record["binary_executed"])
                self.assertEqual(
                    json.loads((output / "prepared.json").read_text()), record
                )

    def test_wrong_stage_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, revisions, snapshots = self.fixture(directory)
            with (
                mock.patch.object(stage_tool, "ROOT", root),
                mock.patch.object(
                    stage_tool.subprocess, "check_output", return_value=revisions["E09"]
                ),
                mock.patch.object(
                    stage_tool,
                    "git_bytes",
                    side_effect=lambda revision, name: snapshots[revision][name],
                ),
                self.assertRaises(KeyError),
            ):
                stage_tool.prepare("E10", "E09", root / "wrong-stage")

    def test_changed_input_manifest_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, revisions, snapshots = self.fixture(directory)
            snapshots[revisions["E10"]][
                "transformer_profiling/inputs/manifest.json"
            ] += b"\n"
            output = root / "changed-input"
            with (
                mock.patch.object(stage_tool, "ROOT", root),
                mock.patch.object(
                    stage_tool.subprocess, "check_output", return_value=revisions["E10"]
                ),
                mock.patch.object(
                    stage_tool,
                    "git_bytes",
                    side_effect=lambda revision, name: snapshots[revision][name],
                ),
                self.assertRaisesRegex(RuntimeError, "Source hash mismatch"),
            ):
                stage_tool.prepare("E10", "E10", output)
            self.assertFalse(output.exists())

    def test_manifest_stage_and_binary_must_match_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _ = self.fixture(directory)
            path = root / "transformer_profiling/experiments/E10/source.json"
            original = json.loads(path.read_text())
            for change, error in (
                ({"stage": "E09"}, "Stage manifest mismatch"),
                ({"binary_sha256": "0" * 64}, "Stage binary identity mismatch"),
            ):
                path.write_text(json.dumps(dict(original, **change)))
                with (
                    self.subTest(change=change),
                    mock.patch.object(stage_tool, "ROOT", root),
                    self.assertRaisesRegex(RuntimeError, error),
                ):
                    stage_tool.load_stage_manifest("E10")

    def test_changed_sources_do_not_inherit_a_measured_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root, revisions, snapshots = self.fixture(directory)
            name = "transformer_profiling/experiments/E10/source.json"
            manifest = json.loads((root / name).read_text())
            manifest["reference_measurement"] = {
                "binary_sha256": manifest["binary_sha256"]
            }
            manifest["binary_sha256"] = None
            (root / name).write_text(json.dumps(manifest))
            snapshots[revisions["E10"]][name] = json.dumps(manifest).encode()
            with (
                mock.patch.object(stage_tool, "ROOT", root),
                mock.patch.object(
                    stage_tool.subprocess, "check_output", return_value=revisions["E10"]
                ),
                mock.patch.object(
                    stage_tool,
                    "git_bytes",
                    side_effect=lambda revision, name: snapshots[revision][name],
                ),
            ):
                _, _, record = stage_tool.prepare("E10", "E10", root / "prepared")
            self.assertIsNone(record["expected_binary_sha256"])
            self.assertEqual(record["reference_binary_sha256"], "a" * 64)
            self.assertEqual(record["measurement_status"], "not_measured")

    def test_cli_defaults_to_public_tag_and_honors_explicit_revision(self):
        for stage, revision in (("E09", None), ("E10", None), ("E10", "candidate")):
            arguments = ["prepare_stage.py", "--stage", stage, "--output", "unused"]
            if revision:
                arguments += ["--revision", revision]
            with (
                self.subTest(stage=stage, revision=revision),
                mock.patch("sys.argv", arguments),
                mock.patch("sys.stdout", new_callable=io.StringIO),
                mock.patch.object(
                    stage_tool, "prepare", return_value=(Path("source"), {}, {})
                ) as prepare,
                mock.patch.object(stage_tool, "build") as build,
            ):
                stage_tool.main()
                prepare.assert_called_once_with(
                    stage, revision or stage, Path("unused").resolve()
                )
                build.assert_not_called()

    def test_unmeasured_build_records_hash_without_claiming_old_binary_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "include/gem5"
            include.mkdir(parents=True)
            paths = {
                "A64CXX": root / "compiler",
                "LIBM5_A": root / "libm5.a",
                "M5_INC": include.parent,
            }
            for path in [paths["A64CXX"], paths["LIBM5_A"], include / "m5ops.h"]:
                path.write_bytes(b"registered tool fixture")
            manifest = dict(
                binary_sha256=None,
                build_environment={"SIMD_FLAG": "1"},
                tool_hashes={
                    key: stage_tool.sha256(b"registered tool fixture")
                    for key in ["compiler", "libm5", "m5_header"]
                },
            )
            source = root / "source"
            source.mkdir()
            (source / "transformer.o").write_bytes(b"new unmeasured binary")
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        **{key: str(path) for key, path in paths.items()},
                        "USE_FP32_TRANSFORMER_FLAG": "1",
                        "SIMD_FLAG": "0",
                    },
                ),
                mock.patch.object(
                    stage_tool.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ) as run,
            ):
                stage_tool.build(
                    source,
                    manifest,
                    {"reference_binary_sha256": "a" * 64, "formal_measurements": 0},
                    root,
                )
            build_env = run.call_args.kwargs["env"]
            self.assertNotIn("USE_FP32_TRANSFORMER_FLAG", build_env)
            self.assertEqual(build_env["SIMD_FLAG"], "1")
            record = json.loads((root / "build-result.json").read_text())
            self.assertEqual(
                record["binary_sha256"], stage_tool.sha256(b"new unmeasured binary")
            )
            self.assertIsNone(record["matches_expected_binary"])
            self.assertEqual(record["validation_status"], "not_validated")
            self.assertEqual(record["formal_measurements"], 0)


if __name__ == "__main__":
    unittest.main()
