"""Release boundaries and reproducibility, with no Blender or pip requirement."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
import zipfile


AUTHORING = Path(__file__).resolve().parents[1] / "authoring"
SPEC = importlib.util.spec_from_file_location("build_addon", AUTHORING / "build_addon.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class AddonReleaseTests(unittest.TestCase):
    def source_tree(self, directory):
        source = Path(directory) / "source"
        source.mkdir()
        for filename in BUILDER.REQUIRED_FILES.values():
            contents = b"# release fixture\n" if filename.endswith(".py") else b"release fixture\n"
            if filename == "blender_addon.py":
                contents = b'bl_info = {"version": (1, 24, 0)}\nraise RuntimeError("must not import")\n'
            (source / filename).write_bytes(contents)
        return source

    def test_archive_is_allowlisted_standalone_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source_tree(directory)
            (source / "requirements-addon-gurobi.txt").write_text("gurobipy\n", encoding="utf-8")
            for filename in ("token.json", "license.lic", "unrelated.py", "results.csv", ".env"):
                (source / filename).write_text("private local data", encoding="utf-8")
            output = BUILDER.build_release(source)
            self.assertEqual(output.name, "lightbender_swarm-1.24.0.zip")
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                names = archive.namelist()
                self.assertEqual(names, sorted(names))
                expected = set(BUILDER.REQUIRED_FILES) | {"requirements-addon-gurobi.txt", "release_manifest.json"}
                self.assertEqual(set(names), {f"lightbender_swarm/{name}" for name in expected})
                self.assertEqual(archive.read("lightbender_swarm/__init__.py"), (source / "blender_addon.py").read_bytes())
                manifest = json.loads(archive.read("lightbender_swarm/release_manifest.json"))
                self.assertEqual(manifest["version"], "1.24.0")
                self.assertEqual(set(manifest["files"]), expected - {"release_manifest.json"})
                for name, digest in manifest["files"].items():
                    self.assertEqual(digest, hashlib.sha256(archive.read(f"lightbender_swarm/{name}")).hexdigest())
                self.assertTrue(all(info.date_time == BUILDER.FIXED_ZIP_TIME for info in archive.infolist()))
            expected_digest = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(output.with_suffix(".zip.sha256").read_text().strip(), f"{expected_digest}  {output.name}")

    def test_identical_content_is_reproducible_after_metadata_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source_tree(directory)
            first = BUILDER.build_release(source, Path(directory) / "first.zip").read_bytes()
            for filename in BUILDER.REQUIRED_FILES.values():
                os.utime(source / filename, (1700000000, 1700000000))
            second = BUILDER.build_release(source, Path(directory) / "second.zip").read_bytes()
            self.assertEqual(first, second)

    def test_missing_runtime_fails_before_replacing_existing_release(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source_tree(directory)
            output = Path(directory) / "existing.zip"
            output.write_bytes(b"old release")
            (source / "addon_runtime.py").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "addon_runtime.py"):
                BUILDER.build_release(source, output)
            self.assertEqual(output.read_bytes(), b"old release")

    def test_invalid_python_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source_tree(directory)
            (source / "transform.py").write_text("def broken(\n", encoding="utf-8")
            with self.assertRaises(SyntaxError):
                BUILDER.build_release(source)

    def test_version_is_read_without_executing_source(self):
        self.assertEqual(BUILDER.addon_version('bl_info = {"version": (2, 4)}\nraise RuntimeError()'), "2.4.0")
        for version in ('"1.24.0"', "(1,)", "(1, True, 0)", "(-1, 2, 0)", "(1, 2, 0, 4)"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                BUILDER.addon_version(f'bl_info = {{"version": {version}}}')

    def test_real_release_contains_every_subprocess_entry_point(self):
        version, files = BUILDER.release_files(AUTHORING)
        self.assertTrue(version)
        script_names = {
            node.value for node in ast.walk(ast.parse(files["__init__.py"]))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", node.value)
        }
        # The flight orchestrator is an explicitly separate application.
        script_names.discard("orchestrator.py")
        self.assertTrue(script_names)
        for filename in script_names:
            self.assertTrue(filename in files or f"authoring/{filename}" in files, filename)


if __name__ == "__main__":
    unittest.main()
