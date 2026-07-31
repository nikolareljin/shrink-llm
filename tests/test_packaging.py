"""Tests for packaging invariants that are easy to break and expensive to notice."""

from __future__ import annotations

import pytest

from tests.conftest import REPO_ROOT


class TestDatasetsIsNotAPackage:
    """The repo's datasets/ directory holds data, not code.

    With an __init__.py it became a regular package on the repo root's sys.path entry, which
    shadowed the HuggingFace `datasets` dependency for anything run from the repo root — and
    that is exactly where run_pipeline.py invokes every stage, since it spawns them as
    `scripts/<name>.py`. `import datasets` then returned an empty stub with no load_dataset.
    """

    def test_datasets_dir_has_no_init(self):
        assert not (REPO_ROOT / "datasets" / "__init__.py").exists(), (
            "datasets/__init__.py makes the data directory a package that shadows the "
            "HuggingFace `datasets` dependency."
        )

    def test_datasets_is_not_a_declared_package(self, pyproject):
        include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
        assert not any(entry.startswith("datasets") for entry in include), include

    def test_huggingface_datasets_is_importable_from_the_repo_root(self):
        datasets = pytest.importorskip("datasets")
        assert hasattr(datasets, "load_dataset"), (
            f"`import datasets` resolved to {datasets.__file__}, which is not the "
            "HuggingFace library."
        )


class TestDeclaredDependencies:
    def test_every_declared_package_dir_exists(self, pyproject):
        include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
        for entry in include:
            name = entry.rstrip("*")
            assert (REPO_ROOT / name).is_dir(), f"packaged directory {name!r} does not exist"

    def test_tflite_extra_uses_the_converter_the_code_prefers(self, pyproject):
        """convert_to_tflite.py tries onnx2tf first; onnx-tf 1.10 requires tensorflow-addons,
        archived in May 2024 and capped at TF 2.14, so it cannot satisfy the TF floor here."""
        extra = " ".join(pyproject["project"]["optional-dependencies"]["tflite"])
        assert "onnx2tf" in extra
        assert "onnx-tf" not in extra.replace("onnx2tf", "")

    def test_tomli_is_available_on_the_python_floor(self, pyproject):
        """The test suite parses pyproject.toml, and tomllib is stdlib only from 3.11.

        requires-python allows 3.10 and CI runs a 3.10 leg, so the dev extra must supply tomli
        there or the entire suite fails at collection time.
        """
        dev = " ".join(pyproject["project"]["optional-dependencies"]["dev"])
        assert "tomli" in dev, dev
        assert 'python_version < "3.11"' in dev, "tomli must carry a version marker"


class TestLintCoversPackagedCode:
    def test_ci_lints_every_packaged_directory(self, pyproject):
        """A packaged directory that CI never lints accumulates unchecked code."""
        include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
        packaged = {entry.rstrip("*") for entry in include}

        for workflow in ("ci.yml", "pr-gate.yml"):
            content = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
            lint_lines = [ln for ln in content.splitlines() if "ruff check" in ln]
            assert lint_lines, f"{workflow} has no ruff invocation"
            covered = " ".join(lint_lines)
            missing = sorted(d for d in packaged if f"{d}/" not in covered)
            assert not missing, f"{workflow} does not lint packaged dir(s): {missing}"
