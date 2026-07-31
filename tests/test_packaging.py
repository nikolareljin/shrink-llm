"""Tests for packaging invariants that are easy to break and expensive to notice."""

from __future__ import annotations

import pytest


class TestDatasetsIsNotAPackage:
    """The repo's datasets/ directory holds data, not code.

    With an __init__.py it became a regular package on the repo root's sys.path entry, which
    shadowed the HuggingFace `datasets` dependency for anything run from the repo root — and
    that is exactly where run_pipeline.py invokes every stage, since it spawns them as
    `scripts/<name>.py`. `import datasets` then returned an empty stub with no load_dataset.
    """

    def test_datasets_dir_has_no_init(self, repo_root):
        assert not (repo_root / "datasets" / "__init__.py").exists(), (
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
    def test_every_declared_package_dir_exists(self, pyproject, repo_root):
        include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
        for entry in include:
            name = entry.rstrip("*")
            assert (repo_root / name).is_dir(), f"packaged directory {name!r} does not exist"

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


class TestDocumentationMatchesReality:
    """Documented commands and layouts that silently drift from the code.

    The contributing guide previously told contributors to lint a narrower tree than CI does,
    so they passed locally and failed the gate -- exactly the failure the CI change closed.
    """

    def test_contributing_lint_command_matches_ci(self, repo_root, pyproject):
        contributing = (repo_root / "docs" / "contributing.md").read_text()
        packaged = {
            entry.rstrip("*")
            for entry in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
        }

        lint_lines = [ln for ln in contributing.splitlines() if "ruff check" in ln]
        assert lint_lines, "contributing.md documents no ruff command"
        covered = " ".join(lint_lines)

        missing = sorted(d for d in packaged if f"{d}/" not in covered)
        assert not missing, (
            f"docs/contributing.md lints a narrower tree than CI; missing: {missing}. "
            "A contributor following it passes locally and fails the gate."
        )

    def test_documented_python_versions_match_the_ci_matrix(self, repo_root):
        ci = (repo_root / ".github" / "workflows" / "ci.yml").read_text()
        contributing = (repo_root / "docs" / "contributing.md").read_text()

        for version in ("3.10", "3.11", "3.12"):
            assert version in ci, f"CI no longer covers {version}"
            assert (
                version in contributing
            ), f"docs/contributing.md does not mention {version}, which CI runs"

    def test_architecture_does_not_prescribe_a_datasets_package(self, repo_root):
        """The blueprint used to prescribe datasets/__init__.py -- the shadowing defect."""
        architecture = (repo_root / "docs" / "architecture.md").read_text()
        structure = architecture[
            architecture.index("## 3. REPOSITORY STRUCTURE") : architecture.index(
                "## 4. FULL COMPRESSION PIPELINE"
            )
        ]

        assert "datasets/\n│   ├── __init__.py" not in structure
        assert "not a Python package" in structure or "must not contain" in structure


class TestLintCoversPackagedCode:
    def test_ci_lints_every_packaged_directory(self, pyproject, repo_root):
        """A packaged directory that CI never lints accumulates unchecked code."""
        include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
        packaged = {entry.rstrip("*") for entry in include}

        for workflow in ("ci.yml", "pr-gate.yml"):
            content = (repo_root / ".github" / "workflows" / workflow).read_text()
            lint_lines = [ln for ln in content.splitlines() if "ruff check" in ln]
            assert lint_lines, f"{workflow} has no ruff invocation"
            covered = " ".join(lint_lines)
            missing = sorted(d for d in packaged if f"{d}/" not in covered)
            assert not missing, f"{workflow} does not lint packaged dir(s): {missing}"
