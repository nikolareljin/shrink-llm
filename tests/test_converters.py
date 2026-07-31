"""Tests for the mobile converter scripts' failure modes.

These cover behaviour that only shows up when an optional dependency is absent or a model has
an unusual signature -- the cases least likely to be exercised by hand.
"""

from __future__ import annotations

import builtins
import inspect
import sys

import pytest


class TestCoreMLUnavailable:
    """coremltools removed ONNX input in 6.0; this project requires >=7.2."""

    def test_raises_not_implemented_when_coremltools_is_installed(self, tmp_path):
        pytest.importorskip("coremltools")
        from scripts.convert_to_coreml import convert_onnx_to_coreml

        with pytest.raises(NotImplementedError, match="does not accept ONNX input"):
            convert_onnx_to_coreml(tmp_path / "m.onnx", tmp_path / "out", "iOS16", "ALL", "none")

    def test_raises_not_implemented_when_coremltools_is_absent(self, tmp_path, monkeypatch):
        """The common case: no `coreml` extra installed.

        Importing coremltools before raising made this path report "install coremltools",
        sending the user to install a package that cannot do the job.
        """
        from scripts.convert_to_coreml import convert_onnx_to_coreml

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "coremltools":
                raise ImportError("no coremltools here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(NotImplementedError) as excinfo:
            convert_onnx_to_coreml(tmp_path / "m.onnx", tmp_path / "out", "iOS16", "ALL", "none")

        message = str(excinfo.value)
        assert "does not accept ONNX input" in message
        assert "SHRINK-020" in message

    def test_error_names_an_actionable_next_step(self, tmp_path, monkeypatch):
        from scripts.convert_to_coreml import convert_onnx_to_coreml

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "coremltools":
                raise ImportError("absent")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(NotImplementedError, match="stages"):
            convert_onnx_to_coreml(tmp_path / "m.onnx", tmp_path / "out", "iOS16", "ALL", "none")


class TestOnnx2TfImportGuard:
    def test_only_the_import_is_guarded(self):
        """An ImportError raised inside onnx2tf.convert must not be reported as
        "onnx2tf not installed" -- that discards the real cause and names the wrong fix."""
        from scripts.convert_to_tflite import convert_onnx_to_tf

        source = inspect.getsource(convert_onnx_to_tf)
        guard = source[source.index("try:") : source.index("except ImportError")]

        assert "import onnx2tf" in guard
        assert (
            "onnx2tf.convert(" not in guard
        ), "the conversion call must sit outside the ImportError guard"

    def test_fallback_error_points_at_the_extra_not_onnx_tf(self):
        from scripts.convert_to_tflite import convert_onnx_to_tf

        source = inspect.getsource(convert_onnx_to_tf)

        assert '".[tflite]"' in source
        assert "tensorflow-addons" in source, "explain why onnx-tf is not an alternative"


class TestSavedModelInputNames:
    """The representative dataset generator needs the model's inputs in order."""

    def _fake_signature(self, monkeypatch, args, kwargs):
        import scripts.convert_to_tflite as module

        class _Spec:
            def __init__(self, name):
                self.name = name

        class _Signature:
            structured_input_signature = (args, kwargs)

        class _Loaded:
            signatures = {"serving_default": _Signature()}

        class _Nest:
            @staticmethod
            def flatten(structure):
                return list(structure)

        class _FakeTF:
            saved_model = type("sm", (), {"load": staticmethod(lambda _p: _Loaded())})
            nest = _Nest()

        monkeypatch.setitem(sys.modules, "tensorflow", _FakeTF())
        return module, _Spec

    def test_prefers_keyword_inputs(self, monkeypatch, tmp_path):
        module, _ = self._fake_signature(monkeypatch, (), {"input_ids": object()})

        assert module._saved_model_input_names(tmp_path) == ["input_ids"]

    def test_falls_back_to_positional_inputs(self, monkeypatch, tmp_path):
        """Reading only kwargs returned [], and the generator then fed TFLite nothing."""

        class _Spec:
            def __init__(self, name):
                self.name = name

        module, _ = self._fake_signature(monkeypatch, (_Spec("pixel_values"),), {})

        assert module._saved_model_input_names(tmp_path) == ["pixel_values"]

    def test_raises_when_there_are_no_named_inputs(self, monkeypatch, tmp_path):
        module, _ = self._fake_signature(monkeypatch, (), {})

        with pytest.raises(ValueError, match="no named inputs"):
            module._saved_model_input_names(tmp_path)


class TestRepresentativeDatasetGenerator:
    def test_yields_tensors_in_the_models_input_order(self, tmp_path):
        import numpy as np

        from scripts.convert_to_tflite import _representative_dataset_gen

        np.savez(
            tmp_path / "s0.npz",
            attention_mask=np.ones((1, 4), dtype=np.int32),
            input_ids=np.full((1, 4), 7, dtype=np.int32),
        )

        gen = _representative_dataset_gen(tmp_path, ["input_ids", "attention_mask"])
        sample = next(iter(gen()))

        assert len(sample) == 2
        assert sample[0][0][0] == 7, "first tensor must be input_ids, matching the declared order"

    def test_missing_input_names_the_file_and_the_expectation(self, tmp_path):
        import numpy as np

        from scripts.convert_to_tflite import _representative_dataset_gen

        np.savez(tmp_path / "s0.npz", input_ids=np.ones((1, 4), dtype=np.int32))

        gen = _representative_dataset_gen(tmp_path, ["input_ids", "attention_mask"])

        with pytest.raises(ValueError, match="attention_mask"):
            list(gen())
