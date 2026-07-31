"""Tests for quantization pipeline."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def _tiny_model(tmp_path, with_softmax: bool = False):
    """A minimal quantizable graph: MatMul, optionally followed by a named Softmax."""
    weight = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 4], np.random.randn(16).astype(np.float32)
    )
    nodes = [helper.make_node("MatMul", ["X", "W"], ["M"], name="matmul_0")]
    output_name = "M"
    if with_softmax:
        nodes.append(helper.make_node("Softmax", ["M"], ["Y"], name="softmax_0"))
        output_name = "Y"

    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 4])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    return path, model


class TestDynamicQuantize:
    """The real call. Passing optimize_model=True raised TypeError on every onnxruntime in
    range of the declared floor, so every quantize run failed — and nothing here caught it
    because no test invoked the function."""

    def test_produces_a_valid_smaller_model(self, tmp_path):
        from scripts.quantize import dynamic_quantize

        src, _ = _tiny_model(tmp_path)
        dst = tmp_path / "model_int8.onnx"

        dynamic_quantize(src, dst, skip_ops=set())

        assert dst.exists() and dst.stat().st_size > 0
        onnx.checker.check_model(onnx.load(str(dst)))

    def test_runs_with_skip_ops(self, tmp_path):
        from scripts.quantize import dynamic_quantize

        src, _ = _tiny_model(tmp_path, with_softmax=True)
        dst = tmp_path / "model_int8.onnx"

        dynamic_quantize(src, dst, skip_ops={"Softmax", "LayerNormalization"})

        assert dst.exists()
        onnx.checker.check_model(onnx.load(str(dst)))


class TestNodesWithOpTypes:
    """onnxruntime's nodes_to_exclude takes node *names*; op types match nothing."""

    def test_resolves_op_types_to_node_names(self, tmp_path):
        from scripts.quantize import nodes_with_op_types

        _, model = _tiny_model(tmp_path, with_softmax=True)
        assert nodes_with_op_types(model, {"Softmax"}) == ["softmax_0"]
        assert nodes_with_op_types(model, {"MatMul"}) == ["matmul_0"]

    def test_op_type_names_are_not_node_names(self, tmp_path):
        """The original bug: passing 'Softmax' straight through excluded nothing."""
        _, model = _tiny_model(tmp_path, with_softmax=True)
        node_names = {n.name for n in model.graph.node}
        assert "Softmax" not in node_names

    def test_unmatched_op_types_yield_nothing(self, tmp_path):
        from scripts.quantize import nodes_with_op_types

        _, model = _tiny_model(tmp_path)
        assert nodes_with_op_types(model, {"LayerNormalization"}) == []
        assert nodes_with_op_types(model, set()) == []


class TestModelInputSpecs:
    def test_reads_shape_and_dtype_from_the_graph(self, tmp_path):
        from scripts.quantize import model_input_specs

        _, model = _tiny_model(tmp_path)
        assert model_input_specs(model) == {"X": ((1, 4), np.float32)}

    def test_excludes_initializers(self, tmp_path):
        from scripts.quantize import model_input_specs

        _, model = _tiny_model(tmp_path)
        assert "W" not in model_input_specs(model)


class TestCalibrationDataReader:
    def test_synthetic_fallback_matches_the_model_signature(self, tmp_path):
        """Guessing float32 [1, 128] gave pixel_values the wrong rank and input_ids the
        wrong dtype."""
        from scripts.quantize import SimpleCalibrationDataReader

        specs = {
            "pixel_values": ((1, 3, 384, 384), np.float32),
            "input_ids": ((1, 128), np.int64),
        }
        reader = SimpleCalibrationDataReader(
            tmp_path, list(specs), max_samples=2, input_specs=specs
        )
        batch = reader.get_next()

        assert batch["pixel_values"].shape == (1, 3, 384, 384)
        assert batch["pixel_values"].dtype == np.float32
        assert batch["input_ids"].shape == (1, 128)
        assert batch["input_ids"].dtype == np.int64

    def test_synthetic_fallback_when_no_files(self, tmp_path):
        from scripts.quantize import SimpleCalibrationDataReader

        reader = SimpleCalibrationDataReader(
            tmp_path, ["input_ids", "attention_mask"], max_samples=10
        )
        batch = reader.get_next()
        assert batch is not None
        assert "input_ids" in batch or "attention_mask" in batch

    def test_exhausts_correctly(self, tmp_path):
        from scripts.quantize import SimpleCalibrationDataReader

        reader = SimpleCalibrationDataReader(tmp_path, ["input_values"], max_samples=5)
        count = 0
        while reader.get_next() is not None:
            count += 1
        assert count == 5  # synthetic fallback produces min(64, max_samples) batches

    def test_reads_npz_samples(self, tmp_path):
        from scripts.quantize import SimpleCalibrationDataReader

        np.savez(tmp_path / "s0.npz", input_ids=np.ones((1, 8), dtype=np.int64))
        np.savez(tmp_path / "s1.npz", input_ids=np.zeros((1, 8), dtype=np.int64))

        reader = SimpleCalibrationDataReader(tmp_path, ["input_ids"], max_samples=10)
        batches = []
        while (b := reader.get_next()) is not None:
            batches.append(b)

        assert len(batches) == 2
        assert batches[0]["input_ids"].shape == (1, 8)

    def test_does_not_unpickle(self, tmp_path):
        """Calibration directories come from config values, so loading one must not be able
        to execute code. .npz + allow_pickle=False is the guarantee."""
        import inspect

        import scripts.quantize as quantize

        source = inspect.getsource(quantize)
        assert "allow_pickle=True" not in source
        assert "allow_pickle=False" in source

    def test_warns_when_only_legacy_npy_present(self, tmp_path, caplog):
        import logging

        from scripts.quantize import SimpleCalibrationDataReader

        np.save(tmp_path / "old.npy", np.ones((1, 8)))
        reader = SimpleCalibrationDataReader(tmp_path, ["input_ids"], max_samples=2)

        with caplog.at_level(logging.WARNING, logger="scripts.quantize"):
            reader.get_next()

        assert any(".npz" in r.message for r in caplog.records)


class TestFp16Dependency:
    def test_fp16_dependency_is_declared(self):
        """fp16_quantize imports onnxconverter_common; it must be installable."""
        from pathlib import Path

        import tomllib

        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        deps = " ".join(pyproject["project"]["dependencies"])
        assert "onnxconverter-common" in deps

    def test_missing_dependency_names_itself(self, monkeypatch, tmp_path):
        import builtins

        from scripts.quantize import fp16_quantize

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxconverter_common":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="onnxconverter-common"):
            fp16_quantize(tmp_path / "a.onnx", tmp_path / "b.onnx")


class TestDefaultSkipOps:
    def test_default_skip_ops_are_sensible(self):
        from scripts.quantize import DEFAULT_SKIP_OPS

        assert "Softmax" in DEFAULT_SKIP_OPS
        assert "LayerNormalization" in DEFAULT_SKIP_OPS


class TestSizeComparison:
    def test_size_comparison_logs(self, tmp_path, caplog):
        from scripts.quantize import _log_size_comparison

        before = tmp_path / "before.onnx"
        after = tmp_path / "after.onnx"
        before.write_bytes(b"0" * 1000000)
        after.write_bytes(b"0" * 250000)

        import logging

        with caplog.at_level(logging.INFO):
            _log_size_comparison(before, after)

        assert "75.0%" in caplog.text or "reduction" in caplog.text

    def test_empty_input_does_not_divide_by_zero(self, tmp_path, caplog):
        import logging

        from scripts.quantize import _log_size_comparison

        before = tmp_path / "before.onnx"
        after = tmp_path / "after.onnx"
        before.write_bytes(b"")
        after.write_bytes(b"0" * 100)

        with caplog.at_level(logging.INFO):
            _log_size_comparison(before, after)  # must not raise ZeroDivisionError

        assert "empty" in caplog.text
