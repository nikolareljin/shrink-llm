# ShrinkLLM — Roadmap

## Phase 1: Model Selection + ONNX Export (Week 1–2)

- [ ] Set up repository structure and dependencies
- [ ] Implement `export_to_onnx.py` for encoder-only, seq2seq, audio models
- [ ] Validate ONNX export for TrOCR-base, Phi-3-mini, AST
- [ ] Set up CI pipeline (lint, test)
- [ ] Document model selection rationale in `docs/architecture.md`

**Exit criteria**: All three teacher models export cleanly to ONNX opset 17.

---

## Phase 2: Quantization + Pruning (Week 3–5)

- [ ] Implement dynamic INT8 quantization
- [ ] Implement static INT8 quantization with calibration
- [ ] Implement GPTQ INT4 quantization for LLMs
- [ ] Implement attention head pruning
- [ ] Implement MLP neuron pruning
- [ ] Implement layer dropping
- [ ] Validate: size reduction ≥ 70%, accuracy drop ≤ 3%

**Exit criteria**: OCR student < 100 MB with CER ≤ 5%.

---

## Phase 3: Knowledge Distillation + Fine-tuning (Week 6–9)

- [ ] Implement `DistillationLoss` (CE + KL + MSE)
- [ ] Implement `DistillationTrainer` (HuggingFace Trainer subclass)
- [ ] Train OCR student on IIIT-5K + IAM
- [ ] Train legal student on CUAD + ContractNLI
- [ ] Train audio student on Donate-a-cry corpus
- [ ] Post-distillation fine-tuning with QAT where needed

**Exit criteria**: Student reaches ≥ 90% of teacher accuracy on eval sets.

---

## Phase 4: Mobile Conversion (Week 10–12)

- [ ] Implement `convert_to_tflite.py`
- [ ] Implement `convert_to_coreml.py`
- [ ] Implement `convert_to_onnx_mobile.py`
- [ ] Package models for Android (TFLite + ORT) and iOS (CoreML)
- [ ] Write Android and iOS integration guides

**Exit criteria**: All three task models run on-device without crashes.

---

## Phase 5: Benchmarking + Optimization (Week 13–15)

- [ ] Implement full benchmark suite (accuracy + latency + memory)
- [ ] Run benchmarks on target devices (Android mid-range, iPhone 12+)
- [ ] Profile bottlenecks and iterate on compression settings
- [ ] Publish benchmark results in `benchmarks/results/`

**Exit criteria**: All targets in `success_criteria` met for each task.

---

## Phase 6: v1.0 Release (Week 16)

- [ ] Final documentation pass
- [ ] GitHub release with model artifacts (if publishable)
- [ ] Tag v1.0.0
- [ ] Write blog post / README showcase

---

## Future Phases

- **v1.1**: Add support for more tasks (sentiment analysis, image segmentation)
- **v1.2**: GGUF/llama.cpp pipeline integration
- **v1.3**: Automatic pipeline config search (NAS-style compression)
- **v2.0**: Web UI for no-code model compression
