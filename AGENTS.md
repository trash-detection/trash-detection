# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes.
Merge with project-specific instructions as needed.

This repository focuses on:
- computer vision research
- YOLO-based object detection
- model compression
- edge deployment
- TensorRT / ONNX optimization
- reproducible experimentation
- AI-agent-assisted engineering

Prefer practical engineering over theoretical elegance.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- Push back against unnecessary complexity.
- If something is unclear, stop and ask.

For research tasks:
- Distinguish clearly between:
  - established facts
  - implementation guesses
  - experimental hypotheses
- Never present speculative optimization as guaranteed improvement.
- Benchmark claims must be measurable.

For deployment tasks:
- Clarify target hardware before optimizing.
- Distinguish:
  - training optimization
  - inference optimization
  - edge optimization

Do not silently optimize for the wrong constraint.

---

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was requested.
- No abstractions for single-use code.
- No premature modularization.
- No "future-proofing" unless requested.
- No configurable systems for fixed workflows.
- No wrappers around wrappers.
- Avoid magic helper utilities.

If 50 lines solve the problem correctly, do not write 200.

Prefer:
- explicit code
- shallow call stacks
- direct data flow
- readable tensor transformations

Avoid:
- hidden side effects
- unnecessary inheritance
- deeply nested abstractions
- framework worship

Simple systems are easier to:
- benchmark
- profile
- quantize
- prune
- export
- debug
- deploy

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Do not refactor unrelated code.
- Do not reformat unrelated files.
- Match existing project style.
- Do not rename symbols unnecessarily.
- Preserve public interfaces unless asked.

When your changes create unused code:
- Remove only the unused artifacts introduced by YOUR changes.

If unrelated problems are discovered:
- mention them
- do not silently fix them

Every changed line must trace directly to the task.

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform vague tasks into measurable goals.

Examples:
- "Optimize inference"
  →
  "Measure latency/FPS before and after"

- "Improve pruning"
  →
  "Compare mAP, params, FLOPs, latency"

- "Fix TensorRT export"
  →
  "Export successfully and verify inference parity"

For multi-step tasks:

1. [Step] → verify: [measurement]
2. [Step] → verify: [measurement]
3. [Step] → verify: [measurement]

Do not stop at:
- "code compiles"
- "runs on my machine"

Verification should include:
- correctness
- reproducibility
- deployment compatibility

---

## 5. Research Engineering Rules

**Research code must still behave like engineering code.**

Do not:
- hardcode experimental hacks silently
- mix unrelated experiments
- overwrite baseline logic
- hide experimental parameters

Prefer:
- isolated experiments
- reproducible configs
- explicit naming
- versioned checkpoints

Experiments should be:
- comparable
- measurable
- reversible

Ablations matter more than intuition.

Claims require evidence.

---

## 6. Edge AI & Deployment Mindset

**Deployment is part of the design, not an afterthought.**

Always consider:
- latency
- VRAM/RAM usage
- batch-size sensitivity
- export compatibility
- TensorRT support
- ONNX operator compatibility

Avoid operations known to:
- break TensorRT export
- introduce unsupported ONNX ops
- increase memory fragmentation
- create dynamic-shape instability

Inference performance matters more than theoretical elegance.

A smaller model that deploys reliably is often better than a larger "SOTA" model that cannot run efficiently.

---

## 7. Model Compression Principles

**Compression is optimization under constraints.**

Do not optimize only for:
- mAP
- FLOPs
- parameter count

Measure:
- real latency
- memory usage
- deployment behavior
- TensorRT engine performance

For pruning:
- verify structural validity
- verify export still works
- verify inference shape consistency

For quantization:
- verify numerical stability
- verify calibration correctness
- compare FP32 / FP16 / INT8 outputs

For knowledge distillation:
- clearly separate:
  - teacher logic
  - student logic
  - distillation losses

Do not entangle training code unnecessarily.

---

## 8. AI-Agent Collaboration Rules

**The repository should remain understandable by humans.**

AI-generated code tends to:
- overabstract
- overengineer
- duplicate logic
- hallucinate APIs

Actively resist this.

Prefer:
- transparent implementations
- explicit tensor operations
- local reasoning
- debuggable code paths

Do not generate architecture complexity without measurable benefit.

The goal is not "clever AI-generated code".

The goal is:
- maintainable research infrastructure
- reproducible experiments
- deployable systems

Code should be understandable by a competent engineer reading it for the first time.

---

## 9. Quantum Experimentation (Exploratory)

**Quantum optimization is experimental research, not production infrastructure.**

Keep quantum-related code:
- isolated
- modular
- optional

Do not contaminate:
- core training pipeline
- deployment pipeline
- inference runtime

Clearly distinguish:
- classical baselines
- quantum-inspired methods
- actual quantum execution

Benchmark against strong classical methods before claiming improvement.

Do not present exploratory quantum results as production-ready solutions.

---

## 10. Benchmark Before Belief

**Do not trust intuition over measurement.**

Always prefer:
- profiling
- benchmarks
- memory traces
- latency measurements
- ablation studies

Performance assumptions are often wrong.

Especially in:
- CUDA optimization
- TensorRT optimization
- pruning
- quantization
- data loading
- mixed precision

Measure first.
Optimize second.

---

## These guidelines are working if:

- diffs stay small and intentional
- deployment breaks less often
- experiments are reproducible
- exported models behave consistently
- benchmark tables are trustworthy
- AI agents introduce less architectural noise
- code remains understandable months later
- optimization claims are backed by measurements


## Project Overview

YOLOv8 model pruning project using channel pruning based on Batch Normalization gamma coefficients.

## Environment

- **Python 3.x** with PyTorch (torch==2.5.1, torchvision==0.20.1)
- Dependencies in `requirements.txt`: numpy, pyyaml, matplotlib, tqdm, opencv-python, requests, psutil, py-cpuinfo
- Uses a `.venv` virtual environment

## Key Commands

### Training Pipeline (4 steps)

1. **Normal training**: `python train-normal.py` - trains with `sr=0` (no sparsity)
2. **Sparsity training**: `python train-sparsity.py` - trains with `sr=1e-2` to enforce sparsity
3. **Prune**: `python prune.py --weights runs/train-sparsity/weights/last.pt --prune-ratio 0.5` - prunes model
4. **Fine-tune**: `python finetune.py` - fine-tunes pruned model

### Other Scripts

- `vis-bn-weight.py` - visualize BN gamma distribution
- `val.py` - validate model
- `export.py` - export model

## Important Notes

- Default dataset config: `datasets/data.yaml`
- Default model weights: `models/trainedv8s.pt`
- Output dir: `runs/` for training, `weights/` for pruned models
- Search for `===========` in codebase to find custom modifications
- Sparsity training disables AMP, scaler, and grad_clip_norm

## Pruning Details

- Pruning based on BN layer gamma coefficient threshold
- Skip pruning for Bottleneck modules with shortcuts (residual connections)
- Pruned modules: `ultralytics/nn/modules/block_pruned.py`, `head_pruned.py`, `tasks_pruned.py`
- Model sizes: n, s, m, l, x (default: s)

## Architecture

- Forked Ultralytics YOLOv8 with custom pruning modules
- Modified: `Conv-BN-Activation` modules, C2f, SPPF, Detect head