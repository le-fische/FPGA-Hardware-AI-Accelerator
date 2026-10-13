# FPGA Hardware AI Accelerator: NNUE Chess Inference Engine

A hardware accelerator for low-latency neural network inference on FPGA, designed for high-performance evaluation in computer chess (Stockfish-style NNUE).

## Architecture Overview

The accelerator implements **Model B ($768 \to 256 \to 32 \to 1$)**, a non-linear neural network that combines sparse feature accumulation with a dense matrix-multiplication processing engine:

1. **Stage 1 (Sparse Accumulator Engine):**
   * Input: 1 to 32 active piece-square features per perspective (indices $0..767$).
   * Computes two perspective accumulators: Side-to-Move and Opponent.
   * Internal Memory: $768 \times 256 \times 16\text{-bit}$ Block RAM (BRAM).
   * Activation: Clipped ReLU: $\text{clamp}(x, 0, 127)$ $\implies$ 512 bytes.

2. **Stage 2 (Dense FC1 Processing Elements):**
   * Computes a $512 \to 32$ fully connected matrix multiplication.
   * Weights: $32 \times 512$ 8-bit signed integers (16 KB on-chip ROM).
   * Biases: $32 \times 32\text{-bit}$ signed integers.
   * Activation: Pipelined arithmetic shift and Clipped ReLU: $\text{clamp}((x + b) \gg 6, 0, 127)$ $\implies$ 32 bytes.

3. **Stage 3 (Output Head):**
   * Dot product of 32 hidden activations with 8-bit output weights.
   * Output: Final positional evaluation in centipawns (16-bit signed integer).

---

## Repository Structure

```
.
├── mem/               # Formatted .mem files for Verilog $readmemh BRAM initialization
│   ├── weights_acc.mem
│   ├── biases_acc.mem
│   ├── weights_fc1.mem
│   ├── biases_fc1.mem
│   ├── weights_fc2.mem
│   └── biases_fc2.mem
├── rtl/               # Synthesizable Verilog / SystemVerilog RTL modules
├── sim/               # Testbenches and simulation scripts (ModelSim / Verilator)
├── testbench/         # Golden test vectors for cycle-accurate assertion
│   ├── golden_stimulus.hex
│   ├── golden_expected.hex
│   └── golden_vectors.json
├── tools/             # Training, quantization, and memory generator utilities
│   ├── preprocess_dataset.py
│   ├── train_nnue_mps.py
│   ├── train_nnue_v2_mlp.py
│   ├── export_mem_files.py
│   └── golden_model.py
└── weights/           # Quantized INT8/INT16 model checkpoints (.npy)
    ├── v1_linear/     # Baseline 768 -> 256 -> 1
    └── v2_mlp/        # Accelerator Model B (768 -> 256 -> 32 -> 1)
```

---

## Quickstart

### 1. Exporting Memory Initialization Files
```bash
python3 tools/export_mem_files.py
```

### 2. Running the Bit-True Golden Model
```bash
python3 tools/golden_model.py
```
Outputs cycle-accurate intermediate activations and final centipawn scores across 10 benchmark positions into `testbench/`.

### 3. Training / Retraining with PyTorch MPS
```bash
# Preprocess dataset shards into fast packed tensor cache
python3 tools/preprocess_dataset.py

# Train Model B on Apple Silicon GPU (Metal / MPS)
python3 tools/train_nnue_v2_mlp.py
```
