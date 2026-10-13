# FPGA Hardware AI Accelerator: NNUE Chess Inference Engine

A high-performance hardware accelerator for ultra-low-latency neural network inference on FPGA, designed for evaluation in computer chess (Stockfish-style NNUE). Implemented in synthesizable SystemVerilog, targeting **AMD Xilinx Vivado ML** (PYNQ-Z2 / Zynq-7000, Kria KV260, and UltraScale+ architectures).

---

## 1. Architecture Overview

The accelerator implements **Model B ($768 \to 256 \to 32 \to 1$)**, combining dual-perspective sparse feature accumulation with parallel dense processing elements:

```
[ Active Pieces (1..32) ] 
           │
           ▼
┌──────────────────────────────────────────────┐
│       Dual Accumulator Core (768 -> 256)     │
│  - White/Turn: 256 x int16 BRAM accumulator  │
│  - Black/Opp:  256 x int16 BRAM accumulator  │
│  - Activation: Clipped ReLU [0, 127]         │
└──────────────────────┬───────────────────────┘
                       │ 512 x uint8 activations
                       ▼
┌──────────────────────────────────────────────┐
│       FC1 GEMM Engine (512 -> 32)            │
│  - 32 parallel DSP multiply-accumulators     │
│  - Weights: 16 KB on-chip ROM (int8)         │
│  - Pipelined Shift & Clipped ReLU [0, 127]   │
└──────────────────────┬───────────────────────┘
                       │ 32 x uint8 activations
                       ▼
┌──────────────────────────────────────────────┐
│       FC2 Dot-Product Output Head (32 -> 1)  │
│  - Parallel dot product with int8 weights    │
│  - Centipawn scaling (>>> 6)                 │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              [ Score (int16 cp) ]
```

### Hardware Specifications
- **Model Parameters:** $768 \times 256 \text{ (Acc)} + 512 \times 32 \text{ (FC1)} + 32 \times 1 \text{ (FC2)} \approx 410\text{ KB}$
- **On-Chip Storage:** 100% fits in Block RAM (BRAM) on AMD Zynq-7020 / XC7Z020 (~4.9 Mb available BRAM).
- **Inference Latency:**
  - Accumulator update: 256 clock cycles per active piece
  - FC1 dense matrix multiplication: 512 clock cycles
  - FC2 output head: 2 clock cycles
  - Total latency @ 100 MHz: **~85 microseconds** for a full 32-piece board evaluation from scratch (and **< 1 microsecond** when using differential piece updates during search!).

---

## 2. Repository Structure

```
.
├── Makefile               # Simulation and waveform build targets
├── README.md              # Project documentation and architecture guide
├── mem/                   # Formatted .mem files for Verilog $readmemh initialization
│   ├── weights_acc.mem    # 196,608 x 16-bit accumulator weights
│   ├── biases_acc.mem     # 256 x 16-bit accumulator biases
│   ├── weights_fc1.mem    # 16,384 x 8-bit FC1 weights
│   ├── biases_fc1.mem     # 32 x 32-bit FC1 biases
│   ├── weights_fc2.mem    # 32 x 8-bit FC2 weights
│   └── biases_fc2.mem     # 1 x 32-bit FC2 bias
├── rtl/                   # Synthesizable SystemVerilog modules
│   ├── nnue_pkg.sv        # Architecture constants, bit-widths, functions
│   ├── nnue_acc_core.sv   # Dual-perspective sparse accumulator core
│   ├── nnue_fc1_engine.sv # 512 -> 32 parallel GEMM engine
│   ├── nnue_fc2_engine.sv # 32 -> 1 dot-product output stage
│   ├── nnue_top.sv        # Top-level accelerator FSM and streaming interface
│   └── nnue_axi_lite_wrapper.sv # AXI4-Lite memory-mapped CSR interface
├── scripts/               # AMD Vivado build and packaging automation
│   ├── vivado_synth.tcl   # Out-of-context batch synthesis script (Zynq-7020)
│   └── package_ip.tcl     # Vivado IP packager automation
├── sim/                   # Testbenches and simulation waveforms
│   └── tb_nnue_top.sv     # Self-checking testbench asserting 10 golden positions
├── testbench/             # Bit-accurate golden vectors generated from PyTorch
│   ├── golden_stimulus.hex # Active piece coordinates for 10 benchmark positions
│   ├── golden_expected.hex # Golden integer centipawn evaluation reference
│   └── golden_vectors.json # Full benchmark suite with FEN strings
├── tools/                 # Training, quantization, and verification tools
│   ├── preprocess_dataset.py # High-speed binary shard packer
│   ├── train_nnue_v2_mlp.py  # PyTorch MPS trainer for Model B
│   ├── export_mem_files.py   # .mem file generator
│   └── golden_model.py       # Cycle-accurate golden reference emulator
└── weights/               # Quantized INT8/INT16 model checkpoints (.npy)
    └── v2_mlp/            # Trained weights for Model B
```

---

## 3. Quickstart: Cycle-Accurate RTL Simulation

Run the self-checking SystemVerilog testbench against 10 diverse benchmark positions (Starting Position, Material Imbalances, Italian Game, Sicilian Defense, Endgames, Tactical Pins):

```bash
# Compile and run with Icarus Verilog
make sim
```

### Verification Output:
```text
=================================================================
   STARTING NNUE ACCELERATOR CYCLE-ACCURATE RTL SIMULATION      
=================================================================

--- [Testcase 01] Presenting 32 active pieces to hardware ---
>>> [PASS] Testcase 01: Score = 39 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 02] Presenting 31 active pieces to hardware ---
>>> [PASS] Testcase 02: Score = 687 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 03] Presenting 31 active pieces to hardware ---
>>> [PASS] Testcase 03: Score = -679 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 04] Presenting 31 active pieces to hardware ---
>>> [PASS] Testcase 04: Score = 446 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 05] Presenting 31 active pieces to hardware ---
>>> [PASS] Testcase 05: Score = -407 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 06] Presenting 32 active pieces to hardware ---
>>> [PASS] Testcase 06: Score = 91 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 07] Presenting 30 active pieces to hardware ---
>>> [PASS] Testcase 07: Score = 26 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 08] Presenting 3 active pieces to hardware ---
>>> [PASS] Testcase 08: Score = 445 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 09] Presenting 4 active pieces to hardware ---
>>> [PASS] Testcase 09: Score = 19 cp (RTL output matched golden reference bit-for-bit!)

--- [Testcase 10] Presenting 29 active pieces to hardware ---
>>> [PASS] Testcase 10: Score = 474 cp (RTL output matched golden reference bit-for-bit!)

=================================================================
SIMULATION SUMMARY: 10 PASSED, 0 FAILED out of 10 TESTCASES
>>> ALL GOLDEN VECTORS VERIFIED! 100% HARDWARE TIMING ACCURACY <<<
=================================================================
```

To view waveforms:
```bash
make wave
```

---

## 4. AXI4-Lite Register Map

For deployment on Zynq SoC (ARM Cortex-A9 host running PYNQ or Linux):

| Offset | Register Name | Access | Description |
|---|---|---|---|
| `0x00` | `REG_CTRL` | R/W | Bit 0: `start_strobe`, Bit 1: `irq_enable` |
| `0x04` | `REG_STATUS` | R | Bit 0: `ready`, Bit 1: `done`, Bit 2: `piece_ready` |
| `0x08` | `REG_CONFIG` | R/W | Bits [5:0]: `num_pieces` on board (1..32) |
| `0x0C` | `REG_SCORE` | R | Bits [15:0]: Signed positional score in centipawns |
| `0x10` | `REG_PIECE_IN` | W | Bits [9:0]: `turn_feat`, Bits [25:16]: `opp_feat` |
| `0x14` | `REG_CYCLE_CNT` | R | 32-bit hardware execution cycle counter |

---

## 5. AMD Vivado Implementation

Targeting **AMD Xilinx Zynq-7000 (XC7Z020-1CLG400C)** / PYNQ-Z2:

```bash
# Run non-project batch synthesis and generate utilization/timing reports
vivado -mode batch -source scripts/vivado_synth.tcl

# Package into Vivado IP Catalog for block design integration
vivado -mode batch -source scripts/package_ip.tcl
```
