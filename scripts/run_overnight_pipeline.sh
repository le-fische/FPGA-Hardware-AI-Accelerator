#!/bin/bash
# ==============================================================================
# Master Overnight Training Pipeline
# 1. Download & Pack 50 Million Stockfish-Evaluated Positions (Depth >= 12)
# 2. Train Model B (768 -> 256 -> 32 -> 1) on Apple M4 Pro MPS GPU (5 Epochs)
# 3. Quantize Weights to INT8/INT16 for FPGA Block RAM
# 4. Generate Verilog .mem Files & Run Cycle-Accurate SystemVerilog RTL Simulation
# ==============================================================================

set -e

PROJECT_DIR="/Users/le/Documents/Projects/The Silicon Project- FPGA Hardware AI Accelerator"
PYTHON_BIN="/Users/le/Documents/Projects/chessathon/project/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/runs"
LOG_FILE="$LOG_DIR/overnight_training.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

echo "==================================================================" | tee -a "$LOG_FILE"
echo "  STARTING OVERNIGHT ACCELERATOR TRAINING PIPELINE                " | tee -a "$LOG_FILE"
echo "  Date: $(date)                                                   " | tee -a "$LOG_FILE"
echo "  Log File: $LOG_FILE                                             " | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"

# Step 1: Download & Preprocess Dataset
echo "" | tee -a "$LOG_FILE"
echo "[STEP 1/3] Downloading & Packing High-Depth Stockfish Dataset..." | tee -a "$LOG_FILE"
"$PYTHON_BIN" tools/download_stockfish_dataset.py 2>&1 | tee -a "$LOG_FILE"

# Step 2: Run Multi-Shard PyTorch MPS Training
echo "" | tee -a "$LOG_FILE"
echo "[STEP 2/3] Training NNUE Model B on Apple M4 Pro GPU (MPS)..." | tee -a "$LOG_FILE"
"$PYTHON_BIN" tools/train_nnue_large.py 2>&1 | tee -a "$LOG_FILE"

# Step 3: Run Cycle-Accurate SystemVerilog RTL Verification
echo "" | tee -a "$LOG_FILE"
echo "[STEP 3/3] Running Cycle-Accurate RTL Verification (make sim)..." | tee -a "$LOG_FILE"
make sim 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"
echo "  OVERNIGHT PIPELINE COMPLETED SUCCESSFULLY!                      " | tee -a "$LOG_FILE"
echo "  Finished at: $(date)                                            " | tee -a "$LOG_FILE"
echo "==================================================================" | tee -a "$LOG_FILE"
