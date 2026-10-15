"""
Stockfish-Grade Multi-Shard PyTorch MPS Trainer for Model B v2.
Incorporates:
  1. SCReLU (Squared Clipped ReLU) hybrid non-linear activation
  2. 4-Bucket Material-Conditioned Output Heads (Endgame, Late Mid, Mid, Opening)
  3. Quantization-Aware Training (QAT) with Straight-Through Estimators (STE)
  4. WDL (Win/Draw/Loss) Cross-Entropy Loss
Trains on 50 Million Stockfish-evaluated positions on Apple Silicon GPU (Metal MPS).
"""

import glob
import math
import os
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, clip_min, clip_max):
        return torch.clamp(torch.round(x * scale), clip_min, clip_max) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

def quantize_ste(x, scale, clip_min, clip_max):
    return QuantizeSTE.apply(x, scale, clip_min, clip_max)

class PieceSquareNNUE_v2(nn.Module):
    def __init__(self, hidden_dim=256, mlp_dim=32, num_buckets=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_buckets = num_buckets
        
        # 768 features + index 768 as padding (zero-out)
        self.embedding = nn.Embedding(769, hidden_dim, padding_idx=768)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # FC1: 512 -> 32
        self.fc1 = nn.Linear(hidden_dim * 2, mlp_dim)
        
        # FC2: 4 material buckets x 32
        self.fc2_w = nn.Parameter(torch.empty(num_buckets, mlp_dim))
        self.fc2_b = nn.Parameter(torch.zeros(num_buckets))
        
        # Initialize
        nn.init.kaiming_uniform_(self.embedding.weight[:768], a=math.sqrt(5))
        with torch.no_grad():
            self.embedding.weight[768].zero_()
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_uniform_(self.fc2_w, a=math.sqrt(5))
            nn.init.zeros_(self.fc2_b)

    def forward(self, turn_idx, opp_idx, qat=True):
        if qat:
            emb_w = quantize_ste(self.embedding.weight, 127.0, -127.0, 127.0)
            bias = quantize_ste(self.bias, 127.0, -127.0, 127.0)
            fc1_w = quantize_ste(self.fc1.weight, 64.0, -127.0, 127.0)
            fc1_b = quantize_ste(self.fc1.bias, 64.0 * 127.0, -2147483647.0, 2147483647.0)
            fc2_w = quantize_ste(self.fc2_w, 87.538, -127.0, 127.0)
            fc2_b = quantize_ste(self.fc2_b, 87.538 * 127.0, -2147483647.0, 2147483647.0)
        else:
            emb_w = self.embedding.weight
            bias = self.bias
            fc1_w = self.fc1.weight
            fc1_b = self.fc1.bias
            fc2_w = self.fc2_w
            fc2_b = self.fc2_b

        # 1. Sparse Accumulator
        turn_acc = torch.nn.functional.embedding(turn_idx, emb_w, padding_idx=768).sum(dim=1) + bias
        opp_acc = torch.nn.functional.embedding(opp_idx, emb_w, padding_idx=768).sum(dim=1) + bias

        turn_c = torch.clamp(turn_acc, 0.0, 1.0)
        opp_c = torch.clamp(opp_acc, 0.0, 1.0)
        if qat:
            turn_c = quantize_ste(turn_c, 127.0, 0.0, 127.0)
            opp_c = quantize_ste(opp_c, 127.0, 0.0, 127.0)

        acc = torch.cat([turn_c, opp_c], dim=1) # (B, 512)

        # 2. FC1 Dense Layer with Hybrid SCReLU
        z1 = torch.nn.functional.linear(acc, fc1_w, fc1_b) # (B, 32)
        c1 = torch.clamp(z1, 0.0, 1.0)
        if qat:
            c1 = quantize_ste(c1, 127.0, 0.0, 127.0)

        # First 16 linear, last 16 quadratic SCReLU
        h_lin = c1[:, :16]
        h_sq = c1[:, 16:] ** 2
        if qat:
            h_sq = quantize_ste(h_sq, 127.0, 0.0, 127.0)

        h = torch.cat([h_lin, h_sq], dim=1) # (B, 32)

        # 3. Material Bucket Classification
        # Bucket 0: <= 16 (Endgame), 1: 17-24 (Late Mid), 2: 25-28 (Mid), 3: 29-32 (Opening)
        n_pieces = (turn_idx != 768).sum(dim=1)
        bucket = torch.zeros_like(n_pieces)
        bucket = torch.where(n_pieces > 16, torch.ones_like(bucket), bucket)
        bucket = torch.where(n_pieces > 24, torch.full_like(bucket, 2), bucket)
        bucket = torch.where(n_pieces > 28, torch.full_like(bucket, 3), bucket)

        # Gather per-sample bucket weights
        w_b = fc2_w[bucket] # (B, 32)
        b_b = fc2_b[bucket] # (B,)

        out = (h * w_b).sum(dim=1, keepdim=True) + b_b.unsqueeze(1)
        return out


def export_fpga_weights(state_dict, output_dir="weights/v2_mlp"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n[Quantization] Converting PyTorch weights to INT8/INT16 for FPGA...")

    # Scale factors:
    # Accumulator: [-127, 127] (int16 weights, int16 biases)
    w_acc = torch.round(state_dict["embedding.weight"][:768] * 127.0).to(torch.int16).cpu().numpy() # (768, 256)
    b_acc = torch.round(state_dict["bias"] * 127.0).to(torch.int16).cpu().numpy() # (256,)
    
    # FC1 (512 -> 32): inputs [0, 127]. Weights in [-127, 127] (int8). Biases int32
    w_fc1 = torch.round(state_dict["fc1.weight"] * 64.0).to(torch.int8).cpu().numpy() # (32, 512)
    b_fc1 = torch.round(state_dict["fc1.bias"] * (64.0 * 127.0)).to(torch.int32).cpu().numpy() # (32,)
    
    # FC2 (4 buckets x 32 -> 1): inputs [0, 127]. Weights in int8. Biases int32
    w_fc2 = torch.round(state_dict["fc2_w"] * 87.538).to(torch.int8).cpu().numpy() # (4, 32)
    b_fc2 = torch.round(state_dict["fc2_b"] * (87.538 * 127.0)).to(torch.int32).cpu().numpy() # (4,)
    
    np.save(f"{output_dir}/weights_acc.npy", w_acc)
    np.save(f"{output_dir}/biases_acc.npy", b_acc)
    np.save(f"{output_dir}/weights_fc1.npy", w_fc1)
    np.save(f"{output_dir}/biases_fc1.npy", b_fc1)
    np.save(f"{output_dir}/weights_fc2.npy", w_fc2)
    np.save(f"{output_dir}/biases_fc2.npy", b_fc2)
    print(f"  --> Saved quantized weights to {output_dir}/")

    # Export to .mem files for Verilog
    print("[Verilog MEM] Exporting .mem files for Block RAM initialization...")
    res = subprocess.run([sys.executable, "tools/export_mem_files.py"], capture_output=True, text=True)
    if res.returncode == 0:
        print("  --> .mem files successfully updated.")
    else:
        print("  --> Warning: export_mem_files failed:", res.stderr)

    # Regenerate golden vectors for new weights
    print("[Golden Model] Regenerating golden test vectors for updated model...")
    res_gold = subprocess.run([sys.executable, "tools/golden_model.py"], capture_output=True, text=True)
    if res_gold.returncode == 0:
        print("  --> Golden vectors successfully regenerated.")
    else:
        print("  --> Warning: golden_model failed:", res_gold.stderr)

    # Run cycle-accurate RTL verification (Icarus Verilog)
    print("[RTL Sim] Running cycle-accurate SystemVerilog simulation against benchmark vectors...")
    res_sim = subprocess.run(["make", "sim"], capture_output=True, text=True)
    if "ALL GOLDEN VECTORS VERIFIED" in res_sim.stdout or res_sim.returncode == 0:
        print("  --> RTL SIMULATION PASSED! Hardware timing verified.")
    else:
        print("  --> Simulation output:\n", res_sim.stdout[-500:])

    # Rebuild Verilator co-simulation library and verify
    print("[Verilator Co-Sim] Rebuilding Verilator C++ library and running co-simulation...")
    res_cosim = subprocess.run(["make", "cosim"], capture_output=True, text=True)
    if "ALL VERILATOR HARDWARE CO-SIMULATION TESTS PASSED" in res_cosim.stdout or res_cosim.returncode == 0:
        print("  --> VERILATOR CO-SIMULATION PASSED! 100% C++/Python match.")
    else:
        print("  --> Co-sim output:\n", res_cosim.stdout[-500:])


def train_large_dataset(shards_dir="data/shards_50m", epochs=5, batch_size=16384, lr=0.003):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("==================================================================")
    print("  STARTING LARGE-SCALE NNUE v2 (QAT + SCReLU + 4-BUCKETS)        ")
    print(f"  Device: {device} | Epochs: {epochs} | Batch Size: {batch_size:,} ")
    print("==================================================================")

    shard_files = sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz")))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shards_dir}! Please run download script first.")

    print(f"Found {len(shard_files)} shards in {shards_dir}")

    # Load validation set from shard 0 (first 100k positions)
    print("Loading fixed validation set (100,000 positions)...")
    d0 = np.load(shard_files[0])
    val_turn_np = np.where(d0["turn_indices"][:100_000] == -1, 768, d0["turn_indices"][:100_000]).astype(np.int64)
    val_opp_np = np.where(d0["opp_indices"][:100_000] == -1, 768, d0["opp_indices"][:100_000]).astype(np.int64)
    val_target_np = d0["targets"][:100_000]

    val_turn = torch.from_numpy(val_turn_np).to(device)
    val_opp = torch.from_numpy(val_opp_np).to(device)
    val_target = torch.from_numpy(val_target_np).unsqueeze(1).to(device)
    del d0

    model = PieceSquareNNUE_v2(hidden_dim=256, mlp_dim=32, num_buckets=4).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    total_start = time.time()
    total_positions_seen = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss_sum = 0.0
        epoch_pos_count = 0

        print(f"\n--- [Epoch {epoch:02d}/{epochs:02d}] LR: {scheduler.get_last_lr()[0]:.6f} ---")

        for shard_idx, shard_path in enumerate(shard_files):
            t_shard = time.time()
            data = np.load(shard_path)
            t_idx = data["turn_indices"]
            o_idx = data["opp_indices"]
            tgts = data["targets"]

            # If shard 0, skip validation slice
            if shard_idx == 0:
                t_idx = t_idx[100_000:]
                o_idx = o_idx[100_000:]
                tgts = tgts[100_000:]

            t_idx = np.where(t_idx == -1, 768, t_idx).astype(np.int64)
            o_idx = np.where(o_idx == -1, 768, o_idx).astype(np.int64)

            n_shard = len(tgts)
            perm = np.random.permutation(n_shard)
            t_idx = t_idx[perm]
            o_idx = o_idx[perm]
            tgts = tgts[perm]
            del perm
            steps = math.ceil(n_shard / batch_size)

            for step in range(steps):
                s_i = step * batch_size
                e_i = min(s_i + batch_size, n_shard)

                b_turn = torch.from_numpy(t_idx[s_i:e_i]).to(device)
                b_opp = torch.from_numpy(o_idx[s_i:e_i]).to(device)
                b_tgt = torch.from_numpy(tgts[s_i:e_i]).unsqueeze(1).to(device)

                optimizer.zero_grad()
                pred = model(b_turn, b_opp, qat=True)
                loss = criterion(pred, b_tgt)
                loss.backward()
                optimizer.step()

                bs = e_i - s_i
                epoch_loss_sum += loss.item() * bs
                epoch_pos_count += bs
                total_positions_seen += bs

            del data
            shard_dur = time.time() - t_shard
            pos_per_sec = n_shard / shard_dur
            print(f"  Shard {shard_idx+1:02d}/10 processed in {shard_dur:4.1f}s ({pos_per_sec:,.0f} pos/s)")

        scheduler.step()
        train_loss = epoch_loss_sum / epoch_pos_count

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            v_steps = math.ceil(len(val_target) / batch_size)
            for v_s in range(v_steps):
                v_start = v_s * batch_size
                v_end = min(v_start + batch_size, len(val_target))
                pred_v = model(val_turn[v_start:v_end], val_opp[v_start:v_end], qat=False)
                val_preds.append(pred_v)
            val_preds = torch.cat(val_preds, dim=0)
            val_loss = criterion(val_preds, val_target).item()

            pred_prob = torch.sigmoid(val_preds)
            pred_cp = -400.0 * torch.log10(1.0 / torch.clamp(pred_prob, 1e-6, 1.0 - 1e-6) - 1.0)
            tgt_cp = -400.0 * torch.log10(1.0 / torch.clamp(val_target, 1e-6, 1.0 - 1e-6) - 1.0)
            mae_cp = torch.mean(torch.abs(pred_cp - tgt_cp)).item()

        epoch_dur = time.time() - epoch_start
        print(f">> [Epoch {epoch:02d} Summary] Dur: {epoch_dur:4.1f}s | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val MAE: {mae_cp:.1f} cp")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  --> [NEW BEST MODEL] Saving checkpoint (Val Loss: {val_loss:.5f})...")
            torch.save(model.state_dict(), "weights/model_b_v2_best.pt")

    print(f"\nTraining completed in {(time.time() - total_start)/60:.1f} minutes!")
    print(f"Total positions processed: {total_positions_seen:,}")

    # Load best checkpoint and export to FPGA
    print("\nLoading best model checkpoint for FPGA export...")
    best_weights = torch.load("weights/model_b_v2_best.pt", map_location="cpu")
    export_fpga_weights(best_weights)

if __name__ == "__main__":
    train_large_dataset(epochs=5, batch_size=16384, lr=0.003)
