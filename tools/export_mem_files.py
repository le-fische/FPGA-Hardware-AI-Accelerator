import os
import numpy as np

def export_mem():
    weights_dir = "weights/v2_mlp"
    mem_dir = "mem"
    os.makedirs(mem_dir, exist_ok=True)
    
    print(f"Loading weights from {weights_dir}...")
    w_acc = np.load(os.path.join(weights_dir, "weights_acc.npy"))  # (768, 256) int16
    b_acc = np.load(os.path.join(weights_dir, "biases_acc.npy"))   # (256,) int16
    w_fc1 = np.load(os.path.join(weights_dir, "weights_fc1.npy"))  # (32, 512) int8
    b_fc1 = np.load(os.path.join(weights_dir, "biases_fc1.npy"))   # (32,) int32
    w_fc2 = np.load(os.path.join(weights_dir, "weights_fc2.npy"))  # (1, 32) int8
    b_fc2 = np.load(os.path.join(weights_dir, "biases_fc2.npy"))   # (1,) int32
    
    # 1. Accumulator Biases: 256 entries of 16-bit
    with open(os.path.join(mem_dir, "biases_acc.mem"), "w") as f:
        for val in b_acc:
            f.write(f"{int(val) & 0xFFFF:04x}\n")
    print("Exported mem/biases_acc.mem (256 words)")
    
    # 2. Accumulator Weights: 768 x 256 = 196,608 entries of 16-bit
    # Address: feature_idx * 256 + unit_idx
    with open(os.path.join(mem_dir, "weights_acc.mem"), "w") as f:
        for val in w_acc.flatten():
            f.write(f"{int(val) & 0xFFFF:04x}\n")
    print(f"Exported mem/weights_acc.mem ({w_acc.size:,} words)")
    
    # 3. FC1 Weights: 32 x 512 = 16,384 entries of 8-bit
    # Address: output_channel * 512 + input_channel
    with open(os.path.join(mem_dir, "weights_fc1.mem"), "w") as f:
        for val in w_fc1.flatten():
            f.write(f"{int(val) & 0xFF:02x}\n")
    print(f"Exported mem/weights_fc1.mem ({w_fc1.size:,} bytes)")
    
    # 4. FC1 Biases: 32 entries of 32-bit
    with open(os.path.join(mem_dir, "biases_fc1.mem"), "w") as f:
        for val in b_fc1:
            f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
    print("Exported mem/biases_fc1.mem (32 words)")
    
    # 5. FC2 Weights: 32 entries of 8-bit
    with open(os.path.join(mem_dir, "weights_fc2.mem"), "w") as f:
        for val in w_fc2.flatten():
            f.write(f"{int(val) & 0xFF:02x}\n")
    print(f"Exported mem/weights_fc2.mem ({w_fc2.size:,} bytes)")
    
    # 6. FC2 Bias: 1 entry of 32-bit
    with open(os.path.join(mem_dir, "biases_fc2.mem"), "w") as f:
        f.write(f"{int(b_fc2[0]) & 0xFFFFFFFF:08x}\n")
    print("Exported mem/biases_fc2.mem (1 word)")
    
    print("\nAll Verilog memory files successfully generated in mem/!")

if __name__ == "__main__":
    export_mem()
