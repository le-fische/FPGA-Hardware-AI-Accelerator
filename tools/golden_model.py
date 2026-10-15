import os
import json
import chess
import numpy as np

def generate_golden_vectors():
    weights_dir = "weights/v2_mlp"
    tb_dir = "testbench"
    os.makedirs(tb_dir, exist_ok=True)
    
    print("Loading quantized weights...")
    w_acc = np.load(os.path.join(weights_dir, "weights_acc.npy"))  # (768, 256) int16
    b_acc = np.load(os.path.join(weights_dir, "biases_acc.npy"))   # (256,) int16
    w_fc1 = np.load(os.path.join(weights_dir, "weights_fc1.npy"))  # (32, 512) int8
    b_fc1 = np.load(os.path.join(weights_dir, "biases_fc1.npy"))   # (32,) int32
    w_fc2 = np.load(os.path.join(weights_dir, "weights_fc2.npy"))  # (4, 32) int8
    b_fc2 = np.load(os.path.join(weights_dir, "biases_fc2.npy"))   # (4,) int32
    if w_fc2.shape[0] == 1 or len(w_fc2.shape) == 1:
        w_fc2 = np.tile(w_fc2.reshape(1, 32), (4, 1))
    if b_fc2.size == 1:
        b_fc2 = np.tile(b_fc2.reshape(1), 4)
    
    test_cases = [
        {"name": "Starting Position", "fen": chess.STARTING_FEN},
        {"name": "White +Q", "fen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        {"name": "Black +Q", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"},
        {"name": "White +R", "fen": "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"},
        {"name": "Black +R", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w KQkq - 0 1"},
        {"name": "Italian Game", "fen": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"},
        {"name": "Sicilian Defense", "fen": "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6"},
        {"name": "King + Rook vs King", "fen": "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"},
        {"name": "Equal Pawn Endgame", "fen": "8/8/4k3/4p3/4P3/4K3/8/8 w - - 0 1"},
        {"name": "Tactical Pin Position", "fen": "r1b1k2r/pppp1ppp/2n2q2/8/1b1NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 1 7"}
    ]
    
    vectors_json = []
    
    stim_file = open(os.path.join(tb_dir, "golden_stimulus.hex"), "w")
    exp_file = open(os.path.join(tb_dir, "golden_expected.hex"), "w")
    
    print(f"Generating golden vectors for {len(test_cases)} benchmark positions...")
    
    for tc_idx, tc in enumerate(test_cases):
        b = chess.Board(tc["fen"])
        is_white = (b.turn == chess.WHITE)
        
        # Extract piece features
        turn_indices = []
        opp_indices = []
        
        for sq, piece in b.piece_map().items():
            pt = piece.piece_type - 1
            c = 0 if piece.color == chess.WHITE else 1
            w_idx = c * 384 + pt * 64 + sq
            b_idx = (c ^ 1) * 384 + pt * 64 + (sq ^ 56)
            
            if is_white:
                turn_indices.append(w_idx)
                opp_indices.append(b_idx)
            else:
                turn_indices.append(b_idx)
                opp_indices.append(w_idx)
                
        n_pieces = len(turn_indices)
        
        # 1. Hardware Stage 1: Accumulator
        acc_turn = b_acc.astype(np.int32).copy()
        acc_opp = b_acc.astype(np.int32).copy()
        for idx in turn_indices:
            acc_turn += w_acc[idx, :]
        for idx in opp_indices:
            acc_opp += w_acc[idx, :]
            
        clamped_turn = np.clip(acc_turn, 0, 127).astype(np.int32)
        clamped_opp = np.clip(acc_opp, 0, 127).astype(np.int32)
        acc_vec = np.concatenate([clamped_turn, clamped_opp]) # 512 bytes
        
        # 2. Hardware Stage 2: FC1 Matrix Multiply (512 -> 32) with Hybrid SCReLU
        fc1_sums = b_fc1.astype(np.int64) + np.dot(w_fc1.astype(np.int64), acc_vec.astype(np.int64))
        scaled_fc1 = fc1_sums >> 6
        c1 = np.clip(scaled_fc1, 0, 127).astype(np.int32)
        h_lin = c1[:16]
        h_sq = np.clip((c1[16:] * c1[16:]) >> 7, 0, 127).astype(np.int32)
        h = np.concatenate([h_lin, h_sq]) # 32 bytes
        
        # 3. Hardware Stage 3: FC2 Dot Product with 4 Material Buckets
        if n_pieces <= 16:
            bucket = 0
        elif n_pieces <= 24:
            bucket = 1
        elif n_pieces <= 28:
            bucket = 2
        else:
            bucket = 3
        fc2_sum = b_fc2.flatten()[bucket].astype(np.int64) + np.dot(w_fc2.reshape(4, 32)[bucket].astype(np.int64), h.astype(np.int64))
        score_cp = int(fc2_sum >> 6) # 16-bit signed integer
        
        # Format Stimulus for Verilog:
        # Line 1: Number of active pieces (e.g. 0x20 = 32)
        # Next lines: piece indices for turn and opp
        stim_file.write(f"// Testcase {tc_idx+1}: {tc['name']}\n")
        stim_file.write(f"{n_pieces:02x}\n")
        for i in range(n_pieces):
            stim_file.write(f"{turn_indices[i]:03x} {opp_indices[i]:03x}\n")
            
        # Format Expected Output for Verilog:
        # Score in 16-bit hex
        score_16b = score_cp & 0xFFFF
        exp_file.write(f"// Testcase {tc_idx+1}: {tc['name']} ({score_cp} cp)\n")
        exp_file.write(f"{score_16b:04x}\n")
        
        vectors_json.append({
            "id": tc_idx + 1,
            "name": tc["name"],
            "fen": tc["fen"],
            "num_pieces": n_pieces,
            "turn_indices": [int(x) for x in turn_indices],
            "opp_indices": [int(x) for x in opp_indices],
            "acc_turn_sample": [int(x) for x in clamped_turn[:8]],
            "acc_opp_sample": [int(x) for x in clamped_opp[:8]],
            "fc1_h_sample": [int(x) for x in h[:8]],
            "score_cp": score_cp,
            "score_hex": f"0x{score_16b:04x}"
        })
        
        print(f"  [{tc_idx+1:02d}] {tc['name']:<24}: Score = {score_cp:>5d} cp ({score_16b:04x}h), Pieces = {n_pieces}")
        
    # Testcase 11: 1. e4 (Differential Move Update)
    b11 = chess.Board()
    b11.push(chess.Move.from_uci("e2e4"))
    turn_indices_11 = []
    opp_indices_11 = []
    for sq, piece in b11.piece_map().items():
        pt = piece.piece_type - 1
        c = 0 if piece.color == chess.WHITE else 1
        w_idx = c * 384 + pt * 64 + sq
        b_idx = (c ^ 1) * 384 + pt * 64 + (sq ^ 56)
        turn_indices_11.append(b_idx)
        opp_indices_11.append(w_idx)
        
    acc_turn_11 = b_acc.astype(np.int32).copy()
    acc_opp_11 = b_acc.astype(np.int32).copy()
    for idx in turn_indices_11: acc_turn_11 += w_acc[idx, :]
    for idx in opp_indices_11: acc_opp_11 += w_acc[idx, :]
    clamped_turn_11 = np.clip(acc_turn_11, 0, 127).astype(np.int32)
    clamped_opp_11 = np.clip(acc_opp_11, 0, 127).astype(np.int32)
    acc_vec_11 = np.concatenate([clamped_turn_11, clamped_opp_11])
    fc1_sums_11 = b_fc1.astype(np.int64) + np.dot(w_fc1.astype(np.int64), acc_vec_11.astype(np.int64))
    scaled_fc1_11 = fc1_sums_11 >> 6
    c1_11 = np.clip(scaled_fc1_11, 0, 127).astype(np.int32)
    h_11 = np.concatenate([c1_11[:16], np.clip((c1_11[16:] * c1_11[16:]) >> 7, 0, 127).astype(np.int32)])
    fc2_sum_11 = b_fc2.flatten()[3].astype(np.int64) + np.dot(w_fc2.reshape(4, 32)[3].astype(np.int64), h_11.astype(np.int64))
    score_cp_11 = int(fc2_sum_11 >> 6)
    score_16b_11 = score_cp_11 & 0xFFFF
    exp_file.write(f"// Testcase 11: 1. e4 ({score_cp_11} cp)\n")
    exp_file.write(f"{score_16b_11:04x}\n")

    stim_file.close()
    exp_file.close()
    
    with open(os.path.join(tb_dir, "golden_vectors.json"), "w") as f:
        json.dump(vectors_json, f, indent=2)
        
    print(f"\nGenerated {len(test_cases)} golden test vectors successfully!")
    print(f"  - {tb_dir}/golden_stimulus.hex")
    print(f"  - {tb_dir}/golden_expected.hex")
    print(f"  - {tb_dir}/golden_vectors.json")

if __name__ == "__main__":
    generate_golden_vectors()
