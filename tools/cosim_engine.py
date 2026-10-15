#!/usr/bin/env python3
"""
Hardware Co-Simulation Engine for Stockfish-style NNUE Model B.
Wraps the Verilator-compiled cycle-accurate SystemVerilog accelerator core (libnnue_cosim.dylib)
via Python ctypes, enabling bit-accurate and cycle-accurate hardware evaluation in Python chess search.
"""

import os
import sys
import ctypes
import json
import time
import chess
import numpy as np

class NNUEHardwareEngine:
    def __init__(self, lib_path=None, repo_dir=None):
        if repo_dir is None:
            repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.repo_dir = repo_dir

        if lib_path is None:
            lib_path = os.path.join(repo_dir, "sim", "libnnue_cosim.dylib")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Hardware co-simulation library not found at: {lib_path}\n"
                                    "Please build it using Verilator and clang++ first.")

        self.lib = ctypes.CDLL(lib_path)

        # Set up C function prototypes
        self.lib.nnue_cosim_init.argtypes = [ctypes.c_char_p]
        self.lib.nnue_cosim_init.restype = ctypes.c_int

        self.lib.nnue_cosim_eval_full.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16)
        ]
        self.lib.nnue_cosim_eval_full.restype = ctypes.c_int16

        self.lib.nnue_cosim_eval_diff.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint8)
        ]
        self.lib.nnue_cosim_eval_diff.restype = ctypes.c_int16

        self.lib.nnue_cosim_get_cycles.argtypes = []
        self.lib.nnue_cosim_get_cycles.restype = ctypes.c_uint64

        self.lib.nnue_cosim_destroy.argtypes = []
        self.lib.nnue_cosim_destroy.restype = None

        # Initialize hardware simulator with repository working directory for mem/ access
        res = self.lib.nnue_cosim_init(self.repo_dir.encode('utf-8'))
        if res != 0:
            raise RuntimeError(f"Failed to initialize NNUE hardware simulator (code {res})")

    @staticmethod
    def extract_features(board: chess.Board):
        """Extracts turn and opponent halfKP (768-dim) feature indices."""
        is_white = (board.turn == chess.WHITE)
        turn_indices = []
        opp_indices = []

        for sq, piece in board.piece_map().items():
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

        return turn_indices, opp_indices

    @staticmethod
    def piece_feature(piece: chess.Piece, sq: int, is_white_turn: bool):
        pt = piece.piece_type - 1
        c = 0 if piece.color == chess.WHITE else 1
        w_idx = c * 384 + pt * 64 + sq
        b_idx = (c ^ 1) * 384 + pt * 64 + (sq ^ 56)
        if is_white_turn:
            return w_idx, b_idx
        else:
            return b_idx, w_idx

    def eval_board(self, board: chess.Board) -> int:
        """Evaluates a chess position from scratch (full board evaluation mode)."""
        turn_feats, opp_feats = self.extract_features(board)
        n = len(turn_feats)
        c_turn = (ctypes.c_uint16 * n)(*turn_feats)
        c_opp = (ctypes.c_uint16 * n)(*opp_feats)
        return int(self.lib.nnue_cosim_eval_full(n, c_turn, c_opp))

    def eval_diff(self, num_diffs: int, swap_turn: bool, turn_feats, opp_feats, is_remove) -> int:
        """Evaluates an incremental move update (differential mode)."""
        c_turn = (ctypes.c_uint16 * num_diffs)(*turn_feats)
        c_opp = (ctypes.c_uint16 * num_diffs)(*opp_feats)
        c_rem = (ctypes.c_uint8 * num_diffs)(*is_remove)
        return int(self.lib.nnue_cosim_eval_diff(num_diffs, 1 if swap_turn else 0, c_turn, c_opp, c_rem))

    def eval_move(self, board_before: chess.Board, move: chess.Move) -> int:
        """
        Executes a differential move update:
        Identifies pieces moved/captured/promoted/castled, streams diffs to hardware,
        and returns the cycle-accurate hardware score for the child position.
        """
        piece_moved = board_before.piece_at(move.from_square)
        if piece_moved is None:
            # Fallback to full evaluation
            b_after = board_before.copy()
            b_after.push(move)
            return self.eval_board(b_after)

        turn_feats = []
        opp_feats = []
        is_remove = []

        # In differential mode with swap_turn, the hardware perspective has inverted.
        # Features must be presented from the perspective of the NEW side to move!
        new_is_white_turn = (not board_before.turn == chess.WHITE)

        # 1. Remove moving piece from origin
        tw, to = self.piece_feature(piece_moved, move.from_square, new_is_white_turn)
        turn_feats.append(tw)
        opp_feats.append(to)
        is_remove.append(1)

        # 2. Handle capture
        captured_piece = board_before.piece_at(move.to_square)
        if captured_piece is not None:
            cw, co = self.piece_feature(captured_piece, move.to_square, new_is_white_turn)
            turn_feats.append(cw)
            opp_feats.append(co)
            is_remove.append(1)
        elif board_before.is_en_passant(move):
            ep_sq = move.to_square - 8 if (board_before.turn == chess.WHITE) else move.to_square + 8
            ep_p = chess.Piece(chess.PAWN, not board_before.turn)
            cw, co = self.piece_feature(ep_p, ep_sq, new_is_white_turn)
            turn_feats.append(cw)
            opp_feats.append(co)
            is_remove.append(1)

        # 3. Handle castling rook move
        if board_before.is_castling(move):
            if move.to_square == chess.G1:
                r_from, r_to = chess.H1, chess.F1
            elif move.to_square == chess.C1:
                r_from, r_to = chess.A1, chess.D1
            elif move.to_square == chess.G8:
                r_from, r_to = chess.H8, chess.F8
            else: # C8
                r_from, r_to = chess.A8, chess.D8
            r_piece = chess.Piece(chess.ROOK, board_before.turn)
            rw1, ro1 = self.piece_feature(r_piece, r_from, new_is_white_turn)
            turn_feats.append(rw1)
            opp_feats.append(ro1)
            is_remove.append(1)
            rw2, ro2 = self.piece_feature(r_piece, r_to, new_is_white_turn)
            turn_feats.append(rw2)
            opp_feats.append(ro2)
            is_remove.append(0)

        # 4. Add moving piece (or promoted piece) to target square
        if move.promotion:
            target_piece = chess.Piece(move.promotion, board_before.turn)
        else:
            target_piece = piece_moved

        aw, ao = self.piece_feature(target_piece, move.to_square, new_is_white_turn)
        turn_feats.append(aw)
        opp_feats.append(ao)
        is_remove.append(0)

        num_diffs = len(turn_feats)
        # In chess, a move always swaps the side to move (swap_turn = True)
        return self.eval_diff(num_diffs, swap_turn=True,
                              turn_feats=turn_feats, opp_feats=opp_feats, is_remove=is_remove)

    def get_cycles(self) -> int:
        return int(self.lib.nnue_cosim_get_cycles())

    def close(self):
        if hasattr(self, 'lib') and self.lib is not None:
            self.lib.nnue_cosim_destroy()
            self.lib = None

    def __del__(self):
        self.close()


def run_self_test():
    print("=================================================================")
    print("   STOCKFISH NNUE HARDWARE ACCELERATOR CO-SIMULATION TEST        ")
    print("=================================================================")

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    engine = NNUEHardwareEngine(repo_dir=repo_dir)

    # 1. Test Golden Benchmark Positions
    json_path = os.path.join(repo_dir, "testbench", "golden_vectors.json")
    with open(json_path, "r") as f:
        vectors = json.load(f)

    print(f"\n[Phase 1] Verifying {len(vectors)} Golden Positions against Cycle-Accurate RTL...")
    all_passed = True
    start_cyc = engine.get_cycles()
    t0 = time.time()

    for vec in vectors:
        b = chess.Board(vec["fen"])
        exp_score = vec["score_cp"]
        act_score = engine.eval_board(b)

        if act_score == exp_score:
            print(f"  [PASS] {vec['name']:<25} | Hardware: {act_score:4d} cp | Expected: {exp_score:4d} cp")
        else:
            print(f"  [FAIL] {vec['name']:<25} | Hardware: {act_score:4d} cp | Expected: {exp_score:4d} cp")
            all_passed = False

    t1 = time.time()
    end_cyc = engine.get_cycles()
    total_cyc = end_cyc - start_cyc
    print(f"\nPhase 1 Result: {'ALL PASSED (100% BIT-ACCURATE MATCH)' if all_passed else 'FAILURES DETECTED'}")
    print(f"Elapsed Time: {(t1-t0)*1000:.2f} ms | Simulated Cycles: {total_cyc} ({total_cyc/len(vectors):.0f} cyc/eval)")

    # 2. Test Incremental Move Updates (Differential Mode)
    print("\n[Phase 2] Verifying Differential Move Updates (Incremental RTL Mode)...")
    b = chess.Board()
    full_start = engine.eval_board(b)

    # Move 1. e4
    m_e4 = chess.Move.from_uci("e2e4")
    diff_score = engine.eval_move(b, m_e4)

    # Verify against child board full evaluation
    b_child = chess.Board()
    b_child.push(m_e4)
    full_child_score = engine.eval_board(b_child)

    print(f"  Starting Position:              {full_start} cp")
    print(f"  Differential Update (1. e4):    {diff_score} cp")
    print(f"  Full Board Re-eval (1. e4):     {full_child_score} cp")

    if diff_score == full_child_score:
        print("  >>> [PASS] Incremental Move Update matched Full Board Re-eval bit-for-bit! <<<")
    else:
        print("  >>> [FAIL] Incremental Move Update mismatch! <<<")
        all_passed = False

    # 3. Test Multi-Move Game Sequence
    print("\n[Phase 3] Playing 10-move Game Sequence with Differential Hardware Updates...")
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6"]
    game_board = chess.Board()
    engine.eval_board(game_board) # Initial seed

    game_passed = True
    for idx, uci_str in enumerate(moves):
        mv = chess.Move.from_uci(uci_str)
        diff_eval = engine.eval_move(game_board, mv)
        
        # Ground truth
        game_board.push(mv)
        ref_eval = engine.eval_board(game_board)

        match = (diff_eval == ref_eval)
        if not match:
            game_passed = False
        print(f"  Move {idx+1:2d} ({uci_str:<4}): Diff = {diff_eval:4d} cp | Ref = {ref_eval:4d} cp | {'MATCH' if match else 'MISMATCH'}")

    print(f"\nPhase 3 Result: {'ALL 10 MOVES MATCHED PERFECTLY' if game_passed else 'MISMATCH DETECTED'}")

    engine.close()
    if all_passed and game_passed:
        print("\n=================================================================")
        print("   ALL VERILATOR HARDWARE CO-SIMULATION TESTS PASSED (100%)      ")
        print("=================================================================")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(run_self_test())
