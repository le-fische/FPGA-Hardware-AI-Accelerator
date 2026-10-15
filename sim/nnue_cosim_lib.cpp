#include <iostream>
#include <vector>
#include <cstdint>
#include <unistd.h>
#include "Vnnue_top.h"
#include "verilated.h"

// Verilator global time callback
static uint64_t g_sim_time = 0;
double sc_time_stamp() {
    return (double)g_sim_time;
}

static Vnnue_top* g_top = nullptr;
static VerilatedContext* g_context = nullptr;

static void step_clock() {
    g_top->clk = 0;
    g_top->eval();
    g_sim_time += 5;
    
    g_top->clk = 1;
    g_top->eval();
    g_sim_time += 5;
}

extern "C" {

int nnue_cosim_init(const char* base_dir) {
    if (base_dir && base_dir[0] != '\0') {
        if (chdir(base_dir) != 0) {
            std::cerr << "[NNUE CoSim] Failed to chdir to: " << base_dir << std::endl;
        }
    }

    if (g_top != nullptr) {
        delete g_top;
        g_top = nullptr;
    }
    if (g_context != nullptr) {
        delete g_context;
        g_context = nullptr;
    }

    g_context = new VerilatedContext();
    g_top = new Vnnue_top(g_context);
    g_sim_time = 0;

    // Reset sequence
    g_top->clk = 0;
    g_top->rst_n = 0;
    g_top->start = 0;
    g_top->eval_mode = 0;
    g_top->swap_turn = 0;
    g_top->num_pieces = 0;
    g_top->piece_valid = 0;
    g_top->piece_remove = 0;
    g_top->piece_turn = 0;
    g_top->piece_opp = 0;

    for (int i = 0; i < 5; ++i) {
        step_clock();
    }

    g_top->rst_n = 1;
    step_clock();

    // Wait until ready
    int timeout = 1000;
    while (!g_top->ready && timeout-- > 0) {
        step_clock();
    }

    if (!g_top->ready) {
        std::cerr << "[NNUE CoSim] Reset timeout: accelerator not ready" << std::endl;
        return -1;
    }

    return 0;
}

int16_t nnue_cosim_eval_full(int num_pieces, const uint16_t* turn_feats, const uint16_t* opp_feats) {
    if (!g_top) return -9999;

    // Wait until ready
    int timeout = 50000;
    while (!g_top->ready && timeout-- > 0) {
        step_clock();
    }

    g_top->start = 1;
    g_top->eval_mode = 0;
    g_top->swap_turn = 0;
    g_top->num_pieces = (uint8_t)num_pieces;
    step_clock();
    g_top->start = 0;

    // Stream pieces
    for (int i = 0; i < num_pieces; ++i) {
        timeout = 50000;
        while (!g_top->piece_ready && timeout-- > 0) {
            step_clock();
        }

        g_top->piece_turn = turn_feats[i];
        g_top->piece_opp = opp_feats[i];
        g_top->piece_remove = 0;
        g_top->piece_valid = 1;
        step_clock();
        g_top->piece_valid = 0;
    }

    // Wait until done
    timeout = 100000;
    while (!g_top->done && timeout-- > 0) {
        step_clock();
    }

    int16_t score = (int16_t)g_top->score;
    // Step one more cycle so top returns to IDLE/READY
    step_clock();
    return score;
}

int16_t nnue_cosim_eval_diff(int num_diffs, int swap_turn, const uint16_t* turn_feats, const uint16_t* opp_feats, const uint8_t* is_remove) {
    if (!g_top) return -9999;

    int timeout = 50000;
    while (!g_top->ready && timeout-- > 0) {
        step_clock();
    }

    g_top->start = 1;
    g_top->eval_mode = 1;
    g_top->swap_turn = (uint8_t)(swap_turn ? 1 : 0);
    g_top->num_pieces = (uint8_t)num_diffs;
    step_clock();
    g_top->start = 0;

    for (int i = 0; i < num_diffs; ++i) {
        timeout = 50000;
        while (!g_top->piece_ready && timeout-- > 0) {
            step_clock();
        }

        g_top->piece_turn = turn_feats[i];
        g_top->piece_opp = opp_feats[i];
        g_top->piece_remove = is_remove[i];
        g_top->piece_valid = 1;
        step_clock();
        g_top->piece_valid = 0;
    }

    timeout = 100000;
    while (!g_top->done && timeout-- > 0) {
        step_clock();
    }

    int16_t score = (int16_t)g_top->score;
    step_clock();
    return score;
}

uint64_t nnue_cosim_get_cycles() {
    return g_sim_time / 10;
}

void nnue_cosim_destroy() {
    if (g_top) {
        delete g_top;
        g_top = nullptr;
    }
    if (g_context) {
        delete g_context;
        g_context = nullptr;
    }
}

}
