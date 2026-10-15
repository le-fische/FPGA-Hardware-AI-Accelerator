`ifndef NNUE_TOP_SV
`define NNUE_TOP_SV

import nnue_pkg::*;

module nnue_top #(
    parameter BIASES_ACC_MEM_FILE   = "mem/biases_acc.mem",
    parameter WEIGHTS_ACC_MEM_FILE  = "mem/weights_acc.mem",
    parameter BIASES_FC1_MEM_FILE   = "mem/biases_fc1.mem",
    parameter WEIGHTS_FC1_MEM_FILE  = "mem/weights_fc1.mem",
    parameter BIASES_FC2_MEM_FILE   = "mem/biases_fc2.mem",
    parameter WEIGHTS_FC2_MEM_FILE  = "mem/weights_fc2.mem"
)(
    input  logic                          clk,
    input  logic                          rst_n,
    
    // Command Interface
    input  logic                          start,         // Start new position evaluation
    input  logic                          eval_mode,     // 0 = Full board refresh, 1 = Differential move update
    input  logic                          swap_turn,     // 1 = Flip side-to-move perspective (differential mode)
    output logic                          ready,         // Accelerator is ready for new position
    output logic                          done,          // Evaluation complete strobe
    output logic signed [SCORE_WIDTH-1:0] score,         // Final evaluation in centipawns
    
    // Active Pieces / Move Diff Streaming Interface
    input  logic [5:0]                    num_pieces,    // Total pieces (full mode) or diff count (diff mode)
    input  logic                          piece_valid,   // Valid piece/diff input
    output logic                          piece_ready,   // Core ready to accept piece
    input  logic                          piece_remove,  // 0 = Add piece, 1 = Remove piece (differential mode)
    input  logic [FEATURE_ADDR_BITS-1:0]  piece_turn,    // Feature index for turn (0..767)
    input  logic [FEATURE_ADDR_BITS-1:0]  piece_opp      // Feature index for opp (0..767)
);

    // Internal Wires
    logic                          init_acc;
    logic                          acc_swap_turn;
    logic                          acc_feat_valid;
    logic                          acc_feat_remove;
    logic [FEATURE_ADDR_BITS-1:0]  acc_feat_turn;
    logic [FEATURE_ADDR_BITS-1:0]  acc_feat_opp;
    logic                          acc_feat_ready;
    logic                          acc_busy;
    
    logic [TOTAL_ACC_ADDR-1:0]     act_read_addr;
    logic [ACTIVATION_WIDTH-1:0]   act_read_data;
    
    logic                          start_fc1;
    logic                          fc1_done;
    /* verilator lint_off UNUSEDSIGNAL */
    logic                          fc1_busy;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [ACTIVATION_WIDTH-1:0]   fc1_h_out [0:FC1_OUT_DIM-1];
    
    logic                          start_fc2;
    logic                          fc2_done;
    logic signed [SCORE_WIDTH-1:0] fc2_score_out;

    // Submodule 1: Sparse Accumulator Engine
    nnue_acc_core #(
        .BIASES_MEM_FILE(BIASES_ACC_MEM_FILE),
        .WEIGHTS_MEM_FILE(WEIGHTS_ACC_MEM_FILE)
    ) u_acc_core (
        .clk(clk),
        .rst_n(rst_n),
        .init_acc(init_acc),
        .swap_turn(acc_swap_turn),
        .feat_valid(acc_feat_valid),
        .feat_remove(acc_feat_remove),
        .feat_turn(acc_feat_turn),
        .feat_opp(acc_feat_opp),
        .feat_ready(acc_feat_ready),
        .act_read_addr(act_read_addr),
        .act_read_data(act_read_data),
        .acc_busy(acc_busy)
    );

    // Submodule 2: Dense FC1 GEMM Engine (512 -> 32)
    nnue_fc1_engine #(
        .BIASES_FC1_MEM_FILE(BIASES_FC1_MEM_FILE),
        .WEIGHTS_FC1_MEM_FILE(WEIGHTS_FC1_MEM_FILE)
    ) u_fc1_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start_fc1(start_fc1),
        .fc1_done(fc1_done),
        .fc1_busy(fc1_busy),
        .act_read_addr(act_read_addr),
        .act_read_data(act_read_data),
        .h_out(fc1_h_out)
    );

    // Submodule 3: Dense FC2 Dot Product Engine (32 -> 1)
    nnue_fc2_engine #(
        .BIASES_FC2_MEM_FILE(BIASES_FC2_MEM_FILE),
        .WEIGHTS_FC2_MEM_FILE(WEIGHTS_FC2_MEM_FILE)
    ) u_fc2_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start_fc2(start_fc2),
        .fc2_done(fc2_done),
        .h_in(fc1_h_out),
        .score_out(fc2_score_out)
    );

    // Top-Level State Machine
    typedef enum logic [2:0] {
        ST_IDLE      = 3'b000,
        ST_INIT_ACC  = 3'b001,
        ST_ACC_LOAD  = 3'b010,
        ST_FC1_RUN   = 3'b011,
        ST_FC2_RUN   = 3'b100,
        ST_DONE      = 3'b101
    } top_state_t;

    top_state_t state;
    logic [5:0] pieces_loaded;
    logic [5:0] total_pieces_latched;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                <= ST_IDLE;
            init_acc             <= 1'b0;
            acc_feat_valid       <= 1'b0;
            acc_feat_remove      <= 1'b0;
            acc_feat_turn        <= '0;
            acc_feat_opp         <= '0;
            start_fc1            <= 1'b0;
            start_fc2            <= 1'b0;
            done                 <= 1'b0;
            score                <= '0;
            pieces_loaded        <= '0;
            total_pieces_latched <= '0;
        end else begin
            // Default strobe pulses
            init_acc       <= 1'b0;
            acc_swap_turn  <= 1'b0;
            acc_feat_valid <= 1'b0;
            start_fc1      <= 1'b0;
            start_fc2      <= 1'b0;
            done           <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        total_pieces_latched <= num_pieces;
                        pieces_loaded        <= '0;
                        if (eval_mode == 1'b0) begin
                            init_acc <= 1'b1;
                            state    <= ST_INIT_ACC;
                        end else begin
                            // Differential update: flip perspective if requested
                            acc_swap_turn <= swap_turn;
                            state         <= ST_ACC_LOAD;
                        end
                    end
                end

                ST_INIT_ACC: begin
                    // Wait for accumulator core to finish bias load
                    if (!acc_busy && !init_acc) begin
                        state <= ST_ACC_LOAD;
                    end
                end

                ST_ACC_LOAD: begin
                    // Stream in active pieces one by one
                    if (piece_valid && acc_feat_ready && !acc_feat_valid) begin
                        acc_feat_turn   <= piece_turn;
                        acc_feat_opp    <= piece_opp;
                        acc_feat_remove <= piece_remove;
                        acc_feat_valid  <= 1'b1;
                        pieces_loaded   <= pieces_loaded + 1'b1;
                    end else if (acc_feat_valid) begin
                        acc_feat_valid  <= 1'b0;
                    end

                    // Check if all pieces have been accumulated
                    if (pieces_loaded == total_pieces_latched && !acc_busy && !acc_feat_valid) begin
                        start_fc1 <= 1'b1;
                        state     <= ST_FC1_RUN;
                    end
                end

                ST_FC1_RUN: begin
                    if (fc1_done) begin
                        start_fc2 <= 1'b1;
                        state     <= ST_FC2_RUN;
                    end
                end

                ST_FC2_RUN: begin
                    if (fc2_done) begin
                        score <= fc2_score_out;
                        done  <= 1'b1;
                        state <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    assign ready       = (state == ST_IDLE);
    assign piece_ready = (state == ST_ACC_LOAD) && acc_feat_ready && !acc_feat_valid;

endmodule : nnue_top

`endif
