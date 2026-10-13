`ifndef NNUE_ACC_CORE_SV
`define NNUE_ACC_CORE_SV

import nnue_pkg::*;

module nnue_acc_core #(
    parameter BIASES_MEM_FILE  = "mem/biases_acc.mem",
    parameter WEIGHTS_MEM_FILE = "mem/weights_acc.mem"
)(
    input  logic                          clk,
    input  logic                          rst_n,
    
    // Control Handshake
    input  logic                          init_acc,      // Reload biases into accumulator
    input  logic                          feat_valid,    // Active piece valid
    input  logic [FEATURE_ADDR_BITS-1:0]  feat_turn,     // Feature index for turn (0..767)
    input  logic [FEATURE_ADDR_BITS-1:0]  feat_opp,      // Feature index for opp (0..767)
    output logic                          feat_ready,    // Ready for next piece
    
    // Output activation read port (for FC1 engine)
    input  logic [TOTAL_ACC_ADDR-1:0]     act_read_addr, // 0..511 (0..255: turn, 256..511: opp)
    output logic [ACTIVATION_WIDTH-1:0]   act_read_data, // Clamped value in [0, 127]
    
    output logic                          acc_busy
);

    // Initial Biases Memory: 256 x 16-bit
    logic signed [ACC_VAL_WIDTH-1:0] biases_mem [0:ACC_DIM-1];
    
    // Weights Memory: 196,608 x 16-bit (768 features x 256 units)
    logic signed [ACC_VAL_WIDTH-1:0] weights_mem [0:(NUM_FEATURES * ACC_DIM)-1];
    
    // Accumulator Registers: 256 for Turn, 256 for Opponent
    logic signed [ACC_VAL_WIDTH-1:0] acc_turn [0:ACC_DIM-1];
    logic signed [ACC_VAL_WIDTH-1:0] acc_opp  [0:ACC_DIM-1];
    
    // Initialize Memories from .mem files
    initial begin
        $readmemh(BIASES_MEM_FILE, biases_mem);
        $readmemh(WEIGHTS_MEM_FILE, weights_mem);
    end

    // FSM States
    typedef enum logic [1:0] {
        IDLE        = 2'b00,
        INIT_BIAS   = 2'b01,
        ACCUMULATE  = 2'b10
    } state_t;

    state_t state;
    logic [ACC_ADDR_BITS-1:0] unit_counter;
    logic [FEATURE_ADDR_BITS-1:0] active_turn_latched;
    logic [FEATURE_ADDR_BITS-1:0] active_opp_latched;

    // Weight BRAM addressing
    logic [17:0] w_turn_addr;
    logic [17:0] w_opp_addr;
    assign w_turn_addr = (active_turn_latched * ACC_DIM) + unit_counter;
    assign w_opp_addr  = (active_opp_latched  * ACC_DIM) + unit_counter;

    // FSM Controller
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state               <= IDLE;
            unit_counter        <= '0;
            active_turn_latched <= '0;
            active_opp_latched  <= '0;
        end else begin
            case (state)
                IDLE: begin
                    unit_counter <= '0;
                    if (init_acc) begin
                        state <= INIT_BIAS;
                    end else if (feat_valid) begin
                        active_turn_latched <= feat_turn;
                        active_opp_latched  <= feat_opp;
                        state               <= ACCUMULATE;
                    end
                end

                INIT_BIAS: begin
                    acc_turn[unit_counter] <= biases_mem[unit_counter];
                    acc_opp[unit_counter]  <= biases_mem[unit_counter];
                    unit_counter           <= unit_counter + 1'b1;
                    if (unit_counter == ACC_DIM - 1) begin
                        state <= IDLE;
                    end
                end

                ACCUMULATE: begin
                    acc_turn[unit_counter] <= acc_turn[unit_counter] + weights_mem[w_turn_addr];
                    acc_opp[unit_counter]  <= acc_opp[unit_counter]  + weights_mem[w_opp_addr];
                    unit_counter           <= unit_counter + 1'b1;
                    if (unit_counter == ACC_DIM - 1) begin
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign feat_ready = (state == IDLE) && !init_acc;
    assign acc_busy   = (state != IDLE) || init_acc;

    // Activation Readout with Clipped ReLU [0, 127]
    always_comb begin
        logic signed [ACC_VAL_WIDTH-1:0] s_acc;
        if (act_read_addr < ACC_DIM) begin
            s_acc = acc_turn[act_read_addr[ACC_ADDR_BITS-1:0]];
        end else begin
            s_acc = acc_opp[act_read_addr - ACC_DIM];
        end

        if (s_acc < 0)
            act_read_data = 8'd0;
        else if (s_acc > 16'sd127)
            act_read_data = 8'd127;
        else
            act_read_data = s_acc[ACTIVATION_WIDTH-1:0];
    end

endmodule : nnue_acc_core

`endif
