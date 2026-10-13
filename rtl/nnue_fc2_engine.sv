`ifndef NNUE_FC2_ENGINE_SV
`define NNUE_FC2_ENGINE_SV

import nnue_pkg::*;

module nnue_fc2_engine #(
    parameter BIASES_FC2_MEM_FILE  = "mem/biases_fc2.mem",
    parameter WEIGHTS_FC2_MEM_FILE = "mem/weights_fc2.mem"
)(
    input  logic                          clk,
    input  logic                          rst_n,
    
    // Control Handshake
    input  logic                          start_fc2,
    output logic                          fc2_done,
    
    // Hidden Activations from FC1 (32 x 8-bit)
    input  logic [ACTIVATION_WIDTH-1:0]   h_in [0:FC1_OUT_DIM-1],
    
    // Final Centipawn Output
    output logic signed [SCORE_WIDTH-1:0] score_out
);

    // FC2 Bias: 1 x 32-bit
    logic signed [FC2_BIAS_WIDTH-1:0]   bias_fc2_mem [0:0];
    
    // FC2 Weights: 32 x 8-bit
    logic signed [FC2_WEIGHT_WIDTH-1:0] weights_fc2_mem [0:FC1_OUT_DIM-1];
    
    initial begin
        $readmemh(BIASES_FC2_MEM_FILE, bias_fc2_mem);
        $readmemh(WEIGHTS_FC2_MEM_FILE, weights_fc2_mem);
    end

    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        COMPUTE = 2'b01,
        FINISH  = 2'b10
    } state_t;

    state_t state;
    logic signed [FC2_ACC_WIDTH-1:0] total_sum;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            fc2_done  <= 1'b0;
            score_out <= '0;
            total_sum <= '0;
        end else begin
            case (state)
                IDLE: begin
                    fc2_done <= 1'b0;
                    if (start_fc2) begin
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // Combinational sum across 32 elements added to bias
                    logic signed [FC2_ACC_WIDTH-1:0] partial_sum;
                    partial_sum = bias_fc2_mem[0];
                    for (int j = 0; j < FC1_OUT_DIM; j++) begin
                        logic signed [15:0] prod;
                        prod = $signed({1'b0, h_in[j]}) * weights_fc2_mem[j];
                        partial_sum = partial_sum + prod;
                    end
                    total_sum <= partial_sum;
                    state     <= FINISH;
                end

                FINISH: begin
                    // Shift right by 6 for centipawn scale
                    score_out <= total_sum >>> SHIFT_SCALE;
                    fc2_done  <= 1'b1;
                    state     <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule : nnue_fc2_engine

`endif
