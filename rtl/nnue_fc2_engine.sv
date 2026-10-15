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
    
    // Material Bucket Selection (0=Endgame, 1=Late Mid, 2=Mid, 3=Opening)
    input  logic [BUCKET_ADDR_BITS-1:0]   material_bucket,
    
    // Final Centipawn Output
    output logic signed [SCORE_WIDTH-1:0] score_out
);

    // FC2 Biases: 4 x 32-bit (one per material bucket)
    logic signed [FC2_BIAS_WIDTH-1:0]   bias_fc2_mem [0:NUM_MATERIAL_BUCKETS-1];
    
    // FC2 Weights: 4 buckets x 32 weights = 128 x 8-bit
    logic signed [FC2_WEIGHT_WIDTH-1:0] weights_fc2_mem [0:(NUM_MATERIAL_BUCKETS*FC1_OUT_DIM)-1];
    
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
                    // Combinational sum across 32 elements added to bucket bias
                    logic signed [FC2_ACC_WIDTH-1:0] partial_sum;
                    partial_sum = bias_fc2_mem[material_bucket];
                    for (int j = 0; j < FC1_OUT_DIM; j++) begin
                        logic [6:0] w_idx;
                        logic signed [15:0] prod;
                        w_idx = {material_bucket, 5'(j)};
                        prod = $signed({1'b0, h_in[j]}) * weights_fc2_mem[w_idx];
                        partial_sum = partial_sum + 32'(signed'(prod));
                    end
                    total_sum <= partial_sum;
                    state     <= FINISH;
                end

                FINISH: begin
                    // Shift right by 6 for centipawn scale
                    score_out <= 16'(total_sum >>> SHIFT_SCALE);
                    fc2_done  <= 1'b1;
                    state     <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule : nnue_fc2_engine

`endif
