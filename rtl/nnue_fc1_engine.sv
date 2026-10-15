`ifndef NNUE_FC1_ENGINE_SV
`define NNUE_FC1_ENGINE_SV

import nnue_pkg::*;

module nnue_fc1_engine #(
    parameter BIASES_FC1_MEM_FILE  = "mem/biases_fc1.mem",
    parameter WEIGHTS_FC1_MEM_FILE = "mem/weights_fc1.mem"
)(
    input  logic                          clk,
    input  logic                          rst_n,
    
    // Control Handshake
    input  logic                          start_fc1,
    output logic                          fc1_done,
    output logic                          fc1_busy,
    
    // Accumulator Activation Read Interface
    output logic [TOTAL_ACC_ADDR-1:0]     act_read_addr,
    input  logic [ACTIVATION_WIDTH-1:0]   act_read_data,
    
    // Output 32 Hidden Activations [0, 127]
    output logic [ACTIVATION_WIDTH-1:0]   h_out [0:FC1_OUT_DIM-1]
);

    // FC1 Biases Memory: 32 x 32-bit
    logic signed [FC1_BIAS_WIDTH-1:0] biases_fc1_mem [0:FC1_OUT_DIM-1];
    
    // FC1 Weights Memory: 32 outputs x 512 inputs = 16,384 x 8-bit
    // Address: (output_idx * 512) + input_idx
    logic signed [FC1_WEIGHT_WIDTH-1:0] weights_fc1_mem [0:(FC1_OUT_DIM * TOTAL_ACC_DIM)-1];
    
    initial begin
        $readmemh(BIASES_FC1_MEM_FILE, biases_fc1_mem);
        $readmemh(WEIGHTS_FC1_MEM_FILE, weights_fc1_mem);
    end

    // 32 Parallel Accumulators
    logic signed [FC1_ACC_WIDTH-1:0] acc_fc1 [0:FC1_OUT_DIM-1];

    typedef enum logic [1:0] {
        IDLE     = 2'b00,
        INIT     = 2'b01,
        COMPUTE  = 2'b10,
        FINISH   = 2'b11
    } state_t;

    state_t state;
    logic [TOTAL_ACC_ADDR-1:0] in_counter;

    assign act_read_addr = in_counter;

    // Parallel GEMM Multiply-Accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            in_counter <= '0;
            fc1_done   <= 1'b0;
            for (int j = 0; j < FC1_OUT_DIM; j++) begin
                acc_fc1[j] <= '0;
                h_out[j]   <= '0;
            end
        end else begin
            case (state)
                IDLE: begin
                    fc1_done   <= 1'b0;
                    in_counter <= '0;
                    if (start_fc1) begin
                        state <= INIT;
                    end
                end

                INIT: begin
                    // Load initial 32-bit biases
                    for (int j = 0; j < FC1_OUT_DIM; j++) begin
                        acc_fc1[j] <= biases_fc1_mem[j];
                    end
                    in_counter <= '0;
                    state      <= COMPUTE;
                end

                COMPUTE: begin
                    // Multiply-accumulate for all 32 neurons in parallel
                    for (int j = 0; j < FC1_OUT_DIM; j++) begin
                        logic [13:0] w_addr;
                        logic signed [FC1_WEIGHT_WIDTH-1:0] w_val;
                        logic signed [15:0] prod;
                        
                        w_addr = 14'(14'(j) * 14'(TOTAL_ACC_DIM) + 14'(in_counter));
                        w_val  = weights_fc1_mem[w_addr];
                        // act_read_data is unsigned [0, 127], w_val is signed 8-bit
                        prod   = $signed({1'b0, act_read_data}) * w_val;
                        acc_fc1[j] <= acc_fc1[j] + 32'(signed'(prod));
                    end

                    in_counter <= in_counter + 1'b1;
                    if (in_counter == 9'(TOTAL_ACC_DIM - 1)) begin
                        state <= FINISH;
                    end
                end

                FINISH: begin
                    // Arithmetic shift right by 6 and clamp to [0, 127]
                    for (int j = 0; j < FC1_OUT_DIM; j++) begin
                        logic signed [FC1_ACC_WIDTH-1:0] scaled_val;
                        scaled_val = acc_fc1[j] >>> SHIFT_SCALE;
                        h_out[j]   <= clip_to_u7(scaled_val);
                    end
                    fc1_done <= 1'b1;
                    state    <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign fc1_busy = (state != IDLE);

endmodule : nnue_fc1_engine

`endif
