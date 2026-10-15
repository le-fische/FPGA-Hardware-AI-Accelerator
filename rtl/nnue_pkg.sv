`ifndef NNUE_PKG_SV
`define NNUE_PKG_SV

package nnue_pkg;

    // Architecture Dimensions (Model B: 768 -> 256 -> 32 -> 1)
    localparam int NUM_FEATURES      = 768;
    localparam int ACC_DIM           = 256;
    localparam int TOTAL_ACC_DIM     = 512; // 256 Turn + 256 Opponent
    localparam int FC1_OUT_DIM       = 32;
    localparam int MAX_ACTIVE_PIECES = 32;

    // Bit Widths
    localparam int FEATURE_ADDR_BITS = 10;  // 0..767
    localparam int ACC_ADDR_BITS     = 8;   // 0..255
    localparam int TOTAL_ACC_ADDR    = 9;   // 0..511
    localparam int FC1_ADDR_BITS     = 5;   // 0..31

    // Precision & Data Types
    localparam int ACC_WEIGHT_WIDTH  = 16;  // int16
    localparam int ACC_BIAS_WIDTH    = 16;  // int16
    localparam int ACC_VAL_WIDTH     = 16;  // int16 accumulator
    localparam int ACTIVATION_WIDTH  = 8;   // uint8 clamped [0, 127]

    localparam int FC1_WEIGHT_WIDTH  = 8;   // int8
    localparam int FC1_BIAS_WIDTH    = 32;  // int32
    localparam int FC1_ACC_WIDTH     = 32;  // int32 accumulator

    localparam int FC2_WEIGHT_WIDTH  = 8;   // int8
    localparam int FC2_BIAS_WIDTH    = 32;  // int32
    localparam int FC2_ACC_WIDTH     = 32;  // int32

    localparam int NUM_MATERIAL_BUCKETS = 4;
    localparam int BUCKET_ADDR_BITS     = 2;
    localparam int SCORE_WIDTH          = 16;  // int16 centipawns
    localparam int SHIFT_SCALE          = 6;   // >> 6

    // Activation Clipping Function: clamp(x, 0, 127)
    function automatic logic [7:0] clip_to_u7(input signed [31:0] val);
        if (val < 0)
            return 8'd0;
        else if (val > 32'sd127)
            return 8'd127;
        else
            return val[7:0];
    endfunction

    // SqrClippedReLU Activation: (clamp(x, 0, 127) * clamp(x, 0, 127)) >> 7
    function automatic logic [7:0] screlu_u7(input signed [31:0] val);
        logic [7:0] c;
        c = clip_to_u7(val);
        return 8'((16'(c) * 16'(c)) >> 7);
    endfunction

endpackage : nnue_pkg

`endif
